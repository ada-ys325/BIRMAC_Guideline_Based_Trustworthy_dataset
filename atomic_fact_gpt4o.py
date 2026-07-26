#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenAI Batch atomic fact extraction for cited medical answers.

Input:
  data/answers_gpt4o_full_cited_atomic_ready.jsonl

Output:
  data/answers_gpt4o_full_atomic_facts.jsonl

Commands:
  build
  build_split
  submit
  run_split
  status
  status_from_tsv
  consume
  consume_from_tsv
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# Regex / schema
# ============================================================

CUSTOM_ID_RE = re.compile(r"^arow-(\d+)$")

SENT_WITH_CIT_RE = re.compile(
    r"(.+?(?:[.!?]|[。！？])\s*(?:\[\d+\]){1,3})(?=\s+|$)",
    flags=re.DOTALL,
)

CIT_RE = re.compile(r"\[(\d+)\]")


ATOMIC_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "sentence_id": {"type": "integer"},
                    "claims": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["sentence_id", "claims"],
            },
        },
    },
    "required": ["sentences"],
}


SYSTEM_PROMPT = """You extract citation-grounded medical atomic facts from cited patient-facing medical answers.

You must extract ONLY factual medical propositions that are intended to be supported by the cited passages for that sentence.

Do NOT add outside medical knowledge.
Do NOT strengthen uncertainty.
Do NOT remove important uncertainty words such as may, might, can, should, generally, conditional, low certainty, or very low certainty.
Do NOT convert patient-specific wording into unsupported general medical rules.
Do NOT extract empathy, reassurance, discourse, transition phrases, or rhetorical framing.

Return valid JSON only."""


# ============================================================
# Utils
# ============================================================

def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def norm_ws(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip())


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} is not object: {path}")
            rows.append(obj)
    return rows


def ensure_nonempty_file(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{label} not created: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"{label} created but empty: {path}")


def get_client():
    from openai import OpenAI
    return OpenAI()


def extract_citation_ids(text: str) -> List[int]:
    out = []
    for x in CIT_RE.findall(text or ""):
        try:
            n = int(x)
        except Exception:
            continue
        if n > 0 and n not in out:
            out.append(n)
    return out


def strip_sentence_citations(text: str) -> str:
    return re.sub(r"\s*(?:\[\d+\])+\s*$", "", norm_ws(text)).strip()


def normalize_sentence_id(x: Any, fallback_i: int) -> int:
    if isinstance(x, int):
        return x
    s = norm_ws(x)
    m = re.search(r"\d+", s)
    if m:
        return int(m.group(0))
    return fallback_i


def get_cited_sentences(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Prefer structured answer_sentences from answer_batch.py.
    Fallback to regex split over answer/answer_text.
    """
    out = []

    answer_sentences = entry.get("answer_sentences", [])
    if isinstance(answer_sentences, list) and answer_sentences:
        for i, s in enumerate(answer_sentences, start=1):
            if not isinstance(s, dict):
                continue

            raw_text = norm_ws(s.get("text"))
            if not raw_text:
                continue

            text_cites = extract_citation_ids(raw_text)

            field_cites = []
            if isinstance(s.get("citations"), list):
                for c in s.get("citations"):
                    try:
                        field_cites.append(int(c))
                    except Exception:
                        pass

            citation_ids = text_cites or field_cites
            citation_ids = [c for c in citation_ids if c > 0]
            if not citation_ids:
                continue

            out.append({
                "sentence_id": normalize_sentence_id(s.get("sentence_id"), i),
                "raw_sentence": raw_text,
                "sentence_text": strip_sentence_citations(raw_text),
                "citation_ids": citation_ids[:3],
                "role": norm_ws(s.get("role")),
            })

    if out:
        out = sorted(out, key=lambda x: x["sentence_id"])
        for i, s in enumerate(out, start=1):
            s["sentence_id"] = i
        return out

    answer = norm_ws(entry.get("answer") or entry.get("answer_text"))
    if not answer:
        return []

    matches = [m.group(1).strip() for m in SENT_WITH_CIT_RE.finditer(answer)]

    if not matches:
        cites = extract_citation_ids(answer)
        clean = strip_sentence_citations(answer)
        if clean and cites:
            return [{
                "sentence_id": 1,
                "raw_sentence": answer,
                "sentence_text": clean,
                "citation_ids": cites[:3],
                "role": "",
            }]
        return []

    for i, seg in enumerate(matches, start=1):
        cites = extract_citation_ids(seg)
        clean = strip_sentence_citations(seg)
        if clean and cites:
            out.append({
                "sentence_id": i,
                "raw_sentence": norm_ws(seg),
                "sentence_text": clean,
                "citation_ids": cites[:3],
                "role": "",
            })

    return out


def normalize_passages(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    passages = entry.get("passages", [])
    out = []

    if isinstance(passages, list) and passages:
        for i, p in enumerate(passages, start=1):
            if not isinstance(p, dict):
                continue
            try:
                pid = int(p.get("pid", i))
            except Exception:
                pid = i
            obj = dict(p)
            obj["pid"] = pid
            obj["text"] = p.get("text", "")
            out.append(obj)
        return out

    top = entry.get("top_passages", [])
    if isinstance(top, list):
        for i, p in enumerate(top, start=1):
            if not isinstance(p, dict):
                continue
            out.append({
                "pid": i,
                "text": p.get("text", ""),
                "passage_id": p.get("passage_id", ""),
                "guideline_id": p.get("guideline_id", ""),
                "main_topic": p.get("main_topic", ""),
                "section": p.get("section", ""),
                "rerank_score": p.get("rerank_score"),
                "faiss_rank": p.get("faiss_rank"),
                "faiss_score": p.get("faiss_score"),
                "source_pdf": p.get("source_pdf"),
            })

    return out


def build_atomic_prompt(sentences: List[Dict[str, Any]]) -> str:
    blocks = []
    for s in sentences:
        blocks.append(
            f"SENTENCE_ID: {s['sentence_id']}\n"
            f"ROLE: {s.get('role', '')}\n"
            f"TEXT: {s['sentence_text']}\n"
            f"CITATIONS: {s['citation_ids']}\n"
        )

    joined = "\n".join(blocks)

    return f"""
# TASK

Extract citation-grounded medical atomic facts from cited answer sentences.

# GOAL

Extract only minimal factual medical propositions intended to be directly supported by the cited passages.

# EXTRACT A CLAIM ONLY IF BOTH ARE TRUE

1. It is a medical, clinical, guideline, screening, safety, treatment, diagnosis, monitoring, evidence-quality, shared-decision-making, or health-equity factual proposition.
2. It should be directly supportable from the cited passage IDs for that sentence.

# EXTRACTABLE CLAIM TYPES

- definition or property of a disease, intervention, medicine, procedure, test, screening program, or care process
- recommendation, conditional recommendation, contraindication, indication, eligibility rule, timing rule, or monitoring rule
- benefit, harm, side effect, risk, safety issue, uncertainty, evidence quality, or evidence limitation
- diagnostic or screening criteria
- explicitly stated shared-decision-making or patient-preference principles
- explicitly stated health-equity or communication principles
- explicit statements that the cited passages do not specify something

# DO NOT EXTRACT

- empathy or reassurance
- rhetorical framing
- transition phrases
- vague statements such as "this is important" or "this is the safest answer"
- generic patient advice unless the sentence states a citation-supported care action
- patient-specific assumptions that are not stated as guideline facts
- conclusions stronger than the sentence wording

# ATOMICITY RULES

- Split combined factual content into minimal standalone claims.
- Keep each claim concise and self-contained.
- Preserve important qualifiers and uncertainty words.
- Keep claims close to the original sentence wording.
- Do not merge multiple medical facts into one claim.
- Do not invent thresholds, timelines, risks, mechanisms, or eligibility criteria.
- If a sentence has no eligible claim, return an empty claims list.

# EXAMPLES

Sentence:
"MMR is a live vaccine and guidelines suggest against giving it to people on immunosuppressive therapy."
Extract:
- "MMR is a live vaccine."
- "Guidelines suggest against giving MMR to people on immunosuppressive therapy."

Sentence:
"Adding tacrolimus to steroids may improve lung function and muscle strength in dermatomyositis, but the evidence is very low quality."
Extract:
- "Adding tacrolimus to steroids may improve lung function in dermatomyositis."
- "Adding tacrolimus to steroids may improve muscle strength in dermatomyositis."
- "The evidence for adding tacrolimus to steroids in dermatomyositis is very low quality."

Sentence:
"Different patients may make different choices, and that's okay."
Extract:
- "Different patients may make different choices when treatment decisions depend on patient values and preferences."

Sentence:
"This is an important safety question."
Extract:
- []

# OUTPUT

Return STRICT JSON only:

{{
  "sentences": [
    {{
      "sentence_id": 1,
      "claims": ["claim 1", "claim 2"]
    }}
  ]
}}

# SENTENCES

{joined}
""".strip()


def build_request(
    entry: Dict[str, Any],
    sample_idx: int,
    model: str,
    max_output_tokens: int,
    temperature: float,
) -> Optional[Dict[str, Any]]:
    sentences = get_cited_sentences(entry)

    if not sentences:
        return None

    body: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_atomic_prompt(sentences)},
        ],
        "max_output_tokens": int(max_output_tokens),
        "temperature": float(temperature),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "atomic_medical_claims",
                "strict": True,
                "schema": ATOMIC_SCHEMA,
            }
        },
    }

    return {
        "custom_id": f"arow-{sample_idx}",
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def extract_output_text_from_batch_line(obj: Dict[str, Any]) -> str:
    body = obj.get("response", {}).get("body", {})

    if isinstance(body.get("output_text"), str) and body["output_text"].strip():
        return body["output_text"].strip()

    output = body.get("output", [])
    texts = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                t = c.get("text", "")
                if c.get("type") in ("output_text", "text") and isinstance(t, str) and t:
                    texts.append(t)

    text = "\n".join(texts).strip()
    if not text:
        incomplete = body.get("incomplete_details")
        if incomplete:
            raise ValueError(f"Empty model output. incomplete_details={incomplete}")
        raise ValueError("Empty model output in batch result.")
    return text


def parse_json_object(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Empty output")

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj

    raise ValueError(f"Could not parse JSON object:\n{raw[:1000]}")


def build_atomic_output(entry: Dict[str, Any], raw_claims: Dict[str, Any]) -> Dict[str, Any]:
    cited_sents = get_cited_sentences(entry)
    passages = normalize_passages(entry)

    claims_by_sid: Dict[int, List[str]] = {}
    for item in raw_claims.get("sentences", []):
        if not isinstance(item, dict):
            continue
        try:
            sid = int(item.get("sentence_id"))
        except Exception:
            continue
        claims_raw = item.get("claims", [])
        if isinstance(claims_raw, list):
            claims_by_sid[sid] = [norm_ws(c) for c in claims_raw if norm_ws(c)]
        else:
            claims_by_sid[sid] = []

    passage_by_pid = {}
    for p in passages:
        if not isinstance(p, dict):
            continue
        try:
            pid = int(p.get("pid"))
        except Exception:
            continue
        passage_by_pid[pid] = p

    sentences_out = []
    atomic_facts_flat = []

    for s in cited_sents:
        sid = int(s["sentence_id"])
        citation_ids = [int(x) for x in s.get("citation_ids", []) if int(x) > 0]
        claims = claims_by_sid.get(sid, [])
        cited_passages = [passage_by_pid[pid] for pid in citation_ids if pid in passage_by_pid]

        atomic_items = []
        for fact_idx, fact in enumerate(claims, start=1):
            item = {
                "atomic_fact_id": f"s{sid}_a{fact_idx}",
                "fact": fact,
                "source_sentence_id": sid,
                "source_sentence_text": s["sentence_text"],
                "source_sentence_raw": s["raw_sentence"],
                "source_sentence_role": s.get("role", ""),
                "citation_ids": citation_ids,
                "cited_passages": cited_passages,
            }
            atomic_items.append(item)
            atomic_facts_flat.append(item)

        sentences_out.append({
            "sentence_id": sid,
            "raw_sentence": s["raw_sentence"],
            "sentence_text": s["sentence_text"],
            "sentence_role": s.get("role", ""),
            "citation_ids": citation_ids,
            "atomic_facts": atomic_items,
        })

    out = dict(entry)
    out["atomic_model"] = ""
    out["sentence_atomic_decomposition"] = sentences_out
    out["atomic_facts_flat"] = atomic_facts_flat
    out["num_sentences_detected"] = len(sentences_out)
    out["num_atomic_facts"] = len(atomic_facts_flat)
    out["atomic_error"] = ""
    return out


# ============================================================
# Batch helpers
# ============================================================

def submit_one_jsonl(input_jsonl: Path, model: str, completion_window: str, metadata_name: str) -> Dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    if completion_window != "24h":
        raise ValueError("Batch completion_window currently supports '24h' only.")
    if not input_jsonl.exists():
        raise FileNotFoundError(f"input_jsonl not found: {input_jsonl}")

    client = get_client()

    with input_jsonl.open("rb") as f:
        up = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=up.id,
        endpoint="/v1/responses",
        completion_window=completion_window,
        metadata={
            "name": metadata_name,
            "task": "atomic_fact_extraction",
            "model": model,
        },
    )

    return {
        "input_jsonl": str(input_jsonl),
        "input_file_id": up.id,
        "batch_id": batch.id,
        "status": batch.status,
        "endpoint": "/v1/responses",
        "completion_window": completion_window,
        "model": model,
    }


# ============================================================
# Commands
# ============================================================

def cmd_build(args: argparse.Namespace) -> None:
    src = Path(args.src).expanduser()
    out_jsonl = Path(args.out_jsonl).expanduser()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(src)
    start_idx = int(args.start_idx)
    end_idx = len(records) if int(args.end_idx) < 0 else min(int(args.end_idx), len(records))

    written = 0
    skipped = 0

    with out_jsonl.open("w", encoding="utf-8") as fout:
        for sample_idx in range(start_idx, end_idx):
            entry = records[sample_idx]
            req = build_request(
                entry=entry,
                sample_idx=sample_idx,
                model=args.openai_model,
                max_output_tokens=int(args.max_output_tokens),
                temperature=float(args.temperature),
            )
            if req is None:
                skipped += 1
                continue
            fout.write(json.dumps(req, ensure_ascii=False) + "\n")
            written += 1

    ensure_nonempty_file(out_jsonl, "batch input jsonl")
    print(json.dumps({
        "src": str(src),
        "out_jsonl": str(out_jsonl),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "written": written,
        "skipped": skipped,
        "openai_model": args.openai_model,
        "max_output_tokens": int(args.max_output_tokens),
        "temperature": float(args.temperature),
        "size_bytes": out_jsonl.stat().st_size,
    }, ensure_ascii=False, indent=2))


def cmd_build_split(args: argparse.Namespace) -> None:
    src = Path(args.src).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(src)
    start_idx = int(args.start_idx)
    end_idx = len(records) if int(args.end_idx) < 0 else min(int(args.end_idx), len(records))
    chunk_size = int(args.chunk_size)

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    total_written = 0
    total_skipped = 0
    files = []

    for chunk_start in range(start_idx, end_idx, chunk_size):
        chunk_end = min(chunk_start + chunk_size, end_idx)
        out_jsonl = out_dir / f"{args.job_name}_{chunk_start:06d}_{chunk_end:06d}.jsonl"

        written = 0
        skipped = 0

        with out_jsonl.open("w", encoding="utf-8") as fout:
            for sample_idx in range(chunk_start, chunk_end):
                entry = records[sample_idx]
                req = build_request(
                    entry=entry,
                    sample_idx=sample_idx,
                    model=args.openai_model,
                    max_output_tokens=int(args.max_output_tokens),
                    temperature=float(args.temperature),
                )
                if req is None:
                    skipped += 1
                    continue
                fout.write(json.dumps(req, ensure_ascii=False) + "\n")
                written += 1

        if written > 0:
            ensure_nonempty_file(out_jsonl, "split batch input jsonl")
            files.append(str(out_jsonl))
        else:
            out_jsonl.unlink(missing_ok=True)

        total_written += written
        total_skipped += skipped

    if total_written == 0:
        raise RuntimeError("build_split wrote 0 requests")

    print(json.dumps({
        "src": str(src),
        "out_dir": str(out_dir),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "chunk_size": chunk_size,
        "num_files": len(files),
        "written": total_written,
        "skipped": total_skipped,
        "files": files,
    }, ensure_ascii=False, indent=2))


def cmd_submit(args: argparse.Namespace) -> None:
    payload = submit_one_jsonl(
        input_jsonl=Path(args.input_jsonl).expanduser(),
        model=args.openai_model,
        completion_window=args.completion_window,
        metadata_name=args.metadata_name,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.batch_meta_out:
        meta_path = Path(args.batch_meta_out).expanduser()
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {meta_path}")


def cmd_run_split(args: argparse.Namespace) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp = argparse.Namespace(
        src=args.src,
        out_dir=str(out_dir),
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        chunk_size=args.chunk_size,
        job_name=args.job_name,
        openai_model=args.openai_model,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
    )
    cmd_build_split(tmp)

    records = load_jsonl(Path(args.src).expanduser())
    start_idx = int(args.start_idx)
    end_idx = len(records) if int(args.end_idx) < 0 else min(int(args.end_idx), len(records))
    chunk_size = int(args.chunk_size)

    tag = now_tag()
    map_path = out_dir / f"{args.job_name}.batches_{tag}.tsv"
    lines = ["jsonl\tmeta\tbatch_id\tstatus\tinput_file_id"]

    for chunk_start in range(start_idx, end_idx, chunk_size):
        chunk_end = min(chunk_start + chunk_size, end_idx)
        jp = out_dir / f"{args.job_name}_{chunk_start:06d}_{chunk_end:06d}.jsonl"
        if not jp.exists() or jp.stat().st_size <= 0:
            continue

        meta_path = jp.with_suffix(".meta.json")
        payload = submit_one_jsonl(
            input_jsonl=jp,
            model=args.openai_model,
            completion_window=args.completion_window,
            metadata_name=jp.stem,
        )
        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"[saved] {meta_path}")
        lines.append(f"{jp}\t{meta_path}\t{payload['batch_id']}\t{payload['status']}\t{payload['input_file_id']}")

    map_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] Submitted all. Map saved: {map_path}")
    print("[INFO] Latest TSV:")
    print(map_path)


def cmd_status(args: argparse.Namespace) -> None:
    client = get_client()
    batch = client.batches.retrieve(args.batch_id)
    rc = getattr(batch, "request_counts", None)

    try:
        rc_payload = {"total": rc.total, "completed": rc.completed, "failed": rc.failed} if rc else None
    except Exception:
        rc_payload = str(rc)

    print(json.dumps({
        "batch_id": batch.id,
        "status": batch.status,
        "input_file_id": getattr(batch, "input_file_id", None),
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "request_counts": rc_payload,
    }, ensure_ascii=False, indent=2))


def cmd_status_from_tsv(args: argparse.Namespace) -> None:
    tsv_path = Path(args.tsv_path).expanduser()
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV not found: {tsv_path}")

    client = get_client()
    total = 0

    with tsv_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("jsonl\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue

            batch_id = parts[2]
            total += 1
            batch = client.batches.retrieve(batch_id)
            rc = getattr(batch, "request_counts", None)

            try:
                rc_payload = {"total": rc.total, "completed": rc.completed, "failed": rc.failed} if rc else None
            except Exception:
                rc_payload = str(rc)

            print(json.dumps({
                "batch_id": batch.id,
                "status": batch.status,
                "output_file_id": getattr(batch, "output_file_id", None),
                "error_file_id": getattr(batch, "error_file_id", None),
                "request_counts": rc_payload,
            }, ensure_ascii=False, indent=2))

    print(f"[INFO] checked batches: {total}")


def consume_batch_to_jsonl(
    batch_id: str,
    src: Path,
    out_jsonl: Path,
    raw_out_jsonl: Optional[Path],
    failed_ids_out: Optional[Path],
    start_idx: int,
    end_idx: int,
    append: bool,
    model: str,
) -> Dict[str, Any]:
    client = get_client()
    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        return {
            "batch_id": batch.id,
            "status": batch.status,
            "ok": 0,
            "failed": 0,
            "skipped": True,
            "reason": f"Batch is not completed. Current status: {batch.status}",
        }

    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        return {
            "batch_id": batch.id,
            "status": batch.status,
            "ok": 0,
            "failed": 0,
            "skipped": True,
            "reason": "Batch completed but output_file_id is missing.",
        }

    records = load_jsonl(src)
    text = client.files.content(output_file_id).text

    if raw_out_jsonl:
        raw_out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        raw_out_jsonl.write_text(text, encoding="utf-8")
        print(f"[saved raw batch output] {raw_out_jsonl}")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    failed_rows = []
    ok = 0
    failed = 0

    mode = "a" if append else "w"
    with out_jsonl.open(mode, encoding="utf-8") as fout:
        for line in text.splitlines():
            if not line.strip():
                continue

            obj = json.loads(line)
            custom_id = obj.get("custom_id", "")

            try:
                m = CUSTOM_ID_RE.match(custom_id)
                if not m:
                    raise ValueError(f"Bad custom_id: {custom_id}")

                sample_idx = int(m.group(1))

                if sample_idx < start_idx:
                    continue
                if end_idx >= 0 and sample_idx >= end_idx:
                    continue
                if sample_idx < 0 or sample_idx >= len(records):
                    raise IndexError(f"sample_idx={sample_idx} out of range len={len(records)}")

                entry = records[sample_idx]
                raw_text = extract_output_text_from_batch_line(obj)
                raw_claims = parse_json_object(raw_text)

                out_entry = build_atomic_output(entry, raw_claims)
                out_entry["atomic_model"] = model

                fout.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
                ok += 1

            except Exception as e:
                failed += 1
                failed_rows.append(f"{custom_id}\t{repr(e)}")
                print(f"[FAILED] {custom_id}: {repr(e)}")

    if failed_ids_out:
        failed_ids_out.parent.mkdir(parents=True, exist_ok=True)
        failed_ids_out.write_text("\n".join(failed_rows) + ("\n" if failed_rows else ""), encoding="utf-8")
        print(f"[saved failed ids] {failed_ids_out}")

    return {
        "batch_id": batch.id,
        "status": batch.status,
        "ok": ok,
        "failed": failed,
        "out_jsonl": str(out_jsonl),
        "start_idx": start_idx,
        "end_idx": end_idx,
    }


def cmd_consume(args: argparse.Namespace) -> None:
    payload = consume_batch_to_jsonl(
        batch_id=args.batch_id,
        src=Path(args.src).expanduser(),
        out_jsonl=Path(args.out_jsonl).expanduser(),
        raw_out_jsonl=Path(args.raw_out_jsonl).expanduser() if args.raw_out_jsonl else None,
        failed_ids_out=Path(args.failed_ids_out).expanduser() if args.failed_ids_out else None,
        start_idx=int(args.start_idx),
        end_idx=int(args.end_idx),
        append=not args.overwrite,
        model=args.openai_model,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_consume_from_tsv(args: argparse.Namespace) -> None:
    tsv_path = Path(args.tsv_path).expanduser()
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV not found: {tsv_path}")

    src = Path(args.src).expanduser()
    out_jsonl = Path(args.out_jsonl).expanduser()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and out_jsonl.exists():
        out_jsonl.unlink()
        print(f"[INFO] removed existing out_jsonl: {out_jsonl}")

    summaries = []
    total = 0

    with tsv_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("jsonl\t"):
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue

            jsonl_path = Path(parts[0])
            batch_id = parts[2]
            stem = jsonl_path.stem

            m = re.search(r"_(\d{6})_(\d{6})$", stem)
            if m:
                local_start = int(m.group(1))
                local_end = int(m.group(2))
            else:
                local_start = int(args.start_idx)
                local_end = int(args.end_idx)

            raw_out = Path(args.raw_out_dir).expanduser() / f"{stem}.output.jsonl" if args.raw_out_dir else None
            failed_out = Path(args.failed_out_dir).expanduser() / f"{stem}.failed_ids.txt" if args.failed_out_dir else None

            total += 1
            print(f"========== consume {stem} batch={batch_id} range={local_start}-{local_end} ==========")

            summary = consume_batch_to_jsonl(
                batch_id=batch_id,
                src=src,
                out_jsonl=out_jsonl,
                raw_out_jsonl=raw_out,
                failed_ids_out=failed_out,
                start_idx=local_start,
                end_idx=local_end,
                append=True,
                model=args.openai_model,
            )
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))

    print(json.dumps({
        "tsv_path": str(tsv_path),
        "out_jsonl": str(out_jsonl),
        "batches_seen": total,
        "summaries": summaries,
    }, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_build = sub.add_parser("build")
    ap_build.add_argument("--src", required=True)
    ap_build.add_argument("--out_jsonl", required=True)
    ap_build.add_argument("--start_idx", type=int, default=0)
    ap_build.add_argument("--end_idx", type=int, default=-1)
    ap_build.add_argument("--openai_model", default="gpt-4o")
    ap_build.add_argument("--max_output_tokens", type=int, default=1600)
    ap_build.add_argument("--temperature", type=float, default=0)
    ap_build.set_defaults(func=cmd_build)

    ap_build_split = sub.add_parser("build_split")
    ap_build_split.add_argument("--src", required=True)
    ap_build_split.add_argument("--out_dir", required=True)
    ap_build_split.add_argument("--start_idx", type=int, default=0)
    ap_build_split.add_argument("--end_idx", type=int, default=-1)
    ap_build_split.add_argument("--chunk_size", type=int, default=50)
    ap_build_split.add_argument("--job_name", default="atomic_fact_batch")
    ap_build_split.add_argument("--openai_model", default="gpt-4o")
    ap_build_split.add_argument("--max_output_tokens", type=int, default=1600)
    ap_build_split.add_argument("--temperature", type=float, default=0)
    ap_build_split.set_defaults(func=cmd_build_split)

    ap_submit = sub.add_parser("submit")
    ap_submit.add_argument("--input_jsonl", required=True)
    ap_submit.add_argument("--openai_model", default="gpt-4o")
    ap_submit.add_argument("--completion_window", default="24h")
    ap_submit.add_argument("--metadata_name", default="atomic-fact-batch")
    ap_submit.add_argument("--batch_meta_out", default="")
    ap_submit.set_defaults(func=cmd_submit)

    ap_run = sub.add_parser("run_split")
    ap_run.add_argument("--src", required=True)
    ap_run.add_argument("--out_dir", required=True)
    ap_run.add_argument("--start_idx", type=int, default=0)
    ap_run.add_argument("--end_idx", type=int, default=-1)
    ap_run.add_argument("--chunk_size", type=int, default=50)
    ap_run.add_argument("--job_name", default="atomic_fact_batch")
    ap_run.add_argument("--openai_model", default="gpt-4o")
    ap_run.add_argument("--max_output_tokens", type=int, default=1600)
    ap_run.add_argument("--temperature", type=float, default=0)
    ap_run.add_argument("--completion_window", default="24h")
    ap_run.set_defaults(func=cmd_run_split)

    ap_status = sub.add_parser("status")
    ap_status.add_argument("--batch_id", required=True)
    ap_status.set_defaults(func=cmd_status)

    ap_status_tsv = sub.add_parser("status_from_tsv")
    ap_status_tsv.add_argument("--tsv_path", required=True)
    ap_status_tsv.set_defaults(func=cmd_status_from_tsv)

    ap_consume = sub.add_parser("consume")
    ap_consume.add_argument("--batch_id", required=True)
    ap_consume.add_argument("--src", required=True)
    ap_consume.add_argument("--out_jsonl", required=True)
    ap_consume.add_argument("--raw_out_jsonl", default="")
    ap_consume.add_argument("--failed_ids_out", default="")
    ap_consume.add_argument("--start_idx", type=int, default=0)
    ap_consume.add_argument("--end_idx", type=int, default=-1)
    ap_consume.add_argument("--openai_model", default="gpt-4o")
    ap_consume.add_argument("--overwrite", action="store_true")
    ap_consume.set_defaults(func=cmd_consume)

    ap_consume_tsv = sub.add_parser("consume_from_tsv")
    ap_consume_tsv.add_argument("--tsv_path", required=True)
    ap_consume_tsv.add_argument("--src", required=True)
    ap_consume_tsv.add_argument("--out_jsonl", required=True)
    ap_consume_tsv.add_argument("--raw_out_dir", default="")
    ap_consume_tsv.add_argument("--failed_out_dir", default="")
    ap_consume_tsv.add_argument("--start_idx", type=int, default=0)
    ap_consume_tsv.add_argument("--end_idx", type=int, default=-1)
    ap_consume_tsv.add_argument("--openai_model", default="gpt-4o")
    ap_consume_tsv.add_argument("--overwrite", action="store_true")
    ap_consume_tsv.set_defaults(func=cmd_consume_from_tsv)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
