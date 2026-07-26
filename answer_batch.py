#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch answer generation for BIRMAC retrieved medical questions.

Input JSONL:
  data/group_questions_gpt5_20250807_full_tok2000_retrieved_top5_qwen3rerank.jsonl

Each line:
  {
    "question_id": "...",
    "question": "...",
    "question_type": "...",
    "perspective": "...",
    "patient_keywords": [...],
    "retrieval_anchor_terms": [...],
    "retrieval_config": {...},
    "retrieved_passages": [
      {
        "passage_id": "...",
        "guideline_id": "...",
        "main_topic": "...",
        "section": "...",
        "rerank_score": ...,
        "text": "..."
      }
    ]
  }

Output JSONL:
  one answered question per line:
  {
    "question_idx": 0,
    "question_id": "...",
    "question": "...",
    "answer": "...",
    "answer_text": "...",
    "answer_sentences": [...],
    "evidence_gaps": [...],
    "top_passages": [...],
    "passages": [
      {"pid": 1, "text": "...", "passage_id": "...", "guideline_id": "..."}
    ]
  }

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
# Prompt + schema
# ============================================================

ANSWER_ROLES = [
    "direct_answer",
    "eligibility_or_condition",
    "benefit_or_effectiveness",
    "risk_or_safety",
    "monitoring_or_follow_up",
    "procedure_or_preparation",
    "medication_use",
    "diagnosis_or_screening",
    "lifestyle_or_self_management",
    "shared_decision_making",
    "equity_or_communication",
    "evidence_limitation",
    "patient_action_supported_by_evidence",
]

SYSTEM_PROMPT = """You are a clinical answer generation assistant for a medical QA dataset.

Your task is to answer patient questions using ONLY the provided guideline passages.

The answer must be:
1. Patient-readable.
2. Medically careful.
3. Easy to decompose into citation-grounded atomic medical facts.
4. Fully grounded in the provided passages.

Do NOT output chain-of-thought.
Do NOT mention that you are an AI.
Do NOT use outside medical knowledge.
Do NOT add diagnoses, treatments, drug doses, risks, benefits, eligibility criteria, timelines, contraindications, or follow-up advice unless directly supported by the provided passages.
Do NOT give generic advice such as "talk to your doctor" unless the passages support clinician discussion, individualized assessment, shared decision-making, or follow-up.

Every sentence in answer_text must:
1. State one main medical fact, recommendation, condition, risk, benefit, limitation, or supported patient action.
2. Be understandable to a patient.
3. Be directly supported by the cited passage IDs.
4. End with 1 to 3 citation brackets, such as [1], [2][4], or [1][3][5].
5. Avoid vague pronouns when repeating the medical term would make the sentence clearer.

If the passages do not answer part of the question, do not guess.
Instead, include the missing issue in evidence_gaps.
When useful, add a cited sentence explaining what the provided passages do support.

Return valid JSON only."""

USER_TEMPLATE = """# TASK

Answer the patient's question using only the provided guideline passages.

# PATIENT QUESTION

{question}

# QUESTION METADATA

question_id: {question_id}
question_type: {question_type}
retrieval_anchor_terms: {retrieval_anchor_terms}
patient_keywords: {patient_keywords}

# GUIDELINE PASSAGES

Use the local passage IDs [1]...[N] as citations.
Only cite these local passage IDs.

{context_block}

# ANSWER STYLE

Write a short patient-friendly answer.

The answer should usually contain 3 to 6 sentences.

The first sentence should directly answer the patient's main question.

Each sentence should contain only one main medical fact, recommendation, condition, risk, benefit, limitation, or supported patient action.

Use plain language, but keep important medical terms when they are needed for clinical accuracy.

Every sentence in answer_text must end with citation brackets.

Do not use bullet points inside answer_text.

Do not cite a passage unless it directly supports the sentence.

Do not combine unrelated facts into one long sentence.

Do not invent exact numbers, risk levels, treatment effects, contraindications, eligibility rules, test criteria, or timelines unless they are explicitly stated in the passages.

If the question asks about a choice between options, explain what the passages support for each option.

If the question asks about safety or side effects, separate safety facts from action steps.

If the question asks about diagnosis or screening, distinguish what the passages support from what remains uncertain.

If the question involves a conditional recommendation, shared decision-making, patient preferences, equity, communication, or self-management, explain that clearly using only supported evidence.

If the evidence is insufficient, say what is missing in evidence_gaps instead of making a stronger claim.

# OUTPUT JSON FORMAT

Return exactly one JSON object with this structure:

{{
  "question_id": "{question_id}",
  "answer_text": "A patient-readable answer. Every sentence ends with citations like [1] or [1][2].",
  "answer_sentences": [
    {{
      "sentence_id": "s1",
      "text": "The exact first sentence from answer_text, including citation brackets.",
      "citations": [1],
      "role": "direct_answer"
    }}
  ],
  "evidence_gaps": []
}}

# STRICT OUTPUT RULES

answer_text must be a single natural paragraph.

Every sentence in answer_text must end with 1 to 3 citation brackets.

The text field of each answer_sentences item must exactly match the corresponding sentence in answer_text, including citation brackets.

The citations field must contain the same citation IDs that appear in that sentence.

Do not include citations that are not present in the sentence text.

Do not include any markdown.

Do not include bullet points.

Do not include extra keys.

# ALLOWED SENTENCE ROLES

Use one of:
- direct_answer
- eligibility_or_condition
- benefit_or_effectiveness
- risk_or_safety
- monitoring_or_follow_up
- procedure_or_preparation
- medication_use
- diagnosis_or_screening
- lifestyle_or_self_management
- shared_decision_making
- equity_or_communication
- evidence_limitation
- patient_action_supported_by_evidence

# FINAL OUTPUT

Return valid JSON only."""

ANSWER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "question_id": {"type": "string"},
        "answer_text": {"type": "string"},
        "answer_sentences": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "sentence_id": {"type": "string"},
                    "text": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {"type": "integer"},
                    },
                    "role": {
                        "type": "string",
                        "enum": ANSWER_ROLES,
                    },
                },
                "required": ["sentence_id", "text", "citations", "role"],
            },
        },
        "evidence_gaps": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "missing_issue": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["missing_issue", "reason"],
            },
        },
    },
    "required": ["question_id", "answer_text", "answer_sentences", "evidence_gaps"],
}


# ============================================================
# Utilities
# ============================================================

CUSTOM_ID_RE = re.compile(r"^qrow-(\d+)$")
CIT_RE = re.compile(r"\[(\d+)\]")


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def norm_ws(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())


def truncate_text(s: Any, max_chars: int) -> str:
    s = norm_ws(s)
    if max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + " ...[truncated]"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} is not a JSON object in {path}")
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


def compact_json(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False)


def select_passages(row: Dict[str, Any], top_n: int) -> List[Dict[str, Any]]:
    passages = row.get("retrieved_passages", [])
    if not isinstance(passages, list):
        return []
    out = []
    for p in passages[:top_n]:
        if isinstance(p, dict) and norm_ws(p.get("text")):
            out.append(p)
    return out


def build_context_block(passages: List[Dict[str, Any]], max_passage_chars: int) -> str:
    blocks = []
    for i, p in enumerate(passages, start=1):
        text = truncate_text(p.get("text", ""), max_passage_chars)
        if not text:
            continue

        blocks.append(
            f"[{i}]\n"
            f"passage_id: {norm_ws(p.get('passage_id'))}\n"
            f"guideline_id: {norm_ws(p.get('guideline_id'))}\n"
            f"main_topic: {norm_ws(p.get('main_topic'))}\n"
            f"section: {norm_ws(p.get('section'))}\n"
            f"rerank_score: {norm_ws(p.get('rerank_score'))}\n"
            f"text: {text}\n"
        )
    return "\n".join(blocks).strip()


def make_user_prompt(row: Dict[str, Any], top_n_passages: int, max_passage_chars: int) -> str:
    passages = select_passages(row, top_n=top_n_passages)
    context_block = build_context_block(passages, max_passage_chars=max_passage_chars)
    if not context_block:
        raise ValueError("Empty context_block")

    return USER_TEMPLATE.format(
        question=norm_ws(row.get("question")),
        question_id=norm_ws(row.get("question_id")),
        question_type=norm_ws(row.get("question_type")),
        retrieval_anchor_terms=compact_json(row.get("retrieval_anchor_terms", [])),
        patient_keywords=compact_json(row.get("patient_keywords", [])),
        context_block=context_block,
    )


def build_messages(row: Dict[str, Any], top_n_passages: int, max_passage_chars: int) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": make_user_prompt(
                row,
                top_n_passages=top_n_passages,
                max_passage_chars=max_passage_chars,
            ),
        },
    ]


def build_request(
    row: Dict[str, Any],
    sample_idx: int,
    model: str,
    max_output_tokens: int,
    top_n_passages: int,
    max_passage_chars: int,
    temperature: Optional[float],
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": model,
        "input": build_messages(
            row,
            top_n_passages=top_n_passages,
            max_passage_chars=max_passage_chars,
        ),
        "max_output_tokens": int(max_output_tokens),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "patient_answer_with_citations",
                "strict": True,
                "schema": ANSWER_SCHEMA,
            }
        },
    }

    if temperature is not None:
        body["temperature"] = float(temperature)

    return {
        "custom_id": f"qrow-{sample_idx}",
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def extract_output_text_from_batch_line(obj: Dict[str, Any]) -> str:
    body = obj.get("response", {}).get("body", {})

    if isinstance(body.get("output_text"), str) and body["output_text"].strip():
        return body["output_text"].strip()

    output = body.get("output", [])
    texts: List[str] = []
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
            raise ValueError(f"Empty model output in batch result. incomplete_details={incomplete}")
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

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")

    obj = json.loads(raw[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not object")
    return obj


def citations_from_sentence_text(text: str) -> List[int]:
    return [int(x) for x in CIT_RE.findall(text or "")]


def normalize_citations(citations: Any, max_pid: int) -> List[int]:
    out: List[int] = []
    if isinstance(citations, list):
        for x in citations:
            try:
                n = int(x)
            except Exception:
                continue
            if 1 <= n <= max_pid and n not in out:
                out.append(n)
    return out[:3]


def fallback_split_sentences(answer_text: str, max_pid: int) -> List[Dict[str, Any]]:
    # Conservative fallback: split only cited sentences ending with brackets.
    sent_re = re.compile(
        r"(.+?(?:[.!?]|[。！？])\s*(?:\[\d+\]){1,3})(?=\s+|$)",
        flags=re.DOTALL,
    )
    out = []
    for i, m in enumerate(sent_re.finditer(norm_ws(answer_text)), start=1):
        txt = norm_ws(m.group(1))
        citations = normalize_citations(citations_from_sentence_text(txt), max_pid=max_pid)
        if txt and citations:
            out.append({
                "sentence_id": f"s{i}",
                "text": txt,
                "citations": citations,
                "role": "direct_answer" if i == 1 else "evidence_limitation",
            })
    return out


def validate_answer_payload(raw: Dict[str, Any], row: Dict[str, Any], sample_idx: int, top_n_passages: int) -> Dict[str, Any]:
    passages = select_passages(row, top_n=top_n_passages)
    max_pid = len(passages)

    qid = norm_ws(row.get("question_id")) or f"qrow_{sample_idx:06d}"

    answer_text = norm_ws(raw.get("answer_text"))
    if not answer_text:
        raise ValueError("Missing answer_text")

    raw_sents = raw.get("answer_sentences", [])
    answer_sentences: List[Dict[str, Any]] = []

    if isinstance(raw_sents, list):
        for i, s in enumerate(raw_sents, start=1):
            if not isinstance(s, dict):
                continue
            txt = norm_ws(s.get("text"))
            if not txt:
                continue

            citations = normalize_citations(s.get("citations"), max_pid=max_pid)
            text_cites = normalize_citations(citations_from_sentence_text(txt), max_pid=max_pid)

            # Prefer citations explicitly present in the sentence text.
            if text_cites:
                citations = text_cites

            if not citations:
                continue

            role = norm_ws(s.get("role"))
            if role not in ANSWER_ROLES:
                role = "direct_answer" if i == 1 else "evidence_limitation"

            answer_sentences.append({
                "sentence_id": f"s{len(answer_sentences) + 1}",
                "text": txt,
                "citations": citations,
                "role": role,
            })

    if not answer_sentences:
        answer_sentences = fallback_split_sentences(answer_text, max_pid=max_pid)

    if not answer_sentences:
        raise ValueError("No valid answer_sentences with citations")

    # Check each sentence has visible citation brackets.
    for s in answer_sentences:
        if not citations_from_sentence_text(s["text"]):
            raise ValueError(f"Sentence lacks visible citation brackets: {s['text']}")

    evidence_gaps = []
    raw_gaps = raw.get("evidence_gaps", [])
    if isinstance(raw_gaps, list):
        for g in raw_gaps[:5]:
            if not isinstance(g, dict):
                continue
            missing_issue = norm_ws(g.get("missing_issue"))
            reason = norm_ws(g.get("reason"))
            if missing_issue or reason:
                evidence_gaps.append({
                    "missing_issue": missing_issue,
                    "reason": reason,
                })

    return {
        "question_id": qid,
        "answer_text": answer_text,
        "answer_sentences": answer_sentences,
        "evidence_gaps": evidence_gaps,
    }


def make_passages_for_atomic(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, p in enumerate(passages, start=1):
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
            "page_start": p.get("page_start"),
            "page_end": p.get("page_end"),
            "source_pdf": p.get("source_pdf"),
        })
    return out


# ============================================================
# OpenAI Batch helpers
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
            "task": "patient_answer_generation_with_citations",
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

    temperature = None if args.temperature < 0 else float(args.temperature)

    with out_jsonl.open("w", encoding="utf-8") as fout:
        for sample_idx in range(start_idx, end_idx):
            row = records[sample_idx]
            question = norm_ws(row.get("question"))
            passages = select_passages(row, top_n=int(args.top_n_passages))

            if not question or not passages:
                skipped += 1
                continue

            req = build_request(
                row=row,
                sample_idx=sample_idx,
                model=args.openai_model,
                max_output_tokens=int(args.max_output_tokens),
                top_n_passages=int(args.top_n_passages),
                max_passage_chars=int(args.max_passage_chars),
                temperature=temperature,
            )
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
        "top_n_passages": int(args.top_n_passages),
        "max_passage_chars": int(args.max_passage_chars),
        "temperature": temperature,
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
    files: List[str] = []

    temperature = None if args.temperature < 0 else float(args.temperature)

    for chunk_start in range(start_idx, end_idx, chunk_size):
        chunk_end = min(chunk_start + chunk_size, end_idx)
        out_jsonl = out_dir / f"{args.job_name}_{chunk_start:06d}_{chunk_end:06d}.jsonl"

        written = 0
        skipped = 0
        with out_jsonl.open("w", encoding="utf-8") as fout:
            for sample_idx in range(chunk_start, chunk_end):
                row = records[sample_idx]
                question = norm_ws(row.get("question"))
                passages = select_passages(row, top_n=int(args.top_n_passages))

                if not question or not passages:
                    skipped += 1
                    continue

                req = build_request(
                    row=row,
                    sample_idx=sample_idx,
                    model=args.openai_model,
                    max_output_tokens=int(args.max_output_tokens),
                    top_n_passages=int(args.top_n_passages),
                    max_passage_chars=int(args.max_passage_chars),
                    temperature=temperature,
                )
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
        top_n_passages=args.top_n_passages,
        max_passage_chars=args.max_passage_chars,
        temperature=args.temperature,
    )
    cmd_build_split(tmp)

    records = load_jsonl(Path(args.src).expanduser())
    start_idx = int(args.start_idx)
    end_idx = len(records) if int(args.end_idx) < 0 else min(int(args.end_idx), len(records))
    chunk_size = int(args.chunk_size)

    tag = now_tag()
    map_path = out_dir / f"{args.job_name}.batches_{tag}.tsv"
    lines: List[str] = ["jsonl\tmeta\tbatch_id\tstatus\tinput_file_id"]

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
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")
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
    top_n_passages: int,
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
    failed_rows: List[str] = []
    ok = 0
    failed = 0
    total_answers = 0

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

                row = records[sample_idx]
                raw_text = extract_output_text_from_batch_line(obj)
                raw_payload = parse_json_object(raw_text)
                answer_payload = validate_answer_payload(
                    raw=raw_payload,
                    row=row,
                    sample_idx=sample_idx,
                    top_n_passages=top_n_passages,
                )

                passages = select_passages(row, top_n=top_n_passages)
                passages_for_atomic = make_passages_for_atomic(passages)

                out_obj = {
                    "question_idx": sample_idx,
                    "question_id": answer_payload["question_id"],
                    "question": norm_ws(row.get("question")),
                    "question_type": norm_ws(row.get("question_type")),
                    "perspective": norm_ws(row.get("perspective")),
                    "patient_keywords": row.get("patient_keywords", []),
                    "retrieval_anchor_terms": row.get("retrieval_anchor_terms", []),
                    "answer": answer_payload["answer_text"],
                    "answer_text": answer_payload["answer_text"],
                    "answer_sentences": answer_payload["answer_sentences"],
                    "evidence_gaps": answer_payload["evidence_gaps"],
                    "top_passages": passages,
                    "passages": passages_for_atomic,
                }

                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                ok += 1
                total_answers += 1

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
        "total_answers": total_answers,
        "out_jsonl": str(out_jsonl),
        "start_idx": start_idx,
        "end_idx": end_idx,
    }


def cmd_consume(args: argparse.Namespace) -> None:
    src = Path(args.src).expanduser()
    out_jsonl = Path(args.out_jsonl).expanduser()
    raw = Path(args.raw_out_jsonl).expanduser() if args.raw_out_jsonl else None
    fail = Path(args.failed_ids_out).expanduser() if args.failed_ids_out else None

    payload = consume_batch_to_jsonl(
        batch_id=args.batch_id,
        src=src,
        out_jsonl=out_jsonl,
        raw_out_jsonl=raw,
        failed_ids_out=fail,
        start_idx=int(args.start_idx),
        end_idx=int(args.end_idx),
        append=not args.overwrite,
        top_n_passages=int(args.top_n_passages),
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

    summaries: List[Dict[str, Any]] = []
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
                top_n_passages=int(args.top_n_passages),
            )
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))

    print(json.dumps({
        "tsv_path": str(tsv_path),
        "out_jsonl": str(out_jsonl),
        "batches_seen": total,
        "summaries": summaries,
    }, ensure_ascii=False, indent=2))


# ============================================================
# Main
# ============================================================

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
    ap_build.add_argument("--top_n_passages", type=int, default=5)
    ap_build.add_argument("--max_passage_chars", type=int, default=1800)
    ap_build.add_argument("--temperature", type=float, default=0.2)
    ap_build.set_defaults(func=cmd_build)

    ap_build_split = sub.add_parser("build_split")
    ap_build_split.add_argument("--src", required=True)
    ap_build_split.add_argument("--out_dir", required=True)
    ap_build_split.add_argument("--start_idx", type=int, default=0)
    ap_build_split.add_argument("--end_idx", type=int, default=-1)
    ap_build_split.add_argument("--chunk_size", type=int, default=30)
    ap_build_split.add_argument("--job_name", default="answer_batch")
    ap_build_split.add_argument("--openai_model", default="gpt-4o")
    ap_build_split.add_argument("--max_output_tokens", type=int, default=1600)
    ap_build_split.add_argument("--top_n_passages", type=int, default=5)
    ap_build_split.add_argument("--max_passage_chars", type=int, default=1800)
    ap_build_split.add_argument("--temperature", type=float, default=0.2)
    ap_build_split.set_defaults(func=cmd_build_split)

    ap_submit = sub.add_parser("submit")
    ap_submit.add_argument("--input_jsonl", required=True)
    ap_submit.add_argument("--openai_model", default="gpt-4o")
    ap_submit.add_argument("--completion_window", default="24h")
    ap_submit.add_argument("--metadata_name", default="answer-batch")
    ap_submit.add_argument("--batch_meta_out", default="")
    ap_submit.set_defaults(func=cmd_submit)

    ap_run = sub.add_parser("run_split")
    ap_run.add_argument("--src", required=True)
    ap_run.add_argument("--out_dir", required=True)
    ap_run.add_argument("--start_idx", type=int, default=0)
    ap_run.add_argument("--end_idx", type=int, default=-1)
    ap_run.add_argument("--chunk_size", type=int, default=30)
    ap_run.add_argument("--job_name", default="answer_batch")
    ap_run.add_argument("--openai_model", default="gpt-4o")
    ap_run.add_argument("--max_output_tokens", type=int, default=1600)
    ap_run.add_argument("--top_n_passages", type=int, default=5)
    ap_run.add_argument("--max_passage_chars", type=int, default=1800)
    ap_run.add_argument("--temperature", type=float, default=0.2)
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
    ap_consume.add_argument("--top_n_passages", type=int, default=5)
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
    ap_consume_tsv.add_argument("--top_n_passages", type=int, default=5)
    ap_consume_tsv.add_argument("--overwrite", action="store_true")
    ap_consume_tsv.set_defaults(func=cmd_consume_from_tsv)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()