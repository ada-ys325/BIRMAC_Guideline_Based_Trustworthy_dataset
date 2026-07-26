#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch semantic seed extraction for guideline passage JSONL.

Input:
  data/passages_2025_test_60_100_regex_fixed.jsonl

Output:
  data/semantic_question_seeds_*.jsonl

Purpose:
  passage -> lightweight semantic seed for later patient-facing question generation.

Commands:
  build            Build one OpenAI Batch input JSONL.
  build_split      Build multiple Batch input JSONLs.
  submit           Submit one Batch input JSONL.
  run_split        Build split JSONLs and submit all.
  status           Check one batch status.
  status_from_tsv  Check statuses from TSV produced by run_split.
  consume          Consume one completed batch into semantic seed JSONL.
  consume_from_tsv Consume completed batches from TSV into one semantic seed JSONL.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Prompt + schema
# ============================================================

SYSTEM_PROMPT = """You are preparing medical guideline passages for patient-facing question generation.

Given one passage from a medical guideline, produce a lightweight semantic seed that describes what the passage is about and how it could be used to generate patient-facing questions.

Do not generate questions.
Do not generate answers.
Do not extract detailed PICO fields.
Do not verify evidence.
Do not over-structure the passage.

Your goal is only to help a later question-generation model understand:
1. what the passage discusses,
2. whether it is useful for patient-facing question generation,
3. what question angles would be natural,
4. what patient-friendly words might be used.

Return usable=false if the passage is mainly:
- references or bibliography,
- author affiliations,
- copyright or journal metadata,
- funding, conflicts of interest, acknowledgments,
- search strategy or guideline methods without patient-relevant clinical content,
- administrative or implementation information without clinical guidance.

If the passage contains both clinical content and noise, set usable=true as long as the clinical content is enough to support patient-facing questions.

Do not invent facts not present in the passage.
Keep the semantic summary concise and faithful.
"""

SEMANTIC_SEED_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "passage_id": {"type": "string"},
        "usable": {"type": "boolean"},
        "discard_reason": {"type": "string"},
        "content_type": {
            "type": "string",
            "enum": [
                "clinical_recommendation",
                "clinical_background",
                "diagnosis_or_screening",
                "treatment_or_management",
                "risk_or_prognosis",
                "monitoring_or_follow_up",
                "methods_or_search_strategy",
                "admin_or_metadata",
                "reference_like",
                "mixed",
            ],
        },
        "main_topic": {"type": "string"},
        "semantic_summary": {"type": "string"},
        "question_angles": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "string"},
        },
        "patient_language_keywords": {
            "type": "array",
            "maxItems": 8,
            "items": {"type": "string"},
        },
    },
    "required": [
        "passage_id",
        "usable",
        "discard_reason",
        "content_type",
        "main_topic",
        "semantic_summary",
        "question_angles",
        "patient_language_keywords",
    ],
}


# ============================================================
# Utilities
# ============================================================

CUSTOM_ID_RE = re.compile(r"^passage-(\d+)$")


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


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


def make_user_prompt(row: Dict[str, Any]) -> str:
    return f"""Passage metadata:
passage_id: {row.get('passage_id', '')}
guideline_id: {row.get('guideline_id', '')}
section_title: {row.get('section_title', '')}
page_start: {row.get('page_start', '')}
page_end: {row.get('page_end', '')}
word_count: {row.get('word_count', '')}
reference_list_score: {row.get('reference_list_score', '')}

Passage text:
{row.get('text', '')}

Produce a lightweight semantic seed for patient-facing question generation.

Output requirements:
1. semantic_summary should be one concise sentence starting with "This passage describes..." or "This passage discusses...".
2. main_topic should be a short phrase, not a full sentence.
3. question_angles should contain 1 to 4 short angle labels, not full questions.
   Good examples: "benefits", "safety", "who is eligible", "treatment choice", "follow-up care".
   Bad examples: "What are the benefits of treatment?" or "Should I choose this treatment?"
4. patient_language_keywords should contain 3 to 8 simple terms a patient or caregiver might use.
5. If the passage is not useful for patient-facing question generation, set usable=false and explain briefly in discard_reason.
"""


def build_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": make_user_prompt(row)},
    ]


def build_request(row: Dict[str, Any], sample_idx: int, model: str, max_output_tokens: int) -> Dict[str, Any]:
    return {
        "custom_id": f"passage-{sample_idx}",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": build_messages(row),
            "max_output_tokens": int(max_output_tokens),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "semantic_question_seed",
                    "strict": True,
                    "schema": SEMANTIC_SEED_SCHEMA,
                }
            },
        },
    }


def extract_output_text_from_batch_line(obj: Dict[str, Any]) -> str:
    body = obj.get("response", {}).get("body", {})

    # Responses API convenience field sometimes appears here.
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


def validate_seed(seed: Dict[str, Any], passage_id: str) -> Dict[str, Any]:
    """Light cleanup only; structured output should already obey schema."""
    out = dict(seed)
    out["passage_id"] = str(out.get("passage_id") or passage_id)
    out["usable"] = bool(out.get("usable", False))

    for key in ["discard_reason", "content_type", "main_topic", "semantic_summary"]:
        out[key] = norm_ws(str(out.get(key, "")))

    if not isinstance(out.get("question_angles"), list):
        out["question_angles"] = []
    out["question_angles"] = [norm_ws(str(x)) for x in out["question_angles"] if norm_ws(str(x))][:4]

    if not isinstance(out.get("patient_language_keywords"), list):
        out["patient_language_keywords"] = []
    out["patient_language_keywords"] = [norm_ws(str(x)) for x in out["patient_language_keywords"] if norm_ws(str(x))][:8]

    if not out["usable"]:
        out["main_topic"] = out.get("main_topic", "")
        out["semantic_summary"] = out.get("semantic_summary", "")
        out["question_angles"] = []
        out["patient_language_keywords"] = []

    return out


def attach_source_metadata(seed: Dict[str, Any], row: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
    out = dict(seed)
    out["sample_idx"] = sample_idx
    out["guideline_id"] = row.get("guideline_id", "")
    out["source_pdf"] = row.get("source_pdf", "")
    out["section_title"] = row.get("section_title", "")
    out["page_start"] = row.get("page_start", None)
    out["page_end"] = row.get("page_end", None)
    out["word_count"] = row.get("word_count", None)
    out["reference_list_score"] = row.get("reference_list_score", None)
    out["source_text"] = row.get("text", "")
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
            "task": "semantic_question_seed_extraction",
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
            row = records[sample_idx]
            text = norm_ws(str(row.get("text", "")))
            pid = str(row.get("passage_id", ""))
            if not pid or not text:
                skipped += 1
                continue
            req = build_request(row, sample_idx, args.openai_model, int(args.max_output_tokens))
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

    for chunk_start in range(start_idx, end_idx, chunk_size):
        chunk_end = min(chunk_start + chunk_size, end_idx)
        out_jsonl = out_dir / f"{args.job_name}_{chunk_start:06d}_{chunk_end:06d}.jsonl"

        written = 0
        skipped = 0
        with out_jsonl.open("w", encoding="utf-8") as fout:
            for sample_idx in range(chunk_start, chunk_end):
                row = records[sample_idx]
                text = norm_ws(str(row.get("text", "")))
                pid = str(row.get("passage_id", ""))
                if not pid or not text:
                    skipped += 1
                    continue
                req = build_request(row, sample_idx, args.openai_model, int(args.max_output_tokens))
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


def cmd_status(args: argparse.Namespace) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    client = get_client()
    batch = client.batches.retrieve(args.batch_id)

    rc = getattr(batch, "request_counts", None)
    if rc is not None:
        try:
            rc = {"total": rc.total, "completed": rc.completed, "failed": rc.failed}
        except Exception:
            rc = str(rc)

    print(json.dumps({
        "batch_id": batch.id,
        "status": batch.status,
        "input_file_id": getattr(batch, "input_file_id", None),
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "request_counts": rc,
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
                seed_raw = parse_json_object(raw_text)
                seed = validate_seed(seed_raw, str(row.get("passage_id", "")))
                payload = attach_source_metadata(seed, row, sample_idx)
                fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
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
    ap_build.add_argument("--max_output_tokens", type=int, default=500)
    ap_build.set_defaults(func=cmd_build)

    ap_build_split = sub.add_parser("build_split")
    ap_build_split.add_argument("--src", required=True)
    ap_build_split.add_argument("--out_dir", required=True)
    ap_build_split.add_argument("--start_idx", type=int, default=0)
    ap_build_split.add_argument("--end_idx", type=int, default=-1)
    ap_build_split.add_argument("--chunk_size", type=int, default=100)
    ap_build_split.add_argument("--job_name", default="semantic_seed_batch")
    ap_build_split.add_argument("--openai_model", default="gpt-4o")
    ap_build_split.add_argument("--max_output_tokens", type=int, default=500)
    ap_build_split.set_defaults(func=cmd_build_split)

    ap_submit = sub.add_parser("submit")
    ap_submit.add_argument("--input_jsonl", required=True)
    ap_submit.add_argument("--openai_model", default="gpt-4o")
    ap_submit.add_argument("--completion_window", default="24h")
    ap_submit.add_argument("--metadata_name", default="semantic-seed-batch")
    ap_submit.add_argument("--batch_meta_out", default="")
    ap_submit.set_defaults(func=cmd_submit)

    ap_run = sub.add_parser("run_split")
    ap_run.add_argument("--src", required=True)
    ap_run.add_argument("--out_dir", required=True)
    ap_run.add_argument("--start_idx", type=int, default=0)
    ap_run.add_argument("--end_idx", type=int, default=-1)
    ap_run.add_argument("--chunk_size", type=int, default=100)
    ap_run.add_argument("--job_name", default="semantic_seed_batch")
    ap_run.add_argument("--openai_model", default="gpt-4o")
    ap_run.add_argument("--max_output_tokens", type=int, default=500)
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
    ap_consume_tsv.add_argument("--overwrite", action="store_true")
    ap_consume_tsv.set_defaults(func=cmd_consume_from_tsv)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
