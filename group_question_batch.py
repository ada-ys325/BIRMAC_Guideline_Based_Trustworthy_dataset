#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch group-level patient question generation from BIRMAC evidence candidate pools.

Input:
  data/evidence_candidate_pools_2025_all_within_guideline_top30_min15.jsonl

Output:
  question-only JSONL, one question per line, e.g.

  {
    "question_id": "q_000001",
    "question": "I've tried quitting smoking before. Should I consider using e-cigarettes?",
    "question_type": "treatment_choice",
    "perspective": "patient_self",
    "patient_keywords": ["quitting smoking", "e-cigarettes", "treatment options"],
    "retrieval_anchor_terms": ["quitting smoking", "e-cigarettes"]
  }

Purpose:
  30-passage candidate pool
  -> identify coherent patient-relevant topics
  -> generate 0-3 realistic patient-voice questions
  -> NO supporting_passage_ids
  -> NO support_roles
  -> later FAISS retrieval decides evidence

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

QUESTION_TYPES = [
    "treatment_choice",
    "medication_use",
    "safety_or_side_effects",
    "eligibility",
    "diagnosis_or_screening",
    "monitoring_or_follow_up",
    "recovery_or_aftercare",
    "when_to_seek_care",
    "risk_or_prognosis",
    "procedure_preparation",
    "lifestyle_or_self_management",
    "other",
]

PERSPECTIVES = [
    "patient_self",
    "caregiver_or_family",
    "general_health_info_seeker",
]

SYSTEM_PROMPT = """You are generating realistic medical questions in the voice of ordinary patients, caregivers, or health information seekers.

The input contains 15-30 related clinical guideline passages from the same guideline. This is a candidate pool used only to inspire question generation.

Your task:
Generate 0 to 1 realistic patient-style question that captures the strongest shared patient-relevant topic across the candidate pool.

Important:
You do NOT need to select supporting passage IDs.
You do NOT need to cite passages.
You do NOT need to output support roles.
The generated question only needs to be natural, clinically meaningful, and retrievable later from the full passage corpus.

The goal is not to write textbook-style medical questions.
The goal is to write one question that a real patient might type into Google, ChatGPT, or ask their doctor, while still keeping enough medical anchor terms for retrieval.

Question generation requirements:
1. Identify the single strongest patient-relevant topic shared by multiple passages in the candidate pool.
2. Generate at most one natural question based on that shared topic.
3. The question must be answerable in principle from clinical guideline passages.
4. Do not invent unsupported patient details, diagnoses, treatments, risks, symptoms, ages, pregnancy status, prior treatment history, or medical history.
5. If a personal context is useful, prefer conditional wording such as "If I...", "I was told...", "My doctor mentioned...", or "I have..." only when clearly supported by the pool.
6. Avoid questions about guideline methodology, GRADE certainty, author opinions, study design, or professional-only technical procedures.
7. Avoid simply rewriting a recommendation as a question.
8. Avoid overly broad questions such as "What is this disease?"
9. Avoid overly narrow questions that depend on one statistic, one trial detail, or one isolated sentence.
10. Do not mention "guideline", "passage", "document", "according to this", or "based on these passages" in the question.

Patient voice requirements:
1. The question should sound like a real person asking about their own situation.
2. Prefer natural first-person wording when appropriate: "I", "my", "I'm", "I've been", "I was told", "my doctor said".
3. Caregiver questions are allowed when natural: "my partner", "my child", "my parent".
4. Use everyday wording where possible.
5. Avoid stiff professional phrasing such as "key considerations", "regimen selection", "risk-benefit assessment", "management strategy", "therapeutic intervention", "clinical outcomes", or "treatment modality".
6. Do not make the question sound like a patient-education handout title.
7. Usually write one sentence, around 8-30 words.
8. Mild worry, confusion, or next-step uncertainty is good when natural.
9. Do not over-pack the question with many clinical details.

Retrieval anchor requirements:
1. Each question must include 1 to 3 clear retrieval anchor terms.
2. Good anchor terms include condition names, procedure names, medicine names, symptoms, tests, treatment names, or safety issues.
3. Keep anchor terms that a patient might realistically know from a prescription, appointment reminder, discharge note, doctor conversation, or online search.
4. Do not make the question so casual that the clinical topic disappears.
5. retrieval_anchor_terms should be the strongest terms for later FAISS retrieval.
6. patient_keywords should include retrieval_anchor_terms first, then add other patient-friendly words if useful.

Good examples:
- "I'm having a colonoscopy soon. How do I choose a bowel prep that's safe for me?"
- "How do I know if my bowel prep worked well enough before my colonoscopy?"
- "I've been taking benzodiazepines for a while. How can I stop without bad withdrawal symptoms?"
- "I'm starting cancer treatment soon. Should I freeze eggs or embryos before treatment?"
- "Should I bank sperm before starting cancer treatment?"
- "Can erythromycin help my gastroparesis symptoms, and what side effects should I know about?"
- "My pain is still bad after a C-section. When are opioids actually needed?"
- "I've tried quitting smoking before. Should I consider using e-cigarettes?"
- "If I have Barrett's Esophagus, how often should I get an endoscopy?"

Bad examples:
- "What should I consider when choosing a bowel preparation regimen?"
- "How can I ensure my bowel preparation is adequate?"
- "What are the key considerations for tapering benzodiazepines safely?"
- "What are the options and considerations for sperm banking?"
- "What is the role of erythromycin in gastroparesis management?"
- "How do I get ready for my test?"
- "Is this medicine safe for me?"
- "What should I do next?"
- "I'm over 65 with early-stage breast cancer. Can I skip the sentinel lymph node biopsy?" unless age is clearly central in the passages.
- "I get nauseous easily. How can I minimize side effects?" unless nausea is clearly central in the passages.

Preferred question types:
- treatment_choice
- medication_use
- safety_or_side_effects
- eligibility
- diagnosis_or_screening
- monitoring_or_follow_up
- recovery_or_aftercare
- when_to_seek_care
- risk_or_prognosis
- procedure_preparation
- lifestyle_or_self_management

Generate exactly one strong question when the pool has a coherent patient-relevant topic.
Return questions=[] only when the pool is mostly administrative, methodological, reference-like, or lacks a coherent patient-relevant topic.
"""

GROUP_QUESTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "pool_id": {"type": "string"},
        "guideline_id": {"type": "string"},
        "questions": {
            "type": "array",
            "maxItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "question_id": {"type": "string"},
                    "question": {"type": "string"},
                    "question_type": {
                        "type": "string",
                        "enum": QUESTION_TYPES,
                    },
                    "perspective": {
                        "type": "string",
                        "enum": PERSPECTIVES,
                    },
                    "patient_keywords": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {"type": "string"},
                    },
                    "retrieval_anchor_terms": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "question_id",
                    "question",
                    "question_type",
                    "perspective",
                    "patient_keywords",
                    "retrieval_anchor_terms",
                ],
            },
        },
        "skip_reason": {"type": "string"},
    },
    "required": [
        "pool_id",
        "guideline_id",
        "questions",
        "skip_reason",
    ],
}


# ============================================================
# Utilities
# ============================================================

CUSTOM_ID_RE = re.compile(r"^pool-(\d+)$")


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def norm_ws(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())


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


def compact_list(x: Any, max_items: int = 8) -> str:
    if isinstance(x, list):
        return "; ".join(norm_ws(i) for i in x[:max_items] if norm_ws(i))
    return norm_ws(x)


def truncate_text(s: Any, max_chars: int) -> str:
    s = norm_ws(s)
    if max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + " ...[truncated]"


def make_user_prompt(pool: Dict[str, Any], max_source_chars: int) -> str:
    passages = pool.get("passages", [])
    if not isinstance(passages, list):
        passages = []

    passages = passages[:30]

    passage_blocks: List[str] = []
    for i, p in enumerate(passages, start=1):
        if not isinstance(p, dict):
            continue

        block = f"""[{i}]
content_type: {norm_ws(p.get('content_type'))}
main_topic: {norm_ws(p.get('main_topic'))}
semantic_summary: {norm_ws(p.get('semantic_summary'))}
question_angles: {compact_list(p.get('question_angles'))}
patient_keywords: {compact_list(p.get('patient_language_keywords'))}
source_text: {truncate_text(p.get('source_text'), max_source_chars)}
"""
        passage_blocks.append(block)

        return f"""Pool metadata:
pool_id: {norm_ws(pool.get('group_id'))}
guideline_id: {norm_ws(pool.get('guideline_id'))}
candidate_topic_hint: {norm_ws(pool.get('group_topic_hint'))}
candidate_summary_hint: {norm_ws(pool.get('group_summary_hint'))}
candidate_question_angles_hint: {compact_list(pool.get('group_question_angles_hint'))}
candidate_patient_keywords_hint: {compact_list(pool.get('group_patient_keywords_hint'))}
num_candidate_passages: {len(passages)}

Candidate passages:
{chr(10).join(passage_blocks)}

Generate 0 to 1 realistic patient-voice question from this candidate pool.

Important:
The passages above are only used to inspire patient-style question generation.
Do not output passage IDs.
Do not cite passage numbers.
Do not create supporting_passage_ids.
Do not create support_roles.
The final question will later be used to retrieve passages from the full FAISS corpus.

Output requirements:
1. Generate at most one question from this pool.
2. Choose the single strongest patient-relevant topic shared by several passages in the pool.
3. Do not generate multiple variants of the same clinical topic.
4. The question must sound like a real patient or caregiver typing a question, not like a medical writer or clinical educator.
5. The question must keep 1 to 3 clear retrieval anchor terms, such as a condition, procedure, medicine, symptom, test, treatment, or safety issue.
6. Use natural wording, but do not remove important medical anchor terms.
7. Prefer first-person wording when natural, such as "I", "my", "I'm", "I've been", "I was told", or "my doctor said".
8. Do not invent specific personal details such as exact age, pregnancy status, symptoms, prior treatment, or medical history unless clearly supported by the pool.
9. Prefer conditional patient wording when needed, such as "If I...", "I was told...", or "My doctor mentioned..." instead of inventing a personal history.
10. Caregiver wording is allowed when natural, such as "my partner", "my child", or "my parent".
11. Avoid stiff phrases such as "key considerations", "regimen", "adequate", "management", "surveillance", "contraindications", "efficacy", "risk-benefit", "administration", "intervention", "protocol", or "outcomes", unless the term is truly necessary.
12. Do not make the question vague, such as "What should I do next?", "Is this safe for me?", or "How do I get ready for my test?"
13. Do not mention "guideline", "passage", "document", "according to this", or "based on these passages" in the question.
14. Usually write one sentence, around 8 to 30 words.
15. retrieval_anchor_terms must contain the 1 to 3 strongest terms for later FAISS retrieval.
16. patient_keywords should include the retrieval_anchor_terms first, then add other patient-friendly terms if useful.
17. Return questions=[] only if the pool is mostly administrative/methodological/reference-like or has no coherent patient-relevant topic.

Good question style:
- "I'm having a colonoscopy soon. How do I choose a bowel prep that's safe for me?"
- "How do I know if my bowel prep worked well enough before my colonoscopy?"
- "I've been taking benzodiazepines for a while. How can I stop without bad withdrawal symptoms?"
- "I'm starting cancer treatment soon. Should I freeze eggs or embryos before treatment?"
- "Should I bank sperm before starting cancer treatment?"
- "Can erythromycin help my gastroparesis symptoms, and what side effects should I know about?"
- "My pain is still bad after a C-section. When are opioids actually needed?"
- "I've tried quitting smoking before. Should I consider using e-cigarettes?"
- "If I have Barrett's Esophagus, how often should I get an endoscopy?"

Bad question style:
- "What should I consider when choosing a bowel preparation regimen?"
- "How can I ensure my bowel preparation is adequate?"
- "What are the key considerations for tapering benzodiazepines safely?"
- "What are the options and considerations for sperm banking?"
- "What is the role of erythromycin in gastroparesis management?"
- "How do I get ready for my test?"
- "Is this medicine safe for me?"
- "What should I do next?"
- "I'm over 65 with early-stage breast cancer. Can I skip the sentinel lymph node biopsy?" unless age is clearly central in the passages.
- "I get nauseous easily. How can I minimize side effects?" unless nausea is clearly central in the passages.
"""

def build_messages(pool: Dict[str, Any], max_source_chars: int) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": make_user_prompt(pool, max_source_chars=max_source_chars)},
    ]


def build_request(
    pool: Dict[str, Any],
    sample_idx: int,
    model: str,
    max_output_tokens: int,
    max_source_chars: int,
) -> Dict[str, Any]:
    return {
        "custom_id": f"pool-{sample_idx}",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": build_messages(pool, max_source_chars=max_source_chars),
            "max_output_tokens": int(max_output_tokens),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "group_patient_questions",
                    "strict": True,
                    "schema": GROUP_QUESTION_SCHEMA,
                }
            },
        },
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


def validate_group_questions(raw: Dict[str, Any], pool: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
    pool_id = norm_ws(pool.get("group_id"))
    guideline_id = norm_ws(pool.get("guideline_id"))

    out: Dict[str, Any] = {
        "pool_id": norm_ws(raw.get("pool_id")) or pool_id,
        "guideline_id": norm_ws(raw.get("guideline_id")) or guideline_id,
        "questions": [],
        "skip_reason": norm_ws(raw.get("skip_reason")),
    }

    qs = raw.get("questions", [])
    if not isinstance(qs, list):
        qs = []

    forbidden_phrases = [
        "according to the guideline",
        "according to this guideline",
        "according to the passage",
        "according to this passage",
        "according to the document",
        "based on these passages",
        "this guideline",
        "this passage",
        "the passage",
        "the guideline",
    ]

    for q in qs[:1]:
        if not isinstance(q, dict):
            continue

        question = norm_ws(q.get("question"))
        if not question:
            continue

        lower_q = question.lower()
        if any(bad in lower_q for bad in forbidden_phrases):
            continue

        qtype = norm_ws(q.get("question_type"))
        if qtype not in QUESTION_TYPES:
            qtype = "other"

        perspective = norm_ws(q.get("perspective"))
        if perspective not in PERSPECTIVES:
            perspective = "general_health_info_seeker"

        keywords_raw = q.get("patient_keywords", [])
        if isinstance(keywords_raw, list):
            keywords = [norm_ws(x) for x in keywords_raw if norm_ws(x)][:8]
        else:
            keywords = []

        anchors_raw = q.get("retrieval_anchor_terms", [])
        if isinstance(anchors_raw, list):
            anchors = [norm_ws(x) for x in anchors_raw if norm_ws(x)][:3]
        else:
            anchors = []

        if not anchors:
            anchors = keywords[:3]

        if not anchors:
            continue

        qid = f"q_{sample_idx:06d}_{len(out['questions']) + 1:02d}"

        out["questions"].append(
            {
                "question_id": qid,
                "question": question,
                "question_type": qtype,
                "perspective": perspective,
                "patient_keywords": keywords,
                "retrieval_anchor_terms": anchors,
            }
        )

    if not out["questions"] and not out["skip_reason"]:
        out["skip_reason"] = "No valid patient-style retrievable question remained after validation."

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
            "task": "group_patient_question_generation_question_only",
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
            pool = records[sample_idx]
            pool_id = norm_ws(pool.get("group_id"))
            passages = pool.get("passages", [])
            if not pool_id or not isinstance(passages, list) or len(passages) < 5:
                skipped += 1
                continue

            req = build_request(
                pool=pool,
                sample_idx=sample_idx,
                model=args.openai_model,
                max_output_tokens=int(args.max_output_tokens),
                max_source_chars=int(args.max_source_chars),
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
        "max_source_chars": int(args.max_source_chars),
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
                pool = records[sample_idx]
                pool_id = norm_ws(pool.get("group_id"))
                passages = pool.get("passages", [])
                if not pool_id or not isinstance(passages, list) or len(passages) < 5:
                    skipped += 1
                    continue

                req = build_request(
                    pool=pool,
                    sample_idx=sample_idx,
                    model=args.openai_model,
                    max_output_tokens=int(args.max_output_tokens),
                    max_source_chars=int(args.max_source_chars),
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
        max_source_chars=args.max_source_chars,
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
    total_questions = 0

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

                pool = records[sample_idx]
                raw_text = extract_output_text_from_batch_line(obj)
                q_raw = parse_json_object(raw_text)
                q_payload = validate_group_questions(q_raw, pool, sample_idx)

                ok += 1
                qs = q_payload.get("questions", [])
                if isinstance(qs, list):
                    for q_obj in qs:
                        if isinstance(q_obj, dict):
                            fout.write(json.dumps(q_obj, ensure_ascii=False) + "\n")
                            total_questions += 1

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
        "total_questions": total_questions,
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
    ap_build.add_argument("--max_output_tokens", type=int, default=900)
    ap_build.add_argument("--max_source_chars", type=int, default=1000)
    ap_build.set_defaults(func=cmd_build)

    ap_build_split = sub.add_parser("build_split")
    ap_build_split.add_argument("--src", required=True)
    ap_build_split.add_argument("--out_dir", required=True)
    ap_build_split.add_argument("--start_idx", type=int, default=0)
    ap_build_split.add_argument("--end_idx", type=int, default=-1)
    ap_build_split.add_argument("--chunk_size", type=int, default=30)
    ap_build_split.add_argument("--job_name", default="group_question_batch")
    ap_build_split.add_argument("--openai_model", default="gpt-4o")
    ap_build_split.add_argument("--max_output_tokens", type=int, default=900)
    ap_build_split.add_argument("--max_source_chars", type=int, default=1000)
    ap_build_split.set_defaults(func=cmd_build_split)

    ap_submit = sub.add_parser("submit")
    ap_submit.add_argument("--input_jsonl", required=True)
    ap_submit.add_argument("--openai_model", default="gpt-4o")
    ap_submit.add_argument("--completion_window", default="24h")
    ap_submit.add_argument("--metadata_name", default="group-question-batch")
    ap_submit.add_argument("--batch_meta_out", default="")
    ap_submit.set_defaults(func=cmd_submit)

    ap_run = sub.add_parser("run_split")
    ap_run.add_argument("--src", required=True)
    ap_run.add_argument("--out_dir", required=True)
    ap_run.add_argument("--start_idx", type=int, default=0)
    ap_run.add_argument("--end_idx", type=int, default=-1)
    ap_run.add_argument("--chunk_size", type=int, default=30)
    ap_run.add_argument("--job_name", default="group_question_batch")
    ap_run.add_argument("--openai_model", default="gpt-4o")
    ap_run.add_argument("--max_output_tokens", type=int, default=900)
    ap_run.add_argument("--max_source_chars", type=int, default=1000)
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