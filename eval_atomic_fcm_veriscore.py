import os
import json
import argparse
from pathlib import Path
from statistics import mean
from collections import Counter, defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


AUTOAIS_MODEL = os.environ.get(
    "AUTOAIS_MODEL",
    "google/t5_xxl_true_nli_mixture"
)

LOCAL_FILES_ONLY = os.environ.get("LOCAL_FILES_ONLY", "1") != "0"
FCM_MAX_INPUT_TOKENS = int(os.environ.get("FCM_MAX_INPUT_TOKENS", "4096"))

autoais_model = None
autoais_tokenizer = None
AUTOAIS_CACHE = {}


def load_jsonl(path):
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_autoais_loaded():
    global autoais_model, autoais_tokenizer

    if autoais_model is not None:
        return

    print(f"[INFO] Loading FCM/NLI model: {AUTOAIS_MODEL}")
    print(f"[INFO] local_files_only={LOCAL_FILES_ONLY}")
    print(f"[INFO] FCM_MAX_INPUT_TOKENS={FCM_MAX_INPUT_TOKENS}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    autoais_model = AutoModelForSeq2SeqLM.from_pretrained(
        AUTOAIS_MODEL,
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=LOCAL_FILES_ONLY,
    )
    autoais_tokenizer = AutoTokenizer.from_pretrained(
        AUTOAIS_MODEL,
        use_fast=False,
        local_files_only=LOCAL_FILES_ONLY,
    )
    autoais_model.eval()

    print("[INFO] Model loaded.")


def get_model_input_device():
    try:
        return autoais_model.device
    except Exception:
        return next(autoais_model.parameters()).device


def run_fcm_entailment(passage, claim):
    """
    FCM / AutoAIS-style NLI.

    Input:
      premise = cited guideline passage(s)
      hypothesis = atomic medical claim

    Return:
      entailment = 1 if premise entails hypothesis, else 0
      raw_output = raw decoded model output
    """
    global AUTOAIS_CACHE

    passage = passage or ""
    claim = claim or ""
    cache_key = (passage, claim)

    if cache_key in AUTOAIS_CACHE:
        return AUTOAIS_CACHE[cache_key]

    input_text = f"premise: {passage} hypothesis: {claim}"

    encoded = autoais_tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=FCM_MAX_INPUT_TOKENS,
    )
    input_ids = encoded.input_ids.to(get_model_input_device())

    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=10)

    raw = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    entailment = 1 if raw == "1" else 0

    AUTOAIS_CACHE[cache_key] = (entailment, raw)
    return entailment, raw


def format_passage(p):
    """
    Format one cited guideline passage as FCM premise.
    Keep metadata because it can help debugging, but the key content is text.
    """
    if not p:
        return ""

    pid = p.get("pid", "")
    passage_id = p.get("passage_id", "")
    guideline_id = p.get("guideline_id", "")
    main_topic = p.get("main_topic", "")
    section = p.get("section", "")
    text = p.get("text") or p.get("passage") or p.get("content") or ""

    return (
        f"Passage {pid}\n"
        f"passage_id: {passage_id}\n"
        f"guideline_id: {guideline_id}\n"
        f"main_topic: {main_topic}\n"
        f"section: {section}\n"
        f"{text}"
    )


def get_cited_passages(item, fact):
    """
    Prefer fact['cited_passages'].
    Fallback: map fact['citation_ids'] to item['passages'] by pid.
    """
    cited = fact.get("cited_passages") or []
    if cited:
        return cited

    citation_ids = fact.get("citation_ids") or []
    passages = item.get("passages") or item.get("top_passages") or []

    by_pid = {}
    for i, p in enumerate(passages, start=1):
        pid = p.get("pid", i)
        try:
            pid = int(pid)
        except Exception:
            pid = i
        by_pid[pid] = p

    out = []
    for cid in citation_ids:
        try:
            cid = int(cid)
        except Exception:
            continue
        if cid in by_pid:
            out.append(by_pid[cid])

    return out


def is_evaluable_claim(item, fact):
    """
    Minimal engineering filter only.

    We DO NOT filter by:
      - source_sentence_role
      - advice-like phrasing
      - shared_decision_making
      - clinical importance
      - recommendation wording

    We only require:
      1. non-empty claim text
      2. at least one cited passage
    """
    claim = (fact.get("fact") or "").strip()
    if not claim:
        return False, "empty_claim"

    cited_passages = get_cited_passages(item, fact)
    if not cited_passages:
        return False, "no_cited_passage"

    return True, "kept"


def verify_one_fact(item, fact, args):
    claim = (fact.get("fact") or "").strip()
    cited_passages = get_cited_passages(item, fact)

    keep, filter_reason = is_evaluable_claim(item, fact)

    base = {
        "question_idx": item.get("question_idx"),
        "question_id": item.get("question_id"),
        "question_type": item.get("question_type"),
        "question": item.get("question"),
        "answer": item.get("answer") or item.get("answer_text"),

        "atomic_fact_id": fact.get("atomic_fact_id"),
        "claim": claim,
        "source_sentence_id": fact.get("source_sentence_id"),
        "source_sentence_role": fact.get("source_sentence_role"),
        "citation_ids": fact.get("citation_ids") or [],
        "num_cited_passages": len(cited_passages),

        "filter_keep": keep,
        "filter_reason": filter_reason,
    }

    if not keep:
        base.update({
            "joint_entailment": None,
            "joint_raw_output": None,
            "individual_entailments": [],
        })
        return base

    # VeriScore / FactScore-style main score:
    # verify claim against the full evidence list jointly.
    joint_premise = "\n\n".join(format_passage(p) for p in cited_passages)
    joint_entailment, joint_raw = run_fcm_entailment(joint_premise, claim)

    individual = []
    if args.individual_diagnostics:
        for p in cited_passages:
            p_text = format_passage(p)
            ent, raw = run_fcm_entailment(p_text, claim)
            individual.append({
                "pid": p.get("pid"),
                "passage_id": p.get("passage_id"),
                "guideline_id": p.get("guideline_id"),
                "main_topic": p.get("main_topic"),
                "entailment": ent,
                "raw_output": raw,
            })

    base.update({
        "joint_entailment": joint_entailment,
        "joint_raw_output": joint_raw,
        "individual_entailments": individual,
    })
    return base


def safe_rate(num, den):
    if den == 0:
        return None
    return num / den


def aggregate_item(item, claim_rows):
    evaluated = [r for r in claim_rows if r["filter_keep"]]
    supported = [r for r in evaluated if r["joint_entailment"] == 1]
    unsupported = [r for r in evaluated if r["joint_entailment"] == 0]
    skipped = [r for r in claim_rows if not r["filter_keep"]]

    role_counter_raw = Counter((r.get("source_sentence_role") or "") for r in claim_rows)
    role_counter_eval = Counter((r.get("source_sentence_role") or "") for r in evaluated)
    role_counter_supported = Counter((r.get("source_sentence_role") or "") for r in supported)

    role_support_rates = {}
    for role, n_eval in role_counter_eval.items():
        role_support_rates[role] = safe_rate(role_counter_supported[role], n_eval)

    return {
        "question_idx": item.get("question_idx"),
        "question_id": item.get("question_id"),
        "question_type": item.get("question_type"),
        "question": item.get("question"),
        "answer": item.get("answer") or item.get("answer_text"),

        "num_atomic_facts_raw": len(claim_rows),
        "num_atomic_facts_evaluated": len(evaluated),
        "num_atomic_facts_supported": len(supported),
        "num_atomic_facts_unsupported": len(unsupported),
        "num_atomic_facts_skipped": len(skipped),

        "atomic_fcm_support_rate": safe_rate(len(supported), len(evaluated)),

        "skipped_reasons": dict(Counter(r["filter_reason"] for r in skipped)),
        "source_sentence_role_counts_raw": dict(role_counter_raw),
        "source_sentence_role_counts_evaluated": dict(role_counter_eval),
        "source_sentence_role_counts_supported": dict(role_counter_supported),
        "source_sentence_role_support_rates": role_support_rates,
    }


def aggregate_dataset(per_claim_rows, per_item_rows):
    evaluated = [r for r in per_claim_rows if r["filter_keep"]]
    supported = [r for r in evaluated if r["joint_entailment"] == 1]
    unsupported = [r for r in evaluated if r["joint_entailment"] == 0]
    skipped = [r for r in per_claim_rows if not r["filter_keep"]]

    item_rates = [
        r["atomic_fcm_support_rate"]
        for r in per_item_rows
        if r["atomic_fcm_support_rate"] is not None
    ]

    role_raw = Counter((r.get("source_sentence_role") or "") for r in per_claim_rows)
    role_eval = Counter((r.get("source_sentence_role") or "") for r in evaluated)
    role_supported = Counter((r.get("source_sentence_role") or "") for r in supported)

    role_support_rates = {}
    for role, n_eval in role_eval.items():
        role_support_rates[role] = safe_rate(role_supported[role], n_eval)

    qtype_eval = Counter((r.get("question_type") or "") for r in evaluated)
    qtype_supported = Counter((r.get("question_type") or "") for r in supported)

    question_type_support_rates = {}
    for qt, n_eval in qtype_eval.items():
        question_type_support_rates[qt] = safe_rate(qtype_supported[qt], n_eval)

    return {
        "num_items": len(per_item_rows),

        "num_claims_raw": len(per_claim_rows),
        "num_claims_evaluated": len(evaluated),
        "num_claims_supported": len(supported),
        "num_claims_unsupported": len(unsupported),
        "num_claims_skipped": len(skipped),

        "claim_level_atomic_support_rate_micro": safe_rate(len(supported), len(evaluated)),
        "answer_level_atomic_support_rate_macro": float(mean(item_rates)) if item_rates else None,

        "num_items_with_no_evaluated_claims": sum(
            1 for r in per_item_rows if r["atomic_fcm_support_rate"] is None
        ),
        "num_items_support_rate_gt_0_8": sum(
            1 for r in per_item_rows
            if r["atomic_fcm_support_rate"] is not None and r["atomic_fcm_support_rate"] > 0.8
        ),
        "num_items_support_rate_ge_0_8": sum(
            1 for r in per_item_rows
            if r["atomic_fcm_support_rate"] is not None and r["atomic_fcm_support_rate"] >= 0.8
        ),
        "num_items_support_rate_eq_1": sum(
            1 for r in per_item_rows
            if r["atomic_fcm_support_rate"] is not None and r["atomic_fcm_support_rate"] == 1.0
        ),

        "skipped_reasons": dict(Counter(r["filter_reason"] for r in skipped)),

        "source_sentence_role_counts_raw": dict(role_raw),
        "source_sentence_role_counts_evaluated": dict(role_eval),
        "source_sentence_role_counts_supported": dict(role_supported),
        "source_sentence_role_support_rates": role_support_rates,

        "question_type_counts_evaluated": dict(qtype_eval),
        "question_type_counts_supported": dict(qtype_supported),
        "question_type_support_rates": question_type_support_rates,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL, e.g. data/answers_gpt4o_full_atomic_facts.jsonl"
    )
    parser.add_argument(
        "--out_prefix",
        required=True,
        help="Output prefix, e.g. data/answers_gpt4o_full_atomic_fcm_veriscore"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start row index in input JSONL."
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=-1,
        help="End row index, exclusive. -1 means until end."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Debug only: after start/end slicing, keep first N rows. -1 means no limit."
    )
    parser.add_argument(
        "--individual_diagnostics",
        action="store_true",
        help="Also run FCM on each cited passage individually. Slower; useful for debugging over-citation."
    )
    parser.add_argument(
        "--write_augmented",
        action="store_true",
        help="Also write an augmented item-level JSONL with FCM results attached to each atomic fact."
    )

    args = parser.parse_args()

    all_rows = load_jsonl(args.input)

    start = max(args.start_idx, 0)
    end = len(all_rows) if args.end_idx is None or args.end_idx < 0 else min(args.end_idx, len(all_rows))
    rows = all_rows[start:end]

    if args.limit is not None and args.limit > 0:
        rows = rows[:args.limit]

    print(f"[INFO] Input file: {args.input}")
    print(f"[INFO] Total rows in input: {len(all_rows)}")
    print(f"[INFO] Evaluating rows: start={start}, end={end}, actual={len(rows)}")

    ensure_autoais_loaded()

    per_claim_rows = []
    per_item_rows = []
    augmented_rows = []

    for item in tqdm(rows, desc="Atomic claim FCM verification"):
        facts = item.get("atomic_facts_flat") or []

        item_claim_rows = []
        fact_eval_by_id = {}

        for fact in facts:
            out = verify_one_fact(item, fact, args)
            item_claim_rows.append(out)
            per_claim_rows.append(out)
            fact_eval_by_id[out.get("atomic_fact_id")] = out

        per_item = aggregate_item(item, item_claim_rows)
        per_item_rows.append(per_item)

        if args.write_augmented:
            new_item = dict(item)
            new_facts = []
            for fact in facts:
                nf = dict(fact)
                fid = nf.get("atomic_fact_id")
                nf["atomic_fcm_eval"] = fact_eval_by_id.get(fid)
                new_facts.append(nf)
            new_item["atomic_facts_flat"] = new_facts
            new_item["atomic_fcm_item_eval"] = per_item
            augmented_rows.append(new_item)

    summary = aggregate_dataset(per_claim_rows, per_item_rows)
    summary.update({
        "input": args.input,
        "out_prefix": args.out_prefix,
        "model": AUTOAIS_MODEL,
        "local_files_only": LOCAL_FILES_ONLY,
        "fcm_max_input_tokens": FCM_MAX_INPUT_TOKENS,
        "start_idx": start,
        "end_idx": end,
        "limit": args.limit,
        "individual_diagnostics": bool(args.individual_diagnostics),
        "filters": [
            "drop only empty claims",
            "drop only claims with no cited passage"
        ],
    })

    out_prefix = Path(args.out_prefix)

    per_claim_path = str(out_prefix) + ".per_claim.jsonl"
    per_item_path = str(out_prefix) + ".per_item.jsonl"
    summary_path = str(out_prefix) + ".summary.json"

    write_jsonl(per_claim_path, per_claim_rows)
    write_jsonl(per_item_path, per_item_rows)
    write_json(summary_path, summary)

    if args.write_augmented:
        augmented_path = str(out_prefix) + ".augmented.jsonl"
        write_jsonl(augmented_path, augmented_rows)
    else:
        augmented_path = None

    print("\n[DONE]")
    print("per_claim :", per_claim_path)
    print("per_item  :", per_item_path)
    print("summary   :", summary_path)
    if augmented_path:
        print("augmented :", augmented_path)

    print("\n[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()