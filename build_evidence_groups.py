#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_guideline_id(r: Dict[str, Any]) -> str:
    gid = r.get("guideline_id")
    if gid:
        return str(gid)
    pid = str(r.get("passage_id", ""))
    return pid.split("_p")[0] if "_p" in pid else "unknown"


def compact_text(x: Any) -> str:
    if isinstance(x, list):
        return "; ".join(str(i).strip() for i in x if str(i).strip())
    if x is None:
        return ""
    return str(x).strip()


def make_embed_text(r: Dict[str, Any], use_source_text: bool = False, max_source_chars: int = 800) -> str:
    """
    Use semantic seed as the embedding text.
    This avoids boilerplate / citation / long surgical technical noise dominating similarity.
    """
    parts = [
        f"Topic: {compact_text(r.get('main_topic'))}",
        f"Summary: {compact_text(r.get('semantic_summary'))}",
        f"Question angles: {compact_text(r.get('question_angles'))}",
        f"Patient keywords: {compact_text(r.get('patient_language_keywords'))}",
        f"Content type: {compact_text(r.get('content_type'))}",
    ]

    if use_source_text:
        source = compact_text(r.get("source_text") or r.get("text") or r.get("passage_text"))
        if source:
            parts.append(f"Source text: {source[:max_source_chars]}")

    return "\n".join(p for p in parts if p.strip())


def normalize_embeddings(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def embed_rows(rows: List[Dict[str, Any]], model_name: str, batch_size: int, use_source_text: bool) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    texts = [make_embed_text(r, use_source_text=use_source_text) for r in rows]
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype("float32")


def topk_within_matrix(
    emb: np.ndarray,
    local_indices: List[int],
    anchor_local_pos: int,
    top_k: int,
) -> Tuple[List[int], List[float]]:
    """
    Return global indices and cosine scores for one anchor within a local guideline subset.
    emb is global normalized embedding matrix.
    """
    global_anchor_idx = local_indices[anchor_local_pos]
    local_emb = emb[local_indices]
    anchor_vec = emb[global_anchor_idx]
    scores = local_emb @ anchor_vec

    order = np.argsort(-scores)
    order = order[: min(top_k, len(order))]

    out_indices = [local_indices[i] for i in order]
    out_scores = [float(scores[i]) for i in order]
    return out_indices, out_scores


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_groups(
    rows: List[Dict[str, Any]],
    emb: np.ndarray,
    min_passages: int,
    max_passages: int,
    min_sim: float,
    dedup_jaccard: float,
    same_guideline_only: bool,
) -> List[Dict[str, Any]]:
    by_gid = defaultdict(list)

    if same_guideline_only:
        for i, r in enumerate(rows):
            by_gid[get_guideline_id(r)].append(i)
    else:
        by_gid["ALL"] = list(range(len(rows)))

    candidate_groups = []

    for gid, idxs in by_gid.items():
        if len(idxs) < min_passages:
            continue

        for local_pos, global_i in enumerate(idxs):
            top_indices, top_scores = topk_within_matrix(
                emb=emb,
                local_indices=idxs,
                anchor_local_pos=local_pos,
                top_k=max_passages,
            )

            # Keep anchor even if score = 1.0, filter neighbors by min_sim.
            selected = []
            selected_scores = []
            for j, score in zip(top_indices, top_scores):
                if j == global_i or score >= min_sim:
                    selected.append(j)
                    selected_scores.append(score)

            if len(selected) < min_passages:
                continue

            selected = selected[:max_passages]
            selected_scores = selected_scores[:max_passages]

            group_set = set(rows[j]["passage_id"] for j in selected)
            avg_sim = float(np.mean(selected_scores[1:])) if len(selected_scores) > 1 else 1.0
            min_group_sim = float(min(selected_scores)) if selected_scores else 0.0

            candidate_groups.append({
                "_set": group_set,
                "_score": avg_sim,
                "_min_sim": min_group_sim,
                "_indices": selected,
                "_scores": selected_scores,
                "_anchor_idx": global_i,
                "_guideline_id": gid,
            })

    # Higher average similarity first, then larger group.
    candidate_groups.sort(key=lambda x: (x["_score"], len(x["_indices"])), reverse=True)

    kept = []
    kept_sets = []

    for g in candidate_groups:
        s = g["_set"]
        too_similar = False
        for ks in kept_sets:
            if jaccard(s, ks) >= dedup_jaccard:
                too_similar = True
                break
        if too_similar:
            continue

        kept.append(g)
        kept_sets.append(s)

    output = []
    for n, g in enumerate(kept, start=1):
        anchor = rows[g["_anchor_idx"]]
        passages = []

        for rank, (idx, score) in enumerate(zip(g["_indices"], g["_scores"]), start=1):
            r = rows[idx]
            passages.append({
                "rank": rank,
                "similarity_to_anchor": round(float(score), 6),
                "passage_id": r.get("passage_id"),
                "guideline_id": get_guideline_id(r),
                "content_type": r.get("content_type"),
                "main_topic": r.get("main_topic"),
                "semantic_summary": r.get("semantic_summary"),
                "question_angles": r.get("question_angles"),
                "patient_language_keywords": r.get("patient_language_keywords"),
                "source_text": r.get("source_text") or r.get("text") or r.get("passage_text") or "",
                "page_start": r.get("page_start"),
                "page_end": r.get("page_end"),
                "source_pdf": r.get("source_pdf"),
            })

        output.append({
            "group_id": f"grp_{n:06d}",
            "guideline_id": g["_guideline_id"],
            "anchor_passage_id": anchor.get("passage_id"),
            "num_passages": len(passages),
            "avg_similarity_to_anchor": round(float(g["_score"]), 6),
            "min_similarity_to_anchor": round(float(g["_min_sim"]), 6),
            "passage_ids": [p["passage_id"] for p in passages],
            "group_topic_hint": anchor.get("main_topic", ""),
            "group_summary_hint": anchor.get("semantic_summary", ""),
            "group_question_angles_hint": anchor.get("question_angles", []),
            "group_patient_keywords_hint": anchor.get("patient_language_keywords", []),
            "passages": passages,
        })

    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_stats", default=None)
    ap.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--min_passages", type=int, default=5)
    ap.add_argument("--max_passages", type=int, default=10)
    ap.add_argument("--min_sim", type=float, default=0.55)
    ap.add_argument("--dedup_jaccard", type=float, default=0.70)
    ap.add_argument("--cross_guideline", action="store_true")
    ap.add_argument("--use_source_text_for_embedding", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input_jsonl)
    out_path = Path(args.out_jsonl)
    out_stats = Path(args.out_stats) if args.out_stats else out_path.with_suffix(".stats.json")

    rows = load_jsonl(input_path)

    # Extra safety: only use strict usable rows.
    rows = [
        r for r in rows
        if r.get("usable") is True
        and r.get("content_type") not in {"admin_or_metadata", "reference_like", "methods_or_search_strategy"}
    ]

    print(json.dumps({
        "input_jsonl": str(input_path),
        "usable_rows_after_safety_filter": len(rows),
        "model_name": args.model_name,
        "min_passages": args.min_passages,
        "max_passages": args.max_passages,
        "min_sim": args.min_sim,
        "dedup_jaccard": args.dedup_jaccard,
        "same_guideline_only": not args.cross_guideline,
        "use_source_text_for_embedding": args.use_source_text_for_embedding,
    }, ensure_ascii=False, indent=2))

    emb = embed_rows(
        rows,
        model_name=args.model_name,
        batch_size=args.batch_size,
        use_source_text=args.use_source_text_for_embedding,
    )

    groups = build_groups(
        rows=rows,
        emb=emb,
        min_passages=args.min_passages,
        max_passages=args.max_passages,
        min_sim=args.min_sim,
        dedup_jaccard=args.dedup_jaccard,
        same_guideline_only=not args.cross_guideline,
    )

    write_jsonl(out_path, groups)

    stats = {
        "input_jsonl": str(input_path),
        "out_jsonl": str(out_path),
        "num_input_rows_after_safety_filter": len(rows),
        "num_groups": len(groups),
        "avg_group_size": float(np.mean([g["num_passages"] for g in groups])) if groups else 0,
        "min_group_size": min([g["num_passages"] for g in groups]) if groups else 0,
        "max_group_size": max([g["num_passages"] for g in groups]) if groups else 0,
        "avg_similarity": float(np.mean([g["avg_similarity_to_anchor"] for g in groups])) if groups else 0,
        "min_similarity": float(np.min([g["min_similarity_to_anchor"] for g in groups])) if groups else 0,
        "max_similarity": float(np.max([g["avg_similarity_to_anchor"] for g in groups])) if groups else 0,
        "params": {
            "model_name": args.model_name,
            "min_passages": args.min_passages,
            "max_passages": args.max_passages,
            "min_sim": args.min_sim,
            "dedup_jaccard": args.dedup_jaccard,
            "same_guideline_only": not args.cross_guideline,
            "use_source_text_for_embedding": args.use_source_text_for_embedding,
        }
    }

    out_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()