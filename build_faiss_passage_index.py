#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build FAISS passage index for BIRMAC guideline passages.

Input:
  data/qg_input_2025_all_60_100_gpt4o_strict.jsonl

Output dir:
  data/faiss_passage_index_2025_strict_qwen3emb4b/
    index.faiss
    passages.jsonl
    embeddings.npy
    meta.json

Design:
  - Use Qwen/Qwen3-Embedding-4B as dense embedding model.
  - Normalize embeddings.
  - Use FAISS IndexFlatIP, so inner product == cosine similarity.
  - Each FAISS row aligns with one row in passages.jsonl.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# IO helpers
# ============================================================

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Bad JSON at {path}:{line_no}: {e}") from e
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def norm_ws(x: Any) -> str:
    return " ".join(str(x or "").replace("\n", " ").split())


def as_list_str(x: Any) -> List[str]:
    if isinstance(x, list):
        return [norm_ws(v) for v in x if norm_ws(v)]
    if isinstance(x, str) and norm_ws(x):
        return [norm_ws(x)]
    return []


# ============================================================
# Passage text construction
# ============================================================

def get_passage_id(row: Dict[str, Any], idx: int) -> str:
    for k in ["passage_id", "id", "chunk_id"]:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return f"passage_{idx:06d}"


def get_guideline_id(row: Dict[str, Any]) -> str:
    for k in ["guideline_id", "pmid", "document_id", "doc_id", "source_id"]:
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()

    passage_id = row.get("passage_id")
    if isinstance(passage_id, str) and "_" in passage_id:
        return passage_id.split("_")[0]

    return ""


def get_source_text(row: Dict[str, Any]) -> str:
    for k in ["source_text", "text", "passage_text", "content", "chunk_text"]:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return norm_ws(v)
    return ""


def build_embedding_text(row: Dict[str, Any]) -> str:
    """
    Build retrieval-friendly passage representation.

    We include semantic seed fields because patient questions are natural-language,
    while guideline passages can be technical.
    """
    main_topic = norm_ws(row.get("main_topic"))
    semantic_summary = norm_ws(row.get("semantic_summary"))

    keywords = []
    for k in [
        "patient_language_keywords",
        "patient_keywords",
        "keywords",
        "question_angles",
        "angles",
    ]:
        keywords.extend(as_list_str(row.get(k)))

    # Dedup keywords while preserving order.
    seen = set()
    keywords_clean = []
    for kw in keywords:
        kk = kw.lower()
        if kk in seen:
            continue
        seen.add(kk)
        keywords_clean.append(kw)

    source_text = get_source_text(row)

    parts = []
    if main_topic:
        parts.append(f"Topic: {main_topic}")
    if semantic_summary:
        parts.append(f"Summary: {semantic_summary}")
    if keywords_clean:
        parts.append("Patient keywords: " + "; ".join(keywords_clean[:20]))
    if source_text:
        parts.append("Guideline passage: " + source_text)

    return "\n".join(parts).strip()


def build_metadata(row: Dict[str, Any], idx: int, embedding_text: str, keep_embedding_text: bool) -> Dict[str, Any]:
    passage_id = get_passage_id(row, idx)
    guideline_id = get_guideline_id(row)
    source_text = get_source_text(row)

    meta = {
        "row_idx": idx,
        "passage_id": passage_id,
        "guideline_id": guideline_id,
        "source_pdf": row.get("source_pdf", row.get("source", "")),
        "page_start": row.get("page_start", row.get("page", None)),
        "page_end": row.get("page_end", row.get("page", None)),
        "section": row.get("section", ""),
        "word_count": row.get("word_count", None),
        "content_type": row.get("content_type", ""),
        "main_topic": row.get("main_topic", ""),
        "semantic_summary": row.get("semantic_summary", ""),
        "question_angles": row.get("question_angles", row.get("angles", [])),
        "patient_language_keywords": row.get("patient_language_keywords", row.get("patient_keywords", [])),
        "text": source_text,
    }

    if keep_embedding_text:
        meta["embedding_text"] = embedding_text

    return meta


# ============================================================
# Model helpers
# ============================================================

def resolve_dtype(name: str):
    name = (name or "").lower().strip()
    if name in ["float16", "fp16", "half"]:
        return torch.float16
    if name in ["bfloat16", "bf16"]:
        return torch.bfloat16
    if name in ["float32", "fp32", ""]:
        return None
    raise ValueError(f"Unsupported torch dtype: {name}")


def load_model(args) -> SentenceTransformer:
    dtype = resolve_dtype(args.torch_dtype)

    model_kwargs = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    print(f"[INFO] loading model: {args.model_name}", flush=True)
    print(f"[INFO] device       : {args.device}", flush=True)
    print(f"[INFO] torch_dtype  : {args.torch_dtype}", flush=True)

    # Different sentence-transformers versions have slightly different __init__ signatures.
    try:
        model = SentenceTransformer(
            args.model_name,
            device=args.device,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        )
    except TypeError:
        print("[WARN] SentenceTransformer did not accept model_kwargs/trust_remote_code combo; retrying simpler load.", flush=True)
        try:
            model = SentenceTransformer(
                args.model_name,
                device=args.device,
                trust_remote_code=True,
            )
        except TypeError:
            model = SentenceTransformer(
                args.model_name,
                device=args.device,
            )

    if args.max_seq_length > 0:
        try:
            old_len = getattr(model, "max_seq_length", None)
            model.max_seq_length = int(args.max_seq_length)
            print(f"[INFO] max_seq_length: {old_len} -> {model.max_seq_length}", flush=True)
        except Exception as e:
            print(f"[WARN] failed to set max_seq_length: {e}", flush=True)

    return model


def encode_corpus(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    print(f"[INFO] encoding texts: {len(texts)}", flush=True)
    print(f"[INFO] batch_size    : {batch_size}", flush=True)

    t0 = time.time()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    dt = time.time() - t0

    emb = np.asarray(emb, dtype="float32")
    print(f"[INFO] encoded shape : {emb.shape}", flush=True)
    print(f"[INFO] encoded dtype  : {emb.dtype}", flush=True)
    print(f"[INFO] encode seconds : {dt:.2f}", flush=True)

    return emb


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-Embedding-4B")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_seq_length", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--torch_dtype", default="float16")
    ap.add_argument("--save_embeddings", action="store_true")
    ap.add_argument("--keep_embedding_text", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--smoke_query", default="Should I use acetaminophen and NSAIDs instead of opioids for pain?")
    ap.add_argument("--smoke_top_k", type=int, default=5)
    args = ap.parse_args()

    input_path = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)

    index_path = out_dir / "index.faiss"
    passages_path = out_dir / "passages.jsonl"
    embeddings_path = out_dir / "embeddings.npy"
    meta_path = out_dir / "meta.json"

    if not input_path.exists():
        raise FileNotFoundError(f"missing input_jsonl: {input_path}")

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"out_dir exists and is not empty: {out_dir}\n"
            f"Use --overwrite to replace files."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[INFO] Build FAISS passage index")
    print(f"[INFO] input_jsonl : {input_path}")
    print(f"[INFO] out_dir     : {out_dir}")
    print(f"[INFO] model_name  : {args.model_name}")
    print("=" * 80)

    rows = read_jsonl(input_path)
    print(f"[INFO] loaded rows : {len(rows)}", flush=True)

    metas: List[Dict[str, Any]] = []
    texts: List[str] = []
    skipped = 0
    seen_passage_ids = set()

    for idx, row in enumerate(rows):
        passage_id = get_passage_id(row, idx)

        # Keep first occurrence only.
        if passage_id in seen_passage_ids:
            skipped += 1
            continue
        seen_passage_ids.add(passage_id)

        emb_text = build_embedding_text(row)
        source_text = get_source_text(row)

        if not source_text or not emb_text:
            skipped += 1
            continue

        meta = build_metadata(
            row=row,
            idx=len(metas),
            embedding_text=emb_text,
            keep_embedding_text=args.keep_embedding_text,
        )

        metas.append(meta)
        texts.append(emb_text)

    print(f"[INFO] usable passages : {len(texts)}", flush=True)
    print(f"[INFO] skipped passages: {skipped}", flush=True)

    if not texts:
        raise RuntimeError("No usable passages found.")

    model = load_model(args)
    embeddings = encode_corpus(model, texts, batch_size=args.batch_size)

    if embeddings.ndim != 2:
        raise RuntimeError(f"bad embedding shape: {embeddings.shape}")

    n, dim = embeddings.shape
    if n != len(metas):
        raise RuntimeError(f"embedding rows != metas: {n} vs {len(metas)}")

    print(f"[INFO] building FAISS IndexFlatIP dim={dim}", flush=True)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"[INFO] faiss ntotal: {index.ntotal}", flush=True)

    faiss.write_index(index, str(index_path))
    write_jsonl(metas, passages_path)

    if args.save_embeddings:
        np.save(embeddings_path, embeddings)

    meta = {
        "input_jsonl": str(input_path),
        "out_dir": str(out_dir),
        "model_name": args.model_name,
        "num_passages": int(n),
        "embedding_dim": int(dim),
        "faiss_index": "IndexFlatIP",
        "normalized_embeddings": True,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "index_path": str(index_path),
        "passages_path": str(passages_path),
        "embeddings_path": str(embeddings_path) if args.save_embeddings else None,
        "created_at_unix": time.time(),
    }
    save_json(meta, meta_path)

    print("[OK] saved index    :", index_path)
    print("[OK] saved passages :", passages_path)
    if args.save_embeddings:
        print("[OK] saved embeddings:", embeddings_path)
    print("[OK] saved meta     :", meta_path)

    # Smoke query search.
    if args.smoke_query:
        print()
        print("=" * 80)
        print("[INFO] smoke search")
        print("[QUERY]", args.smoke_query)

        q_emb = model.encode(
            [args.smoke_query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        q_emb = np.asarray(q_emb, dtype="float32")

        scores, ids = index.search(q_emb, int(args.smoke_top_k))
        for rank, (pid, score) in enumerate(zip(ids[0].tolist(), scores[0].tolist()), start=1):
            if pid < 0:
                continue
            m = metas[pid]
            print("-" * 80)
            print(f"rank={rank} score={score:.4f} row={pid}")
            print("passage_id:", m.get("passage_id"))
            print("guideline_id:", m.get("guideline_id"))
            print("main_topic:", m.get("main_topic"))
            print("summary:", norm_ws(m.get("semantic_summary"))[:300])
            print("text:", norm_ws(m.get("text"))[:500])

    print()
    print("[DONE] FAISS passage index built successfully.")


if __name__ == "__main__":
    main()