#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Retrieve top passages for generated BIRMAC questions.

Pipeline:
  questions JSONL
  -> encode questions with Qwen3-Embedding-4B
  -> FAISS top-k recall from prebuilt passage index
  -> rerank candidates with Qwen3-Reranker-4B
  -> save top-5 passages per question

Input:
  data/group_questions_gpt5_20250807_full_tok2000_raw.jsonl
  data/faiss_passage_index_2025_strict_qwen3emb4b/index.faiss
  data/faiss_passage_index_2025_strict_qwen3emb4b/passages.jsonl

Output:
  data/group_questions_gpt5_20250807_full_tok2000_retrieved_top5_qwen3rerank.jsonl
"""

import argparse
import json
import os
import time
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import faiss

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Basic IO
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


def append_jsonl(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def norm_ws(x: Any) -> str:
    return " ".join(str(x or "").replace("\n", " ").split())


def as_list_str(x: Any) -> List[str]:
    if isinstance(x, list):
        return [norm_ws(v) for v in x if norm_ws(v)]
    if isinstance(x, str) and norm_ws(x):
        return [norm_ws(x)]
    return []


def load_done_ids(out_jsonl: Path) -> set:
    done = set()
    if not out_jsonl.exists():
        return done
    with out_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            qid = obj.get("question_id")
            if isinstance(qid, str) and qid.strip():
                done.add(qid.strip())
    return done


# ============================================================
# Question helpers
# ============================================================

def get_question_id(row: Dict[str, Any], idx: int) -> str:
    qid = row.get("question_id")
    if isinstance(qid, str) and qid.strip():
        return qid.strip()
    return f"q_{idx + 1:06d}"


def get_question(row: Dict[str, Any]) -> str:
    for k in ["question", "query"]:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return norm_ws(v)
    return ""


def build_question_search_text(row: Dict[str, Any], include_anchors: bool = True) -> str:
    q = get_question(row)

    if not include_anchors:
        return q

    anchors = []
    anchors.extend(as_list_str(row.get("retrieval_anchor_terms")))
    anchors.extend(as_list_str(row.get("patient_keywords")))

    # Dedup
    out = []
    seen = set()
    for a in anchors:
        k = a.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(a)

    if out:
        return q + "\nRelevant terms: " + "; ".join(out[:12])
    return q


def build_rerank_query(row: Dict[str, Any], include_anchors: bool = False) -> str:
    q = get_question(row)
    if not include_anchors:
        return q

    anchors = []
    anchors.extend(as_list_str(row.get("retrieval_anchor_terms")))
    anchors.extend(as_list_str(row.get("patient_keywords")))

    out = []
    seen = set()
    for a in anchors:
        k = a.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(a)

    if out:
        return q + "\nRelevant medical terms: " + "; ".join(out[:8])
    return q


# ============================================================
# Passage helpers
# ============================================================

def build_passage_for_rerank(p: Dict[str, Any], max_chars: int) -> str:
    parts = []

    main_topic = norm_ws(p.get("main_topic"))
    if main_topic:
        parts.append(f"Topic: {main_topic}")

    summary = norm_ws(p.get("semantic_summary"))
    if summary:
        parts.append(f"Summary: {summary}")

    keywords = as_list_str(p.get("patient_language_keywords"))
    if keywords:
        parts.append("Patient keywords: " + "; ".join(keywords[:12]))

    text = norm_ws(p.get("text"))
    if text:
        parts.append("Guideline passage: " + text)

    doc = "\n".join(parts).strip()

    if max_chars > 0 and len(doc) > max_chars:
        doc = doc[:max_chars].rsplit(" ", 1)[0].strip()

    return doc


def compact_passage_output(
    p: Dict[str, Any],
    rank: int,
    faiss_rank: int,
    faiss_score: float,
    rerank_score: float,
    text_max_chars: int,
) -> Dict[str, Any]:
    text = norm_ws(p.get("text"))
    if text_max_chars > 0 and len(text) > text_max_chars:
        text = text[:text_max_chars].rsplit(" ", 1)[0].strip()

    return {
        "rank": int(rank),
        "passage_id": p.get("passage_id", ""),
        "guideline_id": p.get("guideline_id", ""),
        "source_pdf": p.get("source_pdf", ""),
        "page_start": p.get("page_start", None),
        "page_end": p.get("page_end", None),
        "section": p.get("section", ""),
        "content_type": p.get("content_type", ""),
        "main_topic": p.get("main_topic", ""),
        "semantic_summary": p.get("semantic_summary", ""),
        "patient_language_keywords": p.get("patient_language_keywords", []),
        "faiss_rank": int(faiss_rank),
        "faiss_score": float(faiss_score),
        "rerank_score": float(rerank_score),
        "text": text,
    }


# ============================================================
# Qwen3 Reranker
# ============================================================

class Qwen3YesNoReranker:
    """
    Direct Qwen3-Reranker scoring.

    Score = P("yes" | query, document) over yes/no logits.
    This is not a calibrated probability, but works well for ranking.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: str = "float16",
        max_length: int = 2048,
        batch_size: int = 8,
        instruction: str = "",
    ):
        self.model_name = model_name
        self.device = device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.instruction = instruction or (
            "Given a patient medical question and a clinical guideline passage, "
            "judge whether the passage contains information that is useful for answering the question."
        )

        dtype = None
        if torch_dtype.lower() in {"float16", "fp16", "half"}:
            dtype = torch.float16
        elif torch_dtype.lower() in {"bfloat16", "bf16"}:
            dtype = torch.bfloat16
        elif torch_dtype.lower() in {"float32", "fp32"}:
            dtype = torch.float32

        print(f"[INFO] loading reranker: {model_name}", flush=True)
        print(f"[INFO] reranker device : {self.device}", flush=True)
        print(f"[INFO] reranker dtype  : {torch_dtype}", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            trust_remote_code=True,
            local_files_only=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }
        if dtype is not None:
            kwargs["torch_dtype"] = dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **kwargs,
        ).eval()

        self.model.to(self.device)

        self.yes_id = self._single_token_id("yes")
        self.no_id = self._single_token_id("no")

        print(f"[INFO] yes_id={self.yes_id} no_id={self.no_id}", flush=True)

    def _single_token_id(self, word: str) -> int:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            raise RuntimeError(f"Could not tokenize word: {word}")
        if len(ids) > 1:
            print(f"[WARN] word {word!r} tokenized into {ids}; using last token {ids[-1]}", flush=True)
        return int(ids[-1])

    def _format_prompt(self, query: str, document: str) -> str:
        system = (
            'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
            'Note that the answer can only be "yes" or "no".'
        )

        return (
            "<|im_start|>system\n"
            f"{system}<|im_end|>\n"
            "<|im_start|>user\n"
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )

    @torch.inference_mode()
    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        scores: List[float] = []

        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            prompts = [self._format_prompt(q, d) for q, d in batch]

            inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            out = self.model(**inputs)
            logits = out.logits[:, -1, :]

            yes_no = torch.stack(
                [logits[:, self.no_id], logits[:, self.yes_id]],
                dim=1,
            )
            probs = torch.softmax(yes_no.float(), dim=1)[:, 1]
            scores.extend(probs.detach().cpu().numpy().astype("float32").tolist())

        return [float(x) for x in scores]


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--questions_jsonl", required=True)
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--out_jsonl", required=True)

    ap.add_argument("--embedding_model", default="Qwen/Qwen3-Embedding-4B")
    ap.add_argument("--reranker_model", default="Qwen/Qwen3-Reranker-4B")

    ap.add_argument("--top_k_recall", type=int, default=50)
    ap.add_argument("--top_k_output", type=int, default=5)

    ap.add_argument("--embed_batch_size", type=int, default=32)
    ap.add_argument("--rerank_batch_size", type=int, default=8)
    ap.add_argument("--embedding_max_seq_length", type=int, default=1024)
    ap.add_argument("--rerank_max_length", type=int, default=2048)
    ap.add_argument("--rerank_doc_max_chars", type=int, default=2500)
    ap.add_argument("--output_text_max_chars", type=int, default=2000)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--torch_dtype", default="float16")

    ap.add_argument("--include_anchors_in_faiss_query", action="store_true")
    ap.add_argument("--include_anchors_in_rerank_query", action="store_true")

    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--limit", type=int, default=-1)

    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    questions_path = Path(args.questions_jsonl)
    index_dir = Path(args.index_dir)
    out_path = Path(args.out_jsonl)

    index_path = index_dir / "index.faiss"
    passages_path = index_dir / "passages.jsonl"
    meta_path = index_dir / "meta.json"

    if not questions_path.exists():
        raise FileNotFoundError(f"missing questions_jsonl: {questions_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"missing FAISS index: {index_path}")
    if not passages_path.exists():
        raise FileNotFoundError(f"missing passages jsonl: {passages_path}")

    if args.overwrite and out_path.exists():
        out_path.unlink()

    print("=" * 90)
    print("[INFO] retrieve questions with FAISS + Qwen3 reranker")
    print(f"[INFO] questions_jsonl : {questions_path}")
    print(f"[INFO] index_dir       : {index_dir}")
    print(f"[INFO] out_jsonl       : {out_path}")
    print(f"[INFO] embedding_model : {args.embedding_model}")
    print(f"[INFO] reranker_model  : {args.reranker_model}")
    print(f"[INFO] top_k_recall    : {args.top_k_recall}")
    print(f"[INFO] top_k_output    : {args.top_k_output}")
    print("=" * 90)

    questions_all = read_jsonl(questions_path)
    print(f"[INFO] loaded questions: {len(questions_all)}", flush=True)

    if args.start_idx > 0 or args.limit > 0:
        start = max(0, args.start_idx)
        end = len(questions_all) if args.limit < 0 else min(len(questions_all), start + args.limit)
        questions = questions_all[start:end]
        print(f"[INFO] sliced questions: start={start}, end={end}, n={len(questions)}", flush=True)
    else:
        questions = questions_all

    done_ids = load_done_ids(out_path)
    if done_ids and not args.overwrite:
        before = len(questions)
        questions = [
            r for i, r in enumerate(questions)
            if get_question_id(r, i) not in done_ids
        ]
        print(f"[INFO] resume mode: done={len(done_ids)}, pending={len(questions)} from slice_n={before}", flush=True)

    if not questions:
        print("[DONE] no pending questions.", flush=True)
        return

    passages = read_jsonl(passages_path)
    print(f"[INFO] loaded passages: {len(passages)}", flush=True)

    index = faiss.read_index(str(index_path))
    print(f"[INFO] faiss ntotal   : {index.ntotal}", flush=True)

    if index.ntotal != len(passages):
        raise RuntimeError(f"FAISS ntotal != passages: {index.ntotal} vs {len(passages)}")

    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"[INFO] index model    : {meta.get('model_name')}", flush=True)
        print(f"[INFO] index dim      : {meta.get('embedding_dim')}", flush=True)

    # ------------------------------------------------------------
    # Encode all pending questions, then FAISS search.
    # ------------------------------------------------------------
    print()
    print("=" * 90)
    print("[INFO] loading embedding model for question encoding", flush=True)

    embed_device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"

    emb_model_kwargs = {}
    if args.torch_dtype.lower() in {"float16", "fp16", "half"}:
        emb_model_kwargs["torch_dtype"] = torch.float16
    elif args.torch_dtype.lower() in {"bfloat16", "bf16"}:
        emb_model_kwargs["torch_dtype"] = torch.bfloat16

    try:
        emb_model = SentenceTransformer(
            args.embedding_model,
            device=embed_device,
            trust_remote_code=True,
            model_kwargs=emb_model_kwargs,
        )
    except TypeError:
        emb_model = SentenceTransformer(
            args.embedding_model,
            device=embed_device,
            trust_remote_code=True,
        )

    if args.embedding_max_seq_length > 0:
        old_len = getattr(emb_model, "max_seq_length", None)
        emb_model.max_seq_length = int(args.embedding_max_seq_length)
        print(f"[INFO] embedding max_seq_length: {old_len} -> {emb_model.max_seq_length}", flush=True)

    q_texts = [
        build_question_search_text(r, include_anchors=args.include_anchors_in_faiss_query)
        for r in questions
    ]

    t0 = time.time()
    q_emb = emb_model.encode(
        q_texts,
        batch_size=args.embed_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    q_emb = np.asarray(q_emb, dtype="float32")
    print(f"[INFO] encoded questions shape: {q_emb.shape}", flush=True)
    print(f"[INFO] question encode seconds: {time.time() - t0:.2f}", flush=True)

    recall_k = min(int(args.top_k_recall), int(index.ntotal))
    print(f"[INFO] FAISS search top_k_recall={recall_k}", flush=True)
    faiss_scores, faiss_ids = index.search(q_emb, recall_k)

    # Free embedding model before loading reranker.
    del emb_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # Load reranker.
    # ------------------------------------------------------------
    print()
    print("=" * 90)
    print("[INFO] loading Qwen3 reranker", flush=True)

    reranker = Qwen3YesNoReranker(
        model_name=args.reranker_model,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_length=args.rerank_max_length,
        batch_size=args.rerank_batch_size,
    )

    # ------------------------------------------------------------
    # Rerank per question and save incrementally.
    # ------------------------------------------------------------
    print()
    print("=" * 90)
    print("[INFO] reranking and saving", flush=True)

    num_saved = 0

    for local_i, qrow in enumerate(questions):
        global_i = args.start_idx + local_i if args.start_idx > 0 else local_i
        qid = get_question_id(qrow, global_i)
        question = get_question(qrow)

        if not question:
            print(f"[SKIP] empty question qid={qid}", flush=True)
            continue

        ids = faiss_ids[local_i].tolist()
        scores = faiss_scores[local_i].tolist()

        candidates = []
        seen_pid = set()

        for faiss_rank, (pid, fscore) in enumerate(zip(ids, scores), start=1):
            if pid < 0 or pid >= len(passages):
                continue

            p = passages[pid]
            passage_id = str(p.get("passage_id", ""))
            if passage_id in seen_pid:
                continue
            seen_pid.add(passage_id)

            doc = build_passage_for_rerank(
                p,
                max_chars=args.rerank_doc_max_chars,
            )
            if not doc:
                continue

            candidates.append({
                "faiss_row": int(pid),
                "faiss_rank": int(faiss_rank),
                "faiss_score": float(fscore),
                "passage": p,
                "doc_for_rerank": doc,
            })

        rerank_query = build_rerank_query(
            qrow,
            include_anchors=args.include_anchors_in_rerank_query,
        )

        pairs = [(rerank_query, c["doc_for_rerank"]) for c in candidates]
        rerank_scores = reranker.score_pairs(pairs) if pairs else []

        for c, rs in zip(candidates, rerank_scores):
            c["rerank_score"] = float(rs)

        candidates.sort(
            key=lambda x: (x.get("rerank_score", -1.0), x.get("faiss_score", -1.0)),
            reverse=True,
        )

        top = candidates[: int(args.top_k_output)]

        retrieved_passages = []
        for rank, c in enumerate(top, start=1):
            retrieved_passages.append(
                compact_passage_output(
                    p=c["passage"],
                    rank=rank,
                    faiss_rank=c["faiss_rank"],
                    faiss_score=c["faiss_score"],
                    rerank_score=c["rerank_score"],
                    text_max_chars=args.output_text_max_chars,
                )
            )

        out_item = {
            "question_id": qid,
            "question": question,
            "question_type": qrow.get("question_type", ""),
            "perspective": qrow.get("perspective", ""),
            "patient_keywords": qrow.get("patient_keywords", []),
            "retrieval_anchor_terms": qrow.get("retrieval_anchor_terms", []),
            "retrieval_config": {
                "embedding_model": args.embedding_model,
                "reranker_model": args.reranker_model,
                "index_dir": str(index_dir),
                "top_k_recall": int(args.top_k_recall),
                "top_k_output": int(args.top_k_output),
                "include_anchors_in_faiss_query": bool(args.include_anchors_in_faiss_query),
                "include_anchors_in_rerank_query": bool(args.include_anchors_in_rerank_query),
            },
            "retrieved_passages": retrieved_passages,
        }

        append_jsonl(out_item, out_path)
        num_saved += 1

        if num_saved % 10 == 0 or num_saved == 1:
            print(
                f"[SAVE] {num_saved}/{len(questions)} qid={qid} "
                f"candidates={len(candidates)} top={len(retrieved_passages)} "
                f"best_rerank={retrieved_passages[0]['rerank_score'] if retrieved_passages else None}",
                flush=True,
            )

    print()
    print("[DONE] saved questions:", num_saved)
    print("[DONE] out_jsonl:", out_path)


if __name__ == "__main__":
    main()