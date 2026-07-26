#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build guideline passages from PDFs.

Pipeline:
1. Read input PDFs.
2. Extract page text with pypdf.
3. Normalize text.
4. Split into paragraphs.
5. Preserve simple section headings.
6. Merge paragraphs into section-aware chunks.
7. Filter only reference-list / bibliography-like chunks.
8. Optionally use Qwen3 reranker-style yes/no scoring to remove reference chunks.
9. Save all kept passages to JSONL.

This script does NOT:
- generate questions
- retrieve passages
- classify disease categories
- filter non-reference clinical content

Important fix:
- QwenYesNoScorer no longer uses @torch.no_grad() at class-definition time.
  This means regex-only mode will not fail just because torch is not imported globally.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from pypdf import PdfReader


# ============================================================
# Basic text helpers
# ============================================================

_WORD_RE = re.compile(r"\b[0-9A-Za-z][0-9A-Za-z'\-]*\b")


def clean_text(s: str) -> str:
    s = (s or "").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n\s+", "\n", s)
    return s.strip()


def clean_passage_text(s: str) -> str:
    s = (s or "").replace("\r", "\n")
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def normalize_pdf_text(t: str) -> str:
    """
    Normalize extracted PDF text while preserving paragraph breaks when possible.
    """
    t = (t or "").replace("\r", "\n")

    # Join hyphenated line breaks: "treat-\nment" -> "treatment"
    t = re.sub(r"-\n(?=[a-z])", "", t)

    # Preserve paragraph breaks
    t = re.sub(r"\n\s*\n+", "\n\n", t)
    t = t.replace("\n\n", "<PARA>")

    # Collapse ordinary line breaks
    t = re.sub(r"\n+", " ", t)
    t = re.sub(r"[ \t]+", " ", t).strip()

    # Restore paragraph breaks
    t = t.replace("<PARA>", "\n\n")
    return t


# ============================================================
# PDF reading
# ============================================================

def read_pdf_pages(pdf_path: Path) -> List[str]:
    reader = PdfReader(str(pdf_path))
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(normalize_pdf_text(text))

    return pages


def split_paragraphs(page_text: str) -> List[str]:
    """
    Prefer paragraph breaks if present.
    Otherwise fallback to sentence-like splitting.
    """
    page_text = clean_text(page_text)

    if not page_text:
        return []

    if "\n\n" in page_text:
        paras = [p.strip() for p in page_text.split("\n\n") if p.strip()]
    else:
        paras = re.split(r"(?<=[。\.!?])\s+", page_text)
        paras = [p.strip() for p in paras if p.strip()]

    return paras


# ============================================================
# Reference-list detection
# ============================================================

REF_HEADING_RE = re.compile(
    r"^\s*(references|bibliography|literature cited|works cited|reference list)\s*$",
    re.IGNORECASE,
)

REF_HEADING_LOOSE_RE = re.compile(
    r"^\s*(references|bibliography|literature cited|works cited|reference list)\b",
    re.IGNORECASE,
)

DOI_RE = re.compile(r"\bdoi\b|10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
ETAL_RE = re.compile(r"\bet al\.?", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
BRACKET_CIT_RE = re.compile(r"\[[0-9,\s\-]+\]")
JOURNAL_RE = re.compile(
    r"\b("
    r"journal|j\.|bmj|jama|lancet|nejm|annals|ann|chest|radiology|"
    r"gastroenterology|neurology|circulation|pediatrics|surgery|"
    r"clinical|clin|medicine|med|nursing|pharmacology|"
    r"intensive care|emergency medicine|rheumatology|oncology|surg|"
    r"obstet|gynecol|anesth|analg"
    r")\b",
    re.IGNORECASE,
)

AUTHOR_PATTERN_RE = re.compile(
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+[A-Z]\.?"
    r"(?:,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+[A-Z]\.?){1,}"
)


def is_reference_heading(text: str) -> bool:
    s = clean_passage_text(text)
    if not s:
        return False

    # Very strict heading
    if REF_HEADING_RE.match(s[:80]):
        return True

    # Loose heading but short
    if len(s.split()) <= 6 and REF_HEADING_LOOSE_RE.match(s[:80]):
        return True

    return False


def reference_list_score(text: str) -> float:
    """
    Conservative heuristic score for bibliography/reference-list-like chunks.
    Higher means more reference-like.
    """
    s = clean_passage_text(text)
    if not s:
        return 1.0

    wc = max(1, count_words(s))
    score = 0.0

    doi_count = len(DOI_RE.findall(s))
    url_count = len(URL_RE.findall(s))
    etal_count = len(ETAL_RE.findall(s))
    year_count = len(YEAR_RE.findall(s))
    bracket_cit_count = len(BRACKET_CIT_RE.findall(s))

    if doi_count >= 1:
        score += 0.35
    if doi_count >= 2:
        score += 0.25

    if url_count >= 1:
        score += 0.20

    if etal_count >= 1:
        score += 0.25

    if year_count >= 3:
        score += 0.20
    if year_count >= 5:
        score += 0.15

    if bracket_cit_count >= 3:
        score += 0.20

    if JOURNAL_RE.search(s):
        score += 0.15

    if AUTHOR_PATTERN_RE.search(s):
        score += 0.30

    # Many punctuation-heavy citation fragments.
    punct_density = sum(1 for ch in s if ch in ".,;:()[]") / max(1, len(s))
    if punct_density > 0.12 and wc < 180:
        score += 0.15

    # A short chunk with many years is probably a reference fragment.
    if wc < 120 and year_count >= 3:
        score += 0.15

    return min(score, 1.0)


# ============================================================
# Section-aware chunking
# ============================================================

COMMON_SECTION_HEADINGS = {
    "abstract",
    "background",
    "introduction",
    "methods",
    "methodology",
    "recommendations",
    "recommendation",
    "diagnosis",
    "treatment",
    "management",
    "monitoring",
    "follow-up",
    "follow up",
    "prevention",
    "discussion",
    "conclusion",
    "conclusions",
    "summary",
    "key recommendations",
    "practice recommendations",
    "guideline",
    "guidelines",
    "overview",
    "scope",
    "results",
    "evidence",
    "clinical question",
    "clinical questions",
}


def looks_like_section_heading(text: str) -> bool:
    s = clean_passage_text(text)
    if not s:
        return False

    s_lower = s.lower().strip(" :.-")

    # Exact common heading
    if s_lower in COMMON_SECTION_HEADINGS:
        return True

    # Numbered headings, e.g. "1. Introduction", "2.3 Treatment"
    if re.match(r"^\d+(\.\d+)*\.?\s+[A-Z][A-Za-z ,/\-:]{2,100}$", s):
        return True

    # Short title-like line without final period
    wc = count_words(s)
    if wc <= 10 and len(s) <= 100 and not s.endswith("."):
        alpha_chars = re.sub(r"[^A-Za-z]", "", s)
        if alpha_chars and sum(c.isupper() for c in alpha_chars[:10]) >= 1:
            return True

    return False


def sentence_split(text: str) -> List[str]:
    text = clean_passage_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def tail_overlap(text: str, overlap_sentences: int = 1, max_words: int = 40) -> str:
    if overlap_sentences <= 0:
        return ""

    sents = sentence_split(text)
    if not sents:
        return ""

    overlap = " ".join(sents[-overlap_sentences:])
    words = overlap.split()

    if len(words) > max_words:
        overlap = " ".join(words[-max_words:])

    return overlap.strip()


def make_chunk(
    guideline_id: str,
    chunk_idx: int,
    text: str,
    section_title: str,
    start_page: int,
    end_page: int,
    start_para_idx: int,
    end_para_idx: int,
    source_pdf: str,
    ref_score_threshold: float,
) -> Dict[str, Any]:
    text = clean_passage_text(text)
    ref_score = reference_list_score(text)

    return {
        "guideline_id": guideline_id,
        "passage_id": f"{guideline_id}_p{chunk_idx:05d}",
        "source_pdf": source_pdf,
        "section_title": section_title,
        "page_start": int(start_page),
        "page_end": int(end_page),
        "para_start": int(start_para_idx),
        "para_end": int(end_para_idx),
        "text": text,
        "word_count": count_words(text),
        "reference_list_score": float(ref_score),
        "is_reference_like_by_regex": bool(ref_score >= ref_score_threshold),
    }


def build_chunks_from_pdf(
    pdf_path: Path,
    guideline_id: Optional[str] = None,
    min_words: int = 100,
    max_words: int = 220,
    overlap_sentences: int = 1,
    ref_score_threshold: float = 0.75,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build section-aware chunks.

    Important:
    - We do NOT filter clinical/background/method content here.
    - We only remove reference-list/bibliography-like passages.
    """
    guideline_id = guideline_id or pdf_path.stem

    pages = read_pdf_pages(pdf_path)

    chunks: List[Dict[str, Any]] = []

    current_section = ""
    in_reference_section = False

    buf: List[str] = []
    buf_start_page = 1
    buf_end_page = 1
    buf_start_para = 1
    buf_end_para = 1

    raw_candidate_chunks = 0
    removed_by_ref_heading_zone = 0
    removed_by_regex_ref = 0

    def flush() -> None:
        nonlocal buf, buf_start_page, buf_end_page, buf_start_para, buf_end_para
        nonlocal raw_candidate_chunks, removed_by_regex_ref

        if not buf:
            return

        merged = clean_passage_text(" ".join(buf))
        buf = []

        if not merged:
            return

        raw_candidate_chunks += 1

        chunk_idx = len(chunks) + 1
        chunk = make_chunk(
            guideline_id=guideline_id,
            chunk_idx=chunk_idx,
            text=merged,
            section_title=current_section,
            start_page=buf_start_page,
            end_page=buf_end_page,
            start_para_idx=buf_start_para,
            end_para_idx=buf_end_para,
            source_pdf=str(pdf_path),
            ref_score_threshold=ref_score_threshold,
        )

        if chunk["reference_list_score"] >= ref_score_threshold:
            removed_by_regex_ref += 1
            return

        chunks.append(chunk)

    for page_no, page_text in enumerate(pages, start=1):
        paras = split_paragraphs(page_text)

        for para_idx, para in enumerate(paras, start=1):
            para_clean = clean_passage_text(para)
            if not para_clean:
                continue

            # If we hit references heading, flush previous clinical chunk and skip the rest.
            if is_reference_heading(para_clean):
                flush()
                in_reference_section = True
                removed_by_ref_heading_zone += 1
                continue

            # Once references section starts, skip all following paragraphs.
            # For journal PDFs this is normally safe.
            if in_reference_section:
                removed_by_ref_heading_zone += 1
                continue

            # Section heading: flush previous chunk, update section title.
            if looks_like_section_heading(para_clean):
                flush()
                current_section = para_clean[:160]
                continue

            # Start new buffer.
            if not buf:
                buf_start_page = page_no
                buf_end_page = page_no
                buf_start_para = para_idx
                buf_end_para = para_idx

            buf.append(para_clean)
            buf_end_page = page_no
            buf_end_para = para_idx

            merged = clean_passage_text(" ".join(buf))
            wc = count_words(merged)

            # Flush when reaching max_words.
            # This preserves paragraph boundaries better than hard token slicing.
            if wc >= max_words:
                previous_text = merged
                flush()

                # Optional light overlap to reduce boundary loss.
                overlap = tail_overlap(previous_text, overlap_sentences=overlap_sentences)
                if overlap:
                    buf = [overlap]
                    buf_start_page = page_no
                    buf_end_page = page_no
                    buf_start_para = para_idx
                    buf_end_para = para_idx

    flush()

    meta = {
        "guideline_id": guideline_id,
        "source_pdf": str(pdf_path),
        "num_pages": len(pages),
        "num_kept_chunks_before_tail_merge": len(chunks),
        "num_raw_candidate_chunks": raw_candidate_chunks,
        "removed_by_ref_heading_zone": removed_by_ref_heading_zone,
        "removed_by_regex_ref": removed_by_regex_ref,
        "min_words": min_words,
        "max_words": max_words,
        "overlap_sentences": overlap_sentences,
        "ref_score_threshold": ref_score_threshold,
    }

    # Merge too-short tail chunks with previous chunk when possible.
    final_chunks: List[Dict[str, Any]] = []
    for ch in chunks:
        if final_chunks and ch["word_count"] < min_words:
            prev = final_chunks[-1]
            merged_text = clean_passage_text(prev["text"] + " " + ch["text"])

            if (
                prev.get("section_title") == ch.get("section_title")
                and count_words(merged_text) <= max_words + 80
            ):
                prev["text"] = merged_text
                prev["word_count"] = count_words(merged_text)
                prev["page_end"] = ch["page_end"]
                prev["para_end"] = ch["para_end"]
                prev["reference_list_score"] = reference_list_score(merged_text)
                prev["is_reference_like_by_regex"] = (
                    prev["reference_list_score"] >= ref_score_threshold
                )
                continue

        final_chunks.append(ch)

    # Final safety filter after tail merge.
    # Some chunks may become reference-like only after a short tail is merged.
    # Without this pass, a merged chunk can keep reference_list_score >= threshold
    # and still leak into the final JSONL.
    post_merge_removed_by_regex_ref = 0
    post_filtered_chunks: List[Dict[str, Any]] = []

    for ch in final_chunks:
        ch["text"] = clean_passage_text(ch["text"])
        ch["word_count"] = count_words(ch["text"])
        ch["reference_list_score"] = reference_list_score(ch["text"])
        ch["is_reference_like_by_regex"] = bool(ch["reference_list_score"] >= ref_score_threshold)

        if ch["reference_list_score"] >= ref_score_threshold:
            post_merge_removed_by_regex_ref += 1
            continue

        post_filtered_chunks.append(ch)

    final_chunks = post_filtered_chunks

    # Reassign passage_id after merging and final filtering.
    for i, ch in enumerate(final_chunks, start=1):
        ch["passage_id"] = f"{guideline_id}_p{i:05d}"

    meta["removed_by_regex_ref_after_tail_merge"] = post_merge_removed_by_regex_ref
    meta["num_kept_chunks_after_tail_merge"] = len(final_chunks)

    return final_chunks, meta


# ============================================================
# Optional Qwen reference filter
# ============================================================

class QwenYesNoScorer:
    """
    Reuses the reranker idea:
    <Instruct> + <Query> + <Document> -> yes/no probability.

    Here yes = "this passage is mainly a reference list / bibliography".
    """

    def __init__(
        self,
        model_path: str,
        max_length: int = 2048,
        batch_size: int = 8,
        device: Optional[str] = None,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            padding_side="left",
        )

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.model.eval()

        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        yes_ids = self.tokenizer("yes", add_special_tokens=False).input_ids
        no_ids = self.tokenizer("no", add_special_tokens=False).input_ids

        if len(yes_ids) != 1 or len(no_ids) != 1:
            raise ValueError(f"Expected single-token yes/no, got yes={yes_ids}, no={no_ids}")

        self.yes_id = yes_ids[0]
        self.no_id = no_ids[0]

        self.prefix_tokens, self.suffix_tokens = self._build_prefix_suffix()

    def _build_prefix_suffix(self):
        prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\"."
            "<|im_end|>\n"
            "<|im_start|>user\n"
        )

        suffix = (
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n"
            "</think>\n\n"
        )

        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        return prefix_tokens, suffix_tokens

    @staticmethod
    def format_instruction(instruction: str, query: str, doc: str) -> str:
        return (
            f"<Instruct>: {instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
        )

    def _process_inputs(self, pairs: List[Tuple[str, str]], instruction: str):
        formatted = [
            self.format_instruction(instruction=instruction, query=q, doc=d)
            for q, d in pairs
        ]

        enc = self.tokenizer(
            formatted,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )

        input_ids = [
            self.prefix_tokens + ids + self.suffix_tokens
            for ids in enc["input_ids"]
        ]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        )

        # If device_map=auto, send inputs to first parameter device.
        device = next(self.model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        return batch

    def score_reference_probability(self, docs: List[str]) -> List[float]:
        """
        Return P(yes), where yes means:
        the passage is mainly a reference list / bibliography fragment.
        """
        if not docs:
            return []

        query = (
            "Is this passage mainly a reference list, bibliography, citation list, "
            "DOI list, URL list, author list, or journal reference metadata, "
            "rather than clinical guideline content?"
        )

        instruction = (
            "Answer yes only if the Document should be removed because it is mainly "
            "a reference list, bibliography, citation list, DOI/URL metadata, or author/journal reference. "
            "Answer no if the Document contains normal clinical guideline content, recommendations, "
            "background, methods, diagnostic criteria, treatment information, tables, or patient-relevant content."
        )

        scores: List[float] = []

        with self.torch.no_grad():
            for start in range(0, len(docs), self.batch_size):
                batch_docs = docs[start:start + self.batch_size]
                pairs = [(query, d) for d in batch_docs]

                inputs = self._process_inputs(pairs, instruction=instruction)

                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :].float()

                yes_logits = logits[:, self.yes_id]
                no_logits = logits[:, self.no_id]

                pair_logits = self.torch.stack([no_logits, yes_logits], dim=1)
                probs = self.torch.nn.functional.softmax(pair_logits, dim=1)
                yes_probs = probs[:, 1].detach().cpu().tolist()

                scores.extend([float(x) for x in yes_probs])

        return scores


def apply_qwen_reference_filter(
    chunks: List[Dict[str, Any]],
    scorer: QwenYesNoScorer,
    threshold: float = 0.80,
) -> Tuple[List[Dict[str, Any]], int]:
    if not chunks:
        return [], 0

    texts = [c["text"] for c in chunks]
    scores = scorer.score_reference_probability(texts)

    kept = []
    removed = 0

    for ch, score in zip(chunks, scores):
        ch["qwen_reference_score"] = float(score)
        ch["is_reference_like_by_qwen"] = bool(score >= threshold)

        if score >= threshold:
            removed += 1
            continue

        kept.append(ch)

    return kept, removed


# ============================================================
# IO
# ============================================================

def write_jsonl(items: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_json(obj: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--pdf_dir", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--out_meta_json", type=str, default="")

    ap.add_argument("--min_words", type=int, default=100)
    ap.add_argument("--max_words", type=int, default=220)
    ap.add_argument("--overlap_sentences", type=int, default=1)

    ap.add_argument("--ref_score_threshold", type=float, default=0.75)

    ap.add_argument("--qwen_model", type=str, default="")
    ap.add_argument("--qwen_ref_threshold", type=float, default=0.80)
    ap.add_argument("--qwen_batch_size", type=int, default=8)
    ap.add_argument("--qwen_max_length", type=int, default=2048)

    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_jsonl = Path(args.out_jsonl)

    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"Missing pdf_dir: {pdf_dir}")

    pdfs = sorted(pdf_dir.glob("*.pdf"))

    if not pdfs:
        raise RuntimeError(f"No PDFs found in {pdf_dir}")

    print("=" * 80)
    print(f"[INFO] pdf_dir       = {pdf_dir}")
    print(f"[INFO] num_pdfs      = {len(pdfs)}")
    print(f"[INFO] out_jsonl     = {out_jsonl}")
    print(f"[INFO] min_words     = {args.min_words}")
    print(f"[INFO] max_words     = {args.max_words}")
    print(f"[INFO] ref_threshold = {args.ref_score_threshold}")
    print(f"[INFO] qwen_model    = {args.qwen_model or '<disabled>'}")

    qwen_scorer = None
    if args.qwen_model.strip():
        qwen_scorer = QwenYesNoScorer(
            model_path=args.qwen_model.strip(),
            max_length=args.qwen_max_length,
            batch_size=args.qwen_batch_size,
        )

    all_chunks: List[Dict[str, Any]] = []
    all_meta: List[Dict[str, Any]] = []

    total_removed_by_qwen = 0

    for i, pdf_path in enumerate(pdfs, start=1):
        guideline_id = pdf_path.stem

        print(f"[RUN] {i}/{len(pdfs)} {guideline_id}")

        try:
            chunks, meta = build_chunks_from_pdf(
                pdf_path=pdf_path,
                guideline_id=guideline_id,
                min_words=args.min_words,
                max_words=args.max_words,
                overlap_sentences=args.overlap_sentences,
                ref_score_threshold=args.ref_score_threshold,
            )

            if qwen_scorer is not None and chunks:
                before = len(chunks)
                chunks, removed_by_qwen = apply_qwen_reference_filter(
                    chunks=chunks,
                    scorer=qwen_scorer,
                    threshold=args.qwen_ref_threshold,
                )
                total_removed_by_qwen += removed_by_qwen
                meta["removed_by_qwen_ref_filter"] = removed_by_qwen
                meta["num_kept_after_qwen"] = len(chunks)
                print(f"      chunks regex={before}, qwen_removed={removed_by_qwen}, kept={len(chunks)}")
            else:
                meta["removed_by_qwen_ref_filter"] = 0
                meta["num_kept_after_qwen"] = len(chunks)
                print(f"      chunks kept={len(chunks)}")

            all_chunks.extend(chunks)
            all_meta.append(meta)

        except Exception as e:
            print(f"[FAIL] {pdf_path}: {repr(e)}")
            all_meta.append({
                "guideline_id": guideline_id,
                "source_pdf": str(pdf_path),
                "error": repr(e),
            })

    write_jsonl(all_chunks, out_jsonl)

    if args.out_meta_json:
        meta_path = Path(args.out_meta_json)
    else:
        meta_path = out_jsonl.with_suffix(".meta.json")

    summary = {
        "pdf_dir": str(pdf_dir),
        "num_pdfs": len(pdfs),
        "num_total_passages": len(all_chunks),
        "total_removed_by_qwen": total_removed_by_qwen,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "overlap_sentences": args.overlap_sentences,
        "ref_score_threshold": args.ref_score_threshold,
        "qwen_model": args.qwen_model,
        "qwen_ref_threshold": args.qwen_ref_threshold,
        "per_pdf": all_meta,
    }

    write_json(summary, meta_path)

    print("=" * 80)
    print(f"[DONE] saved passages -> {out_jsonl}")
    print(f"[DONE] saved meta     -> {meta_path}")
    print(f"[DONE] total passages = {len(all_chunks)}")
    print(f"[DONE] total qwen removed = {total_removed_by_qwen}")


if __name__ == "__main__":
    main()