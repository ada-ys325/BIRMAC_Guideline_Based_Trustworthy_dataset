<p align="center">
  <a href="https://www.birmingham.ac.uk/">
    <img
      src="assets/university-of-birmingham-logo.jpg"
      alt="University of Birmingham"
      height="90"
    >
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.mcmaster.ca/">
    <img
      src="assets/mcmaster-university-logo.jpg"
      alt="McMaster University"
      height="90"
    >
  </a>
</p>

<h1 align="center">
  BIRMAC Guideline-Based Trustworthy Medical QA Dataset
</h1>

<p align="center">
  A guideline-grounded benchmark for evaluating the factual reliability,
  citation faithfulness, and patient accessibility of AI-generated health
  information.
</p>

## Overview

Clinical practice guidelines are among the most authoritative reference sources
for evidence-based healthcare. However, they are primarily written for clinical
and professional audiences and often contain dense medical terminology,
technical recommendations, eligibility criteria, and evidence-grading language.
This makes their content difficult for patients and caregivers to access
directly.

This project transforms recent clinical practice guidelines into a
patient-oriented question-answering dataset while preserving traceability to the
original guideline evidence.

Using guidelines published between **2021 and 2025**, we constructed
approximately **6,500 guideline-grounded question-answer pairs**. Questions are
written to resemble the language used by patients, caregivers, and general
health-information seekers. Answers are expressed in plain, patient-readable
language while restricting their medical content to information supported by
retrieved guideline passages.

Each answer is linked to its supporting passages through sentence-level
citations. The cited answer is subsequently decomposed into atomic medical
claims, allowing the support for each factual proposition to be evaluated
independently.

---

## Dataset at a Glance

| Property | Description |
| --- | --- |
| Guideline period | 2021–2025 |
| Dataset size | Approximately 6,500 QA pairs |
| Source material | Evidence-based clinical practice guidelines |
| Example sources | WHO, NICE, and other trusted guideline providers |
| Question style | Patient, caregiver, or general health-information seeker |
| Answer style | Concise, plain-language, and patient-readable |
| Evidence grounding | Retrieved guideline passages |
| Citation granularity | Sentence-level passage citations |
| Verification granularity | Atomic medical claims |
| Primary quality signal | Atomic factual consistency support rate |
| Storage format | JSONL with passage and model provenance |

---

## Motivation

A medically correct answer is not automatically understandable to a patient,
and a fluent answer is not automatically supported by medical evidence. This
dataset is designed to study both requirements:

1. **Evidence grounding:** Are the medical claims supported by authoritative
   guideline passages?
2. **Citation faithfulness:** Do the cited passages actually entail the claims
   attributed to them?
3. **Patient accessibility:** Are questions and answers expressed in language
   that patients and caregivers can understand?
4. **Traceability:** Can every generated claim be traced back to its originating
   guideline evidence?
5. **Evaluator reliability:** Can automated factual consistency models identify
   unsupported medical claims at atomic-claim granularity?

---

## Dataset Construction Pipeline

```text
Clinical guideline PDFs
        |
        v
Section-aware guideline passages
        |
        v
Semantic passage annotations
        |
        v
Within-guideline topic pools
        |
        v
Patient-style questions
        |
        v
Question-specific FAISS retrieval
        |
        v
Qwen3 passage reranking
        |
        v
Cited patient-facing answers
        |
        v
Atomic medical claims
        |
        v
Entailment-based factual consistency verification
```

---

### 1. Guideline Passage Construction

Guideline PDFs are converted into normalized text and divided into
section-aware passages. Each passage retains provenance information such as its
guideline identifier, source PDF, section title, page range, and passage
identifier. Reference-list and bibliography-like content is removed
conservatively.

Relevant files:

- `build_guideline_passages.py`
- `run_build_guideline_passages.sh`

---

### 2. Semantic Passage Annotation

An LLM produces a lightweight semantic representation for each passage,
including:

- patient-question suitability;
- broad clinical content type;
- main topic;
- concise semantic summary;
- possible question angles;
- patient-language keywords.

These annotations improve matching between natural patient questions and
technically written guideline passages.

Relevant files:

- `semantic_seed_batch.py`
- `run_semantic_seed_batch.sh`

---

### 3. Topic-Pool Construction

Usable passages are embedded with
`sentence-transformers/all-MiniLM-L6-v2`. Semantically related passages from
the same guideline are assembled into local topic pools using nearest-neighbor
similarity and overlap-based deduplication.

The topic pools are used only to inspire question generation. They do not act
as oracle evidence for the final answer.

Relevant files:

- `build_evidence_groups.py`
- `run_build_evidence_groups.sh`

---

### 4. Patient-Style Question Generation

The strongest shared patient-relevant topic in each pool is converted into a
question resembling what a patient or caregiver might ask a clinician or type
into a health-information system.

Questions retain medically meaningful retrieval anchors, such as condition,
treatment, medicine, procedure, symptom, or test names, while avoiding
unnecessarily technical wording.

Relevant files:

- `group_question_batch.py`
- `run_group_question_batch.sh`

---

### 5. Passage Indexing, Retrieval, and Reranking

All usable passages are indexed with `Qwen/Qwen3-Embedding-4B` and FAISS.
For every generated question:

1. FAISS retrieves a broad passage candidate set.
2. `Qwen/Qwen3-Reranker-4B` scores passage usefulness.
3. The highest-ranked passages are retained as the answer context.

Question retrieval is independent of the passages used to inspire question
generation.

Relevant files:

- `build_faiss_passage_index.py`
- `run_build_faiss_passage_index.sh`
- `retrieve_questions_with_faiss_rerank.py`
- `run_retrieve_questions_with_faiss_rerank.sh`

---

### 6. Guideline-Grounded Answer Generation

Patient-readable answers are generated using only the retrieved guideline
passages. Each answer sentence contains one principal medical proposition and
one or more local passage citations.

The resulting answers are synthetic guideline-grounded reference answers. They
should not be interpreted as human-authored clinical gold standards.

Relevant files:

- `answer_batch.py`
- `run_answer_batch.sh`

---

### 7. Atomic Medical Claim Extraction

Each cited answer sentence is decomposed into minimal medical claims. Important
qualifiers, conditional language, and uncertainty expressions are preserved.
Every atomic claim inherits the citations attached to its source sentence.

Relevant files:

- `atomic_fact_gpt4o.py`
- `run_atomic_fact_gpt4o.sh`

---

### 8. Factual Consistency Verification

Each evaluable atomic claim is assessed using the
`google/t5_xxl_true_nli_mixture` natural language inference model.

For each claim:

- the cited guideline passage or passages form the premise;
- the atomic medical claim forms the hypothesis;
- the model produces a binary entailment decision.

The answer-level atomic factual consistency support rate is calculated as:

$$
\text{Atomic FCM Support Rate}
=
\frac{\text{Number of supported atomic claims}}
{\text{Number of evaluable atomic claims}}
$$

Relevant files:

- `eval_atomic_fcm_veriscore.py`
- `run_atomic_fcm_veriscore.sh`

---

## Repository Structure

```text
.
├── assets/
│   ├── mcmaster-university-logo.jpg
│   └── university-of-birmingham-logo.jpg
├── answer_batch.py
├── atomic_fact_gpt4o.py
├── build_evidence_groups.py
├── build_faiss_passage_index.py
├── build_guideline_passages.py
├── eval_atomic_fcm_veriscore.py
├── group_question_batch.py
├── retrieve_questions_with_faiss_rerank.py
├── semantic_seed_batch.py
├── run_answer_batch.sh
├── run_atomic_fact_gpt4o.sh
├── run_atomic_fcm_veriscore.sh
├── run_build_evidence_groups.sh
├── run_build_faiss_passage_index.sh
├── run_build_guideline_passages.sh
├── run_group_question_batch.sh
├── run_retrieve_questions_with_faiss_rerank.sh
└── run_semantic_seed_batch.sh
```

---

## Running the Pipeline

The scripts are intended to be executed in the following order:

```bash
bash run_build_guideline_passages.sh
bash run_semantic_seed_batch.sh
bash run_build_evidence_groups.sh
bash run_group_question_batch.sh
bash run_build_faiss_passage_index.sh
bash run_retrieve_questions_with_faiss_rerank.sh
bash run_answer_batch.sh
bash run_atomic_fact_gpt4o.sh
bash run_atomic_fcm_veriscore.sh
```

OpenAI Batch API stages require an API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

Do not store API keys, access tokens, model credentials, or private filesystem
paths in the repository.

---

## Example Record

A released record may contain fields similar to the following:

```json
{
  "question_id": "q_000001",
  "question": "How often should I be monitored after starting this treatment?",
  "answer_text": "Monitoring should follow the schedule recommended for this treatment [1].",
  "answer_sentences": [
    {
      "sentence_id": "s1",
      "text": "Monitoring should follow the schedule recommended for this treatment [1].",
      "citations": [1],
      "role": "monitoring_or_follow_up"
    }
  ],
  "passages": [
    {
      "pid": 1,
      "passage_id": "guideline_001_p00042",
      "guideline_id": "guideline_001",
      "text": "Supporting guideline passage..."
    }
  ],
  "atomic_facts_flat": [
    {
      "atomic_fact_id": "s1_a1",
      "fact": "Monitoring should follow the schedule recommended for this treatment.",
      "citation_ids": [1]
    }
  ]
}
```

---

## Intended Uses

This resource is intended for research on:

- trustworthy medical question answering;
- retrieval-augmented generation;
- citation faithfulness;
- factual consistency evaluation;
- atomic-claim verification;
- patient-oriented health communication;
- medical information retrieval and reranking;
- automated evaluation of AI-generated health information.

---

## Limitations

- Questions and answers are synthetically generated and may contain systematic
  model artifacts.
- Automated entailment models can produce false-positive and false-negative
  support decisions.
- Guideline recommendations may vary across countries, populations, and
  healthcare systems.
- The dataset reflects the guideline publication period and may not include
  later clinical updates.
- Plain-language transformation may remove nuance present in the original
  guideline wording.
- Guideline-grounded answers are not substitutes for individualized clinical
  assessment.

---

## Medical Disclaimer

This dataset is provided for research and evaluation purposes only. It is not a
medical device, clinical decision-support system, or source of individualized
medical advice. Dataset answers should not be used to diagnose, treat, or
manage any medical condition.

---

## Source and Intellectual Property Notice

Clinical guideline text remains subject to the copyright and licensing terms
of its original publisher. Users are responsible for verifying that their use
and redistribution of guideline-derived material complies with the applicable
source terms.

The University of Birmingham and McMaster University names and logos remain the
property of their respective institutions. Their inclusion identifies the
institutional context of the research and does not imply endorsement of every
dataset item, model output, or downstream use. The logos must not be altered,
reconfigured, or used for commercial purposes.

---

## Acknowledgements

This repository supports research involving collaborators affiliated with the
University of Birmingham and McMaster University. We thank the organizations
that publish and maintain evidence-based clinical practice guidelines and the
researchers contributing to trustworthy, patient-accessible medical AI.
