# Published results

This folder is the evidence for every number in the [README](../README.md) and [docs/](../docs/). These files hold
**scores only**: no answers, no retrieved passages and no policy text. Those are in the private run files, which
quote the policy. Each folder has a `source.json` with the sha256 of the private files it was built from, and
everything here can be rebuilt from them:
- [`final_evaluation/`](final_evaluation/) with [`scripts/export_final_evaluation.py`](../scripts/export_final_evaluation.py);
- the other folders with [`scripts/export_results.py`](../scripts/export_results.py).

[`tests/test_published_results.py`](../tests/test_published_results.py) recomputes the headline numbers from these
files on every push.

| Experiment | Run date (Colab T4) | Result | Evidence |
|---|---|---|---|
| **v3.4 on every question set** (final evaluation, no change to the system) | runs 9–14 Nov 2026; exported 15 Nov 2026 | **255/329 = 77.5% fully correct on new questions**; 101/113 on the development set | [`final_evaluation/`](final_evaluation/) |
| Baseline: [`configs/default.yaml`](../configs/default.yaml) on the 113-question development set | answers 22 Oct 2026; re-measured 29 Oct 2026 | 79/88 fully correct on its test split, 97/113 overall | [`baseline/`](baseline/) |
| Reproduction: package v3.3 re-running the baseline with the published answers | 27 Oct (twice), 29 Oct 2026 | identical answers, contexts and verdicts in all three sessions; 0 answers regenerated | [`reproduction/`](reproduction/) |
| Thinking off ([`configs/experiments/no_thinking.yaml`](../configs/experiments/no_thinking.yaml)) | 25 Oct 2026 | 72/88 on test: 7 lost, 0 gained (p = 0.016); not adopted | [`thinking_off/`](thinking_off/) |
| v2.6 (Mistral-7B, text-file chunks) against v3.1, 24 shared dev questions | 20 and 22 Oct 2026 | 11 → 17 fully correct: 6 fixed, 0 broken (p = 0.031) | [`v2_6_vs_v3_1/`](v2_6_vs_v3_1/) |
| Conversation memory, three runs | 27 Oct (two), 29 Oct 2026 | never significantly better; not adopted, off by default | [`memory/`](memory/) |
| Qwen3 embeddings from Neon against bge-small, text-only chunks (retrieval only) | 2 Nov 2026 | all evidence in context 64/67 for both; dense recall@10 0.955 → 0.993; Qwen3 kept | [`embedder_qwen3/`](embedder_qwen3/) |
| Table route, retrieval only | 2–3 Nov 2026 | table-only questions with every key fact in context 0/12 → 11/12 (12/12 with linked chunks pinned) | [`table_route_retrieval/`](table_route_retrieval/) |
| Neon + table route, full answers (v3.4) | 3 Nov 2026 | 78/88 on test against the baseline's 79/88 (p = 1.0); adopted | [`neon_tables/`](neon_tables/) |

**Which checks scored what.**
- **Up to `neon_tables/`:** the checks of their time. They were unchanged from v3.2 to v3.4.0, and older runs were
  re-scored with them.
- **`final_evaluation/`:** the checks repaired on 12 November 2026 (commit `93b5f9d`). On the development set, the
  repair changes one v3.4 verdict (M04): 100 → 101 of 113.

**Chunk files and models.**
- **Baseline-era runs:** chunk file `74aa551c…`, bge-small-en-v1.5 with FAISS, bge-reranker-v2-m3, Qwen3-8B in 4-bit
  NF4 and HHEM-2.1-Open, scored on test set `7313dd7571`.
- **v3.4 runs:** the 109-chunk text-only file `f92b0ea2…` and Qwen3-Embedding-0.6B vectors in pgvector, with the
  model revisions pinned in each `source.json`.
- **The v2.6 run:** an older text export (`chunks.json`), Mistral-7B-Instruct-v0.3 in 4-bit and bge-reranker-base.

## Private run files behind these results

| Run | File | sha256 (first 16) |
|---|---|---|
| v3.1 notebook, generated the baseline answers, 22 Oct 2026 | `run.json` (`qwen3_8b_think`) | `07b3e8f6228d1608` |
| Package v3.3 single-question check, 27 Oct 2026 (1) | `check_v33/run.json` | `238dbf4481db75af` |
| Package v3.3 single-question check, 27 Oct 2026 (2) | `check_single/run.json` | `d8fec62fa493008f` |
| Package v3.3 single-question check, 29 Oct 2026 | `check_single/run.json` | `eb75e6610ac752e5` |
| Thinking off, package v3.2, 25 Oct 2026 | `no_thinking/run.json` | `fedb335146dfef70` |
| v2.6 notebook compact baseline, 20 Oct 2026 | `compact_baseline.json` | `47ac42d70bb167ac` |
| Memory run 1, 27 Oct 2026 | `memory/dialogues.json` | `8411240f7976cac2` |
| Memory run 2, 27 Oct 2026 | `memory_v2/dialogues.json` | `09201349d959b581` |
| Memory run 3, 29 Oct 2026 | `memory_v3/dialogues.json` | `f45d8c4035c68399` |
| Qwen3 against bge-small, retrieval, 2 Nov 2026 | `drive/step6_record.json` | `bffae64c79eca419` |
| Table route, retrieval, 2–3 Nov 2026 | `drive/step8a_record.json` | `a77c43d62db7911f` |
| v3.4 full answers, 113 questions, 3 Nov 2026 | `tables_full/run.json` | `0d60aa435edf79a7` |
| v3.4 on the development set (answers from the cache), 9 Nov 2026 | `E0_v3_113/run.json` | `4e3ea3d9eb6903b6` |
| v3.4 on fresh v1 (answers of 4 Nov from the cache), 9 Nov 2026 | `E0_fresh_v1/run.json` | `d9c83088ebe82ce3` |
| v3.4 on fresh v2 (answers of 5 Nov from the cache), 9 Nov 2026 | `E0_fresh_v2/run.json` | `ba01f04ce96c27b4` |
| v3.4 on fresh v3, 13 Nov 2026 | `E0_fresh_v3/run.json` | `f9c1e8544c6fa7ab` |
| v3.4 on fresh v4, 14 Nov 2026 | `E0_fresh_v4/run.json` | `1c34503b0fcd4e19` |

To rebuild a folder from your own run (for example after `policy-rag evaluate --name baseline`):

```bash
python scripts/export_results.py run runs/baseline/run.json results/baseline
python scripts/export_results.py compare runs/baseline/run.json runs/<candidate>/run.json results/<candidate> --splits dev test
python scripts/export_results.py dialogues runs/memory/dialogues.json results/memory/run4
```

## What is not here

- **The fresh question sets** (v1–v4) and the decision records of experiments 10 and 11: private, because they
  quote the policy. Their sha256 are in [`final_evaluation/source.json`](final_evaluation/source.json) and
  [docs/experiments.md](../docs/experiments.md).
- **Hard questions** ([`testsets/optima_secure_hard_questions.json`](../testsets/optima_secure_hard_questions.json)):
  the only run was an early development run judged by reading; its outputs were not kept, so no result is published.
- **Human labels on the 113-question set:** not collected. Blind labels exist for the fresh sets: 30 answers in fresh
  v3 and 66 in fresh v4. v3.4's 33 labelled answers are in `final_evaluation/per_question.csv`.
- **Regenerated baseline answers:** every package run of the baseline reused the v3.1 answers from the cache. A fresh
  generation with the package (thinking on) has not been run.
- **R1's runs:** R1 was not adopted. Its numbers are in [docs/experiments.md](../docs/experiments.md) (10 and 11),
  and its code is on the branch `r1-candidate`.
