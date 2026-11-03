# Experiment: Qwen3 embeddings from Neon against bge-small, on the text-only chunks

**Question.** With the tables taken out of the chunk file (they live as tables in Postgres), does
Qwen3-Embedding-0.6B, with its vectors read from Neon (pgvector), find the evidence at least as well as bge-small-en-v1.5
on the same 109 text chunks?

**Method.** Two retrieval-only runs on a Colab T4 (2 November 2026, branch `feature/neon-pgvector`): `bge_text`
([`configs/default.yaml`](../../configs/default.yaml) with only the chunk file changed, FAISS) and `qwen3_text_pgvector`
([`configs/experiments/qwen3_text_pgvector.yaml`](../../configs/experiments/qwen3_text_pgvector.yaml)). Everything else
is the same: BM25, the reranker (bge-reranker-v2-m3), context building, gate, test set. The rule was fixed before the
runs: keep Qwen3 unless it is significantly worse on the primary measure (McNemar p < 0.05), more than 5 points lower,
or needs more than 14 GB of GPU memory. Primary: all evidence passages reach the context, on the 67 answerable test
questions whose evidence is in the text chunk file.

**Result (67 test questions).**

| | bge-small (FAISS) | Qwen3 (pgvector) | Paired difference |
|---|---|---|---|
| All evidence reaching the model | 64/67 | 64/67 | 0 (McNemar p = 1.0) |
| Context recall / search recall | 0.963 / 1.000 | 0.963 / 1.000 | 0 / 0 |
| MRR@10 | 0.917 | 0.918 | +0.001 (95% CI +0.000 to +0.004) |
| Dense recall@5 / @10 / @20 (the embedder alone) | 0.940 / 0.955 / 0.985 | 0.963 / 0.993 / 1.000 | +0.023 / +0.037 / +0.015 |
| Answerable questions refused by the gate (all 94) | 1 | 0 | |
| Run time / peak GPU memory | 1.5 min / 1.54 GB | 2.4 min / 2.44 GB | |

Dev (15 questions, debugging only): 12 and 13. For reference, the published baseline (bge-small, the older chunk file
with the tables written into the chunks) had all evidence in context for 63 of the same 67 questions.

**Decision: keep Qwen3.** Not worse on the primary measure; better on the embedder's own recall.

**Limitations.** Retrieval only (no answers). The 12 questions answered only by tables are outside this comparison:
their evidence is not in the text chunks (see [`../table_route_retrieval/`](../table_route_retrieval/)).

**Files.** [`per_question.csv`](per_question.csv): both runs' retrieval scores per question;
[`source.json`](source.json): the private record it was taken from (sha256), the plan and the decision.
