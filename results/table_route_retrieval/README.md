# Experiment: the table route, retrieval only

**Question.** With the table route on (the facts from the typed tables as the first context passage, and the text
chunks linked to those tables as candidates), do the questions that only the tables answer get their facts, without
losing evidence anywhere else?

**Method.** Two retrieval-only runs in one Colab session (2 November 2026, commit `1ea633b`): `tables_off`
([`configs/experiments/qwen3_text_pgvector.yaml`](../../configs/experiments/qwen3_text_pgvector.yaml)) and `tables_on`
([`configs/experiments/qwen3_text_tables.yaml`](../../configs/experiments/qwen3_text_tables.yaml)). The tables-off run
had to reproduce the Qwen3 run of [`../embedder_qwen3/`](../embedder_qwen3/) exactly, and did (113/113 for context
recall, search recall, MRR@10 and gate). Before that, the route alone was run on all 113 questions against Neon.

**Result.**

| | Tables off | Tables on |
|---|---|---|
| Table-only questions (12) with the table passage in the context | – | 12/12 |
| Table-only questions with every key fact in the context | 0/12 | 11/12 |
| The 67 test questions of the embedder comparison: all evidence in context | 64/67 | 65/67 |
| Other questions that lost evidence or a key fact | | none |
| Answerable questions refused by the gate / not-answerable refused | 0 / 4 of 19 | 0 / 4 of 19 |
| Not-answerable questions given a table passage | | none |

The route alone answered 20 questions; every value it used matched its printed table cell, its answers had no figure
outside the tables, and it answered no not-answerable question. Median time per answered question on Neon: 708 ms.

**Linked chunks pinned (checked before the full run of [`../neon_tables/`](../neon_tables/)).** The one missing fact
(AB02: the Protect Benefit section that pays Annexure B items) was in a linked chunk that the reranker did not place in
the context. Placing linked chunks in the context whatever their score: 19 questions got pinned chunks, AB02 gained its
fact, nothing was lost, the gate did not change; answerable questions with every key fact in context 90 → 91 of 94
(record in [`../neon_tables/comparison_source.json`](../neon_tables/comparison_source.json)).

**Limitations.** Context recall for the table-only questions depends on how the evidence anchors were written (for
text renderings of the tables), so the key facts in the context are the measure used for them. The route's question
patterns were written with these questions in view.

**Files.** [`per_question.csv`](per_question.csv): gate, context recall, key facts in context and route status per
question, tables off and on; [`source.json`](source.json): the private record (sha256), the route's per-question
kinds and the summary.
