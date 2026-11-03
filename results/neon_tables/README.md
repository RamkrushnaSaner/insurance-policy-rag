# Experiment: Neon + pgvector (Qwen3) with the table route, full answers

**Question.** Does the new system answer at least as well as the published baseline? New: Qwen3-Embedding-0.6B
vectors read from Neon (pgvector) over the 109 text-only chunks, plus the table route over the typed tables (facts
first in the context, linked text chunks pinned). Unchanged: the reranker, Qwen3-8B 4-bit with thinking, the prompt,
the gate, the test set.

**Method.** [`configs/experiments/qwen3_text_tables.yaml`](../../configs/experiments/qwen3_text_tables.yaml), all 113
questions answered on a Colab T4 (3 November 2026, commit `32b2fc5`, 2.2 hours, 109 answers generated, none from a
cache). Scored with the same checks as the baseline (unchanged on this branch). The rule was fixed before the run
(2 November 2026, 20:51 UTC): adopt unless fully correct is significantly worse (McNemar p < 0.05) or more than 5 points
lower on the 88 test questions; or groundedness, the share of answers with an unsupported sentence, or correct
abstention is significantly worse (paired bootstrap 95% interval); or the median time per answer is over 150 s; or peak
GPU memory is over 14 GB.

**Result (test split, 88 questions).**

| | Baseline | New | Paired difference |
|---|---|---|---|
| Fully correct | 79/88 (89.8%) | 78/88 (88.6%) | −1.1 points (95% CI −8.0 to +6.8): 5 fixed, 6 broken, McNemar p = 1.0 |
| Key-fact recall (answerable) | 0.929 | 0.945 | |
| Critical errors | 1 | 2 (F01: a figure not in the context; L04: another cover's section) | |
| Not-answerable declined | 15/15 | 15/15 | |
| Groundedness (75 answers scored in both) | 0.895 | 0.900 | no significant change |
| Answers with an unsupported sentence (same 75) | 25.3% | 24.0% | no significant change |
| Seconds per answer, median (all 113) | 47.4 | 49.2 | |
| Peak GPU memory | 10.26 GB | 12.53 GB | |

Now correct: AB01, PR02, X01, X02, X12. Now wrong: C02, D06, F01, L04, L07, M04 (all six fail in the answer, not in
retrieval). Dev (25, debugging only): 18 → 22. All 113: 97 → 100 (p = 0.61).

| Also measured (all 113 unless noted) | Baseline | New |
|---|---|---|
| The 67 test questions with evidence in the text chunks: all evidence in context | 63/67 | 66/67 |
| Same 67: context recall / MRR@10 | 0.948 / 0.892 | 0.985 / 0.918 |
| Table-only questions (12) fully correct | 8/12 | 9/12 |
| Everyday wording / policy wording, answerable, fully correct | 20/28 / 59/66 | 25/28 / 56/66 |
| Not-answerable declined | 18/19 | 19/19 |
| Answerable questions refused | 4 of 94 | 2 of 94 |
| Where failures started: answer / context / retrieval | 8 / 7 / 1 | 10 / 3 / 0 |
| Groundedness, all answers (not paired) | 0.886 | 0.873 |
| Answers with an unsupported sentence, all (not paired) | 28.3% | 32.7% |

**Decision: adopted (not worse than the baseline).** No significant difference either way on the test split; the
interval allows anything from 8 points worse to 7 points better.

**Limitations.**
- 88 test questions can only detect large differences; this run cannot show a small improvement or rule out a small
  loss.
- The table route's question patterns were written with these 113 questions in view, so its results on them are
  optimistic. A new question set is needed for an unbiased figure.
- The remaining failures are mostly in the answer: in G09, L07 and AB02 the facts were in the context and the model
  left part out. Unpaired faithfulness over all answers is slightly lower (dev answers); the paired test comparison
  shows no change.
- In [`scorecard.md`](scorecard.md), "evidence present in the chunk file" counts the text chunks and the printed table
  rows; six table-only questions have evidence anchors written for a text rendering of the tables that the cells do not
  match, although their facts reach the model.

**Files.** [`per_question.csv`](per_question.csv), [`scorecard.md`](scorecard.md) / [`scorecard.csv`](scorecard.csv)
and [`source.json`](source.json) (`scripts/export_results.py run`, with the model revisions and the database version
used); [`paired.csv`](paired.csv): the verdict per question in both systems; [`comparison_source.json`](comparison_source.json):
the inputs (sha256), the rule as fixed, the decision and the pinning check.
