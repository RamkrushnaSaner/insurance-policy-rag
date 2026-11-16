# Evaluation: v3.4 on every question set it answered

**Question.** How well does the adopted system (v3.4,
[`configs/experiments/qwen3_text_tables.yaml`](../../configs/experiments/qwen3_text_tables.yaml)) answer questions it
was never tuned on, and where does it fail?

**Method.** Nothing was generated for this evaluation. The five saved runs of v3.4 were re-scored with this
repository's checks (repaired on 12 November 2026, commit `93b5f9d`):

| Set | Questions (answerable) | Written | Role | Answers |
|---|---|---|---|---|
| `dev_v3` | 113 (94) | October 2026 | the development set: every change was tuned or decided on it | generated earlier, reused from the cache on 9 Nov |
| `fresh_v1` | 72 (64) | 4 Nov | written after v3.4 was built, never tuned on | generated 4 Nov, reused on 9 Nov |
| `fresh_v2` | 62 (56) | 5 Nov | never tuned on | generated 5 Nov, reused on 9 Nov |
| `fresh_v3` | 107 (100) | 12 Nov | unseen before its run | generated 13 Nov on a T4 |
| `fresh_v4` | 88 (80) | 13 Nov | unseen before its run | generated 14 Nov on a T4 |

- **Re-scoring check:** the re-scored verdicts match the stored ones for all 195 answers of fresh v3 and v4, which
  were graded with these checks when they ran. On `dev_v3` the run's stored verdicts are exactly those published in
  [`../neon_tables/`](../neon_tables/) (100/113); the repaired checks change one verdict (M04, now correct): 101/113.
- **The person's labels:** in fresh v4 the owner labelled 33 of these answers blind (all 30 table questions and the
  3 text questions where the two systems of the final round differed). Those labels replace the checks' verdict in
  `fully_correct`; `fully_correct_checks` keeps the checks' verdict.
- **Table and text questions:** in fresh v3 and v4, the table questions listed in the decision rules fixed before
  those runs. Elsewhere, an answerable question whose key fact is printed only in a table.
- **Headline:** the 329 fresh questions only. The development set is shown for comparison.

## Result

| Set | Fully correct | 95% CI | Table questions | Text questions | Not answerable, declined |
|---|---|---|---|---|---|
| Development (tuned on) | 101/113 = 89.4% | 82–94% | 6/8 | 76/86 | 19/19 |
| Fresh v1 | 55/72 = 76.4% | 65–85% | 6/11 | 41/53 | 8/8 |
| Fresh v2 | 44/62 = 71.0% | 59–81% | 5/14 | 33/42 | 6/6 |
| Fresh v3 | 88/107 = 82.2% | 74–88% | 18/29 | 63/71 | 7/7 |
| Fresh v4 | 68/88 = 77.3% | 67–85% | 20/30 | 40/50 | 8/8 |
| **All fresh** | **255/329 = 77.5%** | **73–82%** | **49/84 = 58.3%** | **177/216 = 81.9%** | **29/29** |

With the checks alone (no person's labels): 253/329 = 76.9%.

**Where the 76 failures counted by the checks started** (the first stage that no longer had what the answer needed):

| Stage | Dev | v1 | v2 | v3 | v4 | All fresh |
|---|---|---|---|---|---|---|
| Search (retrieval): every one a fact printed only in a table | 0 | 4 | 9 | 6 | 7 | 26 |
| Context selection: found, not read | 3 | 4 | 2 | 4 | 5 | 15 |
| Relevance gate | 0 | 0 | 1 | 0 | 0 | 1 |
| The answer (the context held every key fact) | 9 | 9 | 6 | 9 | 10 | 34 |

The 34 answer-stage failures were read one by one ([`generation_failure_review.csv`](generation_failure_review.csv);
reviewer: the AI assistant, except the four answers the person labelled):

| What the reading found | Answers |
|---|---|
| The context did not hold the answer after all: the key fact matched a heading or an unrelated word, or the list item the question names was not among the passages read | 8 (+ Y077, which the checks passed and the person failed) |
| The answer is acceptable: the key was too strict, a correct calculation was flagged, or the policy contradicts itself (Annexure B's title against Section 4.3) | 9 (3 of them confirmed by the person's label) |
| The answer applied the right rule in its body but opened with the refusal sentence or said it could not be confirmed | 9 |
| The answer missed or misread a fact that was in its context (a qualifier, an exception, "respectively"; twice with thinking stopped at its budget) | 8 |

So **17 of the 76 are clear model errors** on a context that held the answer. Among the 247 answers whose context held
it, errors were more frequent on false-premise questions (5/14 against 12/233, Fisher p = 0.001) and when thinking
stopped at its 1,024-token budget (3/8 against 14/239, p = 0.012). The reading is one reviewer's judgement, not a
measurement; see [docs/final_evaluation.md](../../docs/final_evaluation.md#5-failure-analysis).

**Also measured** (all fresh questions; every row per set in [`scorecard.md`](scorecard.md)):

| | |
|---|---|
| All evidence among the search candidates / in the context / every key fact in the context | 248/300 / 237/300 / 256/300 |
| Text evidence: first stage (RRF) recall @5 / @10 / @20 / @64 | 0.89 / 0.94 / 0.97 / 0.99 |
| Text evidence: after the reranker, recall @1 / @3 / @5 / @10; MRR@10 | 0.84 / 0.92 / 0.94 / 0.96; 0.89 (211 questions) |
| Table route answered completely: table / text questions | 30/84 / 17/216 |
| Context precision (mean) · passages read · context tokens (median) | 0.21 · 7 · 1,444 (67 contexts over the 1,500 budget, all with a table passage) |
| Answers that decline although the policy answers (refusal or hedged opening) | 50/300 |
| Cites a section / cites a section not in the context / cites a section that holds the answer | 277/297 / 0/277 / 254/277 |
| Answers with a figure that is not in the context | 1/311 |
| HHEM groundedness (mean) / answers it flags | 0.758 / 152/305 (it also flags 29% of the answers the person judged correct: not usable as a hallucination rate here) |
| Seconds per answer (median / p90 / p95 / max), 317 answers on a T4 | 43.3 / 75.9 / 92.6 / 129.4 |
| Thinking stopped at its budget · answers cut off · pipeline errors | 11/317 · 0 · 0 |
| Search repeated within each run · one answer regenerated per run | identical, 442/442 · identical, 5/5 |
| Peak GPU memory (fresh v3, v4) | 9.32 / 9.22 GB |

**Limitations.**

- **Who wrote the questions:** the fresh question sets and their keys were written by the AI assistant that built the
  system. They are kept private (they quote the policy); their sha256 are in [`source.json`](source.json).
- **The checks:** rule-based. On the 33 labelled answers they agree with the person on 29 (kappa 0.75).
- **One run, one GPU type:** each set was answered once, on a Colab T4.
- **Not measured:** one policy only; no real users' questions, no load or concurrency.

## Files

| File | What |
|---|---|
| `per_question.csv` | One row per question and set: stratum, type, the final verdict and the checks' verdict, the person's label, the first failing stage, issue kinds (no text), retrieval metrics as measured in the run, recall at rank k, citations, groundedness, timings |
| `scorecard.csv`, `scorecard.md` | Every measure by set and for all fresh questions |
| `generation_failure_review.csv` | The reading of the 34 answer-stage failures and Y077 |
| `source.json` | sha256 of every input: the five run files, the test sets, the decision rules (table questions) and the decision record holding the person's labels; the chunk file and the models |

Rebuild (needs the private files and the database for the tables' text):

```bash
DATABASE_URL=... python scripts/export_final_evaluation.py results/final_evaluation \
  --run dev_v3=<E0_v3_113/run.json> \
  --run fresh_v1=<E0_fresh_v1/run.json> --testset fresh_v1=<optima_secure_fresh_v1.json> \
  --run fresh_v2=<E0_fresh_v2/run.json> --testset fresh_v2=<optima_secure_fresh_v2.json> \
  --run fresh_v3=<E0_fresh_v3/run.json> --testset fresh_v3=<optima_secure_fresh_v3.json> \
  --run fresh_v4=<E0_fresh_v4/run.json> --testset fresh_v4=<optima_secure_fresh_v4.json> \
  --table-questions fresh_v3=<release_rules_r1.json>:data_roles.decision_set.table_questions \
  --table-questions fresh_v4=<release_rules_final.json>:decision_set.table_questions \
  --labels fresh_v4=<final_decision_record.json>:labels
```
