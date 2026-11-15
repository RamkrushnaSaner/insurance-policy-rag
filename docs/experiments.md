# Experiments

Every experiment run in this project, in the order it was run, with what it measured and where the evidence is. The
numbers come from the files in [`results/`](../results/), and
[`tests/test_published_results.py`](../tests/test_published_results.py) recomputes the headline figures from them.
The decisions of experiments 10 and 11 come from private decision records, whose sha256 are given there. Things that
were planned or claimed earlier but cannot be backed by evidence are listed at the end.

**Words used here.** *Fully correct*: the answer has every key fact, no critical error and is not cut off (for a
question the policy does not answer: it says so at the start). *Test split*: the 88 questions of the 113-question
development set on which changes were adopted or rejected; *dev*: its other 25 questions. Because the whole set was
read during development, its numbers are optimistic: v3.4 scores 89.4% on it and 77.5% on new questions (12). *McNemar p*: the chance of a difference at least
this large between two systems if they were really equally good, counting only the questions where they disagree.

## Summary

| # | Experiment | Date | Main result | Decision | Evidence |
|---|---|---|---|---|---|
| 1 | [Baseline](#1-baseline) | 22 Oct 2026 | 79/88 fully correct on test (89.8%, 95% CI 82–95%), 97/113 overall | the first published system | [`results/baseline/`](../results/baseline/) |
| 2 | [Correcting the checks](#2-correcting-the-checks) | before 24 Oct 2026 (v3.2) | 6 verdicts changed (5 false failures, 1 missed false statement) | checks v3.2 | [`results/baseline/per_question.csv`](../results/baseline/per_question.csv) |
| 3 | [v2.6 against v3.1](#3-v26-against-v31) | 20 and 22 Oct 2026 | 11 → 17 of 24 dev questions, p = 0.031 | v3 kept | [`results/v2_6_vs_v3_1/`](../results/v2_6_vs_v3_1/) |
| 4 | [Reproduction](#4-reproduction) | 23–29 Oct 2026 | same contexts, prompts, verdicts and groundedness in every session | – | [`results/reproduction/`](../results/reproduction/) |
| 5 | [Thinking off](#5-thinking-off) | 25 Oct 2026 | 72/88 on test: 7 lost, 0 gained, p = 0.016; 5× faster | keep thinking on | [`results/thinking_off/`](../results/thinking_off/) |
| 6 | [Conversation memory](#6-conversation-memory) | 27–29 Oct 2026 | three runs, never significantly better (final: 21 vs 19 of 32, p = 0.73) | not adopted, off by default | [`results/memory/`](../results/memory/) |
| 7 | [Qwen3 embeddings from Neon](#7-qwen3-embeddings-from-neon) | 2 Nov 2026 | text-only chunks: all evidence in context 64/67 for both embedders (p = 1.0); dense recall@10 0.955 → 0.993 | keep Qwen3 | [`results/embedder_qwen3/`](../results/embedder_qwen3/) |
| 8 | [Table route, retrieval](#8-table-route-retrieval) | 2–3 Nov 2026 | table-only questions with every key fact in context 0/12 → 11/12 (12/12 with linked chunks pinned); no evidence lost elsewhere | tables on | [`results/table_route_retrieval/`](../results/table_route_retrieval/) |
| 9 | [Neon + table route, full answers](#9-neon--table-route-full-answers) | 3 Nov 2026 | 78/88 on test against 79/88 (5 fixed, 6 broken, p = 1.0); 100/113 overall | adopted: not worse (v3.4) | [`results/neon_tables/`](../results/neon_tables/) |
| 10 | [R1: table cards, on fresh v3](#10-r1-table-cards-on-fresh-v3) | 12–13 Nov 2026 | 94 vs 88 of 107 fully correct; table questions 26 vs 18 (9 fixed, 1 broken, p = 0.02); checks against a person: kappa 0.52 (rule: ≥ 0.7) | not adopted: the check on the checks failed | this page |
| 11 | [Final round: R1 against v3.4, on fresh v4](#11-final-round-r1-against-v34-on-fresh-v4) | 13–15 Nov 2026 | table questions judged by a person 24 vs 20 (6 fixed, 2 broken, p = 0.29); text 42 vs 40; declines 8/8 for both | **not adopted: v3.4 stays**; the last decision | this page |
| 12 | [Evaluation of v3.4 on every question set](#12-evaluation-of-v34-on-every-question-set) | 15 Nov 2026 | 255/329 = 77.5% fully correct on new questions (95% CI 73–82%); 101/113 on the development set | – (no change) | [`results/final_evaluation/`](../results/final_evaluation/) |

## 1. Baseline

- **Objective.** Measure the system as published: how often it answers correctly, where it fails, and how fast.
- **Method.** [`configs/default.yaml`](../configs/default.yaml) on all 113 questions of
  [`testsets/optima_secure_v3.json`](../testsets/optima_secure_v3.json) (version `7313dd7571`). The answers were
  generated once by the v3.1 notebook on a Colab T4 (22 Oct 2026, 17:38 to 19:33). They are scored by deterministic
  checks against hand-written answer keys; retrieval is scored against evidence anchors; groundedness by
  HHEM-2.1-Open. Method details: [evaluation.md](evaluation.md).
- **Result.**

  | | dev (25) | test (88) | all (113) |
  |---|---|---|---|
  | Fully correct | 18 (72.0%) | **79 (89.8%)** | 97 (85.8%) |
  | Key-fact recall (answerable) | 0.885 | 0.929 | 0.919 |
  | Critical errors | 1 | 1 | 2 |
  | Evidence among the candidates / in the context | 0.952 / 0.889 | 0.989 / 0.932 | 0.980 / 0.922 |
  | Context precision (share of context tokens that are evidence) | 0.391 | 0.280 | 0.305 |
  | Groundedness (HHEM) / answers with an unsupported sentence | 0.862 / 38.1% | 0.892 / 25.6% | 0.886 / 28.3% |
  | Not-answerable questions declined | 3/4 | 15/15 | 18/19 |
  | Seconds per answer, median / 95th percentile | 58.3 / 148 | 46.2 / 116 | 47.4 / 131 |

  Peak GPU memory 10.26 GB, 6.09 tokens/s. Of the 16 failures, 8 never had the evidence in front of the model
  (7 context, 1 search) and 8 were answer errors; everyday wording was weaker (20/28 against 59/66 in policy wording).
- **Limitations.** 88 test questions can only detect differences of about 15 points (80% power). The checks are rules
  and have not been compared with independent human labels. One policy only.
- **Evidence.** [`results/baseline/`](../results/baseline/) (scorecard, a row per question, sources). Groundedness was
  re-measured on 29 Oct 2026 with the package; the first published figure (0.851) came from the v3.1 notebook's
  different sentence splitting ([why](../results/baseline/README.md)).
- **Reproduce.** `policy-rag evaluate --name baseline` (about 2 hours on a T4), or re-score saved answers:
  `policy-rag evaluate --name check --mode rescore --rescore-from <run.json>`.

## 2. Correcting the checks

- **Objective.** Find and fix check errors before trusting the scores.
- **Method.** All 113 v3.1 answers were read and judged during development, by the AI assistant that wrote the
  checks (not blind, not independent). Where the reading and the checks disagreed for a clear reason, the checks were
  changed (v3.2), and the answers re-scored.
- **Result.** The v3.1 checks gave 93/113 (75/88 on test); the corrected checks give 97/113 (79/88). Six verdicts
  changed: L04, K04, D02, D03 and D07 had been marked wrong though correct, and X01's false statement ("regardless of
  accidents") had been missed. Agreement between the old and new checks: 107/113, Cohen's kappa 0.80.
- **Limitations.** The reading that guided the corrections is not independent, so this shows the checks match one
  careful reading, not that they match people in general. The 30-answer label sheet for independent labels
  (`labels_to_fill.csv`, `policy-rag agreement`) has not been filled in.
- **Evidence.** The old verdicts are in the private v3.1 run file (sha256 `07b3e8f6…`); the new ones in
  [`results/baseline/per_question.csv`](../results/baseline/per_question.csv).

## 3. v2.6 against v3.1

- **Objective.** Check that the rebuilt system (chunks from the PDF, bge-reranker-v2-m3, Qwen3-8B with thinking)
  beats the earlier notebook (text-file chunks, bge-reranker-base, Mistral-7B-Instruct-v0.3).
- **Method.** The v2.6 compact evaluation answered 24 questions, all of them dev questions of the current test set
  with the same wording. Both systems' answers were re-scored with the current checks and compared per question.
- **Result.** 11 → 17 fully correct: 6 fixed, 0 broken, McNemar p = 0.031. Critical errors 5 → 1.
- **Limitations.** Dev questions only; three components changed at once, so it does not say which change mattered.
- **Evidence.** [`results/v2_6_vs_v3_1/`](../results/v2_6_vs_v3_1/).

## 4. Reproduction

- **Objective.** Show that the package reproduces the notebook that produced the published answers, and that its
  results do not change from session to session.
- **Method.** (a) Without the real models (they could not be downloaded where the package was built): the notebook
  v3.2 and the package were run side by side with small stand-in models, and the saved v3.1 contexts and cache keys
  were rebuilt from the real chunk file. (b) With the real models on a Colab T4: three sessions of package v3.3 (27 Oct
  twice, 29 Oct) ran the baseline with the v3.1 answers in the cache; the thinking-off run (25 Oct, package v3.2)
  reran retrieval with the real models.
- **Result.** (a) Identical retrieval, contexts, gate decisions and retrieval metrics for all 113 questions;
  identical answers, cache keys and verdicts on dev with the stand-in answer model; the 113 v3.1 contexts and 108
  cache keys rebuilt exactly. (b) In all four runs: the same candidates, reranker scores, contexts and gate decisions
  as the published run for 113/113 questions. In the three sessions: 0 answers generated (all 108 found in the cache
  under the same keys, so the prompts were identical), 97/113 fully correct, groundedness 0.886.
- **Limitations.** The published configuration's answers have never been *regenerated* by the package. The chunk
  file's byte-for-byte reproduction from the PDF is not confirmed (it needs the `tiktoken` vocabulary download, which
  was blocked in the build environment).
- **Evidence.** [`results/reproduction/t4_sessions.csv`](../results/reproduction/t4_sessions.csv), and the
  Verification section of [evaluation.md](evaluation.md#verification) for (a).

## 5. Thinking off

- **Objective.** Thinking makes answers slow (median 47 s). Does the model answer as well without it?
- **Method.** One setting changed ([`configs/experiments/no_thinking.yaml`](../configs/experiments/no_thinking.yaml)),
  all 113 questions answered again on a T4 (25 Oct 2026), paired comparison on test with the decision rule fixed in
  the config.
- **Result.** Fully correct 79 → 72 of 88 (−8.0 points, 95% CI −13.6 to −3.4; 7 lost, 0 gained, p = 0.016).
  Key-fact recall 0.929 → 0.884 (worse). Two new critical errors, both wrong yes/no answers (K01, K08).
  Groundedness: no clear change on the 69 answers scored in both (0.940 → 0.930). Retrieval identical. Median time
  47.4 s → 9.5 s; peak GPU 10.26 → 8.69 GB.
- **Decision.** Keep thinking on.
- **Limitations.** One run per setting on one GPU type.
- **Evidence.** [`results/thinking_off/`](../results/thinking_off/).
- **Reproduce.** `policy-rag evaluate --name no_thinking --config configs/experiments/no_thinking.yaml`, then
  `policy-rag compare runs/baseline/run.json runs/no_thinking/run.json --splits test`.

## 6. Conversation memory

- **Objective.** Let people ask follow-ups ("Does it apply after an accident?") by rewriting them into complete
  questions from the conversation before searching.
- **Method.** Three runs on a T4, each with conversations written before the run, a decision rule fixed in advance
  (follow-ups that need the conversation significantly better with memory, McNemar p < 0.05, and no complete question
  or topic switch made wrong), and every follow-up answered without memory, with memory and as the complete question.
  After runs 1 and 2 the rewriter was fixed; run 3 was declared the last attempt.
- **Result.**

  | Run | Deciding follow-ups | Without memory | With memory | Fixed / broken | p | Made wrong elsewhere |
  |---|---|---|---|---|---|---|
  | 1 | 16 | 11 | 12 | 2 / 1 | 1.0 | 0 of 6 |
  | 2 | 28 | 16 | 21 | 6 / 1 | 0.125 | 1 of 14 |
  | 3 | 32 | 19 | 21 | 5 / 3 | 0.73 | 0 of 12 |

  The failure moved each time: copying facts from earlier answers (run 1), vague names such as "the policy" (run 2),
  the wrong referent for "it" or "that" (run 3). A rewrite took a median of 38–48 s with thinking on.
- **Decision.** Not adopted; memory is off by default (`memory.enabled: false`) and can be switched on for
  experiments.
- **Limitations.** One rewrite model; small sets written by the AI assistant that built the system; three attempts.
- **Evidence.** [`results/memory/`](../results/memory/), full write-up in
  [evaluation.md](evaluation.md#conversation-memory).
- **Reproduce.** `policy-rag evaluate-dialogues --name memory --resume-from <baseline run.json>`.

## 7. Qwen3 embeddings from Neon

- **Objective.** The tables were taken out of the chunk file (they are stored as tables in Postgres). On the 109
  text-only chunks, does Qwen3-Embedding-0.6B, with its vectors read from Neon, find the evidence at least as well as
  bge-small-en-v1.5?
- **Method.** Two retrieval-only runs on a T4, everything but the embedder the same; rule fixed before the runs (keep
  Qwen3 unless significantly worse, more than 5 points lower, or over 14 GB). Primary: all evidence reaching the
  context, on the 67 answerable test questions whose evidence is in the text chunks. Before this, the stored-vector
  search was checked against in-memory FAISS search: identical rankings for 113/113 questions.
- **Result.** 64/67 for both (p = 1.0). The embedder alone: dense recall@10 0.955 → 0.993, @5 0.940 → 0.963. Gate:
  1 answerable question refused with bge-small, 0 with Qwen3. Peak GPU 1.54 → 2.44 GB (retrieval only).
- **Decision.** Keep Qwen3.
- **Limitations.** Retrieval only; the 12 questions answered only by tables are outside this comparison.
- **Evidence.** [`results/embedder_qwen3/`](../results/embedder_qwen3/).

## 8. Table route, retrieval

- **Objective.** Get the 12 questions that only the printed tables answer (premiums, claim limits, grace periods,
  lists) to the answer model with exact values, without losing evidence elsewhere.
- **Method.** The table route ([`tables.py`](../src/policy_rag/tables.py)) answers from the typed tables with fixed
  SQL; its facts become the first context passage and the text chunks linked to the tables join the candidates. Two
  retrieval-only runs in one session, tables off and on; the tables-off run had to reproduce experiment 7 exactly
  (it did, 113/113). A second check, before experiment 9, placed the linked chunks in the context whatever their score.
- **Result.** Table-only questions with every key fact in the context: 0/12 → 11/12, and 12/12 with the linked chunks
  pinned (AB02 gained the Protect Benefit section). The 67 questions of experiment 7: 64 → 65 with all evidence in
  context. No question lost evidence or a key fact; the gate did not change; no not-answerable question got a table
  passage. The route alone answered 20 questions, every value matching its printed cell; median 708 ms per answered
  question on Neon.
- **Decision.** Tables on, linked chunks pinned.
- **Limitations.** The route's question patterns were written with these 113 questions in view.
- **Evidence.** [`results/table_route_retrieval/`](../results/table_route_retrieval/).

## 9. Neon + table route, full answers

- **Objective.** Decide whether the new system (experiments 7 and 8 together) can replace the baseline.
- **Method.** [`configs/experiments/qwen3_text_tables.yaml`](../configs/experiments/qwen3_text_tables.yaml), all 113
  questions answered on a T4 (3 Nov 2026, 109 answers generated); same reranker, answer model with thinking, prompt,
  gate and checks as the baseline. Rule fixed before the run: adopt unless fully correct on test is significantly
  worse or more than 5 points lower, or groundedness, unsupported-sentence rate or correct abstention is significantly
  worse, or the median answer takes over 150 s, or peak GPU memory exceeds 14 GB.
- **Result.** Test: 79 → 78 of 88 (−1.1 points, 95% CI −8.0 to +6.8; 5 fixed, 6 broken, p = 1.0). Key-fact recall
  0.929 → 0.945. Groundedness on the 75 answers scored in both: 0.895 → 0.900; answers with an unsupported sentence
  25.3% → 24.0%. Critical errors 1 → 2 (F01, L04). All 113: 97 → 100. Table-only questions 8 → 9 of 12. The 67
  questions of experiment 7: all evidence in context 63 → 66. Everyday wording 20 → 25 of 28; policy wording 59 → 56 of
  66. Median 47.4 → 49.2 s per answer; peak GPU 10.26 → 12.53 GB.
- **Decision.** Adopted (v3.4): not worse than the baseline. Not shown to be better.
- **Limitations.** 88 questions cannot rule out a loss of a few points or show a small gain. The remaining failures are
  mostly in the answer (G09, L07 and AB02 had their facts in the context). Unpaired groundedness over all answers is
  lower (0.886 → 0.873), from the dev answers. Table results are optimistic (see 8).
- **Evidence.** [`results/neon_tables/`](../results/neon_tables/).

## 10. R1: table cards, on fresh v3

- **Objective.** Decide whether R1 can replace v3.4. R1 targets the largest cause of failure found on new questions:
  a fact printed only in a table that the search never reaches.
- **Background.** On the fresh sets v1 and v2 (134 questions written after v3.4), every failure of v3.4 was traced to
  the first stage that lost what the answer needed. Of 35 failures, 13 were in the search (facts printed only in
  tables that the hand-written table patterns did not reach), 8 in context selection, 1 at the relevance gate and 13
  in the answer. A retrieval experiment on those sets (a search card for every table row) removed the search failures.
  Ingestion, chunking and the embeddings were not a cause of any failure.
- **Method.** R1 is v3.4 plus a search card for every data row of the 32 printed table parts (554 cards; large numeric
  grids in their own index), searched for every question with its own quota; each table's role (rule or worked
  example) and the scope of each footnote are fixed when the cards are built; the table route's answer goes to the
  model as evidence instead of replacing its answer (`configs/experiments/r1_cards.yaml`, branch `r1-candidate`). The
  checks were repaired first (amounts such as "30 L", ages without a unit, hedged openings; branch `grader-repair`).
  Decision set: fresh v3, 107 new questions (29 answered only by a table, 71 by the text, 7 not answerable), written
  before R1 was built. Rules D1–D7 were fixed and fingerprinted before any answer: overall not worse; table questions
  significantly better; text questions at most 3 worse; clearly wrong answers at most 2 more; declines not fewer;
  checks agreeing with a person's blind labels on 30 R1 answers at kappa ≥ 0.7; median ≤ 150 s and ≤ 14 GB on a T4.
  Both systems answered once on a T4 (12–13 Nov 2026).
- **Result.** Fully correct 88 → 94 of 107. Table questions 18 → 26 (9 fixed, 1 broken, p = 0.02); text questions
  63 → 61 (0 fixed, 2 broken, p = 0.5). Clearly wrong 0 and 0; not-answerable questions declined 7 and 7; answerable
  questions refused 9 → 4. Median 40.1 → 41.6 s per answer; peak GPU 9.32 → 9.25 GB. Failures by first stage: search
  6 → 0, context 3 and 3, gate 1 and 1, answer 9 and 9. The checks against the owner's 30 blind labels: kappa 0.52.
  In one of the three disagreements, the checks passed a clearly wrong answer: it said a surgery was covered at once
  and used the key words ("24-month", "list") in the sentence that denied them.
- **Decision.** Not adopted: rule D6 failed, so the scores were reported as unreliable.
- **Limitations.** The 30 labelled answers held only 4 that the checks failed, so one or two disagreements decided
  kappa; a sample balanced between passes and failures would have been more informative. The questions and keys were
  written by the AI assistant.
- **Evidence.** The private decision record (sha256 `a165b078e3fccbdf…`), the rules fixed before the run
  (`b239e76613d97693…`) and the question set (`fc7c4ab84d510395…`). v3.4's answers to this set are scored per
  question in [`results/final_evaluation/`](../results/final_evaluation/) (`fresh_v3`).

## 11. Final round: R1 against v3.4, on fresh v4

- **Objective.** Decide once more, with the answers that decide the outcome judged by a person instead of the
  checks. Fixed in advance as the project's last decision: no rerun, no repair of the checks, no new candidate.
- **Method.** Fresh v4: 88 new questions (30 answered only by a table, 50 by the text, 8 not answerable). Rules
  F1–F5 fixed and fingerprinted before any answer: table questions significantly better, judged by the person (F1);
  text questions at most 3 worse (F2); at most 2 more wrong answers by the person's labels (F3); declines not fewer
  (F4); median ≤ 150 s and ≤ 14 GB on a T4 (F5). Both systems answered once on a T4 (13–14 Nov 2026), with no score
  shown before the labels were in. The owner labelled 66 answers blind: both systems' answers to the 30 table
  questions and to the 3 text questions where the checks' verdicts differed, shuffled, with codes instead of system
  names. Elsewhere the checks' verdicts count.
- **Result.** Fully correct 68 → 74 of 88. Table questions 20 → 24 (6 fixed, 2 broken, p = 0.29); text questions
  40 → 42 (2 fixed, 0 broken). Not-answerable questions declined 8 and 8. Median 45.5 → 44.0 s; peak GPU 9.22 →
  9.31 GB. Failures by first stage: search 7 → 0, context 5 and 5, answer 8 and 9. The checks against the owner's
  66 labels: same verdict on 55, kappa 0.63. The labels were given as correct / not correct, so F3 was checked for
  every possible split of the "not correct" answers into incomplete and wrong; it could not change the decision.
- **Decision.** Not adopted: F1 failed. v3.4 stays the default. With 2 answers broken, 10 fixed would have been
  needed for p < 0.05.
- **Limitations.** 30 table questions can only show large differences. Text answers where the checks gave both
  systems the same verdict were not labelled. R1's answers sometimes cite table cards, so the system behind an answer
  could sometimes be guessed. The questions and reference answers were written by the AI assistant; the person
  judged against the policy.
- **Evidence.** The private decision record (sha256 `65c40c37584ee5e0…`, which holds the 66 labels), the rules fixed
  before the run (`9282e14a8a4ce3e1…`) and the question set (`23b6f7f790771875…`). v3.4's answers, with the person's
  labels, are scored per question in [`results/final_evaluation/`](../results/final_evaluation/) (`fresh_v4`).

**Across both rounds.** R1 removed every search failure of a fact printed only in a table (13 of 13), and its
answers to table questions were better in both rounds (+8 by the checks, +4 by a person), but neither round met the
rules fixed in advance. Pooling the two rounds after seeing them would be a test chosen after the fact, so it is not
used. The failures left in both systems are in context selection and in the answers.

## 12. Evaluation of v3.4 on every question set

- **Objective.** Measure how the adopted system performs on questions it was never tuned on, where it fails and why,
  without changing it. Done after the last decision.
- **Method.** The five saved runs of v3.4 were re-scored with the repaired checks, with the person's 33 blind labels
  in fresh v4 replacing the checks' verdict. The runs cover the development set and fresh v1–v4 (329 questions
  written after v3.4). Nothing was regenerated. Each failure was placed at the first stage that lost what the answer
  needed, and the 34 answer-stage failures were read one by one. Full write-up:
  [final_evaluation.md](final_evaluation.md).
- **Result.**
  - **Accuracy:** 255/329 = 77.5% fully correct on new questions (95% CI 73–82%), against 101/113 on the
    development set.
  - **By question kind:** table questions 49/84, text questions 177/216, unanswerable questions declined 29/29.
  - **Where the 76 failures counted by the checks started:** search 26 (every one a fact printed only in a table),
    context 15, gate 1, the answer 34.
  - **The reading of the 34:** 17 clear model errors (9 answers opened with the refusal sentence although they then
    applied the right rule, 8 missed a qualifier), 8 context failures the checks misfiled, and 9 acceptable answers.
- **Limitations.** One run per set on one GPU type. The fresh questions were written by the AI assistant. The reading
  of the failures is one reviewer's judgement, apart from the person's labels.
- **Evidence.** [`results/final_evaluation/`](../results/final_evaluation/) (a row per question and set, a
  scorecard, the reading of the failures, the sha256 of every input).
- **Reproduce.** `scripts/export_final_evaluation.py` with the private run files (command in the results folder).

## Not run, or not verifiable

| Item | Status |
|---|---|
| Hard questions ([`testsets/optima_secure_hard_questions.json`](../testsets/optima_secure_hard_questions.json)) | Only an early development run, judged by reading, under the old single token limit. Its outputs were not kept, so the figures quoted earlier (about 5 of 15 right without thinking, 6 with thinking, 6 answers lost to cut-off thinking) **cannot be verified** and are not used as results. Not rerun since. |
| Independent human labels for the checks | Not collected for the 113-question set. Collected blind on the fresh sets: 30 answers on fresh v3 (kappa 0.52) and 66 on fresh v4 (kappa 0.63); see 10 and 11. |
| Regenerating the baseline answers with the package | Not run; all package runs of the baseline reused the cached v3.1 answers. |
| Chunk file reproduced byte for byte from the PDF | Not confirmed (needs a blocked download). |
| A larger thinking budget, a recalibrated reranker floor, other prompts | Suggested in [evaluation.md](evaluation.md); not run. |
| Groundedness against human judgement | Not measured; HHEM is known to flag correct paraphrases, so its "unsupported" rate is an upper estimate. |
| Model download size (about 20 GB) | An estimate from the models' published sizes, not measured in a run. |
| The table route on new questions | Measured on the fresh sets: it did not reach a fact printed only in a table for 13 of 35 failures on fresh v1–v2, 6 of 29 table questions on fresh v3 and 7 of 30 on fresh v4. |
| Query latency and concurrency in a deployed service | Not measured; only per-question timings inside evaluation runs on Colab. |
| The baseline (`configs/default.yaml`) on the fresh question sets | Not run: the fresh numbers describe v3.4 only. |
| A changed answer instruction (no refusal sentence before an answer, explicit false-premise correction) | Suggested by the evaluation (12); not run. |
| Rebuilding v3.4's chunk file and database from this repository | Not possible: they were made by a private ingestion bundle; this repository has the schema and the code that reads them. |
