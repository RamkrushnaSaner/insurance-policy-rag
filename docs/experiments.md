# Experiments

Every experiment run in this project, in the order it was run, with what it measured and where the evidence is. All
numbers come from the files in [`results/`](../results/), and
[`tests/test_published_results.py`](../tests/test_published_results.py) recomputes the headline figures from them.
Things that were planned or claimed earlier but cannot be backed by evidence are listed at the end.

**Words used here.** *Fully correct*: the answer has every key fact, no critical error and is not cut off (for a
question the policy does not answer: it says so at the start). *Test split*: the 88 held-out questions on which
changes are adopted or rejected; *dev*: 25 questions for development. *McNemar p*: the chance of a difference at least
this large between two systems if they were really equally good, counting only the questions where they disagree.

## Summary

| # | Experiment | Date | Main result | Decision | Evidence |
|---|---|---|---|---|---|
| 1 | [Baseline](#1-baseline) | 22 Oct 2026 | 79/88 fully correct on test (89.8%, 95% CI 82–95%), 97/113 overall | the published system | [`results/baseline/`](../results/baseline/) |
| 2 | [Correcting the checks](#2-correcting-the-checks) | before 24 Oct 2026 (v3.2) | 6 verdicts changed (5 false failures, 1 missed false statement) | checks v3.2 | [`results/baseline/per_question.csv`](../results/baseline/per_question.csv) |
| 3 | [v2.6 against v3.1](#3-v26-against-v31) | 20 and 22 Oct 2026 | 11 → 17 of 24 dev questions, p = 0.031 | v3 kept | [`results/v2_6_vs_v3_1/`](../results/v2_6_vs_v3_1/) |
| 4 | [Reproduction](#4-reproduction) | 23–29 Oct 2026 | same contexts, prompts, verdicts and groundedness in every session | – | [`results/reproduction/`](../results/reproduction/) |
| 5 | [Thinking off](#5-thinking-off) | 25 Oct 2026 | 72/88 on test: 7 lost, 0 gained, p = 0.016; 5× faster | keep thinking on | [`results/thinking_off/`](../results/thinking_off/) |
| 6 | [Conversation memory](#6-conversation-memory) | 27–29 Oct 2026 | three runs, never significantly better (final: 21 vs 19 of 32, p = 0.73) | not adopted, off by default | [`results/memory/`](../results/memory/) |

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

## Not run, or not verifiable

| Item | Status |
|---|---|
| Hard questions ([`testsets/optima_secure_hard_questions.json`](../testsets/optima_secure_hard_questions.json)) | Only an early development run, judged by reading, under the old single token limit. Its outputs were not kept, so the figures quoted earlier (about 5 of 15 right without thinking, 6 with thinking, 6 answers lost to cut-off thinking) **cannot be verified** and are not used as results. Not rerun since. |
| Independent human labels for the checks | Not collected. |
| Regenerating the baseline answers with the package | Not run; all package runs of the baseline reused the cached v3.1 answers. |
| Chunk file reproduced byte for byte from the PDF | Not confirmed (needs a blocked download). |
| A larger thinking budget, a recalibrated reranker floor, other prompts | Suggested in [evaluation.md](evaluation.md); not run. |
| Groundedness against human judgement | Not measured; HHEM is known to flag correct paraphrases, so its "unsupported" rate is an upper estimate. |
| Model download size (about 20 GB) | An estimate from the models' published sizes, not measured in a run. |
