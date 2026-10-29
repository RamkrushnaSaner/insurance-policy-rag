# Baseline results

The published numbers for [`configs/default.yaml`](../../configs/default.yaml) on test set `7313dd7571`. The method
is in [docs/evaluation.md](../../docs/evaluation.md).

| File | Contents |
|---|---|
| [`scorecard.md`](scorecard.md) / [`scorecard.csv`](scorecard.csv) | All metrics by split (dev, test, all) |
| [`per_question.csv`](per_question.csv) | One row per question: verdict, first failing stage, issues found, gate, retrieval metrics, groundedness, thinking tokens, seconds |
| [`source.json`](source.json) | The two private run files these were built from (sha256), models, package versions |

**Provenance.**

- **Answers** were generated on 22 October 2026 on Google Colab (Tesla T4) by the v3.1 notebook (Qwen3-8B 4-bit
  with thinking, bge-reranker-v2-m3, chunk file sha256 `74aa551c…`). Timing, tokens per second, peak GPU memory,
  the evidence-removed test and the repeat check are from that run.
- **Verdicts** use the current checks, which correct five false failures and one missed false statement in the v3.1
  checks (v3.1's own checks gave 93/113; [details](../../docs/evaluation.md#validating-the-checks)).
- **Retrieval metrics and groundedness** are from package v3.3 on a T4 on 29 October 2026, which reran retrieval for
  all 113 questions and took the answers from the v3.1 cache: identical contexts and answers, 0 answers generated
  ([`../reproduction/`](../reproduction/)). Groundedness was measured the same way in three separate sessions with the
  same result (0.886).

**Change on 29 October 2026.** Groundedness was first published as 0.851 (33.7% of answers with an unsupported
sentence), the v3.1 notebook's own measurement. That notebook split answers into sentences differently: it cut
"Rs. 15,000" into two sentences and also scored refusal sentences. The package's splitter (which v3.2 introduced, and
which every later run uses) gives **0.886 (28.3%)** on the same answers. Nothing else in these files changed: a fresh
rebuild matched the earlier files in every other column and row.

These files hold scores and question ids only. The questions are in
[`testsets/optima_secure_v3.json`](../../testsets/optima_secure_v3.json). The answers themselves are not published,
because they quote the policy at length.
