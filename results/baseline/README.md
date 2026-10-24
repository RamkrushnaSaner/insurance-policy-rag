# Baseline results

The published numbers for [`configs/default.yaml`](../../configs/default.yaml) on test set `7313dd7571`. The method
is in [docs/evaluation.md](../../docs/evaluation.md).

| File | Contents |
|---|---|
| [`scorecard.md`](scorecard.md) / [`scorecard.csv`](scorecard.csv) | All metrics by split (dev, test, all) |
| [`per_question.csv`](per_question.csv) | One row per question: verdict, first failing stage, issues found, gate, retrieval metrics, groundedness, thinking tokens, seconds |

**Provenance.** The answers were generated on 22 October 2026 on Google Colab (Tesla T4) by the v3.1 notebook
(Qwen3-8B 4-bit with thinking, bge-reranker-v2-m3, chunk file sha256 `74aa551c…`). They were re-scored with the
current checks, which correct five false failures and one missed false statement in the v3.1 checks; the package
produces the same verdicts. Retrieval metrics, groundedness (HHEM), timing and memory are as measured in that run.

These files hold scores and question ids only. The questions are in
[`testsets/optima_secure_v3.json`](../../testsets/optima_secure_v3.json). The answers themselves are not published,
because they quote the policy at length.
