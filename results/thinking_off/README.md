# Experiment: thinking off

**Question.** Does Qwen3-8B answer as well without its thinking step? Thinking makes answers about five times slower.

**Method.** [`configs/experiments/no_thinking.yaml`](../../configs/experiments/no_thinking.yaml) changes one setting
(`generator.enable_thinking: false`, with Qwen's recommended non-thinking sampling). All 113 questions were answered
again with package v3.2 on a Colab T4 on 25 October 2026. Both runs are scored with the current checks and compared
question by question on the held-out test split (McNemar's exact test; paired bootstrap for the other metrics), with
the decision rule fixed in [`configs/default.yaml`](../../configs/default.yaml) before the run. The baseline side of
the comparison is the published baseline answers, scored the same way ([`../baseline/`](../baseline/)).

**Result (test split, 88 questions).**

| | Thinking on (baseline) | Thinking off | Paired difference (95% CI) |
|---|---|---|---|
| Fully correct | 79/88 (89.8%) | 72/88 (81.8%) | −8.0 points (−13.6 to −3.4): 7 lost, 0 gained, McNemar p = 0.016 |
| Key-fact recall (73 answerable) | 0.929 | 0.884 | −0.046 (−0.087 to −0.014), worse |
| Critical errors | 1 (X01) | 3 (X01, K01, K08: two wrong yes/no answers) | +2.3 points (0 to +5.7), not significant |
| Groundedness (69 answers scored in both runs) | 0.940 | 0.930 | −0.010 (−0.054 to +0.034), no clear change |
| Answers with an unsupported sentence (same 69) | 20.3% | 14.5% | −5.8 points (−15.9 to +4.3), no clear change |
| Context recall, MRR@10 | 0.932, 0.894 | 0.932, 0.894 | identical: retrieval was the same for all 113 questions |
| Seconds per answer, median / 95th percentile (all 113) | 47.4 / 131 | 9.5 / 52 | |
| Peak GPU memory | 10.26 GB | 8.69 GB | |

The answers lost without thinking: P02, C02, K01, K08, D06, D07, PR03 ([`paired.csv`](paired.csv)). Dev (25
questions): 18 fully correct in both runs.

**Decision: keep thinking on.** Fully correct is significantly worse without thinking.

**Limitations.** One run of each setting on one GPU type; sampling is seeded, so a rerun on the same software gives
the same answers, but other library versions may differ. The scorecard's unpaired groundedness
([`scorecard.md`](scorecard.md): 0.927 without thinking against 0.886 with it) averages over different sets of
answers and is not a fair comparison; the paired figures above are.

**Files.** [`comparison.csv`](comparison.csv) (the `compare` output for dev and test), [`secondary.csv`](secondary.csv)
(paired bootstrap intervals for the other metrics), [`paired.csv`](paired.csv)
(verdict per question in both runs), [`per_question.csv`](per_question.csv) and [`scorecard.md`](scorecard.md) for
the thinking-off run, [`source.json`](source.json) and [`comparison_source.json`](comparison_source.json).
