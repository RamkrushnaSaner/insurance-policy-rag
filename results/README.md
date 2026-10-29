# Published results

Evidence for every number in the [README](../README.md) and [docs/](../docs/). These files hold **scores only**: no
answers, no retrieved passages and no policy text (those are in the private run files, which quote the policy).
Each folder has a `source.json` with the sha256 of the private run file(s) it was built from, and everything here can
be rebuilt with [`scripts/export_results.py`](../scripts/export_results.py). The headline numbers are recomputed from
these files by [`tests/test_published_results.py`](../tests/test_published_results.py) on every push.

| Experiment | Run date (Colab T4) | Result | Evidence |
|---|---|---|---|
| Baseline: [`configs/default.yaml`](../configs/default.yaml) on the 113-question test set | answers 22 Oct 2026; re-measured 29 Oct 2026 | 79/88 fully correct on test, 97/113 overall | [`baseline/`](baseline/) |
| Reproduction: package v3.3 re-running the baseline with the published answers | 27 Oct (twice), 29 Oct 2026 | identical answers, contexts and verdicts in all three sessions; 0 answers regenerated | [`reproduction/`](reproduction/) |
| Thinking off ([`configs/experiments/no_thinking.yaml`](../configs/experiments/no_thinking.yaml)) | 25 Oct 2026 | 72/88 on test: 7 lost, 0 gained (p = 0.016); not adopted | [`thinking_off/`](thinking_off/) |
| v2.6 (Mistral-7B, text-file chunks) against v3.1, 24 shared dev questions | 20 and 22 Oct 2026 | 11 → 17 fully correct: 6 fixed, 0 broken (p = 0.031) | [`v2_6_vs_v3_1/`](v2_6_vs_v3_1/) |
| Conversation memory, three runs | 27 Oct (two), 29 Oct 2026 | never significantly better; not adopted, off by default | [`memory/`](memory/) |

All v3 runs use the chunk file with sha256 `74aa551c…`, Qwen3-8B in 4-bit NF4, bge-reranker-v2-m3, bge-small-en-v1.5
and HHEM-2.1-Open, and are scored on test set `7313dd7571` with the current checks (unchanged since v3.2; older runs
are re-scored). The v2.6 run used an older text export (`chunks.json`), Mistral-7B-Instruct-v0.3 in 4-bit and
bge-reranker-base.

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

To rebuild a folder from your own run (for example after `policy-rag evaluate --name baseline`):

```bash
python scripts/export_results.py run runs/baseline/run.json results/baseline
python scripts/export_results.py compare runs/baseline/run.json runs/<candidate>/run.json results/<candidate> --splits dev test
python scripts/export_results.py dialogues runs/memory/dialogues.json results/memory/run4
```

## What is not here

- **Hard questions** ([`testsets/optima_secure_hard_questions.json`](../testsets/optima_secure_hard_questions.json)):
  the only run was an early development run judged by reading; its outputs were not kept, so no result is published.
- **Human labels** for the checks (`policy-rag agreement`): not collected yet.
- **Regenerated baseline answers:** every package run of the baseline reused the v3.1 answers from the cache. A fresh
  generation with the package (thinking on) has not been run.
