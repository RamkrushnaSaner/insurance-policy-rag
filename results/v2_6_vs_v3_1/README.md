# Comparison: v2.6 against v3.1

**Question.** Is the v3 system (chunks from the PDF, bge-reranker-v2-m3, Qwen3-8B with thinking) better than the
earlier v2.6 notebook (text-file chunks, bge-reranker-base, Mistral-7B-Instruct-v0.3)?

**Method.** The v2.6 compact evaluation (Colab T4, 20 October 2026) answered 24 questions; all 24 are dev questions
of the current test set, with the same wording. Its answers and the v3.1 answers to the same 24 questions (22 October
2026) were both re-scored with the current checks and compared question by question.

**Result.** Fully correct 11/24 (v2.6) → 17/24 (v3.1): 6 fixed (G03, G04, G07, K07, M01, N05), none broken, McNemar
exact p = 0.031. Critical errors fell from 5 to 1 (paired bootstrap 95% CI for the difference −33 to −4 points).

**Limitations.** Dev questions only, which guide development and are not used for decisions. Three things changed at
once (chunking, reranker, answer model), so this shows the new system is better on these questions, not which change
mattered. The first version of this comparison used the v3.1 checks and gave the same counts.

**Files.** [`paired.csv`](paired.csv) (verdict per question in both systems), [`comparison.csv`](comparison.csv)
(the `compare` output), [`secondary.csv`](secondary.csv) (other metrics with paired bootstrap intervals),
[`comparison_source.json`](comparison_source.json).
