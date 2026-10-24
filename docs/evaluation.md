# Evaluation

How the system is measured, what the baseline scored, every failure, and how far the numbers can be trusted.

## Approach

Every answer is checked by **deterministic rules** against a hand-written answer key. There is no LLM judge, and no
RAGAS or tracing service. The rules cost nothing to run, give the same verdict every time, and are easy to read
and fix when they are wrong. Their weakness is that they are rules: an answer phrased in a way the key does not
foresee can be scored wrong, and a false statement the key does not foresee can slip through. That is why the
checks themselves are validated against human labels ([below](#validating-the-checks)) and why the per-question
results are published.

## The test set

[`testsets/optima_secure_v3.json`](../testsets/optima_secure_v3.json), version `7313dd7571`: 113 questions written
by hand from the policy wording.

| Type | dev | test | all |
|---|---|---|---|
| fact | 3 | 22 | 25 |
| list | 2 | 11 | 13 |
| conditional ("is X covered if Y?") | 4 | 9 | 13 |
| definition | 1 | 7 | 8 |
| comparison (plans, premiums) | 4 | 3 | 7 |
| procedure | 2 | 3 | 5 |
| multi-section | 1 | 3 | 4 |
| partial (the policy answers only part) | 1 | 0 | 1 |
| everyday wording ("paraphrase") | 2 | 15 | 17 |
| false premise | 2 | 3 | 5 |
| out of scope | 3 | 12 | 15 |
| **all** | **25** | **88** | **113** |

32 questions are in everyday wording: the 17 of type "paraphrase" and 15 of other types (6 conditional, 4 out of
scope, 2 multi-section, 1 each false premise, partial and comparison). 19 cannot be answered from the policy
(15 out of scope, 4 false premises). **dev** (25) is for debugging; **test** (88) is held out and is the
only split on which a change is adopted or rejected.

Each question has:

- **key facts**, each with alternative phrasings or a regular expression ("24 months", "two years");
- **evidence anchors**: short phrases that identify the passage holding the answer, matched on words. Retrieval is
  scored for any chunk file of the same PDF, without depending on chunk ids;
- **extra checks** where they apply: plan-table values (3 questions), the expected yes/no (18), the order of steps
  (1), sections of other covers that must not be used (9), known false statements (11), "must say what is not shown"
  (1), and terms that must not occur in the policy for not-answerable questions (20).

Changing any question changes the version hash; runs on different versions are not compared.

## Metrics

| Group | Metric | Definition |
|---|---|---|
| **Primary** | fully correct | All key facts present, no critical error, nothing missing or cut off. For a not-answerable question: says so within the first two sentences, with no critical error |
| Correctness | key-fact recall | Share of key facts found in the answer |
| | critical error rate | Answers with a wrong table value, a known false statement, another cover's section used, a wrong yes/no, a wrong order, a figure that is not in the context, or an answer to a not-answerable question |
| | list completeness | Key-fact recall on list questions |
| Faithfulness | figures not in the context | Amounts, percentages and durations in the answer that appear in neither the context nor the question |
| | citations not in the context | Section numbers cited that the context does not contain |
| | groundedness (HHEM) | Share of answer sentences that [HHEM-2.1-Open](https://huggingface.co/vectara/hallucination_evaluation_model) scores as supported by the context (at 0.5 or above) |
| Retrieval | search recall | Share of evidence passages among the search candidates |
| | context recall | Share of evidence passages in the context the model read |
| | MRR@10 | 1 / rank of the first evidence chunk among the top 10 reranked |
| | context precision | Share of context tokens that belong to evidence chunks |
| Abstention | correct abstention | Not-answerable questions declined |
| | false refusal | Answerable questions refused |
| | gate AUROC | How well the best reranker score separates answerable from not-answerable questions |
| Robustness | everyday-wording gap | Fully correct on policy wording minus fully correct on everyday wording |
| Efficiency | seconds, tokens, peak GPU | Per generated answer; peak memory for the run |

**Stage attribution.** Every problem is tagged with the stage where it started:
*reliability* (pipeline error) > *corpus* (the fact is not in the chunk file) > *retrieval* (not among the
candidates) > *context* (a candidate, but not in the context) > *gate* (refused without the model) > *generation* (in
the context but not, or wrongly, in the answer). The first failing stage points at the component to change.

**Extra tests** in full runs:

- *Evidence removed.* Up to 4 questions are answered again with their evidence taken out of the context. The answer
  should say the policy text does not show it, not fill the gap from memory.
- *Repeat check.* One answer is regenerated without the cache, to test whether generation is deterministic.
- *Retrieval repeat.* Retrieval runs twice for every question, and both results must be identical.

## Statistics and the decision rule

The unit is the question. Two runs are compared on the same questions:

- **Primary test:** McNemar's exact test on "fully correct" (discordant questions only), with a Holm adjustment
  when several candidates are compared with one baseline.
- **Secondary:** paired bootstrap 95% intervals (10,000 resamples) for key-fact recall, critical errors,
  groundedness, hallucination, abstention, context recall and MRR@10.
- **Intervals** for single proportions: Wilson 95%.
- **Power:** 88 test questions detect a difference of about 15 points in fully correct (80% power, assuming a quarter
  of the questions change outcome); 25 dev questions about 27 points. Smaller real improvements will usually show
  as "not significant".

The decision rule was fixed before any results ([`configs/default.yaml`](../configs/default.yaml), `decision`). A
candidate replaces the baseline only if:

1. fully correct is significantly better on the test split (p < 0.05);
2. groundedness, hallucination and abstention are not significantly worse;
3. the median answer takes at most 150 s and peak GPU memory stays at or below 14 GB.

## Validating the checks

Every full run writes `labels_to_fill.csv`: 30 answers stratified by verdict, without the automatic verdict. A
person marks each one as fully correct (1) or not (0); then `policy-rag agreement runs/<name>/run.json labels.csv`
prints Cohen's kappa (target ≥ 0.7) and every disagreement. A check that disagrees for a clear reason gets fixed,
which makes a new checks version, and runs are re-scored.

**Status:** this has **not yet been done with independent labels**. What has been done:

- During development, all 113 baseline answers were read and judged by hand, not blind. That reading is by the AI
  assistant that wrote the checks, not by an independent person.
- Before the fixes, the v3.1 checks agreed with that reading on 107 of 113 answers (kappa about 0.80). Five correct
  answers had been marked wrong, and one known false statement had been missed.
- The checks were corrected (v3.2) and now agree with that reading on all 113.

## Baseline results

**System:** [`configs/default.yaml`](../configs/default.yaml): bge-small-en-v1.5 + BM25, bge-reranker-v2-m3,
Qwen3-8B 4-bit with thinking (budget 1,024 tokens), 1,500-token context, gate at −5. The chunk file has sha256
`74aa551c…`. **Answers** were generated on 22 October 2026 on Google Colab (Tesla T4) by the v3.1 notebook, then
re-scored with the current checks and test set `7313dd7571`. **Groundedness and retrieval metrics** are as measured
in that run. Full scorecard: [`results/baseline/scorecard.md`](../results/baseline/scorecard.md); per question:
[`results/baseline/per_question.csv`](../results/baseline/per_question.csv).

| | dev (25) | test (88) | all (113) |
|---|---|---|---|
| **Fully correct** | 18/25 = 72.0% (52–86%) | **79/88 = 89.8% (82–95%)** | 97/113 = 85.8% (78–91%) |
| Key-fact recall (answerable) | 0.885 | 0.929 | 0.919 |
| Critical error rate | 4.0% | 1.1% | 1.8% |
| Plan-table values | 13/18 correct, 0 wrong | – | 13/18 correct, 0 wrong |
| Groundedness (HHEM) / answers with an unsupported sentence | 0.809 / 43.5% | 0.863 / 30.9% | 0.851 / 33.7% |
| Figures not in the context | 0% | 0% | 0% |
| Search recall / context recall | 0.952 / 0.889 | 0.989 / 0.932 | 0.980 / 0.922 |
| MRR@10 / context precision | 0.806 / 0.391 | 0.894 / 0.280 | 0.874 / 0.305 |
| Correct abstention | 3/4 | 15/15 | 18/19 |
| False refusal | 4.8% | 4.1% | 4.3% |
| Gate AUROC | 0.619 | 0.952 | |
| Everyday wording, fully correct (answerable) | 6/8 | 14/20 | 20/28 |
| Policy wording, fully correct (answerable) | 9/13 | 50/53 | 59/66 |
| Seconds per answer, median / 95th percentile | 58 / 148 | 46 / 116 | 47 / 131 |
| Thinking tokens, median / budget reached | 418 / 3 | 334 / 5 | 339 / 8 |

Evidence removed: 4/4 answers said the policy text does not show it. Retrieval was identical when repeated (113/113),
and the regenerated answer was identical. There were no pipeline errors, out-of-memory retries or cut-off answers.
Peak GPU memory was 10.26 GB, and the answer model ran at 6.1 tokens/s. The whole run took about 2 hours.

Read the gate AUROC on dev with care: dev has only 4 not-answerable questions.

### The 16 failures

| id | split | wording | first stage | what went wrong |
|---|---|---|---|---|
| X02 | test | everyday | context | The evidence was rank 5 but scored −6.24, below the −5 floor, so it never reached the context (only 708 of 1,500 tokens were used); the model then refused |
| X04 | test | everyday | context | Evidence rank 6, score −5.34, just below the floor; refused |
| FP03 | dev | everyday | context | Evidence rank 26, score −6.65; refused |
| X12 | test | everyday | context | Evidence rank 6 (score −0.97), crowded out of a full context |
| PR02 | test | everyday | context | The right premium row ranked 2nd, but the context filled with other rows; refused |
| G08 | dev | policy | context | A long list: two of its exclusions did not fit in the budget |
| X03 | dev | everyday | retrieval | "Delivery" in the question, "maternity" in the policy: the evidence was not among the candidates |
| AB02 | test | everyday | context | The Protect Benefit section (4.3) was not among the candidates |
| G01 | test | policy | generation | Left out "5 consecutive days" and "economy", and said they were not shown |
| G02 | dev | policy | generation | Left out "irrespective of claims" |
| G09 | dev | policy | generation | Plan comparison: 4 of 14 table values missing |
| C01 | dev | policy | generation | One table value missing (borderline) |
| D05 | test | policy | generation | Left out "sum insured" |
| AB01 | test | policy | generation | Said the Protect Benefit is not shown, though it was in the context |
| X01 | test | everyday | generation | **Critical:** said a waiting period applies "regardless of accidents", but accidents are exempt |
| N04 | dev | policy | generation | **Critical:** answered a question the policy does not cover, instead of declining |

**What the failures point to.** This is an analysis; none of it has been implemented or tested:

- Half of the failures (8) are about retrieval and context, and 7 of those 8 are everyday-wording questions. In
  three of them the evidence was found but scored just below the −5 floor, which was chosen for an earlier reranker.
  Recalibrating the floor on dev data, then testing it on test, is the most direct experiment.
- Another three were crowded out of the context, and one was a search miss caused by vocabulary.
- The generation failures are mostly omissions. Prompt changes might help, but each would need to show a
  significant improvement on test, and with 88 questions only large effects can.

### Compared with the v2.6 baseline

On the 24 dev questions shared with the v2.6 baseline (Mistral-7B, text-file chunks, bge-reranker-base), fully
correct went from 11 to 17: 6 questions were fixed and none broke (McNemar p = 0.031). These numbers come from the
v3.1 checks. The comparison is on dev, and three things changed at once (chunks from the PDF, reranker, answer model),
so it shows the new system is better on these questions, not which change mattered.

## Hard questions (not scored)

[`testsets/optima_secure_hard_questions.json`](../testsets/optima_secure_hard_questions.json) has 15 questions
beyond the test set, such as multi-claim calculations, deductibles across a family floater, deadlines with dates
and interacting waiting periods. Each comes with an expected answer for a person to compare. In an earlier
development run, judged by reading (not by the checks), about 5 of 15 were right without thinking and 6 of 15 with
thinking under the old single token limit. In that run, 6 answers were lost to cut-off thinking, which led to the
thinking budget. They have not been rerun with the current budget.

## Verification

The published answers were generated by the v3.1 notebook, so the package was checked against it. All checks below
ran without the real models (none can be downloaded in the environment where the package was built):

| Check | Result |
|---|---|
| Rebuild the 113 saved v3.1 contexts from their chunk ids (real chunk file) | 113/113 identical |
| Answer-cache keys of the 108 generated v3.1 answers, from their prompts | 108/108 identical, so the v3.1 cache is reused |
| Retrieval, reranking, context and gate on all 113 questions, notebook v3.2 vs package, same small stand-in models | 113/113 identical (candidates, scores, contexts, gate, retrieval metrics) |
| Full run on dev with a small stand-in answer model: two-phase thinking, seeded sampling, cache, checks, evidence-removed test, repeat check, groundedness | 25/25 identical answers, keys and verdicts |
| Re-scoring the saved v3.1 answers, notebook v3.2 vs package | 113/113 identical verdicts: 97/113 fully correct |

Still open:

- a full run of the package on a GPU with the real models;
- a check that the published chunk file is reproduced byte for byte from the PDF. The chunker's text, sections and
  table chunks match, and the remaining check needs the `tiktoken` vocabulary download, which was blocked in the
  build environment.

## Reproducing the scores

- **Re-score saved answers** (minutes, CPU): `policy-rag evaluate --name rescored --mode rescore --rescore-from
  <run.json or results zip>`. This needs the run's saved answers, which contain policy text and are not published.
- **Regenerate** (about 2 hours on a T4): `policy-rag evaluate --name baseline`. With the same software and GPU type,
  seeded sampling should give the same answers. Other versions may differ slightly; use `compare` to see whether the
  difference matters.
