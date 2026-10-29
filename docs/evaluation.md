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
- Before the fixes, the v3.1 checks agreed with that reading on 107 of 113 answers (Cohen's kappa 0.80): five correct
  answers had been marked wrong (L04, K04, D02, D03, D07), and one known false statement had been missed (X01). The
  v3.1 checks gave 93/113 (75/88 on test); the corrected checks give 97/113 (79/88).
- The checks were corrected (v3.2) and now agree with that reading on all 113.

## Baseline results

**System:** [`configs/default.yaml`](../configs/default.yaml): bge-small-en-v1.5 + BM25, bge-reranker-v2-m3,
Qwen3-8B 4-bit with thinking (budget 1,024 tokens), 1,500-token context, gate at −5. The chunk file has sha256
`74aa551c…`. **Answers** were generated on 22 October 2026 on Google Colab (Tesla T4) by the v3.1 notebook, then
re-scored with the current checks and test set `7313dd7571`. **Retrieval metrics and groundedness** were measured by
package v3.3 on a T4 on 29 October 2026, which rebuilt the same contexts and took the answers from the cache (timing,
speed and memory are from the generating run). Full scorecard:
[`results/baseline/scorecard.md`](../results/baseline/scorecard.md); per question:
[`results/baseline/per_question.csv`](../results/baseline/per_question.csv); sources and the groundedness correction:
[`results/baseline/README.md`](../results/baseline/README.md).

| | dev (25) | test (88) | all (113) |
|---|---|---|---|
| **Fully correct** | 18/25 = 72.0% (52–86%) | **79/88 = 89.8% (82–95%)** | 97/113 = 85.8% (78–91%) |
| Key-fact recall (answerable) | 0.885 | 0.929 | 0.919 |
| Critical error rate | 4.0% | 1.1% | 1.8% |
| Plan-table values | 13/18 correct, 0 wrong | – | 13/18 correct, 0 wrong |
| Groundedness (HHEM) / answers with an unsupported sentence | 0.862 / 38.1% | 0.892 / 25.6% | 0.886 / 28.3% |
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
correct went from 11 to 17: 6 questions were fixed (G03, G04, G07, K07, M01, N05) and none broke (McNemar p = 0.031);
critical errors fell from 5 to 1. Both systems' answers are re-scored with the current checks (the v3.1 checks gave
the same counts). The comparison is on dev, and three things changed at once (chunks from the PDF, reranker, answer
model), so it shows the new system is better on these questions, not which change mattered. Evidence:
[`results/v2_6_vs_v3_1/`](../results/v2_6_vs_v3_1/).

## Experiments

**Thinking off** (`configs/experiments/no_thinking.yaml`, run on a Colab T4 on 25 October 2026, package v3.2), judged
by the decision rule above. Speed and memory figures are over all 113 questions; groundedness is compared on the 69
test answers HHEM scored in both runs. Evidence: [`results/thinking_off/`](../results/thinking_off/).

| | Thinking on (baseline) | Thinking off |
|---|---|---|
| Fully correct, test | 79/88 | 72/88: 7 lost, 0 gained (McNemar p = 0.016) |
| Key-fact recall, test | 0.929 | 0.884 |
| Critical errors, test | 1 | 3 (two wrong yes/no answers: K01, K08) |
| Groundedness / answers with an unsupported sentence, test (paired, 69) | 0.940 / 20.3% | 0.930 / 14.5% (no clear change) |
| Seconds per answer, median / 95th percentile | 47 / 131 | 9.5 / 52 |
| Peak GPU memory | 10.3 GB | 8.7 GB |

Without thinking the model gave wrong yes/no answers, left conditions out of definitions and procedures (P02, D06,
D07), said "not shown" for facts in its context (C02, D07) and refused a premium question it had the row for (PR03).
**Decision: keep thinking on.** Retrieval in this run was identical to the published run for all 113 questions.

## Conversation memory

Memory (v3.3) is evaluated on its own conversations, because the 113 questions are all single questions. Turn 1 is a
question; each later turn's complete form is a test-set question, so the existing answer keys score it.
`policy-rag evaluate-dialogues --name memory` answers every later turn three ways:

| Condition | What is asked | Role |
|---|---|---|
| `no_memory` | the follow-up as typed, without the conversation | what the system did before memory |
| `memory` | the system's own rewrite | what memory does |
| `gold` | the complete test-set question | the best a rewrite can achieve (usually cached from a full run) |

Each rewrite is also checked against keywords: the name of what the follow-up refers to (`subject`), and for topic
switches, earlier-topic terms it must not pull in.

### First run: not adopted

Evidence: [`results/memory/run1/`](../results/memory/run1/summary.md).

Colab T4, 27 October 2026, rewrite prompt `condense_v1`, conversations now in
[`testsets/optima_secure_followups_practice.json`](../testsets/optima_secure_followups_practice.json) (version
`861c4640a2`; 16 test follow-ups decide). Rule fixed beforehand: follow-ups significantly more often fully correct with
memory (McNemar exact, p < 0.05) and no topic switch made wrong.

| Test split | Without memory | With memory | Complete question |
|---|---|---|---|
| Follow-ups that decide (16) | 11 | 12: 2 fixed (D01.2, D11.2), 1 broken (D23.2), p = 1.0 | 15 |
| Topic switches (6) | 4 | 4, all left unchanged | 4 |

**Decision: do not adopt.** Two reasons, both fixed for the next run:

1. **The conversations were too easy.** 11 of the 16 follow-ups still named their subject ("Does it pay for gloves and
   food charges in a hospitalization claim?") and were answered correctly without memory. Only 5 could be fixed, and
   5 fixed with none broken gives p = 0.0625: the rule could not be passed even by perfect rewrites.
2. **The rewriter copied from earlier answers.** The prompt allowed adding an "amount" from the conversation, and the
   figure guard accepted figures from earlier answers. 4 of 18 rewrites took figures or sentences from an answer, one
   with a wrong figure (D02.2: "up to 60 days after discharge"; post-hospitalization is 180 days), one pasting most of
   the previous answer into the question (D23.2, the turn memory broke: the answer repeated a figure not in its
   context).

Also seen: the remaining misses were not rewrite failures alone. In D09.3 and D10.2 the rewrite was right and the
evidence was retrieved, but the answer said a fact was not shown (the test-set wording was answered correctly); in
D22.2 the rewrite dropped "premium", so the premium tables were not searched. A rewrite took a median of 48 s
(thinking on), one 6.4 minutes; a follow-up takes about twice as long as a single question. Single questions were
unchanged in the same session: all 113 answers came from the published run's cache.

### Changes after the first run

- Rewrite prompt `condense_v2`: replace "it", "that", "the first one" with the name it stands for, keep what is asked
  (premium, waiting period, documents), take figures only from the user's questions and nothing else from answers.
- Guards: a figure must appear in the user's questions (not in answers); a rewrite repeating 6 or more words in a row
  from an earlier answer, which the user never typed, is rejected (`memory.copy_words`). Replayed on the first run's
  rewrites, these reject exactly the four that copied from answers (D02.2, D12.2, D22.2, D23.2) and none of the others.
- Known cost of the stricter figure rule: a follow-up whose missing piece is a figure from an answer ("when he reaches
  that age") is searched as typed.

### Second run: not adopted

Evidence: [`results/memory/run2/`](../results/memory/run2/summary.md).

Colab T4, 27 October 2026, `condense_v2`, conversations now in
[`testsets/optima_secure_followups_practice2.json`](../testsets/optima_secure_followups_practice2.json) (version
`fbf48a727e`, written before the run). Single questions were unchanged in the same session (all 113 answers from the
cache, 97/113).

| Test split | Without memory | With memory | Complete question |
|---|---|---|---|
| Follow-ups that need the conversation (28) | 16 | 21: 6 fixed, 1 broken (Q13.2), p = 0.125 | 28 |
| Self-contained follow-ups (8) | 8 | 7: Q28.2 made wrong | 8 |
| Topic switches (6) | 6 | 6 | 6 |

**Decision: do not adopt.** Copying from answers was gone (no rewrite took a figure or sentence from an answer), but:

1. **Vague rewrites.** The model often replaced "it" with a general word instead of the name: "What does it not pay
   for?" became "What does the policy not pay for?" (the organ donor cover was meant), and "how long is it valid?"
   became "how long is the approval valid?" (cashless pre-authorization; the answer was then refused: the broken turn).
   5 of the 7 failures. The v2 prompt listed "the plan, cover, benefit" as kinds of names, and the model used those
   words literally.
2. **Complete questions were reworded.** 7 of the 14 self-contained follow-ups and topic switches were changed; in
   Q28.2 "my 3-year-old" became "the dependent child" and the answer flipped to a wrong "yes".

Also: the copied-words check rejected a long cover name (harmless here), and one rewrite took 14 minutes (thinking
and the answer ran to their limits). Median rewrite time 38 s.

### Changes after the second run

- **Complete questions are not rewritten** (`memory.skip_complete`): a question without a word that points back
  ("it", "that", "they", "the first one", or a start such as "And ..." / "What about ...") goes straight to search,
  with no model call. Tuned on the 113 single questions (none sent to the rewriter) and the two practice sets (46 of
  their 48 follow-ups that need the conversation recognised; none of their 12 topic switches sent). It does not
  recognise follow-ups that point back only through "the" ("the papers") or "the emergency-only one".
- **Rewrite prompt `condense_v3`:** name the exact cover, benefit, condition, plan or person, never "the policy" or
  "the plan"; keep every other word and detail of the question; two invented examples.
- **Checks:** a rewrite must keep the figures the user typed and must add something specific (one that only adds
  "the policy", "the plan", "the cover" is rejected); the copied-words limit is 8 words, so long cover names pass.
- **Rewrite answer capped at 64 tokens** (`memory.max_new_tokens`; the thinking budget and all other generation
  settings unchanged); a rewrite cut off at the cap is not used. Rewrites with the cap have their own cache keys.
- Replayed on the second run's rewrites, the new rules avoid both turns memory made wrong: Q28.2 is not sent to the
  rewriter, Q13.2's vague rewrite is rejected.

### Final run: not adopted

Evidence: [`results/memory/run3/`](../results/memory/run3/summary.md).

[`testsets/optima_secure_followups.json`](../testsets/optima_secure_followups.json) (version `a9a5aa60b1`), written
and frozen after the changes above and before any run with them: 44 conversations, 44 later turns, all on the test
split, each the complete form of a test question answered correctly in the published baseline.

| Kind | Turns | Notes |
|---|---|---|
| `follow_up`, `after_refusal` | 32 | written as people ask them; 28 contain a word that points back, 4 refer back only implicitly ("Is there a similar window if I am late renewing?") and are answered as typed |
| `self_contained` | 6 | 2 of them start with "So" or say "they", so they reach the rewriter |
| `topic_switch` | 6 | asked exactly as in the test set |

35 of the 44 complete questions were also used in a practice set, in different conversations and wording; the
practice sets were used to design the changes, never this set.

**Decision rule (the same, fixed before the run):** on the 32 follow-ups that need the conversation, memory must be
fully correct significantly more often than without it (McNemar exact, p < 0.05), and no `self_contained` or
`topic_switch` turn may become wrong; this was the last attempt.

Colab T4, 29 October 2026, `condense_v3`. Single questions were unchanged in the same session (all 113 answers from
the cache, 97/113).

| Test split | Without memory | With memory | Complete question |
|---|---|---|---|
| Follow-ups that need the conversation (32) | 19 | 21: 5 fixed (R08.2, R16.2, R26.2, R28.2, R31.2), 3 broken (R01.2, R17.2, R21.2), p = 0.73 | 32 |
| Self-contained follow-ups (6) | 5 | 5 | 6 |
| Topic switches (6) | 6 | 6, all unchanged | 6 |

**Decision: do not adopt; memory ships switched off** (`memory.enabled: false`). The fixes worked as intended:
complete questions were never reworded (14 turns skipped the rewriter; the one harm-check turn it saw changed only
"they" to "the insurer"), no rewrite copied
figures from an answer, and the 64-token cap held (median rewrite 44 s; the longest, 5.5 minutes, was spent
thinking). But the rewriter often chose the wrong
thing or garbled it: "After how many years can they no longer do that?" (the insurer's action on non-disclosure)
became "... no longer Incorporate an additional Waiting Period?", "Are there covers it doesn't apply to?" became
"Are there Individual Policies or Family Policies the Aggregate Deductible doesn't apply to?", "What is it,
exactly?" (Home Health Care) became "What is cashless approval for Home Health Care, exactly?". 8 of the 27 rewrites
pointed at the wrong thing or were garbled, which broke 3 answers and kept 5 others wrong.

### What the three runs show

| Run | Prompt | Follow-ups that decide | Right without memory | Right with memory | Fixed / broken | p | Harm-check turns made wrong |
|---|---|---|---|---|---|---|---|
| 1 | `condense_v1` | 16 | 11 | 12 | 2 / 1 | 1.0 | 0 of 6 |
| 2 | `condense_v2` | 28 | 16 | 21 | 6 / 1 | 0.125 | 1 of 14 |
| 3 | `condense_v3` | 32 | 19 | 21 | 5 / 3 | 0.73 | 0 of 12 |

- Memory helps some follow-ups, but never significantly: each fix to the rewriter removed one failure and exposed
  another (copying from answers, then vague names, then wrong referents). Pooled over the three runs (not a test
  fixed in advance) it fixed 13 follow-ups and broke 5, p = 0.096.
- Many follow-ups are answered correctly as typed (11 of 16, 16 of 28, 19 of 32): the search still finds the right
  passage from the words that remain, so there is less to gain than expected.
- The limit is the rewriter: Qwen3-8B (4-bit) with thinking resolves "it" and "that" unreliably on this document.
  Untried options: a stronger rewrite model, using the sections retrieved in the previous turn instead of a text
  rewrite, or asking the user which benefit they mean.

The code stays: `policy-rag chat`, `rag.ask(..., conversation=...)` and the dialogue evaluation work, and memory can
be switched on for experiments (`--set memory.enabled=true`, `rag.set_memory(True)`); the evaluation turns it on by
itself. With memory off, a conversation still keeps its document choice, and every question is answered as typed.

## Hard questions (not scored)

[`testsets/optima_secure_hard_questions.json`](../testsets/optima_secure_hard_questions.json) has 15 questions
beyond the test set, such as multi-claim calculations, deductibles across a family floater, deadlines with dates
and interacting waiting periods. Each comes with an expected answer for a person to compare. They were only tried in
an early development run under the old single token limit, judged by reading (not by the checks); its outputs were
not kept, so the figures noted then (about 5 of 15 right without thinking and 6 with thinking, 6 answers lost to
cut-off thinking, which led to the thinking budget) **cannot be verified**. They have not been rerun since.

## Verification

The published answers were generated by the v3.1 notebook, so the package was checked against it. All checks but the
last ran without the real models (none can be downloaded in the environment where the package was built):

| Check | Result |
|---|---|
| Rebuild the 113 saved v3.1 contexts from their chunk ids (real chunk file) | 113/113 identical |
| Answer-cache keys of the 108 generated v3.1 answers, from their prompts | 108/108 identical, so the v3.1 cache is reused |
| Retrieval, reranking, context and gate on all 113 questions, notebook v3.2 vs package, same small stand-in models | 113/113 identical (candidates, scores, contexts, gate, retrieval metrics) |
| Full run on dev with a small stand-in answer model: two-phase thinking, seeded sampling, cache, checks, evidence-removed test, repeat check, groundedness | 25/25 identical answers, keys and verdicts |
| Re-scoring the saved v3.1 answers, notebook v3.2 vs package | 113/113 identical verdicts: 97/113 fully correct |
| After adding conversation memory (v3.3): the five checks above repeated (single questions, no conversation) | all identical; prompts and cache keys unchanged |
| Rebuilding the published baseline files from the run files with [`scripts/export_results.py`](../scripts/export_results.py) (29 Oct 2026) | identical in every column and row except groundedness (now the package's measurement; see [results/baseline](../results/baseline/README.md)) |
| **Real models on a T4** (package v3.2, thinking-off experiment, 25 Oct 2026): retrieval vs the published run | 113/113 identical candidates, reranker scores, contexts and gate decisions |
| **Real models on a T4** (package v3.3, three sessions on 27 Oct (two) and 29 Oct 2026): all 113 questions with the default configuration | in every session: 113/113 identical candidates, reranker scores, contexts and gate decisions; all 108 answers found in the v3.1 cache under the same keys (none generated); 97/113 fully correct; groundedness 0.886 ([results/reproduction](../results/reproduction/)) |

Still open:

- regenerating the answers (not from the cache) with the package on a GPU;
- a check that the published chunk file is reproduced byte for byte from the PDF. The chunker's text, sections and
  table chunks match, and the remaining check needs the `tiktoken` vocabulary download, which was blocked in the
  build environment.

## Reproducing the scores

- **Re-score saved answers** (minutes, CPU): `policy-rag evaluate --name rescored --mode rescore --rescore-from
  <run.json or results zip>`. This needs the run's saved answers, which contain policy text and are not published.
- **Regenerate** (about 2 hours on a T4): `policy-rag evaluate --name baseline`. With the same software and GPU type,
  seeded sampling should give the same answers. Other versions may differ slightly; use `compare` to see whether the
  difference matters.
