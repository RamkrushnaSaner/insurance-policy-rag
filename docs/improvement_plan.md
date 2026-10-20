# Improvement plan: baseline, findings and roadmap

This document records what was measured before changing the pipeline, what the measurements show,
and the order in which improvements should be made. It goes with notebook v2.6.

## 1. Where things stand

| Item | Status |
|---|---|
| Code on GitHub | v2.2 |
| Latest code | v2.6 = the v2.3 answer pipeline + a deterministic evaluation workflow + a BM25 switch + three switchable fixes. RAGAS and Langfuse (v2.4/v2.5) are set aside until this workflow is reliable |
| Real-model results | The original notebook's run (11 questions, with contexts) and the recorded v2.2 run (8 answers, with sources and reranker scores) |
| Not yet measured | Everything v2.3 changed (title pinning, neighbour expansion, relevance floor, scope rules) has never been run with the real models |

So the current system has **no measured baseline yet**. Section 10.5 of the notebook produces one.

## 2. What was measured without the models

These numbers do not depend on the embedder, reranker or LLM. They come from the policy text, the
parser, the chunker, BM25 and title matching, which run identically everywhere.

### 2.1 Recorded runs, scored against the new key facts (questions G01–G09)

| | Original notebook | v2.2 |
|---|---|---|
| Key facts in the context | 0.76 | not recorded |
| Key facts in the answer, 8 shared questions | 0.84 | 0.78 |
| Figures in the answers missing from their context | 2 of 23 | 0 of 16 (context approximated) |

The gap between the original and v2.2 comes from two questions, the Plus Benefit and the 24-month
list, which are the two that v2.3 targeted. In 8 of the original's 9 answers, the share of key facts
in the answer **equals** the share in its context, and no answer had fewer. **In the recorded runs,
the bottleneck was what reached the model, not how the model used it.** (Only nine questions, so
this is an indication, not a conclusion.)

*Correction (v2.6):* the original run stores its context chunks numbered from 1; the v2.5 analysis
read them as numbered from 0. Recomputed, the mean is unchanged (0.76) but the per-question values
differ; the statements above use the corrected values.

### 2.1a Generation errors are rare, and they appear where the context is thin

Both invented figures in the original run are the same made-up clause in the standard-exclusions
answers ("hospitalized for the same disease within 90 days prior to the inception of the policy"),
which is not in the policy. That question's context held 17% of the key facts: the model filled the
gap with generic insurance wording. The recorded v2.2 answers contain no figure missing from their
sections' text, and none of them is cut off (the longest is 382 words, well under the 1,000-token
limit; cut-offs only affected the original notebook). v2.3's prompt rule "answer the part the
context covers and say what is not shown" targets exactly this, and v2.6 measures it
(`unsupported_figures`, `unsupported_citations`).

### 2.2 First-stage retrieval barely filters

The document has 108 chunks. Each question sends the union of the dense top 32 and the BM25 top 32
to the reranker: **30% to 59% of the whole document**. The BM25 leg alone finds the chunk holding a
key fact for 96% of facts in heading-worded questions and 88% in everyday-worded ones. So **the
cross-encoder and the context builder decide what the model sees**. A better embedder is unlikely
to matter much; a better reranker could.

### 2.3 The context budget is not the main constraint

The fewest chunks that hold every key fact of a question need a median of 226 tokens and at most
1,384, so every question's key facts fit in the 1,500-token budget. Only "give the complete list"
questions need more: Section 10.1 with its sub-sections is 2,199 tokens. **Raising the budget helps
complete lists, not answers in general**, and on a T4 it needs the 4-bit model.

### 2.4 Title pinning: precise on headings, absent on everyday wording, and with a bug

- It pins a gold section for 34 of the 61 answerable questions worded with heading words, and for
  **0 of the 24** worded in everyday language. For everyday questions, quality depends entirely on
  dense search and the reranker. The `paraphrase` breakdown in Section 10.5 is the best indicator of quality "for any
  question".
- **Bug (v2.3):** in 11 of the 94 questions its pinned groups bring in unrelated sections. A question that
  contains "claims" ("If I made claims last year…") pins all of Section 11: 8 chunks are forced into
  the context before any reranked chunk. The two Global Health Covers differ only in words inside
  parentheses, which were ignored, so a question about one pinned the other.
- **Fix (`title_pinning_rules = "v2.5"`, off by default):** on the 94-question test set, questions
  with unrelated pinned sections go from 11 to 6 (9,015 to 1,531 pinned tokens), 39 instead of 38
  questions get their gold section, and none loses it. Only 6 questions' contexts can change.

### 2.5 BM25 matches exact words only

"notified" does not match "Notification", and "delivery" does not match "deliveries". **Fix
(`bm25_stemming = True`, off by default):** fact recall of the BM25 leg goes from 0.937 to 0.961 in
the top 32 and from 0.851 to 0.897 in the top 10, with no question worse. Because the candidate pool
is large (2.2), the effect on the final context may be small; the experiment run measures it.

### 2.5a Whether BM25 earns its place is not yet known

BM25 is one of the two candidate sources. v2.6 adds `use_bm25` (default `True`, identical candidates
to v2.3 on all 94 questions) and three experiments: `dense_only`, `dense_only_k64` (same pool size as
now) and `bm25_only`. **Decision rule:** remove BM25 only if `dense_only_k64` is not worse than the
baseline on key facts in the context and on routing, with no regression on questions that cite codes
or section numbers ("Excl06", "Section 10.2").

### 2.6 The old gold set mis-scores correct answers

3 of its 9 answerable questions expect keywords that are not in their gold sections ("incurred and
paid overseas" belongs to 4.10.1, not 4.11; "2 years" and "irrespective of a claim" belong to the
introduction, not 4.4; Excl03 is in 10.1.3, which was not listed). The new test set fixes these. The
old table is kept for continuity.

### 2.7 Notes numbered as separate sections are unreachable

The Cumulative Bonus notes are numbered 3.9 and 3.10 with sentence fragments as titles ("In case
where the Policy", "If the Sum Insured has been reduced…"). Title pinning finds Section 3.8 for a
question about the Cumulative Bonus, but neither pinning nor neighbour expansion brings 3.9 or 3.10
with it, because they are not sub-sections of 3.8. Two test questions (K11, K12) were added to
measure this. **Fix (`continuation_sections = True`, off by default):** a section whose title starts
like a clause ("In case", "If", …) and that is not a sub-section of the previous one is treated as
its continuation; it joins that section's pinned group and expansion, and its passage label names
the section it continues. Only 3.9 and 3.10 qualify in this policy; with the fix, pinning reaches the
gold section for both questions.

### 2.8 Failures were invisible to evaluation

A node that raised an error returned empty results, so the question looked like an ordinary refusal;
an answer generated from the shortened out-of-memory retry context looked like a normal answer.
v2.6 records both in the pipeline state (`error`, `oom_retry`), and the evaluation reports them.

## 3. The evaluation workflow (notebook Sections 10.3–10.5)

- **Test set:** 94 questions across 10 types: 85 answerable (24 of them in everyday wording) and 9
  not answerable (7 out of scope, 2 false premises), plus one false-premise question whose answer
  corrects the premise. Each has gold sections, key facts with alternative phrasings, and a
  reference answer. Every fact was checked against the policy text; `check_testset()` re-checks it in
  the notebook.
- **Two modes:** retrieval mode (no LLM, about 1 s per question) measures what reaches the model;
  full mode measures the answers.
- **Metrics (no API keys):** retrieval — key facts in the context (main metric), gold sections in
  the context and among candidates, MRR of the first gold chunk; routing — gate decisions; answer
  correctness — key facts in the answer (main metric), false refusals, correct refusals;
  faithfulness — figures and section numbers in the answer that the context does not contain,
  grounding; reliability — pipeline errors, out-of-memory retries; cost — context size.
- **Comparisons:** paired by question, with a 95% bootstrap interval; "better" or "worse" only when
  the interval excludes zero; per-question regressions listed; every run saved with its full
  configuration and test-set version, and reloadable in a later session (`export_runs()`,
  `upload_runs()`, `load_run_file()`).

## 4. Recommendations, in order

Each step is accepted only if it beats the baseline with no clear regression on the same 94
questions.

1. **Establish the baseline (about 20 min + 45 min).** Run the notebook: Section 10.5 runs the 13
   retrieval experiments. Then set `EVAL_PLAN["full"] = ["baseline"]` for the answer baseline. *Why
   first:* v2.3 has never been measured, and every later decision depends on these numbers.
2. **Keep or drop each v2.3 change.** The ablations (`v22_like`, `no_title_pinning`,
   `no_neighbour_expansion`, `no_seed_floor`) say whether each one helps with the real reranker. Drop
   any that does not. *Why:* they were designed on recorded scores from 8 questions.
3. **Promote the fixes that win** (`pin_rules_v25`, `bm25_stemming`, `continuations`, `all_fixes`).
   *Why:* they target measured defects, and only the targeted questions change.
4. **Decide on BM25** with the rule in 2.5a.
5. **Calibrate the gate** with `routing_sweep()` on the 9 not-answerable questions. *Why:* the −5.0
   threshold was tuned on one out-of-scope question; wrong refusals and confident answers to
   out-of-scope questions are the most damaging errors for an insurance assistant.
6. **Try a stronger reranker** (`reranker_v2_m3`, in a session with the 4-bit model), then re-tune
   the gate for it. *Why:* 2.2 shows the reranker decides what the model sees.
7. **Complete lists:** run `budget_2600` in retrieval mode to measure the gain. If it is worth it,
   switch to the 4-bit model with the larger budget and confirm in full mode that 4-bit does not lower
   answer quality.
8. **Only then, the generator.** If full mode shows answer facts well below context facts, the model
   is losing facts it was given: compare prompts on identical contexts (Section 10.2), then consider a
   stronger instruction model.
9. **Then judge scores and tracing.** Once this workflow is stable, RAGAS (judge-based faithfulness
   and correctness) and Langfuse (per-question traces) can be added back from v2.4/v2.5.

**Not recommended now:** re-chunking (every question's facts already fit in a few chunks, and the
chunk-boundary problems were fixed in v2.1), changing the embedder first (2.2), or tuning the fusion
weights (fusion only orders candidates; all of them are reranked).

## 5. Path to agentic behaviour

The LangGraph graph already has explicit state and a routing point (the gate), so
agentic steps can be added as nodes behind `CONFIG` switches and measured with the same runner. Each
step below starts only when the baseline shows the failure it targets.

| Step | Targets | How it is measured | Add when |
|---|---|---|---|
| 1. Query routing (rules first) | Out-of-scope questions, complete lists, plan comparisons | `refused`, `refusal_ok`, list and comparison types | The gate misroutes, or list/comparison types lag |
| 2. Retry with a rewritten query when the gate refuses or evidence is weak | Everyday wording | Paraphrase `ctx_fact_recall`, `false_refusal`; out-of-scope questions must still be refused | Paraphrase questions lag heading-worded ones |
| 3. Tools for structured content: `get_section("4.4")`, `list_exclusions()`, `plan_table(year)` | Questions citing a code ("Excl06"), complete lists, plan tables | List and comparison types; context size | Lists remain incomplete after step 6 of Section 4 |
| 4. Answer check: every sentence must cite a passage, unsupported sentences are removed | Faithfulness | `faithful`, `unsupported_figures`, `unsupported_citations`, `grounding` | Full mode shows unsupported claims |
| 5. Planning across several sections | Multi-section questions | `multi_section` type | Steps 1–4 are stable |

**Readiness criteria before step 1:** a full-mode baseline exists; the gate is calibrated; the v2.3
changes are either confirmed or removed; results are reported per question type.

**Trade-off to plan for:** Mistral-7B on a T4 takes about 30 seconds per answer and is not built for
tool calling. Agent loops multiply LLM calls, so a planner may need a faster or tool-capable model,
while the final answer can stay with the local model.

## 6. Limitations

- Fact matching is string-based: it can miss a correct answer worded differently ("two years" for
  "24 months") and cannot tell whether a fact is used correctly. Reading the listed regressions covers
  part of this; judge-based scoring can be added later.
- The test set was written from the policy, so even the everyday-worded questions lean on policy
  vocabulary. Grow it from real questions users ask.
- With 85 answerable questions, differences under about 0.03 are within noise.
- One policy document: results may not carry over to other policies.
- The offline checks of the notebook use stand-in models: they verify the mechanics, not quality.
