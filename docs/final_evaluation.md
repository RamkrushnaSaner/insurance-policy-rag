# Final evaluation of v3.4

An evaluation of the adopted system, v3.4, on every question set it answered: 113 development questions and 329 new
questions in four sets written after it was built. It was done on 15 November 2026, after the project's last decision
(experiment 11 in [experiments.md](experiments.md)). Nothing was changed or regenerated. Every answer, ranking,
context and timing comes from the saved runs, re-scored with this repository's checks.

Each finding is marked as one of three kinds. **Measured** findings come from the run files; most are recomputed from
[`results/final_evaluation/`](../results/final_evaluation/) by
[`tests/test_published_results.py`](../tests/test_published_results.py). **Measured privately** findings come from the
policy PDF or chunk file, which are not published. **Judgement** marks a reading or an opinion, not a measurement.

## Summary

- **Accuracy (measured).** 255 of the 329 new questions are fully correct: **77.5%** (95% CI 73–82%). On the 113
  development questions, which every earlier change was tuned or decided on, the same system scores 89.4%. About 12
  points of that figure do not carry over to new questions.
- **What works (measured).**
  - **Finding text:** the first search stage finds 99% of the passages that hold an answer printed in the text, and
    the reranker puts 84% of them first.
  - **Knowing its limits:** it declined all 29 questions the policy does not answer.
  - **Sources:** 93% of answers cite a section, and none cites a section it was not given.
  - **Numbers:** one answer in 311 has a figure that is not in its context (a correct subtraction).
  - **Stability:** no errors in 442 questions. Search, the table lookup and regenerated answers reproduce exactly.
- **What fails (measured).**
  - **Table questions:** only 58% are fully correct (49 of 84). All 26 failures lost in search are facts printed only
    in a table that the hand-written table lookup did not return.
  - **Declined answers:** 50 of the 300 answerable questions (17%) get an answer that opens by saying the policy does
    not give the answer.
  - **Where the 76 failures counted by the checks started:** 34 in the answer itself, 26 in search, 15 in choosing
    what the model reads, and 1 at the relevance gate.
  - **Speed:** a median of 43 s per answer on a T4 GPU (95th percentile 93 s).
- **The answer-stage failures (judgement).** Read one by one, 17 of the 34 are clear model errors. In 9, the answer
  applied the right rule but opened with the refusal sentence. In 8, it missed or misread a qualifier in front of it.
  Of the other 17, 8 are context failures that the checks misfiled, and 9 are acceptable answers.
- **Readiness (judgement).** Usable as an assistant whose answers a person checks against the cited section. Not
  ready to answer policyholders unsupervised: more than 1 answer in 5 (22%) is incomplete or wrong, and table
  questions are the weakest.

## 1. What was evaluated and how

**The system.** [`configs/experiments/qwen3_text_tables.yaml`](../configs/experiments/qwen3_text_tables.yaml), adopted on
3 November 2026 ([`configs/default.yaml`](../configs/default.yaml) remains the earlier baseline, which needs no
database). Search runs over the 109 text chunks: Qwen3-Embedding-0.6B vectors in Postgres
(pgvector, exact search) and BM25, merged by reciprocal-rank fusion. bge-reranker-v2-m3 reorders the candidates. A
relevance gate (best score at least −5) can refuse before the model is called. Hand-written patterns look up the
printed tables in typed Postgres tables with fixed SQL. Qwen3-8B (4-bit, thinking up to 1,024 tokens) answers from at
most 1,500 tokens of context; the table passage comes on top of that budget. The model revisions are pinned and
were the same in every run.

**The question sets.** These are described in [`results/final_evaluation/`](../results/final_evaluation/README.md).
The four fresh sets were written by the AI assistant that built the system, from the policy, after v3.4 existed. Fresh v3
and v4 were not seen by the system before their runs. Each answer key was checked mechanically: its facts are in the
policy, the reference answer passes the checks, and hand-written right and wrong answers score as intended.

**Scoring.** All answers were re-scored with the same checks: the repaired checks of 12 November, commit `93b5f9d`.
The re-scored verdicts match the stored ones for all 195 answers of fresh v3 and v4, which were graded with those
checks when they ran. In fresh v4, 33 answers were labelled blind by the owner, and those labels replace the checks'
verdict.

**Also run for this evaluation:**
- the test suite (all pass, including the tests against Postgres);
- the ingestion coverage against the PDF;
- BM25 search on its own;
- the table lookup, rerun against a copy of the database on all 442 questions: same status, intent and facts for
  442 of 442.

**Not measured:** load and concurrency, other policies, real users' questions, conversations, attacks such as prompt
injection, and cost.

## 2. Answer accuracy

| Set | Fully correct | 95% CI | Table | Text | Declined, not answerable |
|---|---|---|---|---|---|
| Development (tuned on) | 101/113 = 89.4% | 82–94% | 6/8 | 76/86 | 19/19 |
| Fresh v1 | 55/72 = 76.4% | 65–85% | 6/11 | 41/53 | 8/8 |
| Fresh v2 | 44/62 = 71.0% | 59–81% | 5/14 | 33/42 | 6/6 |
| Fresh v3 | 88/107 = 82.2% | 74–88% | 18/29 | 63/71 | 7/7 |
| Fresh v4 | 68/88 = 77.3% | 67–85% | 20/30 | 40/50 | 8/8 |
| **All fresh** | **255/329 = 77.5%** | **73–82%** | **49/84 = 58.3%** | **177/216 = 81.9%** | **29/29** |

(Measured.) More findings on the 329 fresh questions:

- **Checks only.** Without the person's labels, the checks give 253/329 = 76.9%.
- **By question type** (answerable questions). False premises are weakest: 9 of 17 correct. Next come questions
  spanning several sections (5/8), comparisons and lists (7/11 each) and plain facts (109/147). Conditional questions
  (71/85) and procedures (11/13) do best. Most groups are small.
- **Wording.** Everyday wording (117/157 = 74.5%) scores about the same as policy wording (109/143 = 76.2%). The
  development set shows no gap either (89% against 86%).
- **Declined answers.** 50 of the 300 answers to answerable questions open by saying the policy text does not give
  the answer. All 50 fail, and declining is the most common way an answer fails the user. By where the failure
  started:
  - 23 in search;
  - 9 in context selection;
  - 1 at the gate;
  - 17 at the answer stage (every key fact in the context, by the checks).
- **Cut-off answers.** None was cut off at the length limit.

## 3. Retrieval and ranking

(Measured.) Search is strong on text and weak on tables.

| Stage | Measure | Fresh | Development |
|---|---|---|---|
| Search, all evidence | questions with every evidence passage among the candidates (text or table lookup) | 248/300 = 82.7% | 88/94 = 93.6% |
| First stage (vectors + BM25), text evidence | recall @5 / @10 / @20 / @64 | 0.89 / 0.94 / 0.97 / 0.99 | 0.88 / 0.92 / 0.98 / 1.00 |
| BM25 alone (run here), text evidence | recall @1 / @5 / @10 / @32; MRR | 0.56 / 0.79 / 0.84 / 0.92; 0.65 | 0.53 / 0.73 / 0.83 / 0.94; 0.62 |
| Reranker, text evidence | recall @1 / @3 / @5 / @10 | 0.84 / 0.92 / 0.94 / 0.96 | 0.79 / 0.89 / 0.94 / 0.95 |
| Reranker, text evidence | MRR@10 | 0.89 (211 questions) | 0.87 (87) |
| Table lookup | table questions it answered completely | 30/84 = 35.7% | 6/8 |
| Relevance gate | separating answerable from unanswerable questions (AUROC, per set) | 0.84–0.95 | 0.92 |
| Relevance gate | answerable questions it refused | 2/300 | 0/94 |

- **Every search miss was a fact printed only in a table.** The text evidence was almost always found.
- **Vectors add recall over BM25.** BM25 alone finds 92% of text evidence in its top 32; the hybrid first stage
  finds 97% in its top 20. Vectors alone were not measured, because the embedding model cannot be downloaded here.
- **The reranker works.** It puts 84% of text evidence first and raises the share in the top 10 from 94% to 96%.
- **The table lookup is the weak link.** It answered completely for 36% of table questions. On the 26 search
  failures it:
  - found no matching rule (15);
  - answered only part (5);
  - asked for details the question did not give (3);
  - was not called (3).

  It also answered 17 of the 216 text questions completely from the tables.
- **The gate is safe but not sharp.** It refused 2 answerable questions. It caught 10 of the 29 unanswerable ones
  itself, and the model declined the other 19.

## 4. Context relevance and source attribution

(Measured.)

| Measure (fresh questions) | Result |
|---|---|
| Every key fact in the context | 256/300 = 85.3% |
| Every evidence passage in the context | 237/300 = 79.0% |
| Context tokens from passages that hold evidence (context precision) | 0.21 (0.30 on questions with text evidence) |
| Passages read per question (median) | 7 |
| Context length (median) | 1,444 tokens (budget 1,500) |
| Answers that cite at least one section (answerable, not refused) | 277/297 = 93.3% |
| Citations of a section that was not in the context | 0/277 |
| Answers citing a section that holds the answer | 254/277 = 91.7% |

- **The budget is nearly always full.** About 4 tokens in 5 come from passages that do not hold evidence.
- **The table passage sits outside the budget.** 67 contexts exceeded 1,500 tokens (up to 2,372), all of them when
  the table lookup added its passage.
- **What attribution means here.** A cited section counts when the answer key names it, or names its parent or
  sub-section.

## 5. Failure analysis

The checks count 76 failed answers on the 329 fresh questions. The person's labels change four of them, which gives
74. Each failure is placed at the first stage that no longer had what the answer needed (measured).

| Stage | v1 | v2 | v3 | v4 | All fresh |
|---|---|---|---|---|---|
| Search: the fact is printed only in a table and was not found | 4 | 9 | 6 | 7 | 26 (34%) |
| Context selection: found, but not read | 4 | 2 | 4 | 5 | 15 (20%) |
| Relevance gate | 0 | 1 | 0 | 0 | 1 |
| The answer: every key fact was in the context | 9 | 6 | 9 | 10 | 34 (45%) |

### The answer stage, read one by one (judgement)

The 34 answer-stage failures were read next to their contexts, together with Y077 (passed by the checks, failed by
the person). The reading is in
[`generation_failure_review.csv`](../results/final_evaluation/generation_failure_review.csv). The reviewer was the AI
assistant, except for the four answers the person had labelled.

| What the reading found | Answers | Model error? |
|---|---|---|
| **Disowned a correct answer.** It applies the right rule in the body, but opens with "Not specified in the provided policy text." or says the case cannot be confirmed. The policy states a general rule, and the question names a specific case or makes a false premise. | 9 | yes |
| **Missed or misread a fact in front of it**: a qualifier at the end of a sentence, an exception inside an exclusion, "respectively", a condition read backwards. Twice the thinking stopped at its 1,024-token budget mid-deliberation. | 8 | yes |
| **The context did not hold the answer after all.** The key fact matched a heading or an unrelated word, or the list item the question names was not among the passages read. The model was right not to answer confidently. Y077 is one of these. | 8 + Y077 | no: a context failure the checks misfiled |
| **Acceptable answer.** The key was too strict, a correct subtraction was flagged as a figure not in the context, or the answer followed Annexure B's title, which contradicts Section 4.3. The person confirmed 3 of these. | 9 | no |

The 9 acceptable answers are the reader's judgement, except the 3 the person confirmed. Only those 3 count in the
headline. If the other 6 are also accepted, fresh accuracy would be 261/329 (79.3%). That figure is not used here.

**What goes with the 17 model errors (measured, on the 247 answers whose context did hold the answer):**
- false-premise questions: 5 of 14 wrong, against 12 of 233 (Fisher p = 0.001);
- thinking stopped at its budget: 3 of 8 wrong, against 14 of 239 (p = 0.012).

The number of passages read also seemed to matter, but how strongly depends on how passages are counted (p 0.02 to
0.13), so it is not claimed. These are associations, not causes, and harder questions also make for longer thinking.

**Why the model disowns correct answers (judgement, not tested).** The prompt, [`prompts.py`](../src/policy_rag/prompts.py),
says:
- "Do NOT add, assume, or infer details beyond the given text";
- "If the CONTEXT covers only part of the question, answer that part and say what is not shown";
- "If the answer cannot be found, reply: 'Not specified in the provided policy text.'"

In 10 of the 11 answers that open with the refusal sentence, an explanation follows: the sentence is used as a preface,
not as a refusal. The one saved thinking trace among these answers (V036) reaches the right exclusion several times,
then reverts because the policy does not use the question's exact word. A run with these instructions changed would
be needed to test this.

### Root causes, largest first

1. **The answer step does not use all it is given.** This covers 17 clear model errors among the 34 answer-stage
   failures: the disowned answers and the misread qualifiers above.
2. **Table facts can only be reached through the hand-written lookup** (26 failures, measured). None of the facts
   these questions need appears in the searchable text chunks, so when the lookup cannot answer, nothing else finds
   them. This is why table questions score 58%, against 82% for text questions. R1's search cards removed all 13 such
   failures in fresh v3 and v4 but were not adopted (experiments 10 and 11).
3. **Context assembly drops evidence that search found** (15 failures, measured). In 9, the reranker scored the
   evidence passage below the −5 floor. That floor was set in v2 for a different reranker and was never
   recalibrated. In 6, the passage scored above the floor but did not fit in the budget, while most of the budget went
   to passages without evidence (context precision 0.21).
4. **The relevance gate** (1 failure, measured). It refused W05c, whose context held the facts.
5. **Tuning on the development set overstated accuracy** (measured). The system scores 89.4% on that set and 77.5% on
   new questions, so figures measured on it, including the earlier "test split", overstate accuracy by about 12
   points.
6. **The source document contradicts itself.** Annexure B's title says non-medical items are "not available", while
   Section 4.3 (Protect Benefit) pays them on an admissible claim. Four answers that follow the title (V088, V089, Y075,
   Y076) fail the checks, and the person judged two of them correct.
7. **The checks match words, not claims.** Loose matches made 8 context failures look like answer failures and let
   one (Y077) pass. Strict keys failed acceptable answers.

## 6. Chunking, ingestion, embeddings and indexing

(Measured privately, on the PDF and the chunk file, except where marked.) None of the failures measured here starts
in ingestion or chunking.

- **Text extraction.** 362 of the 373 sentences (10 words or more) on the policy's running-text pages reach the
  chunks. The other 11 are tables that the PDF text extraction flattens; their cells are in the database (32 table
  parts, 27 tables, 4,408 cells). The page header and footer repeated on all 26 pages are left out on purpose.
- **Key facts.** Every key fact of the 329 fresh questions is in the text chunks or the tables. Three
  development-set keys are written in a form the policy never prints ("30 lakh" for "30,00,000"). That is a wording
  issue in the keys, not a gap in the document.
- **Chunks.** 109 chunks across 70 sections, median 90 words (smallest 12, 90th percentile 230, largest 336).
  Thirteen sections span several chunks, yet none of the 219 text evidence passages of the fresh questions is split
  across two chunks.
- **Embeddings** (measured). The hybrid first stage finds 97% of text evidence in its top 20, against 92% in BM25's
  top 32 on its own.
- **Indexing** (measured). pgvector does exact search. In every run, the search was repeated for every question with
  identical results (442 of 442).

## 7. Faithfulness and hallucination

(Measured, and judgement where marked.) Invented or false content is rare by every available measure, but each
measure is a lower bound.

- **Figures.** 1 of 311 answered questions has a figure that is not in its context (H26). Read, it is a correct
  subtraction of two figures that are in the context (judgement).
- **Critical errors flagged by the checks.** 12 of 329 fresh answers. In 8 of them (fresh v1), the "error" is the
  refusal sentence, which v1's keys list as a forbidden phrase: these are refusals, not invented content. Of the
  other 4, two are wrong yes/no conclusions: H51, whose evidence never reached the model, and Y038. The other two are
  false alarms by the reading (judgement): the correct subtraction above, and H07, which names another section only
  to set it aside.
- **Why a lower bound.** The checks look for key words, not for what the answer claims. They passed Y077, a "not
  specified" answer that happened to contain the key words. In another system's run they passed an answer saying a
  surgery was covered at once when it is not.
- **The person's labels.** 33 of this system's answers: 21 correct, 12 not. They were given as correct or not
  correct, so they cannot separate wrong answers from incomplete ones.
- **HHEM is not usable as a hallucination rate here.** Groundedness is 0.758, and 50% of answers have a sentence it
  flags. But it also flags 48% of the answers the checks judged correct and 29% of those the person judged correct.
  Its tokenizer reports a 512-token limit against contexts of about 1,450 tokens; whether the input is truncated was
  not checked.

## 8. Latency, resources and stability

(Measured.) The system is slow but stable.

| Measure | Result |
|---|---|
| Answer generation per question (317 fresh answers, T4) | median 43.3 s · 90th percentile 75.9 s · 95th 92.6 s · max 129.4 s |
| Thinking before answering | median 336 tokens; stopped at the 1,024-token budget in 11 of 317 (3.5%) |
| Answers cut off at the length limit | 0 |
| Generation speed | 7.0–8.5 tokens/s |
| Whole run, everything included | 95 min for 107 questions; 94 min for 88 (53–64 s per question, with model download, loading and grading) |
| Peak GPU memory | 9.2–9.3 GB of the T4's 15 GB |
| Pipeline errors, table-lookup errors | 0, 0 (442 questions) |
| Search repeated within each run | identical for 442 of 442 |
| One answer regenerated without the cache, per run | identical in 5 of 5 runs |

- **Thinking is the cost.** Almost all of the time goes to Qwen3-8B thinking at 7–8.5 tokens per second. Answer
  generation took 74–83% of each fresh run's wall time. The rest is model download and loading, search and grading,
  which were not timed separately.
- **Determinism is by design.** Answers are sampled with a seed taken from the prompt, so the same prompt gives the
  same answer. A different context gives a different sample, which is one source of verdict changes between system
  versions.
- **Not measured.** Concurrent users, throughput, and Neon's cold start after it suspends an idle database.

## 9. How reliable these measurements are

- **Sample size.** The 95% intervals are 73–82% for all 329 fresh questions, 48–68% for the 84 table questions and
  76–87% for the 216 text questions.
- **The checks against the person.** On this system's 33 labelled answers: the same verdict on 29 (kappa 0.75). On
  all 66 answers labelled in fresh v4 (both systems): kappa 0.63. On 30 answers of another system in fresh v3: 0.52.
- **The person's labels are not perfect either.** In Y075 and Y076 they follow Annexure B's title, against Section
  4.3.
- **The questions.** They were written by the AI assistant that built the system, and checked mechanically. A key
  error was found once before (fresh v2, Protect Benefit). Fresh v1's keys list the refusal sentence as a forbidden
  phrase, so such answers count as refusals here, not as invented content.
- **When the checks changed.** The checks were repaired on 12 November, after studying failures in fresh v1 and v2.
  The system itself was not changed.
- **Not independent.** The same assistant built, tested and evaluated the system. A person checked 33 answers.

## 10. Readiness and recommendations

**Readiness (judgement).**
- **Ready for:** a demonstration, or an internal assistant whose user checks each answer against the section it
  cites (in 92% of answers with a citation, a cited section holds the answer).
- **Not ready to answer policyholders or support claim decisions unsupervised:**
  - 22% of fresh answers are not fully correct;
  - 17% of answerable questions get "not specified";
  - table questions are right only 58% of the time;
  - a median answer takes 43 seconds.
- **Unknown (not measured):** behaviour under load, on other policies and on real users' questions.

**Recommendations, in order of measured impact (judgement).** Judge each change once, on new questions, under rules
fixed before the run, as in the last two decisions.

1. **Table questions** (26 failures; 58% correct). Replace the hand-written patterns with a lookup whose fields are
   checked against the database, or adopt search cards for every table row. R1's cards removed all 13 table search
   failures in its two rounds, but its answer gain (+8 by the checks, +4 by a person) did not meet the rules.
2. **The answer instruction** (the 9 disowned answers).
   Tell the model to apply a general rule that covers the case, never to open with the refusal sentence and then
   answer, and to correct a false premise outright.
3. **Context selection** (15 failures). Recalibrate the reranker floor of −5 (9 cases). Spend some of the GPU headroom
   (9.3 of 15 GB used) on a larger reading budget, count the table passage within it, and measure what that costs in
   time.
4. **Answer completeness** (8 misread facts; false-premise questions 9 of 17). Ask the model to address every
   condition the question raises, and test a larger thinking budget, since the cap went with errors.
5. **The measurement.** Make the checks test what an answer claims, not which words it contains. Stop reporting HHEM
   as a hallucination rate. Label a larger, balanced sample by a person. Settle the Annexure B / Section 4.3 conflict
   before writing keys about non-medical items.
6. **Latency.** The time goes to thinking. Switching thinking off cut accuracy (79 → 72 of 88 in experiment 5), so
   test a smaller thinking budget instead.
