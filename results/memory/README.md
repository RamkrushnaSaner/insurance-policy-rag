# Experiment: conversation memory (three runs)

**Question.** Can follow-up questions ("Does it apply after an accident?") be answered correctly by rewriting them
into complete questions from the conversation, before the usual search and answer?

**Method.** The `condense` step ([`memory.py`](../../src/policy_rag/memory.py)) asks Qwen3-8B, with thinking on, to
rewrite a follow-up from the last 3 turns; safety checks reject bad rewrites. Each run used conversations whose
follow-ups have a test-set question as their complete form, so the existing answer keys score them. Every follow-up
was answered three ways: as typed without memory, rewritten by memory, and as the complete question ("gold"). The
decision rule was fixed before each run: on the test split, the follow-ups that need the conversation must be fully
correct significantly more often with memory than without (McNemar exact test, p < 0.05), and no harm-check turn (a
follow-up that names its subject, or a topic switch) may become wrong. After each failed run the rewriter was changed
and **new** conversations were written before the next run; the third run was declared the last.

| Run | Date | Rewrite prompt | Conversations (version) | Deciding follow-ups | Right without memory | Right with memory | Fixed / broken | McNemar p | Harm-check turns made wrong |
|---|---|---|---|---|---|---|---|---|---|
| [1](run1/summary.md) | 27 Oct 2026 | `condense_v1` | [`practice`](../../testsets/optima_secure_followups_practice.json) (`861c4640a2`) | 16 | 11 | 12 | 2 / 1 | 1.0 | 0 of 6 |
| [2](run2/summary.md) | 27 Oct 2026 | `condense_v2` | [`practice2`](../../testsets/optima_secure_followups_practice2.json) (`fbf48a727e`) | 28 | 16 | 21 | 6 / 1 | 0.125 | 1 of 14 |
| [3](run3/summary.md) | 29 Oct 2026 | `condense_v3` | [`final`](../../testsets/optima_secure_followups.json) (`a9a5aa60b1`) | 32 | 19 | 21 | 5 / 3 | 0.73 | 0 of 12 |

Asked as complete questions, the same follow-ups were fully correct 15/16, 28/28 and 32/32 times. Run 1's set was
renamed after the run (it was then `optima_secure_followups.json`); the version hash identifies it.

**Decision: not adopted; memory is off by default** (`memory.enabled: false`). What each run showed:

1. Run 1: the set was too easy (11 of 16 already right without memory, so at most 5 could be fixed, and 5 fixed with
   none broken gives p = 0.0625), and 4 of 18 rewrites copied figures or sentences from earlier answers.
2. Run 2: copying was gone, but rewrites were often vague ("the policy" instead of the cover's name), and a complete
   question was reworded and lost its key detail (Q28.2).
3. Run 3: complete questions were no longer touched, but 8 of 27 rewrites pointed at the wrong thing or were garbled.

Pooled over the three runs (not a test fixed in advance): 13 fixed, 5 broken, McNemar p = 0.096. Median rewrite time
with thinking on: 47.6 s, 38.1 s and 43.8 s; the longest rewrites took 382 s, 862 s and 330 s.

**Limitations.** One rewrite model (Qwen3-8B, 4-bit). Sets of 16 to 32 deciding follow-ups, written for this
policy by the AI assistant that built the system, not by real users. Each run's conversations were new when its
rewriter was designed (the practice sets are earlier runs' conversations), but the rewriter was redesigned three
times, which raises the chance that one run passes by luck; none passed. The full write-up is in
[docs/evaluation.md](../../docs/evaluation.md#conversation-memory).

**Files per run.** `summary.md` (decision, counts and every turn: the question asked, the question searched, the
three verdicts and the issues found), `turns.csv` (the same per turn), `source.json` (the private `dialogues.json`
it was built from). No answers are included.
