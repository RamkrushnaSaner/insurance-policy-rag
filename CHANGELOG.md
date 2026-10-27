# Changelog

## 3.3.0 (unreleased)

Conversation memory: follow-up questions such as "And for the second policy year?" or "Does that apply to
day care treatment too?" can now be asked in a conversation.

### How it works

- A `Conversation` keeps the last turns of a chat: for each one, the complete question that was searched and the
  start of the answer (by default the last 3 turns and the first 400 characters of each answer).
- When a question is asked inside a conversation that already has turns, a new first step (`condense`) asks the same
  Qwen3-8B model, with thinking on and the usual settings, to rewrite it into a question that can be understood on its
  own. That complete question is searched and answered as before. The answer prompt does not contain the conversation,
  so every answer is still based only on the retrieved policy text.
- The rewrite is checked before it is used; the question as typed is used instead when the rewrite is empty, too long,
  looks like an answer (a refusal, "yes"/"no"), contains a figure the user did not type, or repeats 6 or more words in
  a row from an earlier answer. Earlier answers may only supply a name ("it" -> "the Plus Benefit"). The reason is
  logged.
- Rewrites are cached like answers, so a repeated conversation costs no model calls.
- A conversation keeps its document choice between turns, and can be saved to and loaded from a JSON file.

### Added

- `policy_rag.memory` (`Conversation`, `Turn`, `QuestionRewriter`, `check_rewrite`) and `policy_rag.chat`.
- `RAGSystem.new_conversation()`, `RAGSystem.ask(..., conversation=...)`, `RAGSystem.chat()`;
  `Answer.standalone` holds the rewritten question.
- Commands `policy-rag chat` (`/new`, `/doc`, `/save`, `/history`, `/help`, `/quit`; `--load` a saved chat) and
  `policy-rag evaluate-dialogues`.
- `memory` config section and the rewrite templates `condense_v2` (default) and `condense_v1` (first run).
- [`testsets/optima_secure_followups.json`](testsets/optima_secure_followups.json): the deciding set, 38
  conversations with 42 later turns (28 follow-ups that leave out their subject, 8 self-contained follow-ups,
  6 topic switches), each scored with an existing test-set question's answer key; checked when loaded.
  [`testsets/optima_secure_followups_practice.json`](testsets/optima_secure_followups_practice.json): the 26 turns of
  the first run, kept for trying changes.
- Dialogue evaluation comparing `no_memory`, `memory` and `gold` with a fixed decision rule
  ([docs/evaluation.md](docs/evaluation.md#conversation-memory)).
- `--resume-from` (for `evaluate` and `evaluate-dialogues`) also accepts a `run.json`, whose records hold every
  generated answer.
- 20 tests (51 in total).

### Unchanged

- Questions asked without a conversation (the command `ask`, `evaluate`, and `rag.ask(question)`): same graph path,
  same prompts, same cache keys and the same results as v3.2. Verified with the notebook equivalence checks, the
  108 real cache keys of the v3.1 run, and on a Colab T4 (all 113 answers from the v3.1 cache, 97/113).
- Generator settings: thinking stays on (`enable_thinking: true`, `think_budget: 1024`, `max_new_tokens: 1000`,
  sampling as before), for answers and for rewrites.

### Documentation

- README, [how it works](docs/how_it_works.md#7b-follow-up-questions-conversation-memory),
  [evaluation](docs/evaluation.md) (memory evaluation, thinking-off experiment, real-model retrieval check) and
  [extending](docs/extending.md).

### Evaluation so far

- First run (Colab T4, `condense_v1`, the practice set): **not adopted**. 16 deciding follow-ups: 11 right without
  memory, 12 with it (2 fixed, 1 broken, p = 1.0). The set was too easy to show a gain (at most 5 could be fixed) and
  4 of 18 rewrites copied figures or sentences from earlier answers, which caused the broken turn. Both are fixed:
  the `condense_v2` prompt, the stricter checks, and a new deciding set written before any run with them.
- The second run, on the new set, decides.

### Limitations

- Not adopted yet: the second dialogue evaluation decides.
- A follow-up whose missing piece is a figure from an earlier answer ("when he reaches that age") is searched as typed,
  because rewrites may take figures only from the user's questions.
- Each follow-up costs one more model call with thinking (a median of 48 s on a T4 in the first run, once 6.4 minutes
  when thinking and the answer ran to their limits); the first question of a conversation costs nothing extra.
- Only the last 3 turns are used, and answers are cut to 400 characters; something said earlier can be lost.
- A wrong rewrite that passes the checks leads to a correct answer to the wrong question; the reply shows the
  question that was searched ("searched as: ...") so the user can see it.
- Conversations live in memory unless saved with `/save` or `Conversation.save()`.

## 3.2.0 (24 October 2026)

- The v3.2 notebook as a Python package (`policy_rag`): typed YAML configs with `extends` and `--set` overrides,
  a component registry and plugins, several policy documents with routing, a command line
  (`chunk-pdf`, `info`, `ask`, `evaluate`, `compare`, `agreement`), a Colab quick start, tests and documentation.
- Retrieval, prompts, cache keys and checks equal to the notebook; the v3.1 answers re-scored with the corrected
  checks: 97/113 fully correct (79/88 on the test split).
