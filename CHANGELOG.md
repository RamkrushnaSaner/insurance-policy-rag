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
  looks like an answer (a refusal, "yes"/"no") or adds a figure that is neither in the question nor in the
  conversation. The reason is logged.
- Rewrites are cached like answers, so a repeated conversation costs no model calls.
- A conversation keeps its document choice between turns, and can be saved to and loaded from a JSON file.

### Added

- `policy_rag.memory` (`Conversation`, `Turn`, `QuestionRewriter`, `check_rewrite`) and `policy_rag.chat`.
- `RAGSystem.new_conversation()`, `RAGSystem.ask(..., conversation=...)`, `RAGSystem.chat()`;
  `Answer.standalone` holds the rewritten question.
- Commands `policy-rag chat` (`/new`, `/doc`, `/save`, `/history`, `/help`, `/quit`; `--load` a saved chat) and
  `policy-rag evaluate-dialogues`.
- `memory` config section and the `condense_v1` rewrite template.
- [`testsets/optima_secure_followups.json`](testsets/optima_secure_followups.json): 23 conversations with
  26 follow-ups (follow-ups, topic switches and follow-ups after a refused answer), each scored with an existing
  test-set question's answer key.
- Dialogue evaluation comparing `no_memory`, `memory` and `gold` with a fixed decision rule
  ([docs/evaluation.md](docs/evaluation.md#conversation-memory)).
- `--resume-from` (for `evaluate` and `evaluate-dialogues`) also accepts a `run.json`, whose records hold every
  generated answer.
- 16 tests (47 in total).

### Unchanged

- Questions asked without a conversation (the command `ask`, `evaluate`, and `rag.ask(question)`): same graph path,
  same prompts, same cache keys and the same results as v3.2. Verified with the notebook equivalence checks and the
  108 real cache keys of the v3.1 run.
- Generator settings: thinking stays on (`enable_thinking: true`, `think_budget: 1024`, `max_new_tokens: 1000`,
  sampling as before), for answers and for rewrites.

### Documentation

- README, [how it works](docs/how_it_works.md#7b-follow-up-questions-conversation-memory),
  [evaluation](docs/evaluation.md) (memory evaluation, thinking-off experiment, real-model retrieval check) and
  [extending](docs/extending.md).

### Limitations

- Not yet evaluated with the real models: whether memory is adopted is decided by the dialogue evaluation's rule.
- Each follow-up costs one more model call with thinking (typically tens of seconds on a T4, at most the full thinking
  and answer budgets); the first question of a conversation costs nothing extra.
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
