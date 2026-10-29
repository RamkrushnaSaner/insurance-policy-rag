# Changelog

## 3.3.1 (29 October 2026)

Every experiment re-verified against its run files; no change to how the system answers.

- **Evidence published** in [`results/`](results/) for every experiment, scores only (no policy text): baseline,
  reproduction on a T4, thinking off, v2.6 against v3.1, and the three conversation-memory runs. Each folder records
  the sha256 of the private run file it was built from.
- [`scripts/export_results.py`](scripts/export_results.py) rebuilds those files from run files;
  [`tests/test_published_results.py`](tests/test_published_results.py) recomputes the headline numbers from them
  (80 tests in total).
- [docs/experiments.md](docs/experiments.md): objective, method, result, limitations and evidence for each
  experiment, and what has not been run or cannot be verified.
- **Corrected:** groundedness of the baseline is 0.886 (28.3% of answers with an unsupported sentence), the package's
  own measurement on a T4, the same in three sessions. The earlier 0.851 (33.7%) was the v3.1 notebook's measurement
  with a different sentence splitter. Rebuilding the baseline files changed nothing else.
- **Added to the thinking-off results:** groundedness compared on the same 69 answers (0.940 with thinking, 0.930
  without: no clear change).
- **Labelled as unverified:** the hard-question figures from an early development run (outputs not kept), the
  model download size and the answer model's memory use (estimates).
- **Corrected in 3.3.0's notes:** the 64-token cap shortened rewrites, but the longest rewrite in the final run still
  took 5.5 minutes, because thinking itself is slow.

## 3.3.0 (29 October 2026)

An interactive chat, and conversation memory for follow-up questions such as "Does that apply to day care treatment
too?", built and evaluated three times. **Memory is off by default:** it helped some follow-ups but never
significantly (final run: 21 of 32 right with memory, 19 without; 5 fixed, 3 broken), so it is not adopted. It can
be switched on for experiments (`--set memory.enabled=true`, `rag.set_memory(True)`).

### How memory works (when switched on)

- A `Conversation` keeps the last turns of a chat: for each one, the complete question that was searched and the
  start of the answer (by default the last 3 turns and the first 400 characters of each answer).
- When a question asked inside a conversation points back to it ("it", "that", "the first one", "And for ...?"), a
  new first step (`condense`) asks the same Qwen3-8B model, with thinking on and the usual settings (the rewrite's
  answer capped at 64 tokens), to rewrite it into a question that can be understood on its own. A complete question
  skips this step. The complete question is searched and answered as before. The answer prompt does not contain the conversation,
  so every answer is still based only on the retrieved policy text.
- The rewrite is checked before it is used; the question as typed is used instead when the rewrite is empty, too long,
  cut off, looks like an answer (a refusal, "yes"/"no"), contains a figure the user did not type or drops one the
  user typed, names nothing specific (only "the policy", "the plan"), or repeats 8 or more words in a row from an
  earlier answer. Earlier answers may only supply a name ("it" -> "the Plus Benefit"). The reason is logged.
- Rewrites are cached like answers, so a repeated conversation costs no model calls.
- A conversation keeps its document choice between turns, and can be saved to and loaded from a JSON file.

### Added

- `policy_rag.memory` (`Conversation`, `Turn`, `QuestionRewriter`, `check_rewrite`) and `policy_rag.chat`.
- `RAGSystem.new_conversation()`, `RAGSystem.ask(..., conversation=...)`, `RAGSystem.chat()`,
  `RAGSystem.set_memory()`; `Answer.standalone` holds the rewritten question.
- Commands `policy-rag chat` (`/new`, `/doc`, `/save`, `/history`, `/help`, `/quit`; `--load` a saved chat) and
  `policy-rag evaluate-dialogues`.
- `memory` config section and the rewrite templates `condense_v3` (default), `condense_v2` and `condense_v1`
  (second and first runs).
- [`testsets/optima_secure_followups.json`](testsets/optima_secure_followups.json): the final deciding set, 44
  conversations with 44 later turns (32 follow-ups that leave out their subject, 6 self-contained follow-ups,
  6 topic switches), each scored with an existing test-set question's answer key; checked when loaded. The sets of
  the first and second runs are kept as practice sets (`optima_secure_followups_practice.json`, `..._practice2.json`).
- Dialogue evaluation comparing `no_memory`, `memory` and `gold` with a fixed decision rule
  ([docs/evaluation.md](docs/evaluation.md#conversation-memory)).
- `--resume-from` (for `evaluate` and `evaluate-dialogues`) also accepts a `run.json`, whose records hold every
  generated answer.
- 41 tests (72 in total).

### Unchanged

- Questions asked without a conversation (the command `ask`, `evaluate`, and `rag.ask(question)`): same graph path,
  same prompts, same cache keys and the same results as v3.2. Verified with the notebook equivalence checks, the
  108 real cache keys of the v3.1 run, and on a Colab T4 (all 113 answers from the v3.1 cache, 97/113).
- Generator settings: thinking stays on (`enable_thinking: true`, `think_budget: 1024`, `max_new_tokens: 1000`,
  sampling as before), for answers and for rewrites; only a rewrite's answer is capped (`memory.max_new_tokens: 64`).

### Documentation

- README, [how it works](docs/how_it_works.md#7b-follow-up-questions-conversation-memory),
  [evaluation](docs/evaluation.md) (memory evaluation, thinking-off experiment, real-model retrieval check) and
  [extending](docs/extending.md).

### Evaluation

- First run (Colab T4, `condense_v1`, the practice set): **not adopted**. 16 deciding follow-ups: 11 right without
  memory, 12 with it (2 fixed, 1 broken, p = 1.0). The set was too easy to show a gain (at most 5 could be fixed) and
  4 of 18 rewrites copied figures or sentences from earlier answers, which caused the broken turn. Both are fixed:
  the `condense_v2` prompt, the stricter checks, and a new deciding set written before any run with them.
- Second run (`condense_v2`, practice set 2): **not adopted**. 28 deciding follow-ups: 16 right without memory, 21
  with it (6 fixed, 1 broken, p = 0.125), and one self-contained follow-up made wrong. Copying from answers was gone;
  vague rewrites ("the policy" instead of the cover's name) and rewording of complete questions were the problems.
  Fixed with `condense_v3`, skipping complete questions, two more checks and the 64-token cap.
- Final run (`condense_v3`, the final set): **not adopted**. 32 deciding follow-ups: 19 right without memory, 21 with
  it (5 fixed, 3 broken, p = 0.73); no complete question made worse. The rewriter often chose the wrong referent for
  "it" or "that" ("can they no longer do that?" became "... no longer Incorporate an additional Waiting Period?").
  Memory ships switched off (`memory.enabled: false`).

### Limitations

- Memory is not adopted and is off by default; with it off, every question in a chat is answered as typed.
- Follow-ups that point back only through "the" ("the papers", "the emergency-only one") are not recognised and are
  answered as typed.
- A follow-up whose missing piece is a figure from an earlier answer ("when he reaches that age") is searched as typed,
  because rewrites may take figures only from the user's questions.
- When switched on, each follow-up that points back costs one more model call with thinking (a median of 44 s on a
  T4 in the final run, at most 5.5 minutes; before the 64-token cap the longest rewrites took 6.4 and 14.4 minutes);
  complete questions cost nothing extra.
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
