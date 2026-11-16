# Changelog

## Unreleased: v3.5, numeric tables as SQL (step 4 of the v3.5 plan)

Nothing changes unless a config asks for it; the SQL step is not yet in the pipeline (step 6).

- **Migration 003** ([`db/migrations/003_numeric_tables_sql.sql`](db/migrations/003_numeric_tables_sql.sql)), byte for
  byte the file applied to the development database: drops the typed tables of 001 and 002; keeps the three numeric
  tables (premiums, check-up limits, deductible discounts) as SQL tables in their printed layout; schema
  `policy_sql` with views of the active version, the reference text of each table and the printed column headings;
  group `rag_sql` (reads `policy_sql` only); the chunk link `read_with`. Every other printed table is now text
  chunks with vectors.
- **SQL step** ([`sql_step.py`](src/policy_rag/sql_step.py), `sql.*`, off by default): a gate for premium,
  check-up and deductible questions; a schema card read from the database; a prompt (`prompts.SQL_TEMPLATES`,
  `sql_v1`) asking the answer model for one JSON reply (`query`, `ask`, `not_in_tables`, `text_only`); sqlglot checks
  (one SELECT, policy tables only, listed functions only, no constants, no locks or INTO, at most 50 rows); EXPLAIN
  then the query, read-only with a 5-second timeout, as a `rag_sql` login (`SQL_DATABASE_URL`); up to two corrected
  replies; a result block with printed headings, pages and footnotes, the linked text and a figure check.
- `Database.query_columns` (column names and rows, SQL sent as it is); a statement that fails on a working
  connection (e.g. cancelled at the timeout) is no longer run a second time on a new connection.
- `sql.enabled` and `tables.enabled` cannot both be on (after 003 the table route has nothing to read).
- `sqlglot>=27,<31` joins the `db` extra (tested with 27.29 and 30.31).
- **SQL step evaluation** (step 5; [`evaluation/sql_eval.py`](src/policy_rag/evaluation/sql_eval.py),
  `policy-rag evaluate-sql`, [`configs/experiments/v35_sql_step.yaml`](configs/experiments/v35_sql_step.yaml)):
  runs the SQL step on its own question set (kept private: it holds the policy's figures) and grades each reply
  by type: lookups and calculations must return the expected printed values (a calculation's result computed in
  SQL), questions missing a detail must be asked back, values the tables do not hold must be declined, wording
  questions must not be blocked, and other questions must be skipped by the gate. Rows without the expected
  values, or for a value the tables do not hold, are critical. The held-out half is fingerprinted, and its
  targets are part of the question set, fixed before it runs. Saved rows keep numbers as numbers, and text
  cells (a printed age) never count as values. 28 tests.
- **SQL step, revised after the first dev run** (dev half of the question set only; the held-out half was not run).
  The first run (prompt `sql_v1`) passed 10/18 lookups and 0/7 calculations, with 6 critical rows. Causes and changes:
  - results of calculations were not computed (6 of 7 returned only the inputs): prompt `sql_v2` asks for the result
    as its own column, with its inputs;
  - lakh read as the wrong column or amount (50 lakh as `si_500000`, 5 lakh as 50,000): the card writes amounts of a
    lakh or more with their lakh/crore value (`Sum Insured 5,000,000 (50 lakh)`), lists the values of columns that
    hold few (deductible amounts, limits, discounts), and the prompt gives the conversion;
  - an age compared as a text range (`BETWEEN '66*' AND '89*'`, 26 rows): ages are listed as separate values
    (`'66*', '67*', ... '89*'`), and the checks refuse BETWEEN, <, > on text columns;
  - a general city classification instead of the policy's tiers, an amount outside a column's printed amounts
    (30 lakh read as "20, 25 & 50 Lakhs"), a column split at 20 lakh read on the wrong side, a wording question
    declined as not in the tables: one rule each in `sql_v2`;
  - the gate missed all 4 loosely worded questions. No word list foresees every wording, so the gate gains a second
    half instead of more words: closeness (cosine, with the v3.4 embedder) between the question and the tables'
    reference text (descriptions, printed titles, introducing sentences, column headings), recorded for every
    question. Its threshold (`sql.gate_similarity`) is set by a rule fixed before the run (`evaluate-sql
    --calibrate-gate`): the lowest threshold at which at most 10% of the questions not about the tables pass, with
    the dev questions about the tables as positives and, as negatives, the dev controls and the 113-question
    development set without the sections the tables come from or link to. The held-out half is never used. The word
    list keeps its words and one general rule: a common money word ("cost", "price", "pay") with a plan name or an
    age.
  - the prompt's rules use invented example values, so no rule is written for one question, and one rule covers
    every category the tables list (plans, tiers and their places, policy types): never an outside classification.
  - `sql_v1` is kept for the record; `sql.template` defaults to `sql_v2`.
- **Arithmetic in code** (after the held-out run of step 5, where the calculations missed their target): the model
  only fetches the numbers; code computes, in Decimal, from whatever rows come back:
  - with two rows, the difference of each amount column (higher minus lower, naming both rows) and the total;
    with up to 8 rows, the total;
  - in one row, the difference of two amount columns of the same table (two Sums Insured);
  - in one row, an amount less a percentage from another table (a premium after a discount), and the amount
    taken off.

  Amount and percent columns are recognised by their printed headings; columns that chose the rows (= or IN in the
  query) and columns the model computed are never calculated with. The calculations go into the result block, so
  the figure check accepts them, and grading counts them. Aliases made in subqueries now keep their printed
  heading. The prompt is unchanged (`sql_v2`).
  - Replayed on the saved runs (same model replies, code added; not a test, since the change was made after
    seeing them): held-out calculations 3/7 → 6/7, dev run 1 0/7 → 6/7. The two critical held-out rows are
    unchanged: one difference with the opposite sign (S045) and one amount outside a column's printed amounts
    (S075, the coverage check, planned for later).
  - The change will be judged on new questions (the table questions of the fresh set in step 7).
- **Repository brought up to date:** README, `data/README.md`, `db/README.md`, the v3.4 database configs and
  `docs/how_it_works.md` say what v3.5 changes and that v3.4 needs a database with v3.4's data (the project's Neon
  database moved to v3.5 on 16 November 2026; v3.4's data stays on a snapshot branch). Removed: the archived
  single-notebook version (v3.2; still in the git history, linked from the README) and `requirements.txt` (the
  same as `pip install -e ".[models,ingest]"`; `requirements-lock.txt` stays). References to merged branches now
  name the release instead.
- Tests: 51 without a database (gate, card, reply parsing, SQL checks, retries, result block, config), 4 opt-in tests
  on a Postgres at migration 003 (`POLICY_RAG_TEST_DATABASE_URL_V35`: card, answers, the `rag_sql` login's
  permissions, the timeout). CI keeps a database at 002 for v3.4 and one at 003 for v3.5, and installs the `db`
  extra in the unit-test job.

## 3.4.1 (15 November 2026)

**No change to the system.** Retrieval, prompts, models and the system's configs are unchanged. This release
contains the repaired checks, the last decision, the evaluation of v3.4 on every question set, and a cleaned-up
repository.

### Evaluation

- **Final round: R1 not adopted, v3.4 stays.** The last candidate, R1 (a search card for every printed table row;
  branch `r1-candidate`), was judged on two new question sets under rules fixed before any answer:
  - on fresh v3, the checks failed their check against a person (kappa 0.52);
  - on fresh v4, the gain on table questions, judged by a person, was not significant (6 fixed, 2 broken,
    p = 0.29).

  R1 did remove every search failure of a fact printed only in a table (13 of 13). Details:
  [docs/experiments.md](docs/experiments.md) (10 and 11).
- **Evaluation of v3.4 on every question set** ([docs/final_evaluation.md](docs/final_evaluation.md),
  [`results/final_evaluation/`](results/final_evaluation/)).
  - **Accuracy:** 255/329 = 77.5% fully correct on new questions (95% CI 73–82%), against 101/113 on the
    development set it was tuned on.
  - **By question kind:** table questions 49/84, text questions 177/216.
  - **Where the 76 failures counted by the checks started:** search 26 (all facts printed only in a table), context
    15, gate 1, the answer 34.
  - **The 34 answer-stage failures, read one by one:** 17 clear model errors.

### Checks

**Grader repairs** (evaluation only; no change to retrieval, prompts or models). Reading the answers behind the
experiment's verdicts showed the deterministic checks failing correct answers and passing hedges; each repair is a
general rule, applied alike to every run, and is to be validated against new human labels.

- **Amount abbreviations:** "5 L" (lakh) and "25k" are read as amounts, as the tables print them. The same figure
  check decides when the table route replaces an answer (`tables.on_unsupported_figures`), so an answer that says
  "Rs. 5 lakh" where the table prints "5 L" is no longer replaced.
- **Ages without their unit:** "N years" is supported when N is an age said in the question ("I'm 35") or printed in
  a table line about age ("Age ... (in Years): 75").
- **Plan-table checker:** bold marks and "Rs." no longer break a statement, so "**Rs. 20 lakh** (Year 1)" is read as
  Year 1; "20 L" is read as 20 lakh. Base Sum Insured and Restore amounts are no longer checked against the
  illustration's assumed 10 lakh (the plans offer 5 lakh to 2 crore).
- **Word forms in key facts:** phrases without digits also match other word forms with the small words (a, the, of,
  on, for, ...) left out. "not" and "no" are always kept; figures and regular expressions still match exactly.
- **Hedges:** an answerable question's answer that opens by saying the policy text does not give the answer ("Not
  specified in the provided policy text.", "The policy text does not explicitly list ...") is not fully correct,
  whatever follows (new issue kind `hedge`), except where the reference answer itself says so.

### Repository

- **README:** the headline is now v3.4's accuracy on new questions. The 113-question results are labelled as
  development-set results, the diagram shows v3.4, and the limitations follow the final evaluation.
- **New:** [`scripts/export_final_evaluation.py`](scripts/export_final_evaluation.py) rebuilds
  `results/final_evaluation/` from the private run files with this repository's checks, and three tests recompute its
  numbers.
- **Updated:**
  - [`results/README.md`](results/README.md) lists every results folder and private run file;
  - [docs/experiments.md](docs/experiments.md) adds experiment 12, the decision records' fingerprints, and what was
    not run;
  - [docs/evaluation.md](docs/evaluation.md) says what "held out" meant for the development set.
- **Removed:** `configs/experiments/larger_context.yaml` and `configs/experiments/dense_only.yaml`. Neither was ever
  run, and nothing referred to them.

## 3.4.0 (3 November 2026)

Postgres on Neon for the search (pgvector) and for the policy's tables (typed tables answered with fixed SQL).
Nothing changes unless a config asks for it: [`configs/default.yaml`](configs/default.yaml) is still the published
baseline; [`configs/experiments/qwen3_text_tables.yaml`](configs/experiments/qwen3_text_tables.yaml) is the database
version. **Evaluated** ([`results/neon_tables/`](results/neon_tables/)): 78/88 fully correct on test against the
baseline's 79/88 (p = 1.0): adopted as not worse, not shown better; more evidence reaches the model (66 against 63 of
67 test questions), and table-only questions 8 → 9 of 12.

- **Stored vectors** (`retrieval.dense_backend: pgvector`, `database.*`): the dense leg reads the chunk vectors of
  the active policy version from Postgres instead of computing them at start-up. Exact inner-product search, same
  ranking as FAISS. Start-up refuses a database whose active chunks are not exactly the chunk file's (matched by
  `chunk_uid`), lack a vector, or were embedded by another model or revision. The connection string comes from an
  environment variable (`DATABASE_URL`), sessions are read-only, and a dropped connection (Neon suspends idle
  computes) is reopened once. Needs `pip install -e ".[db]"`.
- **Embedder settings** `revision`, `dtype`, `normalize`, `padding_side` (all off by default), for
  Qwen3-Embedding-0.6B in half precision on a T4.
- **Experiment configs** [`qwen3_text_faiss.yaml`](configs/experiments/qwen3_text_faiss.yaml) and
  [`qwen3_text_pgvector.yaml`](configs/experiments/qwen3_text_pgvector.yaml): Qwen3 over the text-only chunk file,
  vectors in memory or from the database.
- Tests: stored-vector search, start-up checks, reconnects and the embedder settings with stand-ins (24 tests), plus
  3 opt-in tests against a real Postgres (`POLICY_RAG_TEST_DATABASE_URL`). Config fingerprints change because the
  new settings are part of every config.
- **Table route** ([`tables.py`](src/policy_rag/tables.py)): questions about premiums, claim limits, instalment grace
  periods, check-up limits, deductible discounts and city tiers are answered from the typed tables with fixed,
  parameterised SQL. Fields (plan, age, Sum Insured, city or tier, year, frequency) are read from the question and
  checked against the tables: a plan without a premium table, an age or Sum Insured that is not printed, an unknown
  plan are explained, never guessed. Differences are computed in Decimal; every value keeps its table, row, column,
  page and printed cell; `verify()` lists figures in an answer that are not in the facts. Tests: 35 on an invented
  policy, plus 18 opt-in tests that compare the SQL with the in-memory stand-in on a real Postgres.
- **Table route, lists and procedures** (needs migration 002): waiting-period conditions, non-payable items, Major
  Medical Illnesses, Global Health Cover benefits and relationships (a whole list, or the items a question names,
  found by keyword search with a precision rule), claim deadlines, grievance contacts and the Sum Insured examples.
  Each answer carries the printed cells behind it (`evidence_text`) and the text chunks linked to its tables.
- **Table route in the pipeline** (`tables.*`, off by default;
  [`qwen3_text_tables.yaml`](configs/experiments/qwen3_text_tables.yaml) turns it on): a new graph node puts the
  table facts first in the context, adds the linked chunks to the search candidates and places them in the context
  whatever their score (`tables.pin_linked_chunks`, at most `tables.max_linked_tokens`); the gate passes when the
  tables answered (except for single list items); a complete table answer replaces a model answer that has a figure
  not in the context, or that refuses. A failing route is logged and the text alone is used. With tables off, the
  pipeline's results are unchanged (checked question by question on the 113 test questions).
- **Evaluation with tables:** records keep the table passage, a summary of the route (no policy text) and the cells
  behind it; those cells count as evidence for search recall, context recall and facts in context, and the printed
  tables (and the table passage) count as part of the document, so a missing fact that the tables hold is not
  blamed on the chunk file. Runs report how many questions the route answered and every fallback.
- The table passage's first line now closes its parenthesis ("Policy tables (exact values read from the database).").
- Tests: 31 on the invented lists, 16 on the pipeline and evaluation with tables, and 20 more opt-in database tests.
- **Database schema in the repo** ([`db/`](db/)): migrations 001 and 002, byte for byte the files applied to the
  development database (checksums in [`db/README.md`](db/README.md)); no policy text.
- **CI against Postgres:** a second job applies the migrations to a Postgres 18 + pgvector service and runs the 41
  database tests (pgvector search and table route SQL) that are skipped without a database.
- Table route SQL sorts text in byte order (`COLLATE "C"`), so its results do not depend on the database's
  collation (CI's `en_US` sorts `p10-t1` before `p1-t5`); Global Health Cover benefits are listed in section order
  (4.9 before 4.10). No change on the development database, whose collation is `C.UTF-8`.
- `evaluation.grounding_revision` pins the HHEM model's commit (HHEM runs code from its repository). The default
  config pins `de9a5e6b…`, the commit that graded the Step 8b run; `null` loads the latest.

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
