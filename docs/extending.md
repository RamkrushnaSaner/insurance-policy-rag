# Extending and experimenting

## The experiment loop

1. **Baseline**: `policy-rag evaluate --name baseline` (once; its answers are cached).
2. **Change one thing**: a small config file in [`configs/experiments/`](../configs/experiments/) that `extends`
   the default, or `--set section.key=value` on the command line.
3. **Run it** under its own name. Retrieval-only changes first: `--mode retrieval` takes minutes and shows search
   recall, context recall and MRR without generating answers.
4. **Compare** on the held-out split: `policy-rag compare runs/baseline/run.json runs/<name>/run.json --splits test`.
   The decision rule in the config (`decision`) says ADOPT or KEEP THE BASELINE, with the reasons.
5. Debug on **dev** only. `results.csv` has every answer, its reference, the issues found and the thinking.

Answers are cached by prompt and generation settings: if a change leaves a question's context unchanged, its answer
comes from the cache.

## Swapping a model

Most model swaps are a config change:

```yaml
extends: ../default.yaml
reranker:
  model: BAAI/bge-reranker-large        # any Hugging Face sequence-classification cross-encoder
embedder:
  model: BAAI/bge-base-en-v1.5          # any sentence-transformers model
generator:
  model: Qwen/Qwen3-4B                  # any chat model with a chat template
```

Notes:

- **Reranker scores** are raw logits. A different reranker has a different scale, so `gate.min_score` and
  `context.seed_score_floor` (both −5) must be re-chosen on dev. The gate report in every run (lowest answerable
  score, highest not-answerable score) is the data for that.
- **Answer models without thinking**: set `enable_thinking: false`. The thinking budget is used only when the
  tokenizer has a `</think>` token.
- **Comparing answer models on identical contexts**: `policy-rag evaluate --name other_llm --contexts-from
  runs/baseline/run.json --set generator.model=...` answers the saved contexts again, so retrieval cannot explain
  the difference.

## Adding a component type

Each component is selected by its `type` in the config, through [`registry.py`](../src/policy_rag/registry.py). A
new implementation only needs the same methods:

| Kind | Methods |
|---|---|
| embedder | `from_config(cfg)`, `encode_documents(texts) -> float32 array`, `encode_query(text) -> (1, d) array`, `revision()` |
| reranker | `from_config(cfg)`, `score(query, docs) -> float array`, `revision()` |
| generator | `from_config(cfg)`, `signature() -> dict`, `generate(prompt, seed) -> dict`, `count_tokens(text)`, `revision()`, `self_check()`, `tokens_per_second(prompt)` |

For example, an answer model behind an HTTP API:

```python
# my_components.py
from policy_rag.registry import register

@register("generator", "my_api")
class MyAPIGenerator:
    @classmethod
    def from_config(cls, cfg, **_):
        return cls(cfg.model)

    def __init__(self, model):
        self.model = model

    def signature(self):                       # everything that changes the output: it is part of the cache key
        return {"llm": self.model, "api": "my_api v1"}

    def generate(self, prompt, seed):
        text = call_my_api(self.model, prompt, seed=seed)       # your code
        return {"answer": text, "thinking": "", "prompt_tokens": None, "think_tokens": 0,
                "answer_tokens": None, "thinking_forced": False, "hit_limit": False}

    def count_tokens(self, text):              # used for the 1,500-token context budget
        return len(text) // 4

    def revision(self): return None
    def self_check(self): return []
    def tokens_per_second(self, prompt): return None
```

```yaml
plugins: [my_components]          # imported before the system is built
generator:
  type: my_api
  model: some-model
```

The tests use the same mechanism with tiny stand-in components ([`tests/fakes.py`](../tests/fakes.py)).

## Other extension points

- **Prompt**: add a template to `TEMPLATES` in [`prompts.py`](../src/policy_rag/prompts.py) and select it with
  `prompt.template`. Keep the refusal sentence: the checks recognise it.
- **Routing**: another separate index is a new `index` value in the chunk file plus an entry under
  `retrieval.routed_indexes`.
- **A check**: add it to `score_answer` in [`evaluation/checks.py`](../src/policy_rag/evaluation/checks.py) with a
  stage and a kind, and decide whether the kind is critical (`CRITICAL`) or incomplete (`INCOMPLETE`). Changing the
  checks changes the scores, so re-score the baseline (`--mode rescore`) before comparing.
- **Conversation memory**: the `memory` section of the config (`enabled`, `max_turns`, `answer_chars`,
  `max_question_chars`, `copy_words`, `skip_complete`, `max_new_tokens`, `template`). A new rewrite instruction is a template in `CONDENSE_TEMPLATES` in
  [`prompts.py`](../src/policy_rag/prompts.py); a new safety check for rewrites goes in `check_rewrite` in
  [`memory.py`](../src/policy_rag/memory.py), and the words that mark a follow-up are in `needs_rewrite` there. Try
  changes on the practice sets (`--set evaluation.dialogues=testsets/optima_secure_followups_practice2.json`) and
  decide with
  `policy-rag evaluate-dialogues` on [`testsets/optima_secure_followups.json`](../testsets/optima_secure_followups.json).
  For a new policy, write conversations whose later turns' complete forms are questions in that policy's test set; with
  `"follow_ups_need_context": true` the file is checked so that every `follow_up` leaves its `subject` out, and
  follow-ups that name their subject are `self_contained` turns (harm checks, like topic switches).
- **A pipeline step** (a second retrieval round, say): a new node in the LangGraph graph in
  [`pipeline.py`](../src/policy_rag/pipeline.py), as the rewrite step (`condense`) is. Measure it like any other change.

## Running the tests

```bash
pip install -e ".[dev]"
pytest -q
```

The 70 tests use an invented sample policy and stand-in components: no GPU, no model downloads and no policy text.
They cover configuration, chunk loading for one or several documents, routing, title pinning, neighbours, the
budget, the gate, the cache, the checks, the statistics, the evaluation commands and conversation memory (rewriting,
its safety checks, the chat commands and the dialogue evaluation). GitHub Actions runs them on
Python 3.10, 3.12 and 3.13 ([`.github/workflows/tests.yml`](../.github/workflows/tests.yml)).
