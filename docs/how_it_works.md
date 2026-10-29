# How it works

This guide follows one question through the system, stage by stage. For each stage it says what the code does, where
it is, and why it is built that way. Where a design choice came from a measurement, the measurement is given. The
examples assume a built system:

```python
from policy_rag import RAGSystem
rag = RAGSystem.from_config("configs/default.yaml")
```

## The idea in one paragraph

A language model answers well from text it can see, and badly from what it half-remembers. Retrieval-augmented
generation (RAG) therefore splits the job in two. First, *retrieval* finds the few passages of the policy that hold
the answer. Then *generation* writes the answer from those passages only. Most of the engineering is in the first
half: if the right passage does not reach the model, no prompt can save the answer. In the evaluation, 8 of the 16
failures happened before the model saw anything ([evaluation.md](evaluation.md)).

## 0. Chunking the policy (offline)

`policy-rag chunk-pdf` ([`ingest/optima_secure_pdf.py`](../src/policy_rag/ingest/optima_secure_pdf.py)) reads the PDF
once and writes the chunk file.

- **Layout.** The prospectus is printed in two columns with running headers and footers. Words are grouped into
  horizontal bands. A band is read column by column only where both columns hold text, so full-width headings and
  tables keep their place in the reading order.
- **Structure.** Bold numbered lines become sections (`4.4 Plus Benefit`). Inside a section, lettered clauses
  (`a.`, `b.`, ...) and the exclusion codes (`Excl01`–`Excl18`) become units. A long section is split at clause
  boundaries, and only where a clause itself is too long, at sentence boundaries (about 320 tokens, at most 450).
- **Tables become sentences.** A plan table is pivoted into one line per plan
  ("Optima Secure: Base Sum Insured: ₹10 lakh; ..."). A premium table becomes one chunk per plan, city tier and
  5 age bands. A table cell on its own means little to a search engine; a self-describing line can be found and
  quoted.

Every chunk carries a `label` (`Section 4.4 - Plus Benefit`), which the model sees above the passage and which is
also searched. The chunk file behind the results has 208 chunks: 132 of policy text and 76 of premium tables.

*Why from the PDF and not a text export:* the earlier text export had no premium tables and lost a list in Annexure B,
so questions about them could not be answered at all.

## 1. Search: two legs, one union

[`retrieval.py`](../src/policy_rag/retrieval.py)

```python
cands, routes = rag.pipeline.retrieve("Is cataract surgery covered in the first year?")
len(cands), routes
```

- **Dense search** embeds the question with [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
  (with BGE's query instruction in front) and finds the 32 most similar chunks by cosine similarity. It is exact
  search with FAISS `IndexFlatIP`: a few hundred chunks need no approximate index. Dense search can match wordings that share
  no words, such as an everyday "kidney stones" and the policy's "calculus diseases".
- **BM25** finds the 32 chunks that share the most informative words with the question. It catches exact terms
  that embeddings blur, such as section numbers, "Excl06" or plan names. Its tokenizer keeps `4.7.1` and `excl06`
  intact.
- **Fusion.** The two lists are merged and ordered by reciprocal rank fusion (`1 / (60 + rank)`), but nothing is cut:
  every candidate goes to the reranker. The first stage is cheap and only has to *include* the evidence; the reranker
  decides what matters. In the baseline, 98% of evidence passages were among the candidates.

**Routed index (premium tables).** Chunks whose `index` field is `premium` are kept out of the two main lists and
searched as their own index, with 12 candidates per leg, but only when the question asks what the policy costs (the
pattern is in `retrieval.routed_indexes` in the config). The 76 premium chunks are near-identical rows of numbers.
In one BM25 index over all 208 chunks, the rows that answer the three premium questions of the test set (PR01, PR02,
N05) rank 36th to 75th, below the cut-off of 32; in their own small index they are found.

## 2. Reranking: read every candidate with the question

[`reranking.py`](../src/policy_rag/reranking.py)

A *bi-encoder* (the dense search) embeds the question and the passage separately. A *cross-encoder* reads them
together and outputs one relevance score, which is slower but much more precise.
[bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) (568M parameters) scores every candidate,
40 to 73 per question in the baseline (median 49), in fp16, with batches sorted by length so little padding is wasted. It reads
`label: text`, so section and clause names count too. The scores are raw logits: higher means more relevant, and
0 is not a boundary.

## 3. Building the context

[`context.py`](../src/policy_rag/context.py)

```python
st = rag.pipeline.search("What is the Plus Benefit? Explain in detail.")
print(rag.context.format(st["groups"]))
```

The answer model gets at most 1,500 tokens of context, counted with its own tokenizer, labels included. The context
is filled in this order:

1. **Title pinning.** If every content word of a section heading appears in the question ("Plus Benefit",
   "Standard Exclusions"), that section and its sub-sections go first, at most two sections. A user who names a
   section wants that section, even when the reranker prefers a passage that only mentions it.
2. **Seeds.** Then the reranked chunks, best first, but only those scoring at least −5 (`seed_score_floor`). Weak
   chunks are not used as filler.
3. **Neighbours.** Each seed brings the text it depends on: the next chunk of the same section, or a sub-section,
   and once more when a chunk ends with "... as listed below:"; the previous chunk or the parent section; the other
   year of a plan table. A chunk that says "the following conditions are excluded:" is useless without the list
   after it.
4. **Budget.** Groups are added until the budget is used up. Each group reaches the model as one passage under its
   labels, such as `[Section 10 - Exclusions → Section 10.1 - Standard Exclusions]`.

Premium-table chunks are self-contained and get no neighbours. Neighbours never cross from one document into
another.

## 4. The gate: refuse without guessing

If the best reranker score is below −5 (`gate.min_score`), the pipeline answers *"Not specified in the provided policy
text."* without calling the model. On the held-out questions this score separates answerable from not-answerable
questions well (AUROC 0.95). Of the 19 not-answerable questions, the gate refused 5; the model declined 13 more by following the
prompt's instruction, and answered 1 that it should have declined.

## 5. The prompt

[`prompts.py`](../src/policy_rag/prompts.py) holds the template (`policy_analyst_v2_3`). Besides "use only the
context", it has scope rules that came from mistakes in earlier recorded answers:

- a condition, limit or exclusion applies only to the cover named in its own section;
- a list from another section must not be presented as part of the cover asked about;
- when a passage says a list is provided below, include the list;
- compare plans from the same year's table and name the year;
- if the context covers part of the question, answer that part and say what is not shown.

## 6. Generation: think, then answer

[`generation.py`](../src/policy_rag/generation.py)

[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) runs in 4-bit NF4 with fp16 compute (a T4 has no bfloat16). The
published run used 7.9 GB of GPU memory after loading its models and peaked at 10.3 GB. With *thinking* on, it first reasons in a hidden `<think>` block, which helps with conditions,
calculations and "is X covered if Y" questions.

- **Thinking budget.** At first a single 1,000-token limit covered thinking and answer together. In an early
  development run on the 15 hard questions, 6 ran out of tokens while still thinking and produced no answer at all
  (that run's outputs were not kept, so this is a development note, not a published result). Now generation runs in
  two phases. Phase 1 thinks until `</think>`, or for at most 1,024 tokens. At the limit, Qwen3's own budget sentence is
  appended ("Considering the limited time by the user, I have to give the solution based on the thinking directly
  now.") and the thinking block is closed. Phase 2 writes the answer with its own 1,000 tokens. In the baseline the
  median was 339 thinking tokens, the budget was reached 8 times, and no answer was cut off. The thinking is saved
  separately (`results.csv`) and never mixed into the answer.
- **Sampling, seeded.** Qwen advises against greedy decoding in thinking mode, because it loops. The recommended
  settings are temperature 0.6, top-p 0.95 and top-k 20. The random seed comes from a hash of the prompt and the
  settings, so the same question with the same context gives the same answer again.
- **Cache.** Every answer is stored in `runs/answer_cache.jsonl` under a key made of the prompt and all generation
  settings. Changing the model, the precision, the budgets or the decoding changes the key, so an old answer is never
  reused by mistake. Repeating a run costs nothing.

## 7. The pipeline graph

[`pipeline.py`](../src/policy_rag/pipeline.py) wires the steps as a [LangGraph](https://github.com/langchain-ai/langgraph)
state graph: `condense → retrieve → rerank → gate → answer | no_answer`. Each node reads and writes one shared state
(question, candidates, scores, context groups, answer, generation record, errors). `condense` does nothing unless
earlier turns of a conversation are passed (next section). On CUDA out-of-memory, the answer node retries once with a
1,000-token context. Errors are recorded in the state instead of stopping a long evaluation run.

The graph is deliberately linear. Further steps, such as a second retrieval round or a calculator, would be new nodes
and edges in the same graph. None of them is implemented; the evaluation is there to show whether such a step helps.

## 7b. Follow-up questions (conversation memory)

[`memory.py`](../src/policy_rag/memory.py) · **Off by default.** Three evaluations did not show a significant gain
([docs/evaluation.md](evaluation.md#conversation-memory)); switch it on for experiments with
`--set memory.enabled=true` or `rag.set_memory(True)`. With memory off, the `condense` node passes every question
through unchanged.

```python
rag.set_memory(True)
chat = rag.new_conversation()
rag.ask("What is the waiting period for kidney stones?", conversation=chat)
r = rag.ask("Does it apply if the problem is caused by an accident?", conversation=chat)
r.standalone   # e.g. "Does the waiting period for kidney stones apply if the problem is caused by an accident?"
```

A follow-up such as *"does it apply after an accident?"* means nothing to a search engine: it has no topic. The
`condense` node therefore rewrites it into a complete question before anything else happens:

1. A `Conversation` (owned by the caller: `rag.ask(..., conversation=...)`, `rag.chat()`, `policy-rag chat`) keeps
   the turns. For each new question it hands the pipeline the last 3 turns: each turn's complete question and the
   first 400 characters of its answer (`memory` in the config).
2. A question that does not point back to the conversation (no "it", "that", "they", "the first one", and no start
   such as "And ..." or "What about ...") is complete: it skips the rewrite and goes straight to search, with no
   model call (`memory.skip_complete`, `needs_rewrite` in [`memory.py`](../src/policy_rag/memory.py)).
3. Otherwise the answer model gets a short rewrite prompt (`condense_v3` in
   [`prompts.py`](../src/policy_rag/prompts.py)): replace each word that points back with the exact name of what it
   points to (never "the policy" or "the plan"), take what is asked from the user's earlier question when it is left
   out ("And for X?"), keep every other word and detail, take figures only from the user's questions, and don't
   answer. Two invented examples show the form. It runs with the generator's configured settings, thinking
   included, except that the answer of a rewrite is capped at 64 tokens (`memory.max_new_tokens`): a rewrite is one
   line. It goes through the answer cache, so a repeated rewrite costs nothing.
   Guards reject a rewrite that is empty, longer than 400 characters, cut off at the cap, looks like an answer
   ("Yes, ..."), contains a figure the user did not type or drops one the user typed, adds nothing specific (only
   "the policy", "the plan", ...), or repeats 8 or more words in a row from an earlier answer that the user never
   typed (`memory.copy_words`). The question is then searched as typed. Answers may supply only a name ("it" -> "the
   Plus Benefit"). These rules come from the two evaluations so far
   ([docs/evaluation.md](evaluation.md#conversation-memory)).
4. Only the complete question is searched, reranked and answered. The answer prompt has exactly the same form as for
   a single question and never contains the conversation, so answers still come from the policy text alone. The
   rewritten question is shown (`searched as:`) so a wrong rewrite is visible.

The first question of a conversation, and every question asked without one, skips the rewrite completely, so the
test-set evaluation and its cache keys are unchanged. The conversation keeps its document filter across turns.

*Why this design:* putting the whole conversation into the answer prompt would make every prompt longer on a T4 and
let the model answer from its own earlier answers instead of the policy. A running summary needs a model call every
turn and drops details such as the plan or the year. Rewriting only the question keeps the grounded pipeline as it is
and costs one extra model call per follow-up that points back.

## 8. Several documents

[`corpus.py`](../src/policy_rag/corpus.py) gives every chunk the id of its document. Sections are keyed by
`(document, section id)`, so two policies can both have a "Section 3.1". A question can be limited to some
documents. With more than one document loaded, labels start with the policy name and the prompt gets one extra rule.
See [adding_a_policy.md](adding_a_policy.md).

## Where to go next

- [evaluation.md](evaluation.md): how the system is measured, and every failure of the baseline.
- [extending.md](extending.md): swap a model or a component, add a check, run an experiment.
- [adding_a_policy.md](adding_a_policy.md): the chunk-file format, and what another policy needs.
