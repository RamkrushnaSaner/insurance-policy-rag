# Adding a policy

The retrieval, context, gate, generation and evaluation code works for any document that comes as a chunk file. Two
things are specific to my:Optima Secure: the **chunker**, which knows that PDF's layout, and the **test set** (with its
plan-table checker). A new policy needs a chunk file, and a test set if you want numbers.

## 1. The chunk file

A JSON list with one object per chunk (a CSV with the same columns also works):

```json
[
  {"section_id": "3.1", "section_title": "Hospitalization Expenses",
   "text": "Room rent, nursing, ICU charges and surgeon fees are paid for a stay of at least 24 hours.",
   "label": "Section 3.1 - Hospitalization Expenses", "index": "policy", "group": "3.1", "kind": "text"}
]
```

| Field | Required | Meaning |
|---|---|---|
| `text` | yes | The passage the model reads |
| `section_id` | yes | The section number as printed (`3.1`, `10.2`, `Annexure-A`). Used for title pinning, neighbours, citations and scoring |
| `section_title` | no | The heading; title pinning matches its words against the question |
| `label` | no | The passage heading shown to the model (default `Section <id> - <title>`) |
| `embed_text` | no | What is searched and reranked (default `<label>: <text>`) |
| `index` | no | `policy` (default), or the name of a separate index such as `premium` (see below) |
| `group` | no | Chunks that belong together; neighbour expansion stays inside a group (default: the section) |
| `kind` / `is_table` | no | `table` marks table chunks (plan-table neighbours) |
| `clause` | no | A clause or exclusion code, kept for reference |

Practical rules that the published results depend on:

- **Keep section numbers real.** Number sub-sections as the document does (`4.4`, `4.4.1`). Neighbour expansion
  uses the dotted structure, and a list that follows a "provided below" sentence should be the next chunk.
- **Make tables self-describing.** A row such as `Age 40: 10,300 / 13,300` is useless without its header. Repeat
  the plan, tier and column names in each chunk.
- **Size.** About 300 tokens per chunk, never splitting a sentence. Very long chunks crowd the 1,500-token context.
- **Separate indexes** for large blocks of near-identical rows (premium tables): set `index` to a name and add it
  under `retrieval.routed_indexes` with a question pattern. Otherwise they crowd the policy text out of the search
  results.

For a PDF with a different layout, write a chunker that produces these fields. The my:Optima Secure chunker
([`ingest/optima_secure_pdf.py`](../src/policy_rag/ingest/optima_secure_pdf.py)) is a worked example. Its page
bands, heading and clause patterns and table rules are specific to that PDF, so treat it as a template, not a
general tool. Always read a sample of the chunks before indexing: everything downstream depends on them.

## 2. The config

```yaml
extends: default.yaml
documents:
  - id: optima-secure                 # short and stable: stored with every chunk and result
    name: "HDFC ERGO my:Optima Secure"   # shown to the model when several documents are loaded
    chunks: data/policy_chunks.json
  - id: my-second-policy
    name: "Insurer X Health Plus"
    chunks: data/second_policy_chunks.json
```

`policy-rag info --config <file>` shows what was loaded. With two or more documents:

- every passage label starts with the policy name, and the prompt gets one extra rule (keep the policies apart and
  say which one each statement comes from);
- sections, title pinning and neighbours are kept per document, so both policies can have a Section 3.1;
- a question can be limited to some policies: `policy-rag ask "..." --document my-second-policy`, or
  `rag.ask(q, documents=["my-second-policy"])`;
- an evaluation searches only the policy its test set is about (the test set's `document` field).

None of this has been evaluated with a second real policy yet. In particular, how well the system picks the right
policy when a question does not name one is unmeasured.

## 3. A test set (to get numbers)

Copy the format of [`testsets/optima_secure_v3.json`](../testsets/optima_secure_v3.json):

```json
{"name": "my second policy v1", "document": "my-second-policy", "table_checker": null,
 "evidence_removed_candidates": ["Q01", "Q02"],
 "items": [
  {"id": "Q01", "split": "dev", "type": "fact", "answerable": true, "paraphrase": false,
   "question": "What is the waiting period for pre-existing diseases?", "sections": ["10.1"],
   "evidence": [["covered after 48 months of continuous cover"]],
   "facts": [["48 months", "four years"]],
   "reference": "48 months of continuous cover (Section 10.1)."}
 ]}
```

Guidelines from building the published set:

- **Write questions from the document**, including some in the words a customer would use. Add questions the
  policy does not answer (`"answerable": false` with `absent_terms`), and false premises.
- **Give alternatives for every key fact** ("24 months", "two years", `re:` patterns). Then run the reference
  answers through the checks: every reference must pass its own key.
- **Hold out a test split before looking at results**, and never tune on it.
- **Plan tables:** the plan-table checker is specific to my:Optima Secure. For another policy, leave
  `table_checker` empty, or write one in `evaluation/table_checks/`.

Then point the config at it (`evaluation.testset`) and run `policy-rag evaluate --name first --splits dev`.
