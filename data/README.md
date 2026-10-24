# Policy data (not included)

The system reads a **chunk file**: the policy text cut into sections and passages (format:
[`src/policy_rag/corpus.py`](../src/policy_rag/corpus.py)). Neither the policy document nor its chunk file is in this
repository, because the text belongs to HDFC ERGO General Insurance Company Limited. Everything in this folder except
this README is ignored by git, so the policy text is not committed by accident.

## Make the chunk file

1. Download the **my:Optima Secure prospectus** (PDF) from HDFC ERGO's
   [Optima Secure page](https://www.hdfcergo.com/health-insurance/optima-secure).
2. Install the ingestion extra and chunk the PDF (CPU only, a few seconds):

   ```bash
   pip install -e ".[ingest]"
   policy-rag chunk-pdf path/to/prospectus.pdf data/policy_chunks.json
   ```

3. The command prints two fingerprints. The published results used:

   | File | sha256 |
   |---|---|
   | prospectus PDF | `6a2f8822f1569a94687935275151c8ac385634d18e96fb37675f479e295979a7` |
   | chunk file (208 chunks: 132 policy, 76 premium-table; 71 sections) | `74aa551cf3df001813443c3d5c83279b36b57dd25a6f7092e185537725376e56` |

   HDFC ERGO updates its documents. A different PDF gives a different fingerprint and possibly different chunks;
   the system still works, but the published results then do not describe your copy exactly, and `policy-rag
   evaluate` reports any test-set facts it cannot find in your chunk file. With the same PDF but a different
   chunk fingerprint, check the `pdfplumber` and `tiktoken` versions against
   [`requirements-lock.txt`](../requirements-lock.txt).

`configs/default.yaml` expects the file at `data/policy_chunks.json`. To use another path, change
`chunks:` under `documents` in the config.

## What else stays out of git

Results folders (`runs/`), answer caches and results zips contain answers and passages from the policy: they are
ignored by git too. The published results in [`results/`](../results/) hold scores only.
