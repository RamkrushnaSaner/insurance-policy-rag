"""Scoring of saved question records: checks + retrieval metrics, with the current test set and chunk file(s)."""
from __future__ import annotations

from typing import Optional

from ..context import ContextBuilder
from ..corpus import Corpus
from .checks import retrieval_metrics, score_answer
from .table_checks import get_table_checker
from .testset import CorpusText, TestSet, norm_text


class Scorer:
    def __init__(self, testset: TestSet, corpus: Corpus, context: ContextBuilder, doc_id: Optional[str] = None,
                 table_text: str = ""):
        """``context`` provides token counts for context precision; ``doc_id`` = the document the test set is about
        (several documents loaded), used to tell a corpus gap from a retrieval miss; ``table_text`` = every printed
        table row (with the table route on, the tables are part of what the system can answer from)."""
        self.testset, self.corpus, self.context = testset, corpus, context
        self.by_id = testset.by_id
        self.text = CorpusText(corpus, doc_id, table_text)
        self.table_checker = get_table_checker(testset.table_checker)

    def score_record(self, rec: dict, live: bool = True) -> dict:
        """Adds rec["retrieval"] (when the record's chunk ids refer to this corpus) and rec["verdict"]."""
        item = self.by_id[rec["id"]]
        chunks = self.corpus.chunks
        same_ids = live and rec.get("candidates") is not None and rec.get("context_source") is None
        table_text = "\n".join(t for t in (rec.get("facts"), rec.get("table_evidence")) if t)
        if same_ids:
            rec["retrieval"] = retrieval_metrics(item, rec["candidates"], [i for i, _ in rec["ranked"]],
                                                 [i for g in rec["groups"] for i in g], self.text, self.context.block_tokens,
                                                 table_text)
        if rec.get("answer") is not None:
            cand = norm_text(" ".join([chunks[i].text for i in rec["candidates"]] + [table_text])) \
                if same_ids and (rec.get("candidates") or table_text) else None
            rec["verdict"] = score_answer(item, rec["answer"], rec.get("context", ""), self._document(table_text),
                                          rec.get("gen"), rec.get("gate_route") or "answer", rec.get("error"), cand,
                                          self.table_checker)
        return rec

    def _document(self, table_text: str) -> str:
        """The document's normalised text, plus a table passage written from it (amounts as "Rs. 30 lakh" where the
        table prints "30,00,000"), so a fact in the passage is not counted as missing from the document."""
        return self.text.norm + " " + norm_text(table_text) if table_text else self.text.norm

    def rescore_run(self, run: dict) -> dict:
        """Re-score a saved run with these checks (so runs are compared under the same rules)."""
        same_corpus = run.get("chunk_sha256") == self.corpus.fingerprint
        for rec in run["records"]:
            if rec["id"] not in self.by_id:
                continue
            item = self.by_id[rec["id"]]
            rec.update({k: rec.get(k, item[k]) for k in ("type", "answerable", "paraphrase")})
            rec["split"] = item["split"]
            if same_corpus and rec.get("candidates") is not None and rec.get("context_source") is None:
                self.score_record(rec, live=True)
            elif rec.get("answer") is not None:
                table_text = "\n".join(t for t in (rec.get("facts"), rec.get("table_evidence")) if t)
                rec["verdict"] = score_answer(item, rec["answer"], rec.get("context", ""), self._document(table_text),
                                              rec.get("gen"), rec.get("gate_route") or "answer", rec.get("error"),
                                              table_checker=self.table_checker)
        run["records"] = [r for r in run["records"] if r["id"] in self.by_id]
        return run
