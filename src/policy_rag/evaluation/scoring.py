"""Scoring of saved question records: checks + retrieval metrics, with the current test set and chunk file(s)."""
from __future__ import annotations

from typing import Optional

from ..context import ContextBuilder
from ..corpus import Corpus
from .checks import retrieval_metrics, score_answer
from .table_checks import get_table_checker
from .testset import CorpusText, TestSet, norm_text


class Scorer:
    def __init__(self, testset: TestSet, corpus: Corpus, context: ContextBuilder, doc_id: Optional[str] = None):
        """``context`` provides token counts for context precision; ``doc_id`` = the document the test set is about
        (several documents loaded), used to tell a corpus gap from a retrieval miss."""
        self.testset, self.corpus, self.context = testset, corpus, context
        self.by_id = testset.by_id
        self.text = CorpusText(corpus, doc_id)
        self.table_checker = get_table_checker(testset.table_checker)

    def score_record(self, rec: dict, live: bool = True) -> dict:
        """Adds rec["retrieval"] (when the record's chunk ids refer to this corpus) and rec["verdict"]."""
        item = self.by_id[rec["id"]]
        chunks = self.corpus.chunks
        same_ids = live and rec.get("candidates") is not None and rec.get("context_source") is None
        if same_ids:
            rec["retrieval"] = retrieval_metrics(item, rec["candidates"], [i for i, _ in rec["ranked"]],
                                                 [i for g in rec["groups"] for i in g], self.text, self.context.block_tokens)
        if rec.get("answer") is not None:
            cand = norm_text(" ".join(chunks[i].text for i in rec["candidates"])) \
                if same_ids and rec.get("candidates") else None
            rec["verdict"] = score_answer(item, rec["answer"], rec.get("context", ""), self.text.norm, rec.get("gen"),
                                          rec.get("gate_route") or "answer", rec.get("error"), cand, self.table_checker)
        return rec

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
                rec["verdict"] = score_answer(item, rec["answer"], rec.get("context", ""), self.text.norm, rec.get("gen"),
                                              rec.get("gate_route") or "answer", rec.get("error"),
                                              table_checker=self.table_checker)
        run["records"] = [r for r in run["records"] if r["id"] in self.by_id]
        return run
