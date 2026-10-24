"""First-stage search: dense (embeddings + FAISS) and lexical (BM25) over the chunks, fused by reciprocal rank.

Nothing is cut here beyond the top-k of each leg: every candidate is scored by the reranker afterwards.
Chunks in a *routed index* (``index`` field, e.g. the premium tables) are searched as a separate index, and only
when the question matches that index's pattern (``retrieval.routed_indexes`` in the config).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .corpus import Corpus


def bm25_tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9.]+", s.lower())          # keeps "4.7.1" and "excl06" intact


class DenseIndex:
    """Exact inner-product search over L2-normalised embeddings (cosine similarity)."""

    def __init__(self, embedder, texts: Sequence[str]):
        import faiss
        self.embedder = embedder
        emb = embedder.encode_documents(list(texts))
        faiss.normalize_L2(emb)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.n = len(texts)

    def ranking(self, query: str) -> List[int]:
        """All chunk ids, most similar first."""
        import faiss
        q = self.embedder.encode_query(query)
        faiss.normalize_L2(q)
        _, ids = self.index.search(q, self.n)
        return [int(i) for i in ids[0] if i != -1]


class BM25Index:
    def __init__(self, texts: Sequence[str]):
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi([bm25_tokenize(t) for t in texts])

    def ranking(self, query: str) -> List[int]:
        return [int(i) for i in np.argsort(-self.bm25.get_scores(bm25_tokenize(query)), kind="stable")]


class HybridRetriever:
    def __init__(self, corpus: Corpus, embedder, cfg):
        self.corpus, self.cfg = corpus, cfg
        texts = [c.embed_text for c in corpus.chunks]
        self.dense = DenseIndex(embedder, texts)
        self.bm25 = BM25Index(texts) if cfg.use_bm25 else None
        present = {c.index for c in corpus.chunks}
        # routed indexes that exist in this corpus; with routing off they are searched like the main index
        self.routed: Dict[str, object] = {name: rc for name, rc in cfg.routed_indexes.items()
                                          if cfg.routing and name in present}
        self._patterns = {name: (re.compile(rc.query_pattern, re.I) if rc.query_pattern else None,
                                 re.compile(rc.exclude_pattern, re.I) if rc.exclude_pattern else None)
                          for name, rc in self.routed.items()}

    def routes(self, query: str) -> List[str]:
        """The routed indexes this question should also search."""
        out = []
        for name, (want, exclude) in self._patterns.items():
            if want is not None and want.search(query) and not (exclude is not None and exclude.search(query)):
                out.append(name)
        return out

    def retrieve(self, query: str, allowed: Optional[Set[int]] = None) -> Tuple[List[int], List[str]]:
        """Candidate chunk ids (fused order) and the routed indexes searched. ``allowed`` restricts the search to
        some chunk ids (e.g. one document)."""
        routes = self.routes(query)
        chunks = self.corpus.chunks
        order_dense = self.dense.ranking(query)
        order_bm25 = self.bm25.ranking(query) if self.bm25 is not None and self.cfg.k_bm25 > 0 else []
        ok = (lambda i: True) if allowed is None else (lambda i: i in allowed)
        in_main = lambda i: chunks[i].index not in self.routed and ok(i)
        lists = [[i for i in order_dense if in_main(i)][:self.cfg.k_dense],
                 [i for i in order_bm25 if in_main(i)][:self.cfg.k_bm25]]
        for name in routes:
            k = self.routed[name].k
            lists += [[i for i in order_dense if chunks[i].index == name and ok(i)][:k],
                      [i for i in order_bm25 if chunks[i].index == name and ok(i)][:k]]
        rrf: Dict[int, float] = {}
        for ranked in lists:
            for rank, i in enumerate(ranked):
                rrf[i] = rrf.get(i, 0.0) + 1.0 / (self.cfg.rrf_k + rank)
        return sorted(rrf, key=lambda i: (-rrf[i], i)), routes
