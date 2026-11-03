"""Postgres + pgvector: the stored chunk vectors as the dense search leg (``retrieval.dense_backend: pgvector``).

The database is filled by the ingestion loader (schema ``rag``): text chunks per policy version, one vector per chunk
text per embedding model. This module only reads it:

- ``Database``: one read-only connection, opened on first use and reopened once if it was dropped (Neon suspends an
  idle compute after a few minutes, which closes open connections). The connection string comes from an environment
  variable (``database.url_env``) and is never logged.
- ``PgVectorDenseIndex``: exact inner-product search over the vectors of the active version of each document. It
  ranks the same chunks as the in-memory ``DenseIndex`` when the stored vectors were made by the same embedder.
  At start-up it checks that the database holds exactly the chunks of the loaded chunk files (matched by
  ``chunk_uid``), each with a vector, and that those vectors came from the configured model.

Needs ``psycopg`` (``pip install "policy-rag[db]"``).
"""
from __future__ import annotations

import hashlib
import os
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


def chunk_uid(doc_id: str, embed_text: str) -> str:
    """The id a chunk is stored under: same document and search text -> same id in every version (the loader uses
    the same formula, so vectors are reused when a text does not change)."""
    return hashlib.sha256(f"{doc_id}\0{embed_text}".encode()).hexdigest()[:32]


def vector_literal(v: Sequence[float]) -> str:
    """pgvector text form; 9 significant digits give the float32 value back exactly."""
    return "[" + ",".join(f"{float(x):.9g}" for x in v) + "]"


class DatabaseError(RuntimeError):
    pass


class Database:
    """A read-only connection to the policy database. ``connect`` and ``retry_on`` replace psycopg in tests."""

    def __init__(self, cfg, connect: Optional[Callable] = None, retry_on: Tuple[type, ...] = ()):
        self.cfg = cfg
        if connect is None:
            dsn = os.environ.get(cfg.url_env, "")
            if not dsn:
                raise DatabaseError(f"environment variable {cfg.url_env} is not set: it must hold the Postgres "
                                    "connection string (keep it out of config files)")
            try:
                import psycopg
            except ImportError as e:
                raise DatabaseError('retrieval.dense_backend: pgvector needs psycopg: pip install "policy-rag[db]"') from e
            timeout = cfg.connect_timeout

            def connect():
                return psycopg.connect(dsn, autocommit=True, connect_timeout=timeout, application_name="policy-rag")
            retry_on = (psycopg.OperationalError, psycopg.InterfaceError)
        self._connect, self._retry_on = connect, tuple(retry_on)
        self._con = None
        self.reconnects = 0

    def _open(self):
        con = self._connect()
        con.execute("SELECT set_config('default_transaction_read_only', 'on', false), "
                    "set_config('statement_timeout', %s, false)", (str(int(self.cfg.statement_timeout_ms)),))
        return con

    def query(self, sql: str, params=()) -> List[tuple]:
        """All rows of one read-only query. A dropped connection is reopened and the query run again, once."""
        return self._run(sql, params, lambda cur: list(cur.fetchall()))

    def query_dicts(self, sql: str, params=()) -> List[dict]:
        """Like ``query``, each row as a dict keyed by column name."""
        def rows(cur):
            names = [d[0] for d in cur.description]
            return [dict(zip(names, r)) for r in cur.fetchall()]
        return self._run(sql, params, rows)

    def _run(self, sql, params, read):
        for attempt in (0, 1):
            try:
                if self._con is None or self._con.closed:
                    self._con = self._open()
                return read(self._con.execute(sql, params))
            except self._retry_on:
                self.close()
                if attempt:
                    raise
                self.reconnects += 1
        raise AssertionError("unreachable")

    def close(self):
        con, self._con = self._con, None
        if con is not None:
            try:
                con.close()
            except Exception:
                pass


_MODEL_SQL = "SELECT model, revision, dim, normalized, dtype, query_instruction FROM rag.embedding_models WHERE model_key = %s"
_STORED_SQL = """
SELECT v.doc_id, c.chunk_uid, c.label, e.chunk_uid IS NOT NULL
  FROM rag.active_versions v
  JOIN rag.chunks c ON c.version_id = v.version_id
  LEFT JOIN rag.chunk_embeddings e ON e.chunk_uid = c.chunk_uid AND e.model_key = %s
 WHERE v.doc_id = ANY(%s)"""
_ACTIVE_SQL = "SELECT doc_id, version_id, version_label FROM rag.active_versions WHERE doc_id = ANY(%s)"
_SEARCH_SQL = """
SELECT c.chunk_uid, e.embedding <#> %s::vector AS distance
  FROM rag.active_versions v
  JOIN rag.chunks c ON c.version_id = v.version_id
  JOIN rag.chunk_embeddings e ON e.chunk_uid = c.chunk_uid AND e.model_key = %s
 WHERE v.doc_id = ANY(%s)"""


class PgVectorDenseIndex:
    """Exact inner-product search over unit-length vectors stored in Postgres (= cosine similarity, like the FAISS
    ``DenseIndex``). Every stored vector of the active versions is compared (no approximate index), and the ranking
    covers all chunks, so the retriever can filter it exactly as it filters the in-memory ranking."""

    def __init__(self, embedder, corpus, cfg, db: Optional[Database] = None):
        if not cfg.model_key:
            raise DatabaseError("database.model_key is empty: set it to the key the vectors were stored under")
        self.embedder, self.model_key = embedder, cfg.model_key
        self.db = db if db is not None else Database(cfg)
        self.doc_ids = sorted({c.doc_id for c in corpus.chunks})
        uids = [chunk_uid(c.doc_id, c.embed_text) for c in corpus.chunks]
        if len(set(uids)) != len(uids):
            raise DatabaseError("two chunks of the corpus have the same document and search text")
        self._pos = {u: i for i, u in enumerate(uids)}
        self.n = len(uids)
        self.info = self._check(corpus)
        self.dim = self.info["dim"]

    # ------------------------------------------------------------------ start-up checks
    def _check(self, corpus) -> dict:
        rows = self.db.query(_MODEL_SQL, (self.model_key,))
        if not rows:
            raise DatabaseError(f"no embedding model {self.model_key!r} in rag.embedding_models")
        model, revision, dim, normalized, dtype, instruction = rows[0]
        if not normalized:
            raise DatabaseError(f"the vectors of {self.model_key!r} are not stored at unit length")
        name = getattr(self.embedder, "name", None)
        if name != model:
            raise DatabaseError(f"the stored vectors were made by {model!r}; the configured embedder is {name!r}")
        loaded = self.embedder.revision() if hasattr(self.embedder, "revision") else None
        if loaded and revision and not (loaded.startswith(revision) or revision.startswith(loaded)):
            raise DatabaseError(f"the stored vectors were made by revision {revision[:12]} of {model}; "
                                f"the loaded embedder is revision {loaded[:12]}")
        active = self.db.query(_ACTIVE_SQL, (self.doc_ids,))
        missing_docs = sorted(set(self.doc_ids) - {r[0] for r in active})
        if missing_docs:
            raise DatabaseError(f"no active version in the database for document(s) {missing_docs}")
        stored = self.db.query(_STORED_SQL, (self.model_key, self.doc_ids))
        in_db = {r[1]: str(r[2]) for r in stored}
        only_db = sorted(label for uid, label in in_db.items() if uid not in self._pos)
        only_corpus = sorted(corpus.chunks[i].label for u, i in self._pos.items() if u not in in_db)
        no_vector = sorted(str(r[2]) for r in stored if not r[3])
        if only_db or only_corpus or len(stored) != len(in_db):
            raise DatabaseError(
                f"the database's active chunks differ from the chunk files: {len(only_db)} only in the database "
                f"{only_db[:3]}, {len(only_corpus)} only in the chunk files {only_corpus[:3]} (search texts must be "
                "identical; with several documents the texts carry the document name and must be stored that way)")
        if no_vector:
            raise DatabaseError(f"{len(no_vector)} active chunks have no {self.model_key!r} vector: {no_vector[:3]}")
        return {"model_key": self.model_key, "model": model, "revision": revision, "loaded_revision": loaded,
                "revision_checked": bool(loaded and revision), "dim": int(dim), "dtype": dtype,
                "query_instruction": instruction, "chunks": len(stored),
                "versions": {r[0]: {"version_id": int(r[1]), "version_label": r[2]} for r in active}}

    # ------------------------------------------------------------------ search
    def scored(self, query: str) -> List[Tuple[int, float]]:
        """(chunk id, inner product) for all chunks, most similar first; ties in corpus order."""
        q = np.asarray(self.embedder.encode_query(query), dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise DatabaseError(f"query vector has {q.shape[0]} dimensions; the stored vectors have {self.dim}")
        norm = float(np.linalg.norm(q))
        if not np.isfinite(norm) or norm == 0:
            raise DatabaseError("the query vector is zero or not finite")
        rows = self.db.query(_SEARCH_SQL, (vector_literal(q / norm), self.model_key, self.doc_ids))
        out = [(self._pos[uid], -float(distance)) for uid, distance in rows if uid in self._pos]
        if len(out) != self.n:
            raise DatabaseError(f"the search returned {len(out)} of {self.n} chunks: the database changed since start-up")
        return sorted(out, key=lambda t: (-t[1], t[0]))

    def ranking(self, query: str) -> List[int]:
        """All chunk ids, most similar first."""
        return [i for i, _ in self.scored(query)]
