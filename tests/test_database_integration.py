"""Against a real Postgres + pgvector (opt-in): the stored-vector search, its start-up checks, the read-only session and
the reconnect after a dropped connection.

Runs only when POLICY_RAG_TEST_DATABASE_URL is set to a database where the ``rag`` schema (migration 001) is applied,
with a login that may write and delete there (a development database: never production). The test adds an invented
document ``pytest-sample`` with stand-in vectors and removes everything it added, before and after.
"""
import json
import os

import numpy as np
import pytest

from policy_rag.config import DatabaseConfig
from policy_rag.corpus import load_corpus
from policy_rag.database import Database, DatabaseError, PgVectorDenseIndex, chunk_uid, vector_literal
from policy_rag.retrieval import DenseIndex

from conftest import DATA
from test_database import QUESTIONS, DenseFake

ENV = "POLICY_RAG_TEST_DATABASE_URL"
DOC = "pytest-sample"
KEY = "pytest-fake-dense@abc123def456:float32:32"

pytestmark = pytest.mark.skipif(not os.environ.get(ENV), reason=f"{ENV} is not set (opt-in database test)")


def _cleanup(con):
    con.execute("DELETE FROM rag.chunk_embeddings WHERE model_key = %s", (KEY,))
    con.execute("DELETE FROM rag.embedding_models WHERE model_key = %s", (KEY,))
    con.execute("DELETE FROM rag.policy_versions WHERE doc_id = %s", (DOC,))     # chunks go with it
    con.execute("DELETE FROM rag.documents WHERE doc_id = %s", (DOC,))


@pytest.fixture(scope="module")
def stored():
    psycopg = pytest.importorskip("psycopg")
    corpus = load_corpus([{"id": DOC, "chunks": str(DATA / "sample_policy_chunks.json"), "name": "Sample"}])
    emb = DenseFake()
    vecs = emb.encode_documents([c.embed_text for c in corpus.chunks])
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    with psycopg.connect(os.environ[ENV], autocommit=True) as con:
        if not con.execute("SELECT to_regclass('rag.chunk_embeddings') IS NOT NULL").fetchone()[0]:
            pytest.skip("the rag schema (migration 001) is not applied in this database")
        _cleanup(con)
        with con.transaction():
            con.execute("INSERT INTO rag.documents (doc_id, title, insurer, metadata) VALUES (%s, 'Sample', 'Test', '{}')",
                        (DOC,))
            vid = con.execute(
                "INSERT INTO rag.policy_versions (doc_id, version_label, source_file, source_sha256, pages, pipeline, "
                "status, activated_at) VALUES (%s, 'v1', 'sample_policy_chunks.json', %s, 1, '{}', 'active', now()) "
                "RETURNING version_id", (DOC, "0" * 64)).fetchone()[0]
            for c in corpus.chunks:
                record = {"idx": c.idx, "kind": "text", "index": c.index, "section_id": c.section_id,
                          "section_title": c.section_title, "clause": c.clause, "label": c.label, "group": c.unit,
                          "pages": [1], "block_ids": [], "token_count": len(c.text.split()), "text": c.text,
                          "embed_text": c.embed_text}
                con.execute("INSERT INTO rag.chunks (version_id, chunk_uid, record) VALUES (%s, %s, %s)",
                            (vid, chunk_uid(DOC, c.embed_text), json.dumps(record)))
            con.execute("INSERT INTO rag.embedding_models (model_key, model, revision, dim, dtype, pooling, "
                        "query_instruction, normalized) VALUES (%s, %s, 'abc123def456', 32, 'float32', 'sum', '', true)",
                        (KEY, emb.name))
            for c, v in zip(corpus.chunks, vecs):
                con.execute("INSERT INTO rag.chunk_embeddings (chunk_uid, model_key, embed_text_sha256, embedding) "
                            "VALUES (%s, %s, %s, %s::vector)", (chunk_uid(DOC, c.embed_text), KEY, "0" * 64, vector_literal(v)))
    yield corpus, emb
    with psycopg.connect(os.environ[ENV], autocommit=True) as con:
        _cleanup(con)


def test_real_database_ranks_like_faiss(stored):
    corpus, emb = stored
    pg = PgVectorDenseIndex(emb, corpus, DatabaseConfig(url_env=ENV, model_key=KEY))
    mem = DenseIndex(emb, [c.embed_text for c in corpus.chunks])
    assert pg.info["chunks"] == len(corpus.chunks) and pg.info["revision_checked"]
    for q in QUESTIONS:
        assert pg.ranking(q) == mem.ranking(q), q


def test_real_database_session_is_read_only_and_survives_a_drop(stored):
    import psycopg
    corpus, emb = stored
    db = Database(DatabaseConfig(url_env=ENV, model_key=KEY, statement_timeout_ms=4321))
    assert db.query("SELECT current_setting('default_transaction_read_only'), current_setting('statement_timeout')") \
        == [("on", "4321ms")]
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        db.query("INSERT INTO rag.documents (doc_id, title, insurer, metadata) VALUES ('pytest-x', 'x', 'x', '{}')")
    pid = db.query("SELECT pg_backend_pid()")[0][0]
    with psycopg.connect(os.environ[ENV], autocommit=True) as other:          # what a Neon suspend does to a session
        assert other.execute("SELECT pg_terminate_backend(%s)", (pid,)).fetchone()[0]
    assert db.query("SELECT 1") == [(1,)] and db.reconnects == 1
    assert db.query("SELECT pg_backend_pid()")[0][0] != pid
    db.close()


def test_real_database_refuses_other_chunk_texts(stored):
    corpus, emb = stored
    other = load_corpus([{"id": DOC, "chunks": str(DATA / "sample_policy_chunks.json"), "name": "Sample"}])
    other.chunks[0].embed_text += " (edited)"
    with pytest.raises(DatabaseError, match="1 only in the database .* 1 only in the chunk files"):
        PgVectorDenseIndex(emb, other, DatabaseConfig(url_env=ENV, model_key=KEY))
