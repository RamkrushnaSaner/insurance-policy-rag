"""Stored vectors as the dense search leg (retrieval.dense_backend: pgvector), without a database server: a small
in-memory stand-in answers the four queries the index sends. The new embedder settings are tested with a stand-in
sentence-transformers module. A real Postgres is used only by test_database_integration.py (opt-in)."""
import hashlib
import re
import sys
import types

import numpy as np
import pytest

from policy_rag import database as dbmod
from policy_rag.config import ConfigError, DatabaseConfig, load_config
from policy_rag.corpus import load_corpus
from policy_rag.database import Database, DatabaseError, PgVectorDenseIndex, chunk_uid, vector_literal
from policy_rag.retrieval import DenseIndex, HybridRetriever

from conftest import DATA, ROOT, fake_config

MODEL_KEY = "fake-dense@test:float32:32"


class DenseFake:
    """Continuous vectors (sum of a fixed random vector per word), so rankings have no ties."""
    name = "fake-dense"

    def __init__(self, dim=32, revision="abc123def456"):
        self.dim, self._revision = dim, revision

    def _vec(self, text):
        v = np.zeros(self.dim, dtype="float32")
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            seed = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
            v += np.random.default_rng(seed).standard_normal(self.dim).astype("float32")
        return v

    def encode_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def encode_query(self, text):
        return self._vec(text)[None, :]

    def revision(self):
        return self._revision


class FakeStore:
    """What the loader writes, held in memory; answers the index's queries like Postgres would."""

    def __init__(self, corpus, embedder, model_key=MODEL_KEY, model=None, revision="abc123def456", normalized=True):
        self.models = {model_key: (model or embedder.name, revision, embedder.dim, normalized, "float32", "q")}
        self.active = {d: (7, "v1") for d in corpus.documents}
        vecs = embedder.encode_documents([c.embed_text for c in corpus.chunks])
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        self.chunks = [(c.doc_id, chunk_uid(c.doc_id, c.embed_text), c.label) for c in corpus.chunks]
        self.vectors = {(chunk_uid(c.doc_id, c.embed_text), model_key): v for c, v in zip(corpus.chunks, vecs)}
        self.sql = []

    def query(self, sql, params=()):
        self.sql.append(sql)
        if sql == dbmod._MODEL_SQL:
            return [self.models[params[0]]] if params[0] in self.models else []
        if sql == dbmod._ACTIVE_SQL:
            return [(d, *self.active[d]) for d in params[0] if d in self.active]
        if sql == dbmod._STORED_SQL:
            key, docs = params
            return [(d, u, label, (u, key) in self.vectors) for d, u, label in self.chunks if d in docs and d in self.active]
        if sql == dbmod._SEARCH_SQL:
            literal, key, docs = params
            q = np.array([float(x) for x in literal.strip("[]").split(",")], dtype="float32")
            rows = [(u, -float(self.vectors[(u, key)] @ q)) for d, u, _ in self.chunks
                    if d in docs and d in self.active and (u, key) in self.vectors]
            return sorted(rows, key=lambda r: r[1])
        raise AssertionError(f"unexpected query: {sql}")


@pytest.fixture
def corpus():
    return load_corpus([{"id": "sample", "chunks": str(DATA / "sample_policy_chunks.json"), "name": "Sample"}])


QUESTIONS = ["What is the waiting period for kidney stones?", "Explain the Wellness Bonus",
             "What is the annual premium for age 40?", "Are children covered?", "room rent limit",
             "Xylophone tuition reimbursement?"]


# ---------------------------------------------------------------------------------------------------- config
def test_new_settings_default_to_the_old_behaviour():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert cfg.retrieval.dense_backend == "faiss"
    e = cfg.embedder
    assert (e.revision, e.dtype, e.normalize, e.padding_side) == (None, None, False, None)
    assert cfg.database == DatabaseConfig()
    assert "DATABASE_URL" == cfg.database.url_env


def test_qwen3_experiment_configs():
    faiss_cfg = load_config(ROOT / "configs" / "experiments" / "qwen3_text_faiss.yaml")
    pg_cfg = load_config(ROOT / "configs" / "experiments" / "qwen3_text_pgvector.yaml")
    e = pg_cfg.embedder
    assert e.model == "Qwen/Qwen3-Embedding-0.6B" and e.revision.startswith("97b0c614be4d")
    assert (e.dtype, e.normalize, e.padding_side, e.batch_size) == ("float16", True, "left", 16)
    assert e.query_prefix.endswith("retrieve the passages that answer it\nQuery:")
    assert faiss_cfg.retrieval.dense_backend == "faiss" and pg_cfg.retrieval.dense_backend == "pgvector"
    assert pg_cfg.database.model_key == "qwen3-embedding-0.6b@97b0c614be4d:fp16:1024"
    assert faiss_cfg.embedder == pg_cfg.embedder and faiss_cfg.documents == pg_cfg.documents
    assert pg_cfg.documents[0].chunks == "data/policy_text_chunks.json"


def test_bad_database_settings_are_errors():
    with pytest.raises(ConfigError, match="dense_backend"):
        load_config(ROOT / "configs" / "default.yaml", ["retrieval.dense_backend=qdrant"])
    with pytest.raises(ConfigError, match="model_key"):
        load_config(ROOT / "configs" / "default.yaml", ["retrieval.dense_backend=pgvector"])
    with pytest.raises(ConfigError, match="padding_side"):
        load_config(ROOT / "configs" / "default.yaml", ["embedder.padding_side=middle"])
    with pytest.raises(ConfigError, match="unknown setting"):
        load_config(ROOT / "configs" / "default.yaml", ["database.url=postgresql://x"])


# ---------------------------------------------------------------------------------------------------- search
def test_chunk_uid_and_vector_literal_match_the_loader():
    assert chunk_uid("optima-secure", "a: b") == hashlib.sha256(b"optima-secure\x00a: b").hexdigest()[:32]
    v = np.random.default_rng(0).standard_normal(16).astype("float32")
    back = np.array([float(x) for x in vector_literal(v).strip("[]").split(",")], dtype="float32")
    assert np.array_equal(back, v)                      # float32 values survive the text form exactly


def test_stored_vectors_rank_like_faiss(corpus):
    emb = DenseFake()
    store = FakeStore(corpus, emb)
    pg = PgVectorDenseIndex(emb, corpus, DatabaseConfig(model_key=MODEL_KEY), db=store)
    mem = DenseIndex(emb, [c.embed_text for c in corpus.chunks])
    assert pg.n == len(corpus.chunks) and pg.dim == 32 and pg.info["revision_checked"]
    assert pg.info["versions"] == {"sample": {"version_id": 7, "version_label": "v1"}}
    for q in QUESTIONS:
        assert pg.ranking(q) == mem.ranking(q), q
        scores = [s for _, s in pg.scored(q)]
        assert scores == sorted(scores, reverse=True)


def test_hybrid_retriever_gives_the_same_candidates(corpus, monkeypatch):
    emb = DenseFake()
    cfg = load_config(ROOT / "configs" / "default.yaml",
                      ["retrieval.k_dense=8", "retrieval.k_bm25=8"]).retrieval
    mem = HybridRetriever(corpus, emb, cfg)
    store = FakeStore(corpus, emb)
    monkeypatch.setattr(dbmod, "Database", lambda db_cfg: store)
    cfg.dense_backend = "pgvector"
    pg = HybridRetriever(corpus, emb, cfg, DatabaseConfig(model_key=MODEL_KEY))
    assert isinstance(pg.dense, PgVectorDenseIndex)
    for q in QUESTIONS:
        assert pg.retrieve(q) == mem.retrieve(q), q


def test_whole_system_runs_on_stored_vectors(tmp_path, monkeypatch):
    """The fake components end to end, dense leg from the (stand-in) database."""
    from policy_rag.pipeline import RAGSystem
    import fakes
    base = RAGSystem.from_config(fake_config(tmp_path), log=lambda *_: None)
    holder = {}

    def fake_database(db_cfg):
        holder["cfg"] = db_cfg
        emb = fakes.HashEmbedder()
        emb.dim = 64
        return FakeStore(base.corpus, emb, model_key="fake-hash@x:float32:64", revision="")
    monkeypatch.setattr(dbmod, "Database", fake_database)
    cfg = fake_config(tmp_path, retrieval={"dense_backend": "pgvector"}, database={"model_key": "fake-hash@x:float32:64"})
    logs = []
    pg = RAGSystem.from_config(cfg, log=logs.append)
    assert holder["cfg"].model_key == "fake-hash@x:float32:64" and "dense from pgvector" in logs[-1]
    r = pg.ask("What is the waiting period for kidney stones?", show_sources=False)
    assert r.route == "answer" and "24 months" in r.answer
    assert pg.ask("Xylophone tuition reimbursement?", show_sources=False).route == "no_answer"
    # hashed bag-of-words vectors tie a lot, so compare the dense scores: same chunks at each score
    import faiss
    for q in QUESTIONS:
        v = base.embedder.encode_query(q)
        faiss.normalize_L2(v)
        scores, ids = base.retriever.dense.index.search(v, len(base.corpus.chunks))
        assert tie_groups(zip(ids[0], scores[0])) == tie_groups(pg.retriever.dense.scored(q)), q


def tie_groups(pairs, places=5):
    groups = {}
    for i, s in pairs:
        groups.setdefault(round(float(s), places), set()).add(int(i))
    return groups


@pytest.mark.parametrize("change, message", [
    (lambda s: s.chunks.pop(), "only in the chunk files"),
    (lambda s: s.chunks.append(("sample", "f" * 32, "Extra chunk")), "only in the database"),
    (lambda s: s.vectors.pop(next(iter(s.vectors))), "have no"),
    (lambda s: s.active.clear(), "no active version"),
    (lambda s: s.models.clear(), "no embedding model"),
    (lambda s: s.models.update({MODEL_KEY: ("other/model", "abc123def456", 32, True, "float32", "q")}), "made by 'other/model'"),
    (lambda s: s.models.update({MODEL_KEY: ("fake-dense", "999999999999", 32, True, "float32", "q")}), "revision 999999999999"),
    (lambda s: s.models.update({MODEL_KEY: ("fake-dense", "abc123def456", 32, False, "float32", "q")}), "unit length"),
])
def test_start_up_refuses_a_database_that_does_not_match(corpus, change, message):
    emb = DenseFake()
    store = FakeStore(corpus, emb)
    change(store)
    with pytest.raises(DatabaseError, match=message):
        PgVectorDenseIndex(emb, corpus, DatabaseConfig(model_key=MODEL_KEY), db=store)


def test_wrong_query_dimensions_and_later_changes_are_caught(corpus):
    emb = DenseFake()
    store = FakeStore(corpus, emb)
    pg = PgVectorDenseIndex(emb, corpus, DatabaseConfig(model_key=MODEL_KEY), db=store)
    pg.embedder = DenseFake(dim=16)
    with pytest.raises(DatabaseError, match="16 dimensions"):
        pg.ranking("hernia")
    pg.embedder = emb
    store.vectors.pop(next(iter(store.vectors)))
    with pytest.raises(DatabaseError, match="changed since start-up"):
        pg.ranking("hernia")


def test_empty_model_key_is_refused(corpus):
    with pytest.raises(DatabaseError, match="model_key"):
        PgVectorDenseIndex(DenseFake(), corpus, DatabaseConfig(), db=FakeStore(corpus, DenseFake()))


# ---------------------------------------------------------------------------------------------------- connection
class Dropped(Exception):
    pass


class FakeConnection:
    def __init__(self, log, fail_first_query=False):
        self.log, self.closed, self.fail = log, False, fail_first_query

    def execute(self, sql, params=()):
        self.log.append(sql)
        if self.fail and not sql.startswith("SELECT set_config"):
            self.fail, self.closed = False, True
            raise Dropped("server closed the connection unexpectedly")
        return types.SimpleNamespace(fetchall=lambda: [(1,)])

    def close(self):
        self.closed = True


def test_connection_is_read_only_and_reopened_once_after_a_drop():
    log, opened = [], []

    def connect():
        opened.append(FakeConnection(log, fail_first_query=not opened))
        return opened[-1]
    db = Database(DatabaseConfig(statement_timeout_ms=1234), connect=connect, retry_on=(Dropped,))
    assert db.query("SELECT 1") == [(1,)]
    assert len(opened) == 2 and db.reconnects == 1 and opened[0].closed
    assert sum("default_transaction_read_only" in s for s in log) == 2      # every new session is read-only
    assert db.query("SELECT 1") == [(1,)] and len(opened) == 2               # the connection is kept


def test_a_second_drop_is_raised():
    def connect():
        return FakeConnection([], fail_first_query=True)
    db = Database(DatabaseConfig(), connect=connect, retry_on=(Dropped,))
    with pytest.raises(Dropped):
        db.query("SELECT 1")
    assert db.reconnects == 1 and db._con is None


def test_missing_environment_variable_is_explained(monkeypatch):
    monkeypatch.delenv("POLICY_RAG_TEST_NO_SUCH_VAR", raising=False)
    with pytest.raises(DatabaseError, match="POLICY_RAG_TEST_NO_SUCH_VAR is not set"):
        Database(DatabaseConfig(url_env="POLICY_RAG_TEST_NO_SUCH_VAR"))


# ---------------------------------------------------------------------------------------------------- embedder
class StubSentenceTransformer:
    """Records how it was built; encodes to fixed half-precision vectors of length 2."""
    made = []

    def __init__(self, model, device=None, revision=None, model_kwargs=None, processor_kwargs=None):
        self.kwargs = {"model": model, "device": device, "revision": revision, "model_kwargs": model_kwargs,
                       "processor_kwargs": processor_kwargs}
        self.tokenizer = types.SimpleNamespace(padding_side="right")
        self.calls = []
        StubSentenceTransformer.made.append(self)

    def encode(self, texts, **kw):
        self.calls.append(kw)
        return np.array([[3.0, 4.0]] * len(texts), dtype=np.float16)


@pytest.fixture
def stub_st(monkeypatch):
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = StubSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    torch = types.ModuleType("torch")
    torch.dtype = type("dtype", (), {})
    torch.float16 = torch.dtype()
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(__version__="5.20.1"))
    StubSentenceTransformer.made.clear()
    return torch


def test_embedder_defaults_are_unchanged(stub_st):
    from policy_rag.embedding import SentenceTransformerEmbedder
    e = SentenceTransformerEmbedder("bge", "prefix: ", 32, "cpu")
    st = StubSentenceTransformer.made[-1]
    assert {k: v for k, v in st.kwargs.items() if v is not None} == {"model": "bge", "device": "cpu"}
    d = e.encode_documents(["a", "b"])
    assert d.dtype == np.float32 and np.allclose(d, [[3, 4], [3, 4]])          # not normalised
    assert st.calls[-1] == {"convert_to_numpy": True, "batch_size": 32, "show_progress_bar": False}
    e.encode_query("q")
    assert st.calls[-1] == {"convert_to_numpy": True}


def test_embedder_qwen3_settings(stub_st):
    from policy_rag.embedding import SentenceTransformerEmbedder
    from policy_rag.config import EmbedderConfig
    cfg = EmbedderConfig(model="Qwen/Qwen3-Embedding-0.6B", query_prefix="Q:", batch_size=16, device="cpu",
                         revision="97b0c614be4d", dtype="float16", normalize=True, padding_side="left")
    e = SentenceTransformerEmbedder.from_config(cfg)
    st = StubSentenceTransformer.made[-1]
    assert st.kwargs["revision"] == "97b0c614be4d"
    assert st.kwargs["model_kwargs"] == {"dtype": stub_st.float16}
    assert st.kwargs["processor_kwargs"] == {"padding_side": "left"}
    assert st.tokenizer.padding_side == "left"
    q = e.encode_query("hernia")
    assert q.dtype == np.float32 and np.allclose(q, [[0.6, 0.8]]) and abs(np.linalg.norm(q) - 1) < 1e-7
    assert st.calls[-1]["normalize_embeddings"] is True
    assert e.revision() == "97b0c614be4d"                 # the pinned commit, not the newest on the Hub


def test_embedder_older_libraries(stub_st, monkeypatch):
    """sentence-transformers before 5.x takes tokenizer_kwargs; transformers before 4.56 takes torch_dtype."""
    from policy_rag.embedding import SentenceTransformerEmbedder

    class OldST(StubSentenceTransformer):
        def __init__(self, model, device=None, revision=None, model_kwargs=None, tokenizer_kwargs=None):
            super().__init__(model, device, revision, model_kwargs)
            self.kwargs["tokenizer_kwargs"] = tokenizer_kwargs
    sys.modules["sentence_transformers"].SentenceTransformer = OldST
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(__version__="4.51.3"))
    SentenceTransformerEmbedder("m", device="cpu", dtype="float16", padding_side="left")
    st = StubSentenceTransformer.made[-1]
    assert st.kwargs["tokenizer_kwargs"] == {"padding_side": "left"}
    assert st.kwargs["model_kwargs"] == {"torch_dtype": stub_st.float16}


def test_embedder_rejects_an_unknown_dtype(stub_st):
    from policy_rag.embedding import SentenceTransformerEmbedder
    with pytest.raises(ValueError, match="not a torch dtype"):
        SentenceTransformerEmbedder("m", device="cpu", dtype="float17")
