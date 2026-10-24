import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).parent
DATA = TESTS / "data"
ROOT = TESTS.parent
sys.path.insert(0, str(TESTS))          # so that "plugins: [fakes]" can be imported

import fakes  # noqa: E402,F401  (registers the fake component types)

from policy_rag.config import load_config  # noqa: E402


def fake_config(tmp_path, documents=None, **sections):
    """A config that uses the invented sample policy and the fake components."""
    data = {
        "plugins": ["fakes"],
        "documents": documents or [{"id": "sample", "name": "Sample Health Shield",
                                    "chunks": str(DATA / "sample_policy_chunks.json")}],
        "embedder": {"type": "fake_hash"},
        "reranker": {"type": "fake_overlap"},
        "generator": {"type": "fake_extractive"},
        "cache": {"path": str(tmp_path / "cache.jsonl")},
        "evaluation": {"testset": str(DATA / "sample_testset.json"), "output_dir": str(tmp_path / "runs"),
                       "groundedness": False},
        "retrieval": {"k_dense": 8, "k_bm25": 8},
    }
    for k, v in sections.items():
        data.setdefault(k, {}).update(v)
    return load_config(str(ROOT / "configs" / "default.yaml"), data=data)


@pytest.fixture
def cfg(tmp_path):
    return fake_config(tmp_path)


@pytest.fixture
def system(cfg):
    from policy_rag.pipeline import RAGSystem
    return RAGSystem.from_config(cfg, log=lambda *_: None)
