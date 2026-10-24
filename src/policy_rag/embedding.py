"""Embedders turn text into vectors for dense search.

Interface (what the retriever calls): ``encode_documents(texts) -> float32 array (n, d)``,
``encode_query(text) -> float32 array (1, d)``, ``name``, ``revision()``.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .device import model_revision, resolve_device
from .registry import register


@register("embedder", "sentence_transformers")
class SentenceTransformerEmbedder:
    """Any sentence-transformers model. BGE models expect an instruction in front of the query (``query_prefix``);
    the documents are encoded as they are."""

    def __init__(self, model: str, query_prefix: str = "", batch_size: int = 32, device=None):
        from sentence_transformers import SentenceTransformer
        self.name, self.query_prefix, self.batch_size = model, query_prefix, batch_size
        self.device = resolve_device(device)
        self.model = SentenceTransformer(model, device=self.device)

    @classmethod
    def from_config(cls, cfg, **_):
        return cls(cfg.model, cfg.query_prefix, cfg.batch_size, cfg.device)

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True,
                                 show_progress_bar=False).astype("float32")

    def encode_query(self, text: str) -> np.ndarray:
        return self.model.encode([self.query_prefix + text], convert_to_numpy=True).astype("float32")

    def revision(self):
        return model_revision(self.model, self.name)
