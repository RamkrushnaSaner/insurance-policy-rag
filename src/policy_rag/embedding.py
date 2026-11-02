"""Embedders turn text into vectors for dense search.

Interface (what the retriever calls): ``encode_documents(texts) -> float32 array (n, d)``,
``encode_query(text) -> float32 array (1, d)``, ``name``, ``revision()``.
"""
from __future__ import annotations

import inspect
from typing import List, Optional

import numpy as np

from .device import dtype_kw, model_revision, resolve_device
from .registry import register


@register("embedder", "sentence_transformers")
class SentenceTransformerEmbedder:
    """Any sentence-transformers model. BGE models expect an instruction in front of the query (``query_prefix``);
    the documents are encoded as they are.

    Optional settings (all off by default, so existing configs behave as before): ``revision`` pins the model files,
    ``dtype`` loads the weights in another precision (e.g. ``float16`` on a T4), ``normalize`` returns unit-length
    float32 vectors (normalised again after the conversion from half precision), ``padding_side`` sets the tokenizer's
    padding side (``left`` for last-token pooling models such as Qwen3-Embedding)."""

    def __init__(self, model: str, query_prefix: str = "", batch_size: int = 32, device=None,
                 revision: Optional[str] = None, dtype: Optional[str] = None, normalize: bool = False,
                 padding_side: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        self.name, self.query_prefix, self.batch_size = model, query_prefix, batch_size
        self.normalize, self.pinned_revision = bool(normalize), revision
        self.device = resolve_device(device)
        kwargs = {}
        if revision:
            kwargs["revision"] = revision
        if dtype:
            kwargs["model_kwargs"] = dtype_kw(_torch_dtype(dtype))
        if padding_side:
            params = inspect.signature(SentenceTransformer.__init__).parameters
            # sentence-transformers 5.x: processor_kwargs (tokenizer_kwargs before)
            kwargs["processor_kwargs" if "processor_kwargs" in params else "tokenizer_kwargs"] = {
                "padding_side": padding_side}
        self.model = SentenceTransformer(model, device=self.device, **kwargs)
        tokenizer = getattr(self.model, "tokenizer", None)
        if padding_side and tokenizer is not None and getattr(tokenizer, "padding_side", None) != padding_side:
            tokenizer.padding_side = padding_side

    @classmethod
    def from_config(cls, cfg, **_):
        return cls(cfg.model, cfg.query_prefix, cfg.batch_size, cfg.device, revision=cfg.revision, dtype=cfg.dtype,
                   normalize=cfg.normalize, padding_side=cfg.padding_side)

    def _encode(self, texts: List[str], **kw) -> np.ndarray:
        if self.normalize:
            kw["normalize_embeddings"] = True
        v = self.model.encode(texts, convert_to_numpy=True, **kw).astype("float32")
        if self.normalize:                    # half-precision output: unit length again in float32
            norms = np.linalg.norm(v, axis=-1, keepdims=True)
            v = v / np.where(norms > 0, norms, 1.0)
        return v

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts, batch_size=self.batch_size, show_progress_bar=False)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])

    def revision(self):
        if self.pinned_revision:              # the commit of the loaded files (not the newest one on the Hub)
            try:
                loaded = model_revision(self.model[0], None)
            except (TypeError, IndexError, KeyError):
                loaded = None
            return loaded or self.pinned_revision
        return model_revision(self.model, self.name)


def _torch_dtype(name: str):
    import torch
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"embedder.dtype: {name!r} is not a torch dtype (e.g. float16, bfloat16, float32)")
    return value
