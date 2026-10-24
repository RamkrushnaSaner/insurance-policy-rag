"""Rerankers score every (question, passage) pair; the scores order the candidates, fill the context and drive the
relevance gate.

Interface: ``score(query, docs) -> float32 array`` in the order of ``docs`` (higher = more relevant), ``name``,
``revision()``. The gate and the context floor are thresholds on this score, so a reranker with a different
score scale needs new thresholds (``gate.min_score``, ``context.seed_score_floor``).
"""
from __future__ import annotations

from typing import List

import numpy as np

from .device import dtype_kw, model_revision, resolve_device
from .registry import register


@register("reranker", "cross_encoder")
class CrossEncoderReranker:
    """A Hugging Face sequence-classification cross-encoder (e.g. BAAI/bge-reranker-v2-m3); the score is the raw
    logit. Runs in fp16 on a GPU, fp32 on a CPU."""

    def __init__(self, model: str, max_length: int = 1024, batch_size: int = 8, device=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.name, self.max_length, self.batch_size = model, max_length, batch_size
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model, **dtype_kw(torch.float16 if self.device == "cuda" else torch.float32)).to(self.device).eval()

    @classmethod
    def from_config(cls, cfg, **_):
        return cls(cfg.model, cfg.max_length, cfg.batch_size, cfg.device)

    def score(self, query: str, docs: List[str]) -> np.ndarray:
        """Docs are batched by length (less padding); scores come back in the input order."""
        import torch
        order = sorted(range(len(docs)), key=lambda j: len(docs[j]))
        scores = np.zeros(len(docs), dtype=np.float32)
        for i in range(0, len(order), self.batch_size):
            idx = order[i:i + self.batch_size]
            enc = self.tokenizer([query] * len(idx), [docs[j] for j in idx], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                scores[idx] = self.model(**enc).logits.float().view(-1).cpu().numpy()
        return scores

    def revision(self):
        return model_revision(self.model, self.name)
