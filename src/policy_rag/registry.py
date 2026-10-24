"""Component registry: the ``type`` field in the config picks the implementation.

Built-in types: embedder ``sentence_transformers``; reranker ``cross_encoder``; generator ``hf_causal_lm``.
To try another model family or service, write a class with the same methods, decorate it with
``@register("generator", "my_type")``, import it before building the system, and set ``generator.type: my_type``.
"""
from __future__ import annotations

import importlib
from typing import Callable, Dict

_REGISTRY: Dict[str, Dict[str, Callable]] = {"embedder": {}, "reranker": {}, "generator": {}}
_BUILTIN_MODULES = {"embedder": "policy_rag.embedding", "reranker": "policy_rag.reranking",
                    "generator": "policy_rag.generation"}


def register(kind: str, name: str):
    def deco(cls):
        _REGISTRY.setdefault(kind, {})[name] = cls
        return cls
    return deco


def build(kind: str, cfg, **kwargs):
    """Instantiate the component of ``kind`` named by ``cfg.type`` with ``cls.from_config(cfg, **kwargs)``."""
    if cfg.type not in _REGISTRY.get(kind, {}) and kind in _BUILTIN_MODULES:
        importlib.import_module(_BUILTIN_MODULES[kind])
    try:
        cls = _REGISTRY[kind][cfg.type]
    except KeyError:
        raise KeyError(f"unknown {kind} type {cfg.type!r}; registered: {sorted(_REGISTRY.get(kind, {}))}") from None
    return cls.from_config(cfg, **kwargs)


def registered(kind: str):
    if kind in _BUILTIN_MODULES:
        importlib.import_module(_BUILTIN_MODULES[kind])
    return sorted(_REGISTRY.get(kind, {}))
