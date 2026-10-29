"""Grounded question answering over insurance policy documents (retrieval-augmented generation)."""
__version__ = "3.3.1"

from .config import Config, load_config  # noqa: E402,F401
from .corpus import Corpus, load_corpus  # noqa: E402,F401
from .memory import Conversation  # noqa: E402,F401
from .prompts import NO_ANSWER_TEXT, is_no_answer  # noqa: E402,F401


def __getattr__(name):            # heavy imports (LangGraph, models) only when used
    if name in ("RAGSystem", "RAGPipeline", "Answer"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(name)
