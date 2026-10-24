"""The RAG pipeline as a LangGraph state graph: retrieve -> rerank -> gate -> answer | no_answer.

``RAGSystem`` builds every component from a config and is the main entry point::

    from policy_rag import RAGSystem
    rag = RAGSystem.from_config("configs/default.yaml")
    result = rag.ask("What is the waiting period for pre-existing diseases?")
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TypedDict

from . import registry
from .config import Config, DocumentConfig, load_config, resolve_input
from .context import ContextBuilder
from .corpus import Corpus, load_corpus
from .device import free_gpu, is_oom
from .generation import Answerer, AnswerCache
from .prompts import NO_ANSWER_TEXT, build_prompt, is_no_answer
from .retrieval import HybridRetriever


class RAGState(TypedDict, total=False):
    query: str
    doc_ids: List[str]          # restrict the search to these documents (None = all)
    candidates: List[int]       # chunk ids: search results + title-matched sections
    routes: List[str]           # routed indexes searched (e.g. ["premium"])
    scores: List[float]         # reranker score of each candidate
    groups: List[List[int]]     # context passages (chunk ids), most relevant first
    top_score: float            # best reranker score (the gate uses it)
    route: str                  # "answer" or "no_answer"
    answer: str
    gen: dict                   # generation record (tokens, seconds, thinking, from_cache)
    error: str
    oom_retry: bool


@dataclass
class Answer:
    question: str
    answer: str
    sources: List[str]
    route: str
    top_score: Optional[float]
    state: dict = field(repr=False)

    @property
    def refused(self) -> bool:
        return is_no_answer(self.answer)


class RAGPipeline:
    def __init__(self, corpus: Corpus, retriever: HybridRetriever, reranker, context: ContextBuilder, answerer,
                 cfg: Config, log=print):
        self.corpus, self.retriever, self.reranker, self.context = corpus, retriever, reranker, context
        self.answerer, self.cfg, self.log = answerer, cfg, log
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ steps (usable on their own)
    def prompt_for(self, query: str, groups) -> str:
        return build_prompt(query, self.context.format(groups), self.cfg.prompt.template, self.corpus.multi_document)

    def retrieve(self, query: str, doc_ids=None):
        query = query.strip()
        allowed = self.corpus.doc_chunk_ids(doc_ids)
        docs = set(doc_ids) if doc_ids else None
        cands, routes = self.retriever.retrieve(query, allowed)
        if self.cfg.context.title_pinning:                     # pinned sections must be scored too
            have = set(cands)
            for key in self.context.title_matches(query, docs):
                group = self.context.pinned_group(key)
                cands += [i for i in group if i not in have]
                have.update(group)
        return cands, routes

    def rerank_text(self, i: int) -> str:
        c = self.corpus.chunks[i]
        return c.embed_text if self.cfg.reranker.use_label else c.text

    def gate(self, state) -> str:
        return "answer" if state.get("groups") and state.get("top_score", float("-inf")) >= self.cfg.gate.min_score \
            else "no_answer"

    # ------------------------------------------------------------------ graph nodes
    def _retrieve_node(self, state: RAGState):
        try:
            cands, routes = self.retrieve(state["query"], state.get("doc_ids"))
            return {"candidates": cands, "routes": routes}
        except Exception as e:
            traceback.print_exc()
            return {"candidates": [], "routes": [], "error": f"retrieve: {type(e).__name__}: {e}"}

    def _rerank_node(self, state: RAGState):
        empty = {"scores": [], "groups": [], "top_score": float("-inf")}
        try:
            cands = state.get("candidates") or []
            if not cands:
                return empty
            scores = self.reranker.score(state["query"], [self.rerank_text(i) for i in cands])
            docs = set(state["doc_ids"]) if state.get("doc_ids") else None
            groups = self.context.build(state["query"], cands, scores, docs)
            return {"scores": [float(s) for s in scores], "groups": groups, "top_score": float(max(scores))}
        except Exception as e:
            traceback.print_exc()
            return {**empty, "error": f"rerank: {type(e).__name__}: {e}"}

    def _answer_node(self, state: RAGState):
        try:
            try:
                gen = self.answerer.answer(self.prompt_for(state["query"], state["groups"]))
                return {"answer": gen["answer"], "gen": gen, "route": "answer"}
            except Exception as e:
                if not is_oom(e):
                    raise
                free_gpu()
                keep = self.context.shorten(state["groups"], self.cfg.generator.oom_retry_context_tokens)
                self.log(f"out of GPU memory for {state['query'][:50]!r}: retried with {sum(map(len, keep))} chunks")
                gen = self.answerer.answer(self.prompt_for(state["query"], keep))
                return {"answer": gen["answer"], "gen": gen, "groups": keep, "oom_retry": True, "route": "answer"}
        except Exception as e:
            traceback.print_exc()
            return {"answer": NO_ANSWER_TEXT, "error": f"answer: {type(e).__name__}: {e}", "route": "answer"}
        finally:
            free_gpu()

    def _no_answer_node(self, state: RAGState):
        return {"answer": NO_ANSWER_TEXT, "route": "no_answer"}

    def _build_graph(self):
        from langgraph.graph import END, StateGraph
        g = StateGraph(RAGState)
        g.add_node("retrieve", self._retrieve_node)
        g.add_node("rerank", self._rerank_node)
        g.add_node("answer", self._answer_node)
        g.add_node("no_answer", self._no_answer_node)
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "rerank")
        g.add_conditional_edges("rerank", self.gate, {"answer": "answer", "no_answer": "no_answer"})
        g.add_edge("answer", END)
        g.add_edge("no_answer", END)
        return g.compile()

    # ------------------------------------------------------------------ running
    def search(self, query: str, doc_ids=None) -> dict:
        """Retrieve, rerank and build the context; no answer model (what retrieval-only evaluation runs)."""
        st: dict = {"query": query, "doc_ids": doc_ids}
        st.update(self._retrieve_node(st))
        st.update(self._rerank_node(st))
        st["route"] = self.gate(st)
        return st

    def run(self, query: str, doc_ids=None) -> dict:
        return self.graph.invoke({"query": query, "doc_ids": list(doc_ids) if doc_ids else None})


class RAGSystem:
    """All components built from one config."""

    def __init__(self, cfg: Config, corpus: Corpus, embedder, reranker, generator, cache: AnswerCache, log=print):
        self.cfg, self.corpus, self.embedder, self.reranker, self.generator, self.cache = \
            cfg, corpus, embedder, reranker, generator, cache
        self.log = log
        self.retriever = HybridRetriever(corpus, embedder, cfg.retrieval)
        self.context = ContextBuilder(corpus, cfg.context, generator.count_tokens)
        self.answerer = Answerer(generator, cache)
        self.pipeline = RAGPipeline(corpus, self.retriever, reranker, self.context, self.answerer, cfg, log)

    @classmethod
    def from_config(cls, config="configs/default.yaml", overrides=None, corpus: Optional[Corpus] = None,
                    log=print) -> "RAGSystem":
        """Build everything: load the chunk files, the embedder and reranker, index all chunks. The answer model
        loads on the first answer that is not in the cache."""
        cfg = config if isinstance(config, Config) else load_config(config, overrides)
        if corpus is None:
            corpus = load_corpus([DocumentConfig(d.id, str(resolve_input(d.chunks, cfg)), d.name) for d in cfg.documents])
            log(corpus.summary())
        embedder = registry.build("embedder", cfg.embedder)
        reranker = registry.build("reranker", cfg.reranker)
        generator = registry.build("generator", cfg.generator)          # weights load on first use
        cache = AnswerCache(cfg.cache.path, cfg.cache.enabled)
        system = cls(cfg, corpus, embedder, reranker, generator, cache, log)
        routed = ", ".join(system.retriever.routed) or "none"
        log(f"Ready: {len(corpus.chunks)} chunks indexed (dense + {'BM25' if cfg.retrieval.use_bm25 else 'no BM25'}); "
            f"routed indexes: {routed}; answer cache: {len(cache)} answers")
        return system

    def ask(self, question: str, documents=None, show_sources: bool = True) -> Answer:
        """One question through the pipeline (one answer-model call unless the answer is cached or refused)."""
        st = self.pipeline.run(question, documents)
        sources = [self.context.group_label(g) for g in st.get("groups", [])]
        if show_sources:
            for s in sources:
                print("  source:", s[:120])
            print("\nQ:", question, "\nA:", st["answer"])
        top = st.get("top_score")
        return Answer(question, st["answer"], sources, st.get("route") or self.pipeline.gate(st),
                      top if top is not None and top != float("-inf") else None, st)
