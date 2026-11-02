"""The RAG pipeline as a LangGraph state graph:
condense -> retrieve -> tables -> rerank -> gate -> answer | no_answer.

``condense`` acts only when earlier turns of a conversation are passed (see memory.py); otherwise the question goes
straight to retrieval. ``tables`` acts only with ``tables.enabled``: a question about the policy's printed tables
gets the exact values from the database (tables.py) as the first context passage, and the text chunks linked to
those tables become search candidates. ``RAGSystem`` builds every component from a config and is the main entry
point::

    from policy_rag import RAGSystem
    rag = RAGSystem.from_config("configs/default.yaml")
    result = rag.ask("What is the waiting period for pre-existing diseases?")

    chat = rag.new_conversation()                       # follow-up questions
    rag.ask("What is the waiting period for kidney stones?", conversation=chat)
    rag.ask("Does it apply after an accident?", conversation=chat)
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
from .memory import Conversation, QuestionRewriter
from .prompts import NO_ANSWER_TEXT, build_prompt, is_no_answer
from .retrieval import HybridRetriever


class RAGState(TypedDict, total=False):
    query: str                  # the question that is searched and answered (standalone after a rewrite)
    doc_ids: List[str]          # restrict the search to these documents (None = all)
    history: List[dict]         # earlier turns of a conversation ({"question", "answer"}), oldest first
    original_question: str      # the question as asked, when it was given to the rewriter
    rewrite: dict               # the rewriter's record (rewritten?, fallback reason, seconds, thinking tokens)
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
    table: object               # the table route's answer (tables.TableAnswer), when tables are enabled
    facts: str                  # the table passage put ahead of the text passages ("" = none)
    table_linked: List[int]     # chunk ids added as candidates because they are linked to the tables used
    table_check: dict           # the answer's figures checked against the context, and the fallback if one was used
    table_error: str


@dataclass
class Answer:
    question: str
    answer: str
    sources: List[str]
    route: str
    top_score: Optional[float]
    state: dict = field(repr=False)
    standalone: Optional[str] = None   # the rewritten question, when a follow-up was rewritten

    @property
    def refused(self) -> bool:
        return is_no_answer(self.answer)


class RAGPipeline:
    def __init__(self, corpus: Corpus, retriever: HybridRetriever, reranker, context: ContextBuilder, answerer,
                 cfg: Config, log=print, rewriter: Optional[QuestionRewriter] = None, tables=None):
        """``tables``: a ``tables.TableRoute`` (None = no table route)."""
        self.corpus, self.retriever, self.reranker, self.context = corpus, retriever, reranker, context
        self.answerer, self.cfg, self.log, self.rewriter, self.tables = answerer, cfg, log, rewriter, tables
        self._uid_pos = {}
        if tables is not None:
            from .database import chunk_uid
            self._uid_pos = {chunk_uid(c.doc_id, c.embed_text): i for i, c in enumerate(corpus.chunks)}
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ steps (usable on their own)
    def context_text(self, groups, facts: str = "") -> str:
        """The CONTEXT of the prompt: the table passage (if any), then the text passages."""
        return "\n\n".join(p for p in (facts, self.context.format(groups)) if p)

    def prompt_for(self, query: str, groups, facts: str = "") -> str:
        return build_prompt(query, self.context_text(groups, facts), self.cfg.prompt.template, self.corpus.multi_document)

    def table_answer(self, query: str, doc_ids=None):
        """The table route's answer; None when tables are off or the search is limited to other documents."""
        if self.tables is None or (doc_ids and self.tables.data.doc not in set(doc_ids)):
            return None
        return self.tables.answer(query.strip())

    def table_passage(self, table) -> str:
        """The context passage for a table answer: its facts under a label naming the sections they come from, cut
        to ``tables.max_tokens`` (a shortened passage says so)."""
        text = table.facts_block() if table is not None else ""
        if not text:
            return ""
        sections = list(dict.fromkeys(str(r["section_id"]) for r in table.rows if r.get("section_id")))
        label = "[Policy tables" + "".join(f" → Section {s}" for s in sections) + "]"
        limit, count = self.cfg.tables.max_tokens, self.context.count_tokens
        if count(text) > limit:
            lines = text.split("\n")
            tail = [ln for ln in lines[1:] if ln.startswith(("Note:", "Source:"))]
            body = [ln for ln in lines[1:] if ln not in tail]
            n = len(body)
            short = lambda b: "\n".join(lines[:1] + b + [f"(Shortened: {n - len(b)} more lines of this table passage are "
                                                          "not shown.)"] + tail)
            while body and count(short(body)) > limit:
                body.pop()
            text = short(body)
        return f"{label}\n{text}"

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
        """Answer when the best text passage is relevant enough, or when the tables answered the question (items
        found on a list do not count: they only add to the text, which must be relevant on its own)."""
        table = state.get("table")
        if table is not None and table.answered and state.get("facts") and table.fields.get("mode") != "items":
            return "answer"
        return "answer" if state.get("groups") and state.get("top_score", float("-inf")) >= self.cfg.gate.min_score \
            else "no_answer"

    def check_answer(self, query: str, answer: str, table, context: str) -> dict:
        """Figures of an answer that are in neither the context nor the question. When the tables gave the whole
        answer (``table.complete``) and the answer has such a figure, or refuses, the answer written from the tables
        replaces it (``tables.on_unsupported_figures: table_answer``)."""
        from .evaluation.checks import unsupported_figures
        derived = [("inr", int(c["result"])) for c in table.calculations if c.get("unit") == "inr"]
        refused = is_no_answer(answer)
        unsupported = [] if refused else unsupported_figures(answer, context, query, derived)
        check = {"unsupported_figures": unsupported, "refused": refused, "fallback": False}
        if (unsupported or refused) and table.complete and self.cfg.tables.on_unsupported_figures == "table_answer":
            why = "refused" if refused else "has figures not in the context (" + ", ".join(unsupported) + ")"
            self.log(f"the answer to {query[:50]!r} {why}: replaced by the answer written from the tables")
            check.update(fallback=True, model_answer=answer, answer=table.text)
        return check

    # ------------------------------------------------------------------ graph nodes
    def _condense_node(self, state: RAGState):
        """A follow-up becomes a standalone question; without earlier turns (or with memory off) nothing changes."""
        history = state.get("history") or []
        if not history or self.rewriter is None:
            return {}
        try:
            rw = self.rewriter.rewrite(state["query"], history)
        except Exception as e:                         # a failed rewrite must not lose the turn
            if is_oom(e):
                free_gpu()
            traceback.print_exc()
            rw = {"question": state["query"].strip(), "rewritten": False, "turns_used": len(history),
                  "fallback": f"rewrite failed: {type(e).__name__}: {e}"}
        if rw.get("fallback"):
            self.log(f"follow-up not rewritten ({rw['fallback']}): searching {state['query'][:60]!r} as asked")
        return {"original_question": state["query"], "query": rw["question"], "rewrite": rw}

    def _retrieve_node(self, state: RAGState):
        try:
            cands, routes = self.retrieve(state["query"], state.get("doc_ids"))
            return {"candidates": cands, "routes": routes}
        except Exception as e:
            traceback.print_exc()
            return {"candidates": [], "routes": [], "error": f"retrieve: {type(e).__name__}: {e}"}

    def _tables_node(self, state: RAGState):
        """The table route. A failure here is logged and the question is answered from the text passages alone."""
        if self.tables is None:
            return {}
        try:
            table = self.table_answer(state["query"], state.get("doc_ids"))
        except Exception as e:
            traceback.print_exc()
            self.log(f"table route failed ({type(e).__name__}): answering from the text passages only")
            return {"table_error": f"tables: {type(e).__name__}: {e}"}
        if table is None:
            return {}
        out = {"table": table, "facts": self.table_passage(table)}
        if table.answered and self.cfg.tables.link_chunks and table.linked_chunks:
            allowed = self.corpus.doc_chunk_ids(state.get("doc_ids"))
            cands = list(state.get("candidates") or [])
            have = set(cands)
            linked = [i for i in dict.fromkeys(self._uid_pos[u] for u in table.linked_chunks if u in self._uid_pos)
                      if i not in have and (allowed is None or i in allowed)]
            if linked:
                out.update(candidates=cands + linked, table_linked=linked)
        return out

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
        facts = state.get("facts") or ""
        try:
            try:
                groups = state["groups"]
                gen = self.answerer.answer(self.prompt_for(state["query"], groups, facts))
                out = {"answer": gen["answer"], "gen": gen, "route": "answer"}
            except Exception as e:
                if not is_oom(e):
                    raise
                free_gpu()
                groups = self.context.shorten(state["groups"], self.cfg.generator.oom_retry_context_tokens)
                self.log(f"out of GPU memory for {state['query'][:50]!r}: retried with {sum(map(len, groups))} chunks")
                gen = self.answerer.answer(self.prompt_for(state["query"], groups, facts))
                out = {"answer": gen["answer"], "gen": gen, "groups": groups, "oom_retry": True, "route": "answer"}
            if facts and state.get("table") is not None:
                check = self.check_answer(state["query"], gen["answer"], state["table"], self.context_text(groups, facts))
                out["table_check"] = check
                if check["fallback"]:
                    out["answer"] = check["answer"]
            return out
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
        g.add_node("condense", self._condense_node)
        g.add_node("retrieve", self._retrieve_node)
        g.add_node("tables", self._tables_node)
        g.add_node("rerank", self._rerank_node)
        g.add_node("answer", self._answer_node)
        g.add_node("no_answer", self._no_answer_node)
        g.set_entry_point("condense")
        g.add_edge("condense", "retrieve")
        g.add_edge("retrieve", "tables")
        g.add_edge("tables", "rerank")
        g.add_conditional_edges("rerank", self.gate, {"answer": "answer", "no_answer": "no_answer"})
        g.add_edge("answer", END)
        g.add_edge("no_answer", END)
        return g.compile()

    # ------------------------------------------------------------------ running
    def search(self, query: str, doc_ids=None) -> dict:
        """Retrieve, look up the tables, rerank and build the context; no answer model (what retrieval-only
        evaluation runs)."""
        st: dict = {"query": query, "doc_ids": doc_ids}
        st.update(self._retrieve_node(st))
        st.update(self._tables_node(st))
        st.update(self._rerank_node(st))
        st["route"] = self.gate(st)
        return st

    def run(self, query: str, doc_ids=None, history=None) -> dict:
        """The whole graph. ``history`` = earlier turns of a conversation (see ``Conversation.recent``)."""
        state = {"query": query, "doc_ids": list(doc_ids) if doc_ids else None}
        if history:
            state["history"] = list(history)
        return self.graph.invoke(state)


class RAGSystem:
    """All components built from one config."""

    def __init__(self, cfg: Config, corpus: Corpus, embedder, reranker, generator, cache: AnswerCache, log=print,
                 tables=None):
        """``tables``: a ``tables.TableRoute`` to use instead of the one ``tables.enabled`` builds over the database."""
        self.cfg, self.corpus, self.embedder, self.reranker, self.generator, self.cache = \
            cfg, corpus, embedder, reranker, generator, cache
        self.log = log
        self.retriever = HybridRetriever(corpus, embedder, cfg.retrieval, cfg.database)
        self.context = ContextBuilder(corpus, cfg.context, generator.count_tokens)
        self.answerer = Answerer(generator, cache)
        self.rewriter = QuestionRewriter(self.answerer, cfg.memory) if cfg.memory.enabled else None
        self.tables = tables if tables is not None else self._table_route(cfg)
        self.pipeline = RAGPipeline(corpus, self.retriever, reranker, self.context, self.answerer, cfg, log,
                                    self.rewriter, self.tables)

    def _table_route(self, cfg: Config):
        """The table route over the database (sharing the pgvector connection when there is one)."""
        if not cfg.tables.enabled:
            return None
        from .database import Database
        from .tables import TableData, TableRoute
        doc = cfg.tables.doc_id or cfg.documents[0].id
        db = getattr(self.retriever.dense, "db", None) or Database(cfg.database)
        return TableRoute(TableData(db, doc))

    @classmethod
    def from_config(cls, config="configs/default.yaml", overrides=None, corpus: Optional[Corpus] = None,
                    log=print, tables=None) -> "RAGSystem":
        """Build everything: load the chunk files, the embedder and reranker, index all chunks, and (with
        ``tables.enabled``) open the table route. The answer model loads on the first answer that is not in the
        cache. ``tables``: a ``TableRoute`` to use instead of one over the database."""
        cfg = config if isinstance(config, Config) else load_config(config, overrides)
        if corpus is None:
            corpus = load_corpus([DocumentConfig(d.id, str(resolve_input(d.chunks, cfg)), d.name) for d in cfg.documents])
            log(corpus.summary())
        embedder = registry.build("embedder", cfg.embedder)
        reranker = registry.build("reranker", cfg.reranker)
        generator = registry.build("generator", cfg.generator)          # weights load on first use
        cache = AnswerCache(cfg.cache.path, cfg.cache.enabled)
        system = cls(cfg, corpus, embedder, reranker, generator, cache, log, tables)
        routed = ", ".join(system.retriever.routed) or "none"
        dense = "dense" if cfg.retrieval.dense_backend == "faiss" else f"dense from {cfg.retrieval.dense_backend}"
        tables = "" if system.tables is None else (f"; tables of {system.tables.data.doc!r} from the database "
                                                   f"(version {system.tables.cat['version_label']})")
        log(f"Ready: {len(corpus.chunks)} chunks indexed ({dense} + {'BM25' if cfg.retrieval.use_bm25 else 'no BM25'}); "
            f"routed indexes: {routed}; answer cache: {len(cache)} answers{tables}")
        return system

    def ask(self, question: str, documents=None, show_sources: bool = True,
            conversation: Optional[Conversation] = None) -> Answer:
        """One question through the pipeline (one answer-model call unless the answer is cached or refused).

        With ``conversation``, earlier turns are used to rewrite a follow-up into a standalone question (one more
        model call), the conversation's document filter applies unless ``documents`` is given, and the turn is added
        to the conversation."""
        history = None
        if conversation is not None:
            history = conversation.recent(self.cfg.memory.max_turns, self.cfg.memory.answer_chars)
            if documents is None:
                documents = conversation.documents
        st = self.pipeline.run(question, documents, history)
        sources = [self.context.group_label(g) for g in st.get("groups", [])]
        if st.get("facts"):
            sources.insert(0, "[Policy tables] " + "; ".join(st["table"].sources))
        rw = st.get("rewrite") or {}
        standalone = st["query"] if rw.get("rewritten") else None
        if show_sources:
            if standalone:
                print("  searched as:", standalone)
            for s in sources:
                print("  source:", s[:120])
            print("\nQ:", question, "\nA:", st["answer"])
        route = st.get("route") or self.pipeline.gate(st)
        if conversation is not None:
            conversation.add(question, st["query"], st["answer"], sources, route)
        top = st.get("top_score")
        return Answer(question, st["answer"], sources, route,
                      top if top is not None and top != float("-inf") else None, st, standalone)

    def set_memory(self, enabled: bool = True) -> "RAGSystem":
        """Turn conversation memory (follow-up rewriting) on or off for this system. It is off by default: three
        evaluations did not show a significant gain (docs/evaluation.md#conversation-memory)."""
        self.cfg.memory.enabled = bool(enabled)
        self.rewriter = QuestionRewriter(self.answerer, self.cfg.memory) if enabled else None
        self.pipeline.rewriter = self.rewriter
        return self

    def new_conversation(self, documents=None) -> Conversation:
        """An empty conversation (optionally limited to some document ids) to pass to ``ask``."""
        if documents:
            self.corpus.doc_chunk_ids(documents)              # unknown ids fail here, not on the first question
        return Conversation(documents=list(documents) if documents else None)

    def chat(self, conversation: Optional[Conversation] = None, documents=None, input_fn=input) -> Conversation:
        """An interactive chat in the terminal or a notebook cell (type /help for commands)."""
        from .chat import run_chat
        return run_chat(self, conversation or self.new_conversation(documents), input_fn=input_fn)
