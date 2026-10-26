"""One evaluation run: answers (or saved answers), scores, groundedness, report.

Modes:

- ``full``: retrieval pass, then every question through the pipeline (answers come from the cache when the prompt
  and settings match), evidence-removed test, repeat check;
- ``retrieval``: retrieval, reranking, context and gate only (no answer model; minutes) - to compare chunk files,
  retrievers or rerankers;
- ``rescore``: a saved run's answers re-checked with the current checks and test set (no retrieval models).

``contexts_from`` (with ``full``) answers the saved contexts of another run again, e.g. to compare answer models
on identical contexts.
"""
from __future__ import annotations

import datetime
import json
import platform
import time
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .. import __version__, registry
from ..config import Config, DocumentConfig, resolve_input
from ..context import ContextBuilder
from ..corpus import load_corpus
from ..device import gpu_gb, gpu_name, peak_gpu_gb, reset_peak_memory
from ..prompts import NO_ANSWER_TEXT, is_no_answer
from .checks import admits_gap, unsupported_figures
from .grounding import HHEMGrounder
from .labels import label_sheet
from .report import write_report
from .runs import load_run, make_zip, save_run
from .scoring import Scorer
from .stats import auroc
from .testset import anchor_in, check_testset, fact_found, fact_label, load_testset, norm_text

PACKAGES = ["torch", "transformers", "sentence-transformers", "bitsandbytes", "accelerate", "faiss-cpu", "rank-bm25",
            "langgraph", "numpy", "pandas"]


def _version(p):
    try:
        return pkg_version(p)
    except PackageNotFoundError:
        return None


def gate_report(recs, threshold: float) -> dict:
    s = lambda r: r["top_score"] if r["top_score"] is not None else -99.0
    out = {}
    for split in sorted({r["split"] for r in recs}):
        rs = [r for r in recs if r["split"] == split]
        pos, neg = [s(r) for r in rs if r["answerable"]], [s(r) for r in rs if not r["answerable"]]
        out[split] = {"auroc": auroc(pos, neg), "answerable_refused_by_gate": sum(p < threshold for p in pos),
                      "not_answerable_refused_by_gate": sum(q < threshold for q in neg), "n_answerable": len(pos),
                      "n_not_answerable": len(neg), "lowest_answerable_score": round(min(pos), 2) if pos else None,
                      "highest_not_answerable_score": round(max(neg), 2) if neg else None}
    return out


def build_record(item: dict, st: dict, context, gate) -> dict:
    """One question's record (as saved in run.json) from a pipeline state: retrieval, context, gate; no verdict yet."""
    cands, scores = st.get("candidates") or [], st.get("scores") or []
    ranked = sorted(([int(i), round(float(s), 4)] for i, s in zip(cands, scores)), key=lambda x: (-x[1], x[0]))
    groups = [[int(i) for i in g] for g in st.get("groups", [])]
    top = st.get("top_score", -np.inf)
    return {"id": item["id"], "split": item["split"], "type": item["type"], "answerable": item["answerable"],
            "paraphrase": item["paraphrase"], "question": item["question"],
            "premium_routed": "premium" in (st.get("routes") or []), "routes": st.get("routes") or [],
            "candidates": [int(i) for i in cands], "ranked": ranked, "groups": groups,
            "context": context.format(groups), "context_tokens": context.tokens(groups),
            "top_score": float(top) if top is not None and np.isfinite(top) else None,
            "gate_route": gate(st), "answer": None, "gen": None, "error": st.get("error"),
            "oom_retry": bool(st.get("oom_retry")), "retrieval": {}, "verdict": None, "grounding": None}


class Evaluation:
    def __init__(self, cfg: Config, name: str, mode: str = "full", splits: Optional[List[str]] = None,
                 rescore_from=None, contexts_from=None, resume_from: Optional[List[str]] = None, log=print,
                 system=None):
        """``system``: an already built RAGSystem to reuse (e.g. in a notebook, so the models are not loaded twice);
        its config is used."""
        if mode not in ("full", "retrieval", "rescore"):
            raise ValueError(f"mode must be full, retrieval or rescore, not {mode!r}")
        if mode == "rescore" and not rescore_from:
            raise ValueError("mode 'rescore' needs the saved run to re-score (rescore_from)")
        if contexts_from and mode != "full":
            raise ValueError("contexts_from re-answers saved contexts: use mode 'full'")
        self._given_system = system
        cfg = system.cfg if system is not None else cfg
        self.cfg, self.name, self.mode = cfg, name, mode
        self.splits = list(splits or cfg.evaluation.splits)
        self.rescore_from, self.contexts_from, self.resume_from = rescore_from, contexts_from, resume_from or []
        self.run_log: List[str] = []
        self._print = log
        self.out_dir = Path(cfg.evaluation.output_dir) / name

    def log(self, msg: str):
        """A problem worth knowing about: printed now and saved with the results."""
        self._print("  NOTE: " + msg)
        self.run_log.append(msg)

    # ------------------------------------------------------------------ setup
    def _setup(self):
        cfg = self.cfg
        if self._given_system is not None:
            self.corpus = self._given_system.corpus
        else:
            self.corpus = load_corpus([DocumentConfig(d.id, str(resolve_input(d.chunks, cfg)), d.name) for d in cfg.documents])
            self._print(self.corpus.summary())
        self.testset = load_testset(resolve_input(cfg.evaluation.testset, cfg))
        self.items = self.testset.select(self.splits)
        doc = self.testset.document
        self.scope = None
        if self.corpus.multi_document:
            if doc not in self.corpus.documents:
                raise ValueError(f"the test set is about document {doc!r}, which is not loaded "
                                 f"(loaded: {sorted(self.corpus.documents)})")
            self.scope = [doc]                         # questions about one policy search that policy
        if self._given_system is not None:
            self.system = self._given_system
            self.generator, self.context = self.system.generator, self.system.context
            for src in self.resume_from:
                n = self.system.cache.add_from(str(src))
                self._print(f"Resuming from {Path(src).name}: {n} cached answers added")
        elif self.mode == "rescore":
            self.system = None
            self.generator = registry.build("generator", cfg.generator)      # tokenizer only (context token counts)
            self.context = ContextBuilder(self.corpus, cfg.context, self.generator.count_tokens)
        else:
            from ..pipeline import RAGSystem
            self.system = RAGSystem.from_config(cfg, corpus=self.corpus, log=self._print)
            self.generator, self.context = self.system.generator, self.system.context
            for src in self.resume_from:
                n = self.system.cache.add_from(str(src))
                self._print(f"Resuming from {Path(src).name}: {n} cached answers added (used where the prompt and settings match)")
        self.scorer = Scorer(self.testset, self.corpus, self.context, doc if self.corpus.multi_document else None)
        self.problems = check_testset(self.items, self.corpus, self.scorer.text)
        self.evidence_missing = {p["id"] for p in self.problems if p["problem"] == "evidence not in this chunk file"}
        n_ans = sum(t["answerable"] for t in self.items)
        per_split = ", ".join(f"{s} {sum(t['split'] == s for t in self.items)}" for s in self.splits)
        self._print(f"Test set {self.testset.version}: {len(self.testset.items)} questions; running {len(self.items)} "
                    f"({per_split}): {n_ans} "
                    f"answerable, {len(self.items) - n_ans} not answerable, {sum(t['paraphrase'] for t in self.items)} "
                    "in everyday wording")
        for p in self.problems:
            self._print(f"  test set vs chunk file: {p['id']}: {p['problem']} ({p['detail']})")

    def _record(self, item: dict, st: dict) -> dict:
        return build_record(item, st, self.context, self.system.pipeline.gate)

    # ------------------------------------------------------------------ the run
    def run(self, make_zip_file: bool = False) -> dict:
        t_start = time.perf_counter()
        reset_peak_memory()
        self._setup()
        cfg = self.cfg
        started = datetime.datetime.now().isoformat(timespec="seconds")
        res = load_run(self.rescore_from) if self.mode == "rescore" else None
        src = load_run(self.contexts_from) if self.contexts_from else None
        if res is not None:
            self._print(f"Re-scoring the saved answers of run '{res['name']}' ({len(res['records'])} questions) with test "
                        f"set {self.testset.version}; no answer model is used")
            if res.get("chunk_sha256") != self.corpus.fingerprint:
                self.log("the saved run used a different chunk file: its retrieval metrics are kept as saved")
        if src is not None:
            self._print(f"Answering the saved contexts of run '{src['name']}' ({len(src['records'])} questions); "
                        "retrieval is not repeated")

        # ---- retrieval pass (no answer model)
        by_id = self.testset.by_id
        if res is not None:
            ids = {t["id"] for t in self.items}
            records = []
            for rec in res["records"]:
                if rec["id"] in ids:
                    item = by_id[rec["id"]]
                    rec.update({k: item[k] for k in ("split", "type", "answerable", "paraphrase")}, grounding=None)
                    if res.get("chunk_sha256") != self.corpus.fingerprint:
                        rec["context_source"] = res["name"]
                    records.append(rec)
        elif src is None:
            records = []
            for n, item in enumerate(self.items, 1):
                records.append(self._record(item, self.system.pipeline.search(item["question"], self.scope)))
                if n % 25 == 0:
                    self._print(f"  retrieval {n}/{len(self.items)}")
        else:
            saved = {r["id"]: r for r in src["records"]}
            records = []
            for item in self.items:
                r0 = saved.get(item["id"])
                if r0 is None:
                    continue
                rec = {k: r0.get(k) for k in ("candidates", "ranked", "groups", "context", "context_tokens", "top_score",
                                              "gate_route", "premium_routed")}
                rec.update({"id": item["id"], "split": item["split"], "type": item["type"], "answerable": item["answerable"],
                            "paraphrase": item["paraphrase"], "question": item["question"], "answer": None, "gen": None,
                            "error": None, "oom_retry": False, "retrieval": r0.get("retrieval") or {}, "verdict": None,
                            "grounding": None, "context_source": src["name"]})
                records.append(rec)
        gate = gate_report(records, cfg.gate.min_score)
        for split, g in gate.items():
            self._print(f"Gate [{split}]: AUROC {g['auroc'] if g['auroc'] is None else round(g['auroc'], 3)} | at threshold "
                        f"{cfg.gate.min_score}: refuses {g['answerable_refused_by_gate']}/{g['n_answerable']} answerable and "
                        f"{g['not_answerable_refused_by_gate']}/{g['n_not_answerable']} not-answerable questions")

        # ---- answers
        llm_calls, repeat_retrieval, tps = {"generated": 0, "from_cache": 0}, {"identical": 0, "different": []}, None
        evidence_removed, repeat = [], None
        if self.mode == "full":
            pipe, answerer = self.system.pipeline, self.system.answerer
            prompts = [pipe.prompt_for(r["question"], r["groups"]) if src is None else
                       self._prompt_from_context(r["question"], r["context"]) for r in records if r["gate_route"] == "answer"]
            need = sum(not answerer.is_cached(p) for p in prompts)
            if need:
                for problem in self.generator.self_check():
                    self.log(problem)
                tps = self.generator.tokens_per_second(prompts[0])
            self._print(f"{len(prompts)} of {len(records)} questions go to the answer model: {len(prompts) - need} cached, "
                        f"{need} to generate" + (f" ({tps} tokens/s)" if need else ""))
            recs = {r["id"]: r for r in records}
            for n, item in enumerate([by_id[r["id"]] for r in records], 1):
                rec, gen = recs[item["id"]], None
                if src is not None:
                    if rec["gate_route"] == "answer":
                        gen = answerer.answer(self._prompt_from_context(item["question"], rec["context"]))
                        rec["answer"] = gen["answer"]
                    else:
                        rec["answer"] = NO_ANSWER_TEXT
                else:
                    st = pipe.run(item["question"], self.scope)
                    again = self._record(item, st)
                    if st.get("oom_retry"):
                        rec.update({k: again[k] for k in ("groups", "context", "context_tokens")}, oom_retry=True)
                    elif all(again[k] == rec[k] for k in ("candidates", "ranked", "groups", "gate_route")):
                        repeat_retrieval["identical"] += 1
                    else:
                        repeat_retrieval["different"].append(item["id"])
                    rec["answer"], rec["error"], gen = st.get("answer"), st.get("error"), st.get("gen")
                if gen:
                    rec["gen"] = {k: gen.get(k) for k in ("key", "prompt_tokens", "new_tokens", "think_tokens",
                                                          "answer_tokens", "thinking_forced", "hit_limit", "seconds", "from_cache")}
                    rec["thinking"] = gen.get("thinking") or None
                    llm_calls["from_cache" if gen["from_cache"] else "generated"] += 1
                if n % 10 == 0:
                    self._print(f"  answered {n}/{len(records)} ({(time.perf_counter() - t_start) / 60:.1f} min)")
            if repeat_retrieval["different"]:
                self.log(f"retrieval differed between two passes for {repeat_retrieval['different']}")

        # ---- scores
        for rec in records:
            self.scorer.score_record(rec)

        if self.mode == "full" and cfg.evaluation.evidence_removed_test:
            evidence_removed = self._evidence_removed(records, src)
        if self.mode == "full" and cfg.evaluation.repeat_check:
            repeat = self._repeat_check(records, src)

        peak, tokens_per_second = peak_gpu_gb(), tps
        run = {"name": self.name, "version": __version__, "mode": self.mode, "testset": self.testset.name,
               "testset_version": self.testset.version, "chunk_files": [Path(d.path).name for d in self.corpus.documents.values()],
               "chunk_sha256": self.corpus.fingerprint, "documents": {d.id: d.sha256 for d in self.corpus.documents.values()},
               "models": self._models(), "config": cfg.to_dict(), "overrides": getattr(self, "overrides", []),
               "context_source": src["name"] if src else None, "scope": self.scope, "started": started, "gate": gate,
               "llm_calls": llm_calls, "retrieval_repeat": repeat_retrieval, "tokens_per_second": tokens_per_second,
               "peak_gpu_gb": peak, "evidence_removed": evidence_removed, "repeat": repeat, "records": records}
        if res is not None:                            # what the saved run measured, not this session
            run.update({k: res.get(k) for k in ("models", "config", "started", "wall_seconds", "peak_gpu_gb",
                                                 "tokens_per_second", "evidence_removed", "repeat")},
                       llm_calls=res.get("llm_calls") or llm_calls,
                       retrieval_repeat=res.get("retrieval_repeat") or repeat_retrieval, rescored_from=res["name"])
        else:
            run["wall_seconds"] = round(time.perf_counter() - t_start, 1)

        # ---- groundedness
        if self.mode in ("full", "rescore") and cfg.evaluation.groundedness:
            self._ground(run)

        # ---- save
        run["environment"] = self._environment(started)
        save_run(run, self.out_dir)
        tables = write_report(run, self.out_dir, by_id, self.problems, self.run_log)
        if self.mode in ("full", "rescore"):
            label_sheet(run, by_id, self.out_dir / "labels_to_fill.csv")
            fc = [r["verdict"]["fully_correct"] for r in records if r.get("verdict")]
            saved = " (as measured in the saved run)" if res is not None else ""
            self._print(f"Done in {(time.perf_counter() - t_start) / 60:.1f} min: {sum(fc)}/{len(fc)} fully correct | "
                        f"answers generated {run['llm_calls']['generated']}, from cache {run['llm_calls']['from_cache']} "
                        f"| peak GPU {run['peak_gpu_gb']} GB{saved}")
        else:
            self._print(f"Retrieval run done in {(time.perf_counter() - t_start) / 60:.1f} min")
        (self.out_dir / "environment.json").write_text(json.dumps(run["environment"], indent=1, default=str))
        self._print(f"Results in {self.out_dir}/: report.md, results.csv, scorecard.csv, run.json"
                    + (", labels_to_fill.csv" if self.mode != "retrieval" else "")
                    + ". They contain policy text: keep them out of public repositories.")
        if make_zip_file:
            run["zip"] = str(make_zip(self.out_dir, self.name))
        run["tables"] = tables
        return run

    # ------------------------------------------------------------------ parts
    def _prompt_from_context(self, question: str, context: str) -> str:
        from ..prompts import build_prompt
        return build_prompt(question, context, self.cfg.prompt.template, self.corpus.multi_document)

    def _evidence_removed(self, records, src) -> list:
        """Up to 4 answers with the evidence chunks taken out of the context: the answer should say the policy text
        does not show it (refuse or admit the gap) and state no fact from outside the context."""
        if src is not None and src.get("chunk_sha256") != self.corpus.fingerprint:
            self.log("evidence-removed test skipped: the saved contexts come from a different chunk file")
            return []
        out, recs, by_id = [], {r["id"]: r for r in records}, self.testset.by_id
        chunks, words = self.corpus.chunks, self.scorer.text.chunk_words
        for qid in self.testset.evidence_removed_candidates:
            rec, item = recs.get(qid), by_id.get(qid)
            if len(out) >= 4 or not rec or not item or rec["gate_route"] != "answer":
                continue
            holds = lambda i: any(anchor_in(e, words[i]) for e in item["evidence"]) or \
                any(fact_found(f, norm_text(chunks[i].text)) for f in item["facts"])
            keep = [g for g in ([i for i in g if not holds(i)] for g in rec["groups"]) if g]
            ctx = self.context.format(keep)
            removed = sum(map(len, rec["groups"])) - sum(map(len, keep))
            if not keep or not removed or any(fact_found(f, norm_text(ctx)) for f in item["facts"]):
                continue                     # evidence was never in the context, or cannot be taken out cleanly
            ans = self.system.answerer.answer(self._prompt_from_context(item["question"], ctx))["answer"]
            q_norm = norm_text(item["question"])
            stated = [fact_label(f) for f in item["facts"] if fact_found(f, norm_text(ans)) and not fact_found(f, q_norm)]
            unsup = unsupported_figures(ans, ctx, item["question"])
            ok = is_no_answer(ans) or (admits_gap(ans) and not stated and not unsup)
            out.append({"id": qid, "chunks_removed": removed, "pass": ok,
                        "outcome": "refused" if is_no_answer(ans) else "said it is not shown" if ok
                        else "stated facts from outside the context" if stated else "answered without the evidence",
                        "facts_stated": stated, "figures_not_in_context": unsup, "answer": ans})
        self._print(f"Evidence removed: {sum(e['pass'] for e in out)}/{len(out)} answers said the policy text does not "
                    f"show it | " + ", ".join(f"{e['id']}: {e['outcome']}" for e in out))
        return out

    def _repeat_check(self, records, src):
        cands = [r for r in records if r.get("gen") and r["gen"].get("new_tokens")]
        if not cands:
            return None
        r = min(cands, key=lambda r: r["gen"]["new_tokens"])
        prompt = self._prompt_from_context(r["question"], r["context"])
        again = self.system.answerer.answer(prompt, use_cache=False)["answer"]
        out = {"id": r["id"], "identical": again == r["answer"]}
        self._print(f"Repeat check: {r['id']} regenerated without the cache is {'identical' if out['identical'] else 'DIFFERENT'}")
        if not out["identical"]:
            self.log(f"regenerated answer to {r['id']} differs: generation is not fully deterministic")
        return out

    def _ground(self, run):
        cfg = self.cfg.evaluation
        if self.system is not None:
            from ..device import free_gpu
            free_gpu()
        grounder = HHEMGrounder(cfg.grounding_model, cfg.hhem_threshold)
        try:
            grounder.load()
        except Exception as e:
            self.log(f"grounding model could not be loaded ({type(e).__name__}: {e}); groundedness not measured")
            return
        if grounder.ground_run(run, self._print):
            g = [r["grounding"] for r in run["records"] if r.get("grounding") and r["grounding"]["groundedness"] is not None]
            if g:
                self._print(f"  mean groundedness {np.mean([x['groundedness'] for x in g]):.3f} | answers with an "
                            f"unsupported sentence: {sum(x['hallucinated'] for x in g)}/{len(g)}")
            run["grounding_model"], run["grounding_revision"] = cfg.grounding_model, grounder.revision()

    def _models(self) -> dict:
        cfg = self.cfg
        gen = self.generator
        out = {"embed": cfg.embedder.model, "reranker": cfg.reranker.model, "llm": cfg.generator.model,
               "llm_precision": getattr(gen, "precision", None), "enable_thinking": cfg.generator.enable_thinking,
               "decoding": {k: v for k, v in (getattr(gen, "decoding", {}) or {}).items() if v is not None}}
        if self.system is not None:
            out.update(embed_revision=self.system.embedder.revision(), reranker_revision=self.system.reranker.revision(),
                       llm_revision=gen.revision())
        return out

    def _environment(self, started) -> dict:
        return {"policy_rag": __version__, "started": started,
                "finished": datetime.datetime.now().isoformat(timespec="seconds"), "python": platform.python_version(),
                "gpu": gpu_name(), "gpu_in_use_gb": gpu_gb(), "packages": {p: _version(p) for p in PACKAGES},
                "config_source": self.cfg.source, "config_fingerprint": self.cfg.fingerprint(), "run_log": self.run_log}
