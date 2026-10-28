"""Evaluation of conversation memory on follow-up dialogues (``policy-rag evaluate-dialogues``).

A dialogue file lists short conversations. Turn 1 is a test-set question; each later turn is a follow-up whose
complete ("gold") form is another test-set question, so the existing answer key and checks score it. Every
follow-up is answered three ways:

- ``no_memory``: the follow-up exactly as typed, without the conversation (what the system did before memory);
- ``memory``: the system's own rewrite from the conversation (what memory does);
- ``gold``: the hand-written complete question (the best a rewrite could achieve; usually cached from a full run).

Each rewrite is also checked against the dialogue's keywords (``subject``: names of what the follow-up refers to,
one of which the rewrite must contain; ``must_mention``: further groups of alternatives that must appear;
``must_not_mention``: terms of an earlier topic that must not be pulled in).

Kinds of follow-up:

- ``follow_up``: needs the conversation ("Does it cover gloves?"); ``after_refusal``: the same after a refused answer;
- ``self_contained``: a follow-up that names its subject; memory must not make it worse;
- ``topic_switch``: a new, complete question; the rewriter should leave it alone.

A file with ``"follow_ups_need_context": true`` is checked when loaded: every follow_up and after_refusal turn has a
``subject``, and the follow-up as typed contains none of its alternatives (it really leaves its subject out).

Decision rule (fixed before the first run), on the test split: memory is adopted if the answers to follow-ups that
need the conversation (follow_up and after_refusal) are significantly more often fully correct with memory than
without (McNemar exact test, ``decision.alpha``) and no harm-check turn (self_contained, topic_switch) becomes
wrong. Single questions take the same path as before memory existed; that part of the rule is checked by the
unchanged test-set evaluation.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .. import __version__
from ..config import Config, resolve_input
from .report import md_table
from .runner import build_record
from .runs import make_zip
from .scoring import Scorer
from .stats import mcnemar_exact, wilson
from .testset import load_testset, norm_text

KINDS = ("follow_up", "after_refusal", "self_contained", "topic_switch")
NEEDS_CONTEXT = ("follow_up", "after_refusal")          # the primary comparison
HARM_CHECKS = ("self_contained", "topic_switch")         # memory must not make these worse
CONDITIONS = ("no_memory", "memory", "gold")


def load_dialogues(path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [d.get("id") for d in data["dialogues"]]
    if len(ids) != len(set(ids)):
        raise ValueError("dialogue ids must be unique")
    for d in data["dialogues"]:
        if len(d.get("turns", [])) < 2:
            raise ValueError(f"dialogue {d.get('id')}: needs at least two turns")
        for k, t in enumerate(d["turns"][1:], start=2):
            kind = t.get("kind", "follow_up")
            if kind not in KINDS:
                raise ValueError(f"dialogue {d['id']}: unknown kind {kind!r}; use one of {KINDS}")
            if data.get("follow_ups_need_context") and kind in NEEDS_CONTEXT:
                if not t.get("subject"):
                    raise ValueError(f"dialogue {d['id']} turn {k}: a {kind} turn needs a subject")
                named = [a for a in t["subject"] if norm_text(a) in norm_text(t["question"])]
                if named:
                    raise ValueError(f"dialogue {d['id']} turn {k}: the follow-up names its subject ({named[0]!r}), "
                                     "so it does not need the conversation; make it a self_contained turn")
    data["version"] = hashlib.sha1(json.dumps(data["dialogues"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]
    return data


def rewrite_check(turn: dict, rewritten: str):
    """(ok, problem): the rewrite names the subject (if given) and has one alternative of every must_mention group;
    no must_not_mention term was added."""
    text, asked = norm_text(rewritten), norm_text(turn["question"])
    for group in ([turn["subject"]] if turn.get("subject") else []) + turn.get("must_mention", []):
        if not any(norm_text(a) in text for a in group):
            return False, f"does not mention {group[0]!r}"
    for term in turn.get("must_not_mention", []):
        t = norm_text(term)
        if t in text and t not in asked:
            return False, f"pulls in {term!r} from the earlier topic"
    return True, None


class DialogueEvaluation:
    def __init__(self, cfg: Optional[Config], name: str, splits: Optional[List[str]] = None, conditions=None,
                 resume_from: Optional[List[str]] = None, log=print, system=None):
        """``system``: an already built RAGSystem to reuse (its config is used). ``conditions`` always includes
        ``memory`` (later turns need the conversation)."""
        self._given_system = system
        self.cfg = system.cfg if system is not None else cfg
        self.name, self.log = name, log
        self.splits = list(splits or self.cfg.evaluation.splits)
        self.conditions = [c for c in CONDITIONS if c in set(conditions or CONDITIONS) | {"memory"}]
        self.resume_from = resume_from or []
        self.out_dir = Path(self.cfg.evaluation.output_dir) / name

    def _score(self, item: dict, st: dict) -> dict:
        rec = build_record(item, st, self.system.context, self.system.pipeline.gate)
        gen = st.get("gen") or {}
        rec.update(answer=st.get("answer"), error=st.get("error"),
                   gen={k: gen.get(k) for k in ("key", "prompt_tokens", "new_tokens", "think_tokens", "answer_tokens",
                                                "thinking_forced", "hit_limit", "seconds", "from_cache")} if gen else None)
        self.scorer.score_record(rec)
        v, rt = rec["verdict"], rec.get("retrieval") or {}
        return {"question": st.get("query"), "answer": rec["answer"], "fully_correct": bool(v["fully_correct"]),
                "first_stage": v.get("first_stage"), "critical": v.get("critical", []),
                "issues": " | ".join(i["issue"] for i in v.get("issues", [])), "gate_route": rec["gate_route"],
                "top_score": rec["top_score"], "groups": rec["groups"], "context_recall": rt.get("context_recall"),
                "search_recall": rt.get("search_recall"), "seconds": gen.get("seconds"),
                "think_tokens": gen.get("think_tokens"), "from_cache": gen.get("from_cache"), "error": rec["error"]}

    def run(self, make_zip_file: bool = False) -> dict:
        t0 = time.perf_counter()
        started = datetime.datetime.now().isoformat(timespec="seconds")
        cfg = self.cfg
        if self._given_system is not None:
            self.system = self._given_system
        else:
            from ..pipeline import RAGSystem
            self.system = RAGSystem.from_config(cfg, log=self.log)
        for src in self.resume_from:
            n = self.system.cache.add_from(str(src))
            self.log(f"Resuming from {Path(src).name}: {n} cached answers added")
        testset = load_testset(resolve_input(cfg.evaluation.testset, cfg))
        by_id = testset.by_id
        dfile = resolve_input(cfg.evaluation.dialogues, cfg)
        data = load_dialogues(dfile)
        notes = []
        if data.get("testset_version") and data["testset_version"] != testset.version:
            notes.append(f"the dialogues were written for test set {data['testset_version']}, not {testset.version}")
        corpus = self.system.corpus
        scope = [testset.document] if corpus.multi_document else None
        self.scorer = Scorer(testset, corpus, self.system.context, testset.document if corpus.multi_document else None)
        if self.system.rewriter is None:
            notes.append("memory is off (memory.enabled: false): the memory condition only keeps the document filter")
        for n in notes:
            self.log("  NOTE: " + n)
        dialogues = [d for d in data["dialogues"]
                     if any(by_id[t["item"]]["split"] in self.splits for t in d["turns"][1:])]
        n_turns = sum(len(d["turns"]) - 1 for d in dialogues)
        self.log(f"Dialogues {data['version']}: {len(dialogues)} conversations, {n_turns} follow-up turns; "
                 f"conditions: {', '.join(self.conditions)}")
        rows = []
        for n, d in enumerate(dialogues, 1):
            conv = self.system.new_conversation(scope)
            first = d["turns"][0]
            self.system.ask(first.get("question") or by_id[first["item"]]["question"], conversation=conv,
                            show_sources=False)
            for k, t in enumerate(d["turns"][1:], start=2):
                item = by_id[t["item"]]
                row = {"dialogue": d["id"], "turn": k, "kind": t.get("kind", "follow_up"), "item": item["id"],
                       "split": item["split"], "asked": t["question"], "gold": item["question"], "conditions": {}}
                r = self.system.ask(t["question"], conversation=conv, show_sources=False)
                rw = r.state.get("rewrite") or {}
                row.update(rewrite=r.state["query"], rewritten=bool(rw.get("rewritten")), fallback=rw.get("fallback"),
                           skipped=bool(rw.get("skipped")),
                           rewrite_seconds=rw.get("seconds"), rewrite_think_tokens=rw.get("think_tokens"),
                           rewrite_from_cache=rw.get("from_cache"))
                row["rewrite_ok"], row["rewrite_problem"] = rewrite_check(t, r.state["query"])
                row["conditions"]["memory"] = self._score(item, r.state)
                if "no_memory" in self.conditions:
                    row["conditions"]["no_memory"] = self._score(item, self.system.pipeline.run(t["question"], scope))
                if "gold" in self.conditions:
                    row["conditions"]["gold"] = self._score(item, self.system.pipeline.run(item["question"], scope))
                rows.append(row)
            self.log(f"  dialogue {n}/{len(dialogues)} ({(time.perf_counter() - t0) / 60:.1f} min)")
        rows = [r for r in rows if r["split"] in self.splits]
        summary, decision = self._summarize(rows)
        run = {"name": self.name, "kind": "dialogues", "version": __version__, "started": started,
               "wall_seconds": round(time.perf_counter() - t0, 1), "dialogues_file": Path(dfile).name,
               "dialogues_version": data["version"], "testset_version": testset.version,
               "chunk_sha256": corpus.fingerprint, "config": cfg.to_dict(), "conditions": self.conditions,
               "notes": notes, "summary": summary, "decision": decision, "turns": rows}
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "dialogues.json").write_text(json.dumps(run, ensure_ascii=False, default=str), encoding="utf-8")
        self._write_report(run)
        self.log(f"\n{decision}\nResults in {self.out_dir}/: dialogue_report.md, dialogues.csv, dialogues.json. They "
                 "contain policy text: keep them out of public repositories.")
        if make_zip_file:
            run["zip"] = str(make_zip(self.out_dir, self.name))
        return run

    # ------------------------------------------------------------------ summary and decision
    def _summarize(self, rows):
        fc = lambda r, c: r["conditions"].get(c, {}).get("fully_correct")
        out = {"by_condition": {}, "rewrite": {}}
        for split in self.splits + ["all"]:
            rs = [r for r in rows if split == "all" or r["split"] == split]
            for kind in KINDS + ("all",):
                ks = [r for r in rs if kind == "all" or r["kind"] == kind]
                if not ks:
                    continue
                for c in self.conditions:
                    vals = [fc(r, c) for r in ks if fc(r, c) is not None]
                    if vals:
                        out["by_condition"].setdefault(c, {}).setdefault(split, {})[kind] = (sum(vals), len(vals))
        mem = [r for r in rows if r["kind"] in NEEDS_CONTEXT]
        fresh = [r["rewrite_seconds"] for r in rows if r.get("rewrite_seconds") and not r.get("rewrite_from_cache")]
        thinks = [r["rewrite_think_tokens"] for r in rows if r.get("rewrite_think_tokens") is not None]
        out["rewrite"] = {"turns": len(rows), "rewritten": sum(r["rewritten"] for r in rows),
                          "skipped": sum(bool(r.get("skipped")) for r in rows),
                          "fallbacks": sum(bool(r["fallback"]) for r in rows),
                          "keywords_ok_follow_ups": f"{sum(r['rewrite_ok'] for r in mem)}/{len(mem)}",
                          "topic_switch_ok": f"{sum(r['rewrite_ok'] for r in rows if r['kind'] == 'topic_switch')}/"
                                             f"{sum(r['kind'] == 'topic_switch' for r in rows)}",
                          "topic_switch_unchanged": f"{sum(r['rewrite'] == r['asked'] for r in rows if r['kind'] == 'topic_switch')}/"
                                                    f"{sum(r['kind'] == 'topic_switch' for r in rows)}",
                          "median_seconds": round(statistics.median(fresh), 1) if fresh else None,
                          "median_think_tokens": statistics.median(thinks) if thinks else None}
        if "no_memory" not in self.conditions or "test" not in self.splits:
            return out, "No decision: it needs the no_memory condition on the test split."
        test = [r for r in rows if r["split"] == "test"]
        fu = [r for r in test if r["kind"] in NEEDS_CONTEXT]
        fixed = [r for r in fu if fc(r, "memory") and not fc(r, "no_memory")]
        broken = [r for r in fu if fc(r, "no_memory") and not fc(r, "memory")]
        ts_broken = [r for r in test if r["kind"] in HARM_CHECKS and fc(r, "no_memory") and not fc(r, "memory")]
        p = mcnemar_exact(len(broken), len(fixed))
        out["primary"] = {"n": len(fu), "memory": sum(bool(fc(r, "memory")) for r in fu),
                          "no_memory": sum(bool(fc(r, "no_memory")) for r in fu),
                          "gold": sum(bool(fc(r, "gold")) for r in fu) if "gold" in self.conditions else None,
                          "fixed": [f"{r['dialogue']}.{r['turn']}" for r in fixed],
                          "broken": [f"{r['dialogue']}.{r['turn']}" for r in broken], "p_mcnemar": p,
                          "topic_switch_broken": [f"{r['dialogue']}.{r['turn']}" for r in ts_broken]}
        reasons = []
        if not (p < self.cfg.decision.alpha and len(fixed) > len(broken)):
            reasons.append(f"follow-ups not significantly better with memory (fixed {len(fixed)}, broken "
                           f"{len(broken)}, p = {p:.3f})")
        if ts_broken:
            reasons.append(f"{len(ts_broken)} harm-check turn(s) (self-contained follow-up or topic switch) became "
                           f"wrong: {', '.join(out['primary']['topic_switch_broken'])}")
        verdict = "ADOPT MEMORY" if not reasons else "DO NOT ADOPT MEMORY: " + "; ".join(reasons)
        return out, verdict

    def _write_report(self, run):
        rows, s = run["turns"], run["summary"]
        table = []
        for c, by_split in s["by_condition"].items():
            for split, kinds in by_split.items():
                row = {"condition": c, "split": split}
                for kind in KINDS + ("all",):
                    if kind in kinds:
                        k, n = kinds[kind]
                        lo, hi = wilson(k, n)
                        row[kind] = f"{k}/{n}" + (f" ({100 * lo:.0f}–{100 * hi:.0f}%)" if kind == "all" else "")
                    else:
                        row[kind] = ""
                table.append(row)
        rw = s["rewrite"]
        per_turn = pd.DataFrame([{
            "turn": f"{r['dialogue']}.{r['turn']}", "split": r["split"], "kind": r["kind"], "asked": r["asked"],
            "searched as (memory)": r["rewrite"], "rewrite ok": "yes" if r["rewrite_ok"] else f"no: {r['rewrite_problem']}",
            "fallback": r["fallback"] or "",
            **{f"correct: {c}": "yes" if r["conditions"][c]["fully_correct"] else "no"
               for c in self.conditions if c in r["conditions"]},
            "issues (memory)": r["conditions"]["memory"]["issues"]} for r in rows])
        pr = s.get("primary")
        L = [f"# Dialogue evaluation: {run['name']}", "",
             f"policy_rag {run['version']} | dialogues {run['dialogues_version']} | test set {run['testset_version']} | "
             f"{run['started']} | {len(rows)} follow-up turns | conditions {', '.join(run['conditions'])}", "",
             "## Decision", "", run["decision"], "",
             "Rule (fixed before the run): on the test split, answers to follow-ups that need the conversation "
             "(follow_up, after_refusal) must be significantly more often fully correct with memory than without "
             "(McNemar exact test), and no harm-check turn (self_contained, topic_switch) may become wrong. Single "
             "questions take the same path as before memory; the test-set evaluation checks that.", ""]
        if pr:
            L += [f"Test follow-ups that need the conversation: {pr['n']} turns; fully correct without memory "
                  f"{pr['no_memory']}, with memory {pr['memory']}" + (f", gold question {pr['gold']}" if pr['gold'] is not None else "")
                  + f". Fixed by memory: {', '.join(pr['fixed']) or 'none'}; broken: {', '.join(pr['broken']) or 'none'}; "
                  f"McNemar p = {pr['p_mcnemar']:.4f}. Harm-check turns broken: {', '.join(pr['topic_switch_broken']) or 'none'}.", ""]
        L += ["## Fully correct by condition", "", md_table(pd.DataFrame(table)), "",
              "## Rewrites", "",
              f"Rewritten {rw['rewritten']}/{rw['turns']} turns; not sent to the rewriter (complete questions) "
              f"{rw.get('skipped', 0)}; guard fallbacks {rw['fallbacks']}; keywords right on "
              f"follow-ups that need the conversation {rw['keywords_ok_follow_ups']}; topic switches without earlier-topic terms "
              f"{rw['topic_switch_ok']}, unchanged {rw['topic_switch_unchanged']}; median "
              f"rewrite time " + (f"{rw['median_seconds']} s" if rw["median_seconds"] is not None else
                                  "– (all rewrites from the cache)")
              + f", median rewrite thinking tokens {rw['median_think_tokens']}.", "",
              "## Turns", "", md_table(per_turn), ""]
        if run["notes"]:
            L += ["## Notes", ""] + [f"- {n}" for n in run["notes"]]
        (self.out_dir / "dialogue_report.md").write_text("\n".join(L), encoding="utf-8")
        flat = []
        for r in rows:
            f = {k: r[k] for k in ("dialogue", "turn", "kind", "split", "item", "asked", "rewrite", "rewrite_ok",
                                   "rewrite_problem", "fallback", "gold", "rewrite_seconds", "rewrite_think_tokens")}
            for c, v in r["conditions"].items():
                f.update({f"{c}_fully_correct": v["fully_correct"], f"{c}_answer": v["answer"], f"{c}_issues": v["issues"]})
            flat.append(f)
        pd.DataFrame(flat).to_csv(self.out_dir / "dialogues.csv", index=False)
