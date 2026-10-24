"""Scorecard, failure tables and report.md for one run."""
from __future__ import annotations

import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from .checks import STAGES
from .stats import mde, wilson

_mean = lambda xs: float(np.mean(xs)) if len(xs) else None


def _has(r, kind) -> bool:
    return any(i["kind"] == kind for i in (r.get("verdict") or {}).get("issues", []))


def summarize(recs) -> dict:
    """All metrics for a set of question records (one split, or all)."""
    v = [r for r in recs if r.get("verdict")]
    ans = [r for r in v if r["answerable"]]
    neg = [r for r in v if not r["answerable"]]
    ret = [r["retrieval"] for r in recs if r["answerable"] and r.get("retrieval")]
    grd = [r["grounding"] for r in v if r.get("grounding") and r["grounding"].get("groundedness") is not None]
    tabs = [r["verdict"]["table"] for r in ans if r["verdict"].get("table")]
    fc = lambda rs: [bool(r["verdict"]["fully_correct"]) for r in rs]
    out = {"n": len(recs), "n_answerable": sum(r["answerable"] for r in recs)}
    if v:
        k = sum(fc(v))
        out.update(fully_correct=k / len(v), fully_correct_k=k, fully_correct_n=len(v), fully_correct_ci=wilson(k, len(v)),
                   fully_correct_answerable=_mean(fc(ans)), fully_correct_not_answerable=_mean(fc(neg)),
                   key_fact_recall=_mean([r["verdict"]["fact_recall"] or 0.0 for r in ans]),
                   critical_error_rate=_mean([bool(r["verdict"]["critical"]) for r in v]),
                   table_values=f"{sum(t['correct'] for t in tabs)}/{sum(t['expected'] for t in tabs)} correct, "
                                f"{sum(len(t['wrong']) for t in tabs)} wrong" if tabs else None,
                   list_completeness=_mean([r["verdict"]["fact_recall"] or 0.0 for r in ans if r["type"] == "list"]),
                   unsupported_figures=_mean([_has(r, "unsupported_figures") for r in v]),
                   unsupported_citations=_mean([_has(r, "unsupported_citations") for r in v]),
                   correct_abstention=_mean(fc(neg)),
                   false_refusal=_mean([bool(r["verdict"].get("false_refusal")) for r in ans]),
                   refused_by_gate=sum(r["gate_route"] == "no_answer" for r in ans),
                   everyday_gap=(_mean(fc([r for r in ans if not r["paraphrase"]])) or 0) -
                                (_mean(fc([r for r in ans if r["paraphrase"]])) or 0)
                   if any(r["paraphrase"] for r in ans) and any(not r["paraphrase"] for r in ans) else None,
                   cut_off=sum(_has(r, "cut_off") for r in v), errors=sum(bool(r.get("error")) for r in v),
                   oom_retries=sum(bool(r.get("oom_retry")) for r in v))
    if grd:
        out.update(groundedness=_mean([g["groundedness"] for g in grd]), hallucination_rate=_mean([g["hallucinated"] for g in grd]),
                   grounded_answers=len(grd))
    if ret:
        for m in ("context_recall", "search_recall", "mrr10", "context_precision", "evidence_in_chunk_file", "facts_in_context"):
            out[m] = _mean([x[m] for x in ret if x.get(m) is not None])
    gens = [r["gen"] for r in recs if r.get("gen")]
    if gens:
        secs = [g["seconds"] for g in gens if g.get("seconds")]
        out.update(median_seconds=statistics.median(secs) if secs else None,
                   p95_seconds=float(np.percentile(secs, 95)) if secs else None,
                   median_prompt_tokens=statistics.median([g["prompt_tokens"] for g in gens if g.get("prompt_tokens")] or [0]),
                   median_answer_tokens=statistics.median([g.get("answer_tokens") or g.get("new_tokens") or 0 for g in gens]),
                   median_think_tokens=statistics.median([g.get("think_tokens") or 0 for g in gens])
                   if any(g.get("think_tokens") for g in gens) else None,
                   thinking_forced=sum(bool(g.get("thinking_forced")) for g in gens)
                   if any(g.get("think_tokens") for g in gens) else None)
    return out


def _fmt(x, pct=False):
    if x is None:
        return "–"
    if isinstance(x, str):
        return x
    if pct:
        return f"{100 * x:.1f}%"
    if isinstance(x, float):
        return f"{x:.0f}" if abs(x) >= 100 else f"{x:.1f}" if abs(x) >= 10 else f"{x:.3f}"
    return str(x)


SCORE_ROWS = [  # (group, label, key, as percent)
    ("PRIMARY", "Fully correct (all questions)", "fully_correct", True),
    ("Answer correctness", "fully correct: answerable", "fully_correct_answerable", True),
    ("Answer correctness", "key-fact recall (answerable)", "key_fact_recall", False),
    ("Answer correctness", "critical error rate", "critical_error_rate", True),
    ("Answer correctness", "plan-table values", "table_values", False),
    ("Answer correctness", "list completeness (list questions)", "list_completeness", False),
    ("Faithfulness", "groundedness (supported sentences, HHEM)", "groundedness", False),
    ("Faithfulness", "hallucination rate (answers with an unsupported sentence)", "hallucination_rate", True),
    ("Faithfulness", "answers with figures not in the context", "unsupported_figures", True),
    ("Faithfulness", "answers citing section numbers not in the context", "unsupported_citations", True),
    ("Retrieval", "context recall (evidence passages reaching the model)", "context_recall", False),
    ("Retrieval", "search recall (evidence among the candidates)", "search_recall", False),
    ("Retrieval", "MRR@10 (rank of the first relevant chunk)", "mrr10", False),
    ("Retrieval", "context precision (share of context tokens that are evidence)", "context_precision", False),
    ("Retrieval", "evidence present in the chunk file", "evidence_in_chunk_file", False),
    ("Abstention", "correct abstention (not answerable)", "correct_abstention", True),
    ("Abstention", "false refusal rate (answerable)", "false_refusal", True),
    ("Abstention", "answerable questions refused by the gate", "refused_by_gate", False),
    ("Robustness", "everyday-wording gap (policy wording minus everyday wording)", "everyday_gap", True),
    ("Efficiency", "seconds per answer, median", "median_seconds", False),
    ("Efficiency", "seconds per answer, 95th percentile", "p95_seconds", False),
    ("Efficiency", "prompt tokens, median", "median_prompt_tokens", False),
    ("Efficiency", "answer tokens, median", "median_answer_tokens", False),
    ("Efficiency", "thinking tokens, median", "median_think_tokens", False),
    ("Reliability", "thinking stopped at the budget (told to answer)", "thinking_forced", False),
    ("Reliability", "answers cut off at the length limit", "cut_off", False),
    ("Reliability", "pipeline errors / out-of-memory retries", "errors", False),
]


def scorecard(run: dict) -> pd.DataFrame:
    recs = run["records"]
    groups = {s: summarize([r for r in recs if r["split"] == s]) for s in ("dev", "test") if any(r["split"] == s for r in recs)}
    if len(groups) > 1:
        groups["all"] = summarize(recs)
    rows = []
    for grp, label, key, pct in SCORE_ROWS:
        vals = {s: g.get(key) for s, g in groups.items()}
        if all(x is None for x in vals.values()):
            continue
        row = {"group": grp, "measure": label}
        for s, g in groups.items():
            x = g.get(key)
            if key == "fully_correct" and x is not None:
                lo, hi = g["fully_correct_ci"]
                row[s] = f"{g['fully_correct_k']}/{g['fully_correct_n']} = {100 * x:.1f}% (95% CI {100 * lo:.0f}–{100 * hi:.0f}%)"
            elif key == "errors" and x is not None:
                row[s] = f"{x} / {g['oom_retries']}"
            else:
                row[s] = _fmt(x, pct)
        rows.append(row)
    answered = any(r.get("gen") for r in recs)
    first = "all" if "all" in groups else next(iter(groups), "all")
    extra = [
        ("Abstention", "gate separability (AUROC of the best reranker score)",
         {s: _fmt((run.get("gate") or {}).get(s, {}).get("auroc")) for s in groups if s != "all"}),
        ("Abstention", "evidence removed: answer says it is not shown",
         {"dev": f"{sum(e['pass'] for e in run.get('evidence_removed') or [])}/{len(run.get('evidence_removed') or [])}"}
         if run.get("evidence_removed") else {}),
        ("Efficiency", "tokens per second / peak GPU memory (GB)",
         {first: f"{run.get('tokens_per_second') or '–'} / {run.get('peak_gpu_gb') or '–'}"} if answered else {}),
        ("Reliability", "retrieval identical when repeated / regenerated answer identical",
         {first: f"{run['retrieval_repeat']['identical']}/{run['retrieval_repeat']['identical'] + len(run['retrieval_repeat']['different'])}"
                 f" / {(run.get('repeat') or {}).get('identical', 'not run')}"} if answered and run.get("retrieval_repeat") else {})]
    for grp, label, val in extra:
        if val:
            rows.append({"group": grp, "measure": label, **val})
    return pd.DataFrame(rows).fillna("")


def question_table(run: dict, by_id: dict) -> pd.DataFrame:
    rows = []
    for r in run["records"]:
        v, rt, g, gen = r.get("verdict") or {}, r.get("retrieval") or {}, r.get("grounding") or {}, r.get("gen") or {}
        item = by_id.get(r["id"], {})
        rows.append({"id": r["id"], "split": r["split"], "type": r["type"], "answerable": r["answerable"],
                     "fully_correct": v.get("fully_correct"), "first_stage": v.get("first_stage") or "",
                     "critical": ", ".join(v.get("critical", [])), "fact_recall": v.get("fact_recall"),
                     "context_recall": rt.get("context_recall"), "mrr10": rt.get("mrr10"),
                     "context_precision": None if rt.get("context_precision") is None else round(rt["context_precision"], 2),
                     "groundedness": None if g.get("groundedness") is None else round(g["groundedness"], 2),
                     "gate": r["gate_route"], "top_score": r.get("top_score"), "context_tokens": r.get("context_tokens"),
                     "answer_tokens": gen.get("answer_tokens") or gen.get("new_tokens"), "think_tokens": gen.get("think_tokens"),
                     "thinking_forced": gen.get("thinking_forced"), "seconds": gen.get("seconds"),
                     "issues": " | ".join(i["issue"] for i in v.get("issues", [])),
                     "weakest_sentence": g.get("weakest", ""), "question": r["question"], "answer": r.get("answer"),
                     "reference": item.get("reference", ""), "thinking": r.get("thinking") or ""})
    return pd.DataFrame(rows)


def stage_table(run: dict) -> pd.DataFrame:
    rows = [{"stage": i["stage"], "kind": i["kind"], "id": r["id"]} for r in run["records"]
            for i in (r.get("verdict") or {}).get("issues", [])]
    if not rows:
        return pd.DataFrame(columns=["stage", "kind", "questions"])
    t = pd.DataFrame(rows).drop_duplicates().groupby(["stage", "kind"])["id"].apply(
        lambda s: ", ".join(sorted(s))).reset_index(name="questions")
    t["stage"] = pd.Categorical(t["stage"], STAGES, ordered=True)
    return t.sort_values(["stage", "kind"]).reset_index(drop=True)


def _cell(x) -> str:
    if isinstance(x, float):
        if math.isnan(x):
            return ""
        return str(int(x)) if x.is_integer() else f"{x:.3f}"
    return str(x).replace("|", "/").replace("\n", " ")


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(_cell(row[c]) for c in cols) + " |")
    return "\n".join(out)


def write_report(run: dict, out_dir: Path, by_id: dict, testset_problems, run_log) -> dict:
    """report.md, results.csv and scorecard.csv in ``out_dir``; returns the three tables."""
    out_dir = Path(out_dir)
    sc, qt, st = scorecard(run), question_table(run, by_id), stage_table(run)
    mode = run.get("mode")
    n = {s: sum(r["split"] == s for r in run["records"]) for s in ("dev", "test")}
    m = run.get("models") or {}
    head = (f"policy_rag {run.get('version')} | test set {run.get('testset_version')} | {run.get('started')} | GPU "
            f"{(run.get('environment') or {}).get('gpu')} | chunks {', '.join(run.get('chunk_files') or [])} (sha256 "
            f"{(run.get('chunk_sha256') or '')[:12]}) | answer model {m.get('llm')} ({m.get('llm_precision')}, thinking "
            f"{'on' if m.get('enable_thinking') else 'off'}) | reranker {m.get('reranker')} | embedder {m.get('embed')}"
            + (f" | answers from run '{run['rescored_from']}', re-scored" if run.get("rescored_from") else "")
            + (f" | contexts from run '{run['context_source']}'" if run.get("context_source") else "")
            + (f" | overrides {run['overrides']}" if run.get("overrides") else ""))
    L = [f"# Evaluation: {run['name']}", "", head, "", "## Scorecard", "", md_table(sc), "",
         "Reading it: the unit is the question. Smallest difference in *fully correct* a paired test can detect "
         f"(80% power): about {100 * (mde(n['test']) or 0):.0f} points on the {n['test']} test questions, about "
         f"{100 * (mde(n['dev']) or 0):.0f} on the {n['dev']} dev questions. Dev results guide debugging; adopt or "
         "reject only on test. Checks are rules on text; `results.csv` has every answer next to its reference.", ""]
    if mode in ("full", "rescore"):
        failing = qt[qt["fully_correct"] == False]                                # noqa: E712
        L += ["## Where answers go wrong (first failing stage)", "",
              md_table(failing["first_stage"].value_counts().rename_axis("stage").reset_index(name="questions")),
              "", "## Issues by stage", "", md_table(st) if len(st) else "None.", ""]
    L += ["## Gate", "", md_table(pd.DataFrame(run["gate"]).T.reset_index(names="split")), "",
          "## Questions", "", md_table(qt[["id", "split", "type", "fully_correct", "first_stage", "critical", "fact_recall",
                                           "context_recall", "mrr10", "context_precision", "groundedness", "context_tokens",
                                           "think_tokens", "answer_tokens", "issues"]]), ""]
    if run.get("evidence_removed"):
        L += ["## Evidence removed from the context", "",
              md_table(pd.DataFrame(run["evidence_removed"])[["id", "chunks_removed", "outcome", "facts_stated",
                                                               "figures_not_in_context"]]), ""]
    L += ["## Problems noted during the run", ""] + ([f"- {x}" for x in run_log] or ["None."])
    L += ["", "## Test set against this chunk file", "",
          md_table(pd.DataFrame(testset_problems, columns=["id", "problem", "detail"])) if testset_problems
          else "All facts and evidence found; all out-of-scope terms absent."]
    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    qt.to_csv(out_dir / "results.csv", index=False)
    sc.to_csv(out_dir / "scorecard.csv", index=False)
    return {"scorecard": sc, "questions": qt, "stages": st}
