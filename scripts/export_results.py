"""Rebuild the published results under results/ from private run files (scores only, no policy text).

The run files themselves (run.json, dialogues.json) hold the answers and the retrieved passages, which quote the
policy, so they are not published. This script turns them into the files that are: per-question verdicts, scorecards,
paired comparisons and per-turn tables of the follow-up evaluations. Each output folder gets a ``source.json`` with the
sha256 of every input file, so anyone holding the same files can check that the published numbers come from them.

    python scripts/export_results.py run       <run.json> <out_dir>
    python scripts/export_results.py compare   <baseline run.json> <candidate run.json> <out_dir> --splits test
    python scripts/export_results.py dialogues <dialogues.json> <out_dir>
    python scripts/export_results.py reproduction <generating run.json> <out.csv> <run.json> [<run.json> ...]

Runs are re-scored with the current checks and test set before anything is written (``compare`` and ``run``), so all
published verdicts follow the same rules. Retrieval metrics and groundedness are taken from the run as measured.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_rag.config import ContextConfig, load_config  # noqa: E402
from policy_rag.context import ContextBuilder  # noqa: E402
from policy_rag.corpus import load_corpus  # noqa: E402
from policy_rag.evaluation.compare import compare, compare_runs  # noqa: E402
from policy_rag.evaluation.report import md_table, scorecard  # noqa: E402
from policy_rag.evaluation.scoring import Scorer  # noqa: E402
from policy_rag.evaluation.stats import mcnemar_exact, wilson  # noqa: E402
from policy_rag.evaluation.testset import load_testset  # noqa: E402

PER_QUESTION = ["id", "split", "type", "answerable", "everyday_wording", "fully_correct", "first_failing_stage",
                "critical", "issues", "gate", "top_rerank_score", "context_tokens", "search_recall", "context_recall",
                "mrr10", "groundedness_hhem", "think_tokens", "thinking_forced", "answer_tokens", "seconds"]


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_source(out: Path, inputs, notes=None, filename="source.json", **extra):
    info = {"inputs": [{"file": f"{Path(p).parent.name}/{Path(p).name}", "sha256": sha256(p)} for p in inputs],
            "exported_with": "scripts/export_results.py", **extra}
    if notes:
        info["notes"] = notes
    (out / filename).write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def scorer_for(cfg_path="configs/default.yaml"):
    cfg = load_config(ROOT / cfg_path)
    corpus = load_corpus([{"id": d.id, "chunks": str(ROOT / d.chunks), "name": d.name} for d in cfg.documents])
    testset = load_testset(ROOT / cfg.evaluation.testset)
    # token counts only matter for retrieval metrics, which are kept as measured in the run
    return cfg, testset, Scorer(testset, corpus, ContextBuilder(corpus, ContextConfig(), count_tokens=len))


def rescore(run: dict, scorer: Scorer) -> dict:
    """Current checks; retrieval metrics as measured in the run (they need the real tokenizer)."""
    run = copy.deepcopy(run)
    measured = {r["id"]: r.get("retrieval") for r in run["records"]}
    scorer.rescore_run(run)
    for r in run["records"]:
        if measured.get(r["id"]) is not None:
            r["retrieval"] = measured[r["id"]]
    return run


def per_question(run: dict, testset) -> pd.DataFrame:
    rows = []
    for r in run["records"]:
        v, rt, g, gen = r.get("verdict") or {}, r.get("retrieval") or {}, r.get("grounding") or {}, r.get("gen") or {}
        item = testset.by_id[r["id"]]
        rows.append({"id": r["id"], "split": item["split"], "type": item["type"], "answerable": item["answerable"],
                     "everyday_wording": item["paraphrase"], "fully_correct": bool(v.get("fully_correct")),
                     "first_failing_stage": v.get("first_stage") or "", "critical": ", ".join(v.get("critical", [])),
                     "issues": " | ".join(i["issue"] for i in v.get("issues", [])), "gate": r.get("gate_route"),
                     "top_rerank_score": r.get("top_score"), "context_tokens": r.get("context_tokens"),
                     "search_recall": rt.get("search_recall"), "context_recall": rt.get("context_recall"),
                     "mrr10": rt.get("mrr10"), "groundedness_hhem": g.get("groundedness"),
                     "think_tokens": gen.get("think_tokens"), "thinking_forced": gen.get("thinking_forced"),
                     "answer_tokens": gen.get("answer_tokens"), "seconds": gen.get("seconds")})
    return pd.DataFrame(rows, columns=PER_QUESTION)


def cmd_run(args):
    cfg, testset, scorer = scorer_for(args.config)
    run = rescore(json.loads(Path(args.run).read_text(encoding="utf-8")), scorer)
    inputs, notes = [args.run], []
    if args.generation_run:
        # the answers came from another run's cache: take the measurements made while generating them from that run
        gen = json.loads(Path(args.generation_run).read_text(encoding="utf-8"))
        assert {r["id"]: r["answer"] for r in gen["records"]} == {r["id"]: r["answer"] for r in run["records"]}, \
            "the generation run must hold exactly the same answers"
        for k in ("tokens_per_second", "peak_gpu_gb", "evidence_removed", "repeat"):
            run[k] = gen.get(k)
        inputs.append(args.generation_run)
        notes.append("Answers, verdicts, retrieval and groundedness from the first input; tokens per second, peak GPU "
                     "memory, the evidence-removed test and the repeat check from the second, the run that generated "
                     "the same answers (checked identical).")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_question(run, testset).to_csv(out / "per_question.csv", index=False)
    sc = scorecard(run)
    sc.to_csv(out / "scorecard.csv", index=False)
    (out / "scorecard.md").write_text(f"# Scorecard: {args.title or run.get('name')}\n\n" + md_table(sc) + "\n",
                                      encoding="utf-8")
    write_source(out, inputs, notes, run_name=run.get("name"), started=run.get("started"),
                 testset_version=testset.version, chunk_sha256=run.get("chunk_sha256"),
                 models=run.get("models"), packages=(run.get("environment") or {}).get("packages"),
                 gpu=(run.get("environment") or {}).get("gpu"), llm_calls=run.get("llm_calls"))
    print(f"wrote {out}/per_question.csv, scorecard.csv, scorecard.md, source.json")


def cmd_compare(args):
    cfg, testset, scorer = scorer_for(args.config)
    base, cand = (rescore(json.loads(Path(p).read_text(encoding="utf-8")), scorer) for p in (args.baseline, args.candidate))
    if args.names:
        base["name"], cand["name"] = args.names
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = compare_runs(base, [cand], args.splits, cfg.decision)
    pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)
    secondary = [{"split": sp, **row} for sp in args.splits
                 for row in compare(base, cand, sp, cfg.decision.n_bootstrap)["rows"]]
    pd.DataFrame(secondary).to_csv(out / "secondary.csv", index=False)
    a = {r["id"]: bool((r.get("verdict") or {}).get("fully_correct")) for r in base["records"]}
    b = {r["id"]: bool((r.get("verdict") or {}).get("fully_correct")) for r in cand["records"]}
    ids = [i for i in a if i in b]
    paired = pd.DataFrame([{"id": i, "split": testset.by_id[i]["split"], "fully_correct_baseline": a[i],
                            "fully_correct_candidate": b[i]} for i in ids])
    paired.to_csv(out / "paired.csv", index=False)
    write_source(out, [args.baseline, args.candidate], filename="comparison_source.json",
                 baseline=base.get("name"), candidate=cand.get("name"),
                 splits=args.splits, testset_version=testset.version)
    print(f"wrote {out}/comparison.csv, secondary.csv, paired.csv, comparison_source.json")


def cmd_dialogues(args):
    j = json.loads(Path(args.dialogues).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in j["turns"]:
        c = t["conditions"]
        rows.append({"turn": f"{t['dialogue']}.{t['turn']}", "split": t["split"], "kind": t["kind"], "item": t["item"],
                     "asked": t["asked"], "searched_as": t["rewrite"], "skipped_rewriter": bool(t.get("skipped")),
                     "fallback": t.get("fallback") or "", "rewrite_keywords_ok": t.get("rewrite_ok"),
                     "rewrite_seconds": t.get("rewrite_seconds"), "rewrite_think_tokens": t.get("rewrite_think_tokens"),
                     **{f"correct_{k}": c[k]["fully_correct"] for k in ("no_memory", "memory", "gold") if k in c},
                     "memory_first_stage": c["memory"].get("first_stage") or "", "memory_issues": c["memory"].get("issues") or ""})
    df = pd.DataFrame(rows)
    df.to_csv(out / "turns.csv", index=False)
    test = df[df.split == "test"]
    need = test[test.kind.isin(["follow_up", "after_refusal"])]
    harm = test[test.kind.isin(["self_contained", "topic_switch"])]
    fixed = need[need.correct_memory & ~need.correct_no_memory].turn.tolist()
    broken = need[~need.correct_memory & need.correct_no_memory].turn.tolist()
    harm_broken = harm[~harm.correct_memory & harm.correct_no_memory].turn.tolist()
    p = mcnemar_exact(len(broken), len(fixed))
    by = []
    for cond in ("no_memory", "memory", "gold"):
        row = {"condition": cond}
        for kind in ("follow_up", "after_refusal", "self_contained", "topic_switch"):
            k = test[test.kind == kind]
            row[kind] = f"{int(k[f'correct_{cond}'].sum())}/{len(k)}" if len(k) else ""
        k, n = int(need[f"correct_{cond}"].sum()), len(need)
        lo, hi = wilson(k, n)
        row["deciding (follow_up + after_refusal)"] = f"{k}/{n} ({100 * lo:.0f}–{100 * hi:.0f}%)"
        by.append(row)
    secs = df[df.rewrite_seconds.notna()].rewrite_seconds
    L = [f"# Follow-up evaluation: {args.title or j['name']}", "",
         f"policy_rag {j['version']} | {j['started']} | dialogues `{j['dialogues_file']}` version `{j['dialogues_version']}` | "
         f"rewrite template `{j['config']['memory']['template']}` | test set `{j['testset_version']}`", "",
         f"**Decision (rule fixed before the run):** {j['decision']}", "",
         f"Test follow-ups that need the conversation: {len(need)}; fully correct without memory "
         f"{int(need.correct_no_memory.sum())}, with memory {int(need.correct_memory.sum())}, as the complete question "
         f"{int(need.correct_gold.sum())}. Fixed by memory: {', '.join(fixed) or 'none'}; broken: {', '.join(broken) or 'none'}; "
         f"McNemar exact p = {p:.4f}. Harm-check turns (self_contained, topic_switch) made wrong: "
         f"{', '.join(harm_broken) or 'none'} of {len(harm)}.", "",
         "## Fully correct by condition (test split)", "", md_table(pd.DataFrame(by)), "",
         f"Rewrites: {int((df.searched_as != df.asked).sum())} of {len(df)} turns changed; {int(df.skipped_rewriter.sum())} "
         f"not sent to the rewriter; {int((df.fallback != '').sum())} rejected by the checks. Rewrite time (seconds, "
         f"generated): median {secs.median():.1f}, max {secs.max():.0f}. Total run time {j['wall_seconds'] / 60:.0f} min.", "",
         "## Turns", "", md_table(df[["turn", "kind", "asked", "searched_as", "fallback", "correct_no_memory",
                                      "correct_memory", "correct_gold", "memory_issues"]]), ""]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    write_source(out, [args.dialogues], run_name=j["name"], started=j["started"], dialogues_version=j["dialogues_version"],
                 template=j["config"]["memory"]["template"], decision=j["decision"])
    print(f"wrote {out}/turns.csv, summary.md, source.json")


def cmd_reproduction(args):
    """Runs that reused a reference run's answers from the cache: are answers, contexts and verdicts the same?"""
    cfg, testset, scorer = scorer_for(args.config)
    ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    ref_rec = {r["id"]: r for r in ref["records"]}
    rows = []
    for p in args.runs:
        run = rescore(json.loads(Path(p).read_text(encoding="utf-8")), scorer)
        recs = {r["id"]: r for r in run["records"]}
        g = [r["grounding"] for r in run["records"] if (r.get("grounding") or {}).get("groundedness") is not None]
        rows.append({"run": run.get("name"), "started": run.get("started"), "package": run.get("version"),
                     "gpu": (run.get("environment") or {}).get("gpu"),
                     "transformers": ((run.get("environment") or {}).get("packages") or {}).get("transformers"),
                     "answers_generated": (run.get("llm_calls") or {}).get("generated"),
                     "answers_from_cache": (run.get("llm_calls") or {}).get("from_cache"),
                     "answers_identical": sum(recs[i]["answer"] == ref_rec[i]["answer"] for i in ref_rec if i in recs),
                     "candidates_identical": sum(recs[i]["candidates"] == ref_rec[i]["candidates"] for i in ref_rec if i in recs),
                     "rerank_scores_identical": sum(recs[i]["ranked"] == ref_rec[i]["ranked"] for i in ref_rec if i in recs),
                     "contexts_identical": sum(recs[i]["context"] == ref_rec[i]["context"] for i in ref_rec if i in recs),
                     "gate_identical": sum(recs[i]["gate_route"] == ref_rec[i]["gate_route"] for i in ref_rec if i in recs),
                     "questions": len(recs),
                     "fully_correct": sum(bool((r.get("verdict") or {}).get("fully_correct")) for r in run["records"]),
                     "fully_correct_test": sum(bool((r.get("verdict") or {}).get("fully_correct")) for r in run["records"]
                                               if testset.by_id[r["id"]]["split"] == "test"),
                     "groundedness_mean": round(sum(x["groundedness"] for x in g) / len(g), 3) if g else None,
                     "answers_with_unsupported_sentence": f"{sum(bool(x['hallucinated']) for x in g)}/{len(g)}" if g else None,
                     "sha256": sha256(p)})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} (reference {Path(args.reference).name}, sha256 {sha256(args.reference)[:12]})")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="configs/default.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("run")
    r.add_argument("out")
    r.add_argument("--title")
    r.add_argument("--generation-run", help="the run that generated the (cached) answers: its speed, memory, "
                                            "evidence-removed and repeat measurements are used")
    c = sub.add_parser("compare")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.add_argument("out")
    c.add_argument("--splits", nargs="+", default=["test"])
    c.add_argument("--names", nargs=2)
    d = sub.add_parser("dialogues")
    d.add_argument("dialogues")
    d.add_argument("out")
    d.add_argument("--title")
    rp = sub.add_parser("reproduction")
    rp.add_argument("reference", help="the run that generated the answers")
    rp.add_argument("out", help="CSV to write")
    rp.add_argument("runs", nargs="+", help="runs that reused its answers")
    args = p.parse_args(argv)
    {"run": cmd_run, "compare": cmd_compare, "dialogues": cmd_dialogues, "reproduction": cmd_reproduction}[args.cmd](args)


if __name__ == "__main__":
    main()
