"""Command line: ``policy-rag <command>`` (or ``python -m policy_rag <command>``).

    policy-rag chunk-pdf policy.pdf data/policy_chunks.json      # PDF -> chunk file (CPU, seconds)
    policy-rag info                                              # what the config loads
    policy-rag ask "What is the waiting period for pre-existing diseases?"
    policy-rag evaluate --name baseline                          # full evaluation (GPU)
    policy-rag evaluate --name retr --mode retrieval             # retrieval only (no answer model)
    policy-rag evaluate --name check --mode rescore --rescore-from runs/baseline/run.json
    policy-rag compare runs/baseline/run.json runs/candidate/run.json
    policy-rag agreement runs/baseline/run.json labels_filled.csv

Every command takes ``--config`` (default ``configs/default.yaml``) and ``--set key=value`` overrides.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_CONFIG = "configs/default.yaml"


def _config(args):
    from .config import load_config
    return load_config(args.config, args.set)


def cmd_chunk_pdf(args):
    try:
        from .ingest.optima_secure_pdf import main as chunk_main
    except ImportError as e:
        sys.exit(f"{e}: install the ingest extra first: pip install -e \".[ingest]\"")
    chunk_main([args.pdf, args.out] + (["--csv", args.csv] if args.csv else []))


def cmd_info(args):
    from . import registry
    from .config import DocumentConfig, resolve_input
    from .corpus import load_corpus
    cfg = _config(args)
    print(f"Config {cfg.source} (fingerprint {cfg.fingerprint()})")
    corpus = load_corpus([DocumentConfig(d.id, str(resolve_input(d.chunks, cfg)), d.name) for d in cfg.documents])
    print(corpus.summary())
    print(f"Embedder {cfg.embedder.type}: {cfg.embedder.model} | reranker {cfg.reranker.type}: {cfg.reranker.model} | "
          f"generator {cfg.generator.type}: {cfg.generator.model} ({'4-bit' if cfg.generator.load_in_4bit else 'full precision'}, "
          f"thinking {'on, budget ' + str(cfg.generator.think_budget) if cfg.generator.enable_thinking else 'off'})")
    print("Registered types:", {k: registry.registered(k) for k in ("embedder", "reranker", "generator")})


def cmd_ask(args):
    from .pipeline import RAGSystem
    rag = RAGSystem.from_config(args.config, args.set)
    questions = [{"question": q} for q in args.question or []]
    if args.questions_file:
        data = json.loads(Path(args.questions_file).read_text(encoding="utf-8"))
        items = data.get("items", data) if isinstance(data, dict) else data
        questions += [q if isinstance(q, dict) else {"question": q} for q in items]
    if not questions:
        sys.exit("give a question, or --questions-file")
    rows = []
    for item in questions:
        q = item["question"]
        print("=" * 100)
        r = rag.ask(q, documents=args.document)
        if args.show_context:
            print("\nCONTEXT:\n" + rag.context.format(r.state.get("groups", [])))
        gen = r.state.get("gen") or {}
        if item.get("expected"):
            print("\nEXPECTED:", item["expected"])
        rows.append({"id": item.get("id", ""), "question": q, "answer": r.answer, "expected": item.get("expected", ""),
                     "route": r.route, "top_score": r.top_score,
                     "sources": " | ".join(r.sources), "think_tokens": gen.get("think_tokens"),
                     "thinking_forced": gen.get("thinking_forced"), "seconds": gen.get("seconds"),
                     "from_cache": gen.get("from_cache"), "thinking": gen.get("thinking")})
    if args.out:
        import pandas as pd
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nSaved {args.out}")


def cmd_evaluate(args):
    from .evaluation.runner import Evaluation
    cfg = _config(args)
    ev = Evaluation(cfg, args.name, args.mode, args.splits, args.rescore_from, args.contexts_from, args.resume_from)
    ev.overrides = list(args.set or [])
    run = ev.run(make_zip_file=args.zip)
    print("\n=== Scorecard (decisions use the held-out test split)")
    print(run["tables"]["scorecard"].to_string(index=False))
    if run.get("zip"):
        print(f"\nZip: {run['zip']}")


def cmd_compare(args):
    from .config import DocumentConfig, resolve_input
    from .context import ContextBuilder
    from .corpus import load_corpus
    from . import registry
    from .evaluation.compare import blind_sheet, compare_runs
    from .evaluation.runs import load_run
    from .evaluation.scoring import Scorer
    from .evaluation.testset import load_testset
    cfg = _config(args)
    corpus = load_corpus([DocumentConfig(d.id, str(resolve_input(d.chunks, cfg)), d.name) for d in cfg.documents])
    testset = load_testset(resolve_input(cfg.evaluation.testset, cfg))
    gen = registry.build("generator", cfg.generator)             # tokenizer only
    scorer = Scorer(testset, corpus, ContextBuilder(corpus, cfg.context, gen.count_tokens),
                    testset.document if corpus.multi_document else None)
    runs = [scorer.rescore_run(load_run(p)) for p in [args.baseline] + args.candidates]
    rows = compare_runs(runs[0], runs[1:], args.splits, cfg.decision)
    out = Path(args.out or Path(cfg.evaluation.output_dir) / f"compare_{runs[0]['name']}")
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)
    p = blind_sheet(runs[0], runs[1], testset.by_id, out)
    print(f"\nSaved {out}/comparison.csv and {p.name} (answers that differ, as A/B with names hidden; key in blind_key.json)")


def cmd_agreement(args):
    from .evaluation.labels import agreement, read_labels
    from .evaluation.runs import load_run
    cfg = _config(args)
    res = agreement(load_run(args.run), read_labels(args.labels))
    if not res["n"]:
        sys.exit("no labelled answers of this run in the labels file")
    print(f"kappa {res['kappa']:.2f} on {res['n']} answers (agree on {res['n'] - len(res['disagreements'])}); "
          f"target >= {cfg.decision.kappa_target}")
    for d in res["disagreements"]:
        print(f"  {d['id']}: human {d['human']}, automatic {d['automatic']} | {d['issues']}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="policy-rag", description="Grounded question answering over insurance policies.")
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--config", default=DEFAULT_CONFIG, help=f"YAML config (default {DEFAULT_CONFIG})")
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="override a setting, e.g. --set generator.think_budget=1536 (repeatable)")
        return p

    p = sub.add_parser("chunk-pdf", help="chunk the my:Optima Secure PDF into a chunk file")
    p.add_argument("pdf")
    p.add_argument("out")
    p.add_argument("--csv")
    p.set_defaults(func=cmd_chunk_pdf)

    p = common(sub.add_parser("info", help="show what the config loads"))
    p.set_defaults(func=cmd_info)

    p = common(sub.add_parser("ask", help="answer questions"))
    p.add_argument("question", nargs="*")
    p.add_argument("--questions-file", help="JSON list of questions (or a test-set-style file with items)")
    p.add_argument("--document", action="append", help="search only this document id (repeatable)")
    p.add_argument("--show-context", action="store_true", help="print the context the answer model read")
    p.add_argument("--out", help="save the answers as CSV")
    p.set_defaults(func=cmd_ask)

    p = common(sub.add_parser("evaluate", help="run the test set and write a report"))
    p.add_argument("--name", required=True, help="run name: results go to <output_dir>/<name>/")
    p.add_argument("--mode", default="full", choices=["full", "retrieval", "rescore"])
    p.add_argument("--splits", nargs="+", choices=["dev", "test"], help="default: evaluation.splits in the config")
    p.add_argument("--rescore-from", help="mode rescore: the saved run (run.json or results zip)")
    p.add_argument("--contexts-from", help="answer the saved contexts of this run again (run.json or zip)")
    p.add_argument("--resume-from", action="append", help="add answers from an answer_cache.jsonl or results zip")
    p.add_argument("--zip", action="store_true", help="also write one zip of the results folder")
    p.set_defaults(func=cmd_evaluate)

    p = common(sub.add_parser("compare", help="paired comparison of runs and the adopt/keep decision"))
    p.add_argument("baseline")
    p.add_argument("candidates", nargs="+")
    p.add_argument("--splits", nargs="+", default=["dev", "test"], choices=["dev", "test"])
    p.add_argument("--out", help="output folder (default <output_dir>/compare_<baseline>)")
    p.set_defaults(func=cmd_compare)

    p = common(sub.add_parser("agreement", help="kappa between the automatic verdicts and human labels"))
    p.add_argument("run")
    p.add_argument("labels", help="labels_to_fill.csv with human_fully_correct filled in (1/0)")
    p.set_defaults(func=cmd_agreement)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
