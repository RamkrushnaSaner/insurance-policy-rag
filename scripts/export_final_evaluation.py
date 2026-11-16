"""Export the evaluation of v3.4 on every question set it answered (scores only, no policy text).

The saved runs of the v3.4 configuration are re-scored with this repository's checks, the person's blind labels are
added where they exist, and ``<out_dir>`` gets per_question.csv, scorecard.csv, scorecard.md and source.json. Nothing
is generated: answers, rankings, contexts and timings come from the run files, and retrieval metrics are kept as
measured in the run (they need the real tokenizer).

    DATABASE_URL=... python scripts/export_final_evaluation.py results/final_evaluation \\
        --run dev_v3=runs/E0_v3_113/run.json \\
        --run fresh_v1=runs/E0_fresh_v1/run.json --testset fresh_v1=private/optima_secure_fresh_v1.json \\
        --run fresh_v2=... --testset fresh_v2=... --run fresh_v3=... --testset fresh_v3=... \\
        --run fresh_v4=... --testset fresh_v4=... \\
        --table-questions fresh_v3=private/release_rules_r1.json:data_roles.decision_set.table_questions \\
        --table-questions fresh_v4=private/release_rules_final.json:decision_set.table_questions \\
        --labels fresh_v4=private/final_decision_record.json:labels

A set without ``--testset`` uses the config's test set (the 113-question development set). A set without
``--table-questions`` calls an answerable question a table question when one of its key facts is printed only in a
table (not in the text chunks). The chunk file and the database (for the text of the printed tables) are the
config's; the chunk file must be the one the runs used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_rag.config import ContextConfig, load_config  # noqa: E402
from policy_rag.context import ContextBuilder  # noqa: E402
from policy_rag.corpus import load_corpus  # noqa: E402
from policy_rag.database import Database  # noqa: E402
from policy_rag.evaluation.checks import cited_sections, unsupported_citations  # noqa: E402
from policy_rag.evaluation.report import md_table  # noqa: E402
from policy_rag.evaluation.scoring import Scorer  # noqa: E402
from policy_rag.evaluation.stats import wilson  # noqa: E402
from policy_rag.evaluation.testset import anchor_in, fact_found, load_testset, norm_text, words  # noqa: E402
from policy_rag.tables import TableData  # noqa: E402

CONFIG = "configs/experiments/qwen3_text_tables.yaml"
FIRST_STAGE_K, RERANK_K = (5, 10, 20, 64), (1, 3, 5, 10)
BUDGET = 1500  # the context budget of the config; the table passage is added outside it


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pairs(values, what):
    out = {}
    for v in values or []:
        if "=" not in v:
            raise SystemExit(f"--{what} expects SET=VALUE, got {v!r}")
        k, val = v.split("=", 1)
        out[k] = val
    return out


def from_json(spec: str):
    """``file.json:dotted.key`` -> the value at that key (the whole file without a key)."""
    path, _, key = spec.partition(":")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    for part in [p for p in key.split(".") if p]:
        value = value[part]
    return path, value


def key_facts(item):
    """Key facts, without the ones that are figures the answer must work out (not printed anywhere)."""
    return [f for f in item["facts"] if not any(fact_found(f, str(d[1])) for d in item.get("derived_figures", []))]


class SetExporter:
    def __init__(self, cfg, corpus, table_text):
        self.cfg, self.corpus, self.table_text = cfg, corpus, table_text
        chunks = corpus.chunks
        self.chunk_words = [" " + words(c.label + " " + c.text) + " " for c in chunks]
        self.text_norm = norm_text(" ".join(c.label + " " + c.text for c in chunks))

    def recall_at(self, item, order, k):
        """Share of the item's evidence passages that are in the text chunks found among the first k of ``order``."""
        ev = [e for e in item["evidence"] if any(anchor_in(e, w) for w in self.chunk_words)]
        if not ev:
            return None
        top = [self.chunk_words[i] for i in order[:k]]
        return sum(any(anchor_in(e, w) for w in top) for e in ev) / len(ev)

    def stratum(self, item, table_ids):
        if not item["answerable"]:
            return "not_answerable"
        if table_ids is not None:
            return "table" if item["id"] in table_ids else "text"
        return "text" if all(fact_found(f, self.text_norm) for f in key_facts(item)) else "table"

    def rows(self, name, run, testset, table_ids, labels):
        scorer = Scorer(testset, self.corpus, ContextBuilder(self.corpus, ContextConfig(), count_tokens=len),
                        table_text=self.table_text)
        measured = {r["id"]: r.get("retrieval") for r in run["records"]}
        scorer.rescore_run(run)
        out = []
        for rec in run["records"]:
            item = testset.by_id[rec["id"]]
            v, rt = rec.get("verdict") or {}, measured.get(rec["id"]) or {}
            gen, grounding, table = rec.get("gen") or {}, rec.get("grounding") or {}, rec.get("table") or {}
            answer, context = rec.get("answer") or "", rec.get("context") or ""
            ctx_norm = norm_text(context)
            ranked = [i for i, _ in rec.get("ranked") or []]
            candidates = rec.get("candidates") or []
            cited = cited_sections(answer)
            expected = set(item.get("sections") or [])
            label = labels.get(rec["id"])
            checks_ok = bool(v.get("fully_correct"))
            row = {
                "set": name, "id": rec["id"], "stratum": self.stratum(item, table_ids), "type": item["type"],
                "answerable": item["answerable"], "everyday_wording": bool(item.get("paraphrase")),
                "fully_correct": (label == "correct") if label else checks_ok,
                "fully_correct_checks": checks_ok, "person_label": label or "",
                "first_failing_stage": v.get("first_stage") or "",
                "issue_kinds": " ".join(sorted({i["kind"] for i in v.get("issues", [])})),
                "critical": " ".join(v.get("critical") or []),
                "refused": bool(v.get("refused")),
                "declined": bool(v.get("refused") or v.get("false_refusal") or v.get("hedged")),
                "gate": rec.get("gate_route"), "top_rerank_score": rec.get("top_score"),
                "search_recall": rt.get("search_recall"), "context_recall": rt.get("context_recall"),
                "mrr10": rt.get("mrr10"), "context_precision": rt.get("context_precision"),
                "all_key_facts_in_context": (all(fact_found(f, ctx_norm) for f in item["facts"])
                                             if item["answerable"] and item["facts"] else None),
                "passages_read": sum(len(g) for g in rec.get("groups") or []),
                "context_tokens": rec.get("context_tokens"),
                "table_route_status": table.get("status"), "table_route_complete": bool(table.get("complete")),
                "cites_a_section": bool(cited),
                "cites_a_section_holding_the_answer": (any(c == s or c.startswith(s + ".") or s.startswith(c + ".")
                                                           for c in cited for s in expected) if cited else None),
                "citations_not_in_context": len(unsupported_citations(answer, context)),
                "groundedness_hhem": grounding.get("groundedness"), "hhem_flagged": grounding.get("hallucinated"),
                "think_tokens": gen.get("think_tokens"), "thinking_forced": gen.get("thinking_forced"),
                "answer_tokens": gen.get("answer_tokens"), "hit_length_limit": gen.get("hit_limit"),
                "seconds": gen.get("seconds"), "answer_from_cache": gen.get("from_cache"),
                "error": bool(rec.get("error")), "table_error": bool(rec.get("table_error")),
            }
            for k in FIRST_STAGE_K:
                row[f"first_stage_recall@{k}"] = self.recall_at(item, candidates, k) if item["answerable"] else None
            for k in RERANK_K:
                row[f"rerank_recall@{k}"] = self.recall_at(item, ranked, k) if item["answerable"] else None
            out.append(row)
        return out


# ------------------------------------------------------------------------------------------------------ scorecard
def frac(k, n):
    return f"{k}/{n} = {100 * k / n:.1f}%" if n else "–"


def ci(k, n):
    lo, hi = wilson(k, n)
    return f"{100 * lo:.0f}–{100 * hi:.0f}%"


def mean(xs, digits=3):
    xs = [x for x in xs if x is not None and x == x]
    return f"{statistics.mean(xs):.{digits}f}" if xs else "–"


def pct_index(xs, p):
    return xs[int(p * (len(xs) - 1))]


def scorecard(df: pd.DataFrame, runs: dict) -> pd.DataFrame:
    groups = {s: df[df.set == s] for s in df.set.unique()}
    fresh = [s for s in groups if s.startswith("fresh")]
    if fresh:
        groups["fresh_total"] = df[df.set.isin(fresh)]
    measures = []

    def add(measure, fn):
        measures.append({"measure": measure, **{g: fn(d) for g, d in groups.items()}})

    A = lambda d: d[d.answerable]  # noqa: E731
    add("questions", lambda d: str(len(d)))
    add("fully correct (person's label where given)", lambda d: frac(int(d.fully_correct.sum()), len(d)))
    add("95% CI (Wilson)", lambda d: ci(int(d.fully_correct.sum()), len(d)))
    add("fully correct, checks only", lambda d: frac(int(d.fully_correct_checks.sum()), len(d)))
    add("answers labelled by a person", lambda d: str(int((d.person_label != "").sum())))
    add("answerable questions fully correct", lambda d: frac(int(A(d).fully_correct.sum()), len(A(d))))
    for s in ("table", "text"):
        add(f"{s} questions fully correct", lambda d, s=s: frac(int(d[d.stratum == s].fully_correct.sum()),
                                                               int((d.stratum == s).sum())))
    add("not-answerable questions declined", lambda d: frac(int(d[~d.answerable].fully_correct.sum()),
                                                            int((~d.answerable).sum())))
    add("everyday wording, fully correct", lambda d: frac(int(A(d)[A(d).everyday_wording].fully_correct.sum()),
                                                          int(A(d).everyday_wording.sum())))
    add("policy wording, fully correct", lambda d: frac(int(A(d)[~A(d).everyday_wording].fully_correct.sum()),
                                                        int((~A(d).everyday_wording).sum())))
    add("answerable answers that decline (refusal or hedged opening)",
        lambda d: frac(int(A(d).declined.sum()), len(A(d))))
    add("failures by first failing stage (checks)", lambda d: " · ".join(
        f"{k} {v}" for k, v in d[~d.fully_correct_checks].first_failing_stage.value_counts().items()))
    R = lambda d: A(d)[A(d).search_recall.notna()]  # noqa: E731
    add("all evidence among the candidates", lambda d: frac(int((R(d).search_recall == 1).sum()), len(R(d))))
    add("all evidence in the context", lambda d: frac(int((R(d).context_recall == 1).sum()), len(R(d))))
    add("all key facts in the context", lambda d: frac(int(R(d).all_key_facts_in_context.fillna(False).sum()),
                                                       len(R(d))))
    T = lambda d: A(d)[A(d)["rerank_recall@1"].notna()]  # noqa: E731
    add("questions with evidence in the text chunks", lambda d: str(len(T(d))))
    add("first stage (RRF) recall @5 / @10 / @20 / @64", lambda d: " / ".join(
        mean(T(d)[f"first_stage_recall@{k}"], 2) for k in FIRST_STAGE_K))
    add("reranker recall @1 / @3 / @5 / @10", lambda d: " / ".join(
        mean(T(d)[f"rerank_recall@{k}"], 2) for k in RERANK_K))
    add("MRR@10 (text evidence)", lambda d: mean(T(d).mrr10, 2))
    add("context precision (mean)", lambda d: mean(R(d).context_precision, 2))
    add("passages read / context tokens (median)", lambda d: (
        f"{statistics.median(R(d).passages_read):g} / {statistics.median(R(d).context_tokens.dropna()):g}"))
    add(f"contexts over {BUDGET} tokens", lambda d: str(int((d.context_tokens > BUDGET).sum())))
    add("table route answered completely: table / text questions", lambda d: (
        f"{int(d[d.stratum == 'table'].table_route_complete.sum())}/{int((d.stratum == 'table').sum())} / "
        f"{int(d[d.stratum == 'text'].table_route_complete.sum())}/{int((d.stratum == 'text').sum())}"))
    C = lambda d: A(d)[~A(d).refused]  # noqa: E731
    add("cites a section (answerable, not refused)", lambda d: frac(int(C(d).cites_a_section.sum()), len(C(d))))
    add("citations of a section not in the context", lambda d: frac(
        int((C(d)[C(d).cites_a_section].citations_not_in_context > 0).sum()), int(C(d).cites_a_section.sum())))
    add("cites a section that holds the answer", lambda d: frac(
        int(C(d)[C(d).cites_a_section].cites_a_section_holding_the_answer.fillna(False).sum()),
        int(C(d).cites_a_section.sum())))
    N = lambda d: d[~d.refused]  # noqa: E731
    add("answers with a figure not in the context (not refused)", lambda d: frac(
        int(N(d).issue_kinds.str.contains("unsupported_figures").sum()), len(N(d))))
    add("critical issues by kind (checks)", lambda d: " · ".join(
        f"{k} {v}" for k, v in pd.Series(" ".join(d.critical).split()).value_counts().items()) or "none")
    G = lambda d: N(d)[N(d).groundedness_hhem.notna()]  # noqa: E731
    add("HHEM groundedness, mean (not refused)", lambda d: mean(G(d).groundedness_hhem))
    add("answers HHEM flags (not refused)", lambda d: frac(int(G(d).hhem_flagged.fillna(False).astype(bool).sum()),
                                                           len(G(d))))

    def latency(d):
        s = sorted(d.seconds.dropna())
        return (f"{statistics.median(s):.1f} / {pct_index(s, 0.9):.1f} / {pct_index(s, 0.95):.1f} / {max(s):.1f} "
                f"(n={len(s)})") if s else "–"
    add("seconds per answer: median / p90 / p95 / max", latency)
    add("thinking tokens, median", lambda d: f"{statistics.median(d.think_tokens.dropna()):g}")
    add("thinking stopped at the 1,024-token budget", lambda d: frac(
        int(d[d.seconds.notna()].thinking_forced.fillna(False).astype(bool).sum()), int(d.seconds.notna().sum())))
    add("answers cut off at the length limit", lambda d: str(int(d.hit_length_limit.fillna(False).astype(bool).sum())))
    add("pipeline errors / table-route errors", lambda d: f"{int(d.error.sum())} / {int(d.table_error.sum())}")

    def run_info(key, fmt):
        return {g: (fmt(runs[g]) if g in runs else "") for g in groups}
    measures.append({"measure": "run: started (UTC)", **run_info("started", lambda r: (r.get("started") or "")[:16])})
    measures.append({"measure": "run: wall time, all included (min)",
                     **run_info("wall", lambda r: f"{(r.get('wall_seconds') or 0) / 60:.0f}")})
    measures.append({"measure": "run: answers generated in this run / from the cache", **run_info(
        "cache", lambda r: (f"{sum(not (x.get('gen') or {}).get('from_cache') for x in r['records'] if x.get('gen'))}"
                            f" / {sum(bool((x.get('gen') or {}).get('from_cache')) for x in r['records'])}"))})
    measures.append({"measure": "run: peak GPU memory (GB)", **run_info("gpu", lambda r: f"{r.get('peak_gpu_gb')}")})
    measures.append({"measure": "run: search repeated, identical", **run_info(
        "rrep", lambda r: f"{(r.get('retrieval_repeat') or {}).get('identical')}/{len(r['records'])}")})
    measures.append({"measure": "run: one answer regenerated without the cache, identical", **run_info(
        "rep", lambda r: str((r.get("repeat") or {}).get("identical")))})
    return pd.DataFrame(measures)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out")
    p.add_argument("--config", default=CONFIG)
    p.add_argument("--run", action="append", required=True, help="SET=run.json")
    p.add_argument("--testset", action="append", help="SET=testset.json (default: the config's)")
    p.add_argument("--table-questions", action="append", help="SET=file.json:dotted.key (a list of question ids)")
    p.add_argument("--labels", action="append", help="SET=file.json:dotted.key ({'E0:<id>': 'correct'|'not correct'})")
    p.add_argument("--labels-system", default="E0", help="system prefix of the labels to use")
    args = p.parse_args(argv)

    cfg = load_config(ROOT / args.config)
    corpus = load_corpus([{"id": d.id, "chunks": str(ROOT / d.chunks), "name": d.name} for d in cfg.documents])
    table_text = TableData(Database(cfg.database), cfg.documents[0].id).all_rows_text()
    exporter = SetExporter(cfg, corpus, table_text)
    runs_in, testsets_in = pairs(args.run, "run"), pairs(args.testset, "testset")
    tables_in, labels_in = pairs(args.table_questions, "table-questions"), pairs(args.labels, "labels")

    rows, runs, sources = [], {}, {"sets": {}}
    for name, run_path in runs_in.items():
        run = json.loads(Path(run_path).read_text(encoding="utf-8"))
        assert run.get("chunk_sha256") == corpus.fingerprint, f"{name}: the run used another chunk file"
        ts_path = Path(testsets_in.get(name) or ROOT / cfg.evaluation.testset)
        testset = load_testset(ts_path)
        assert run.get("testset_version") == testset.version, f"{name}: the run answered another test-set version"
        info = {"run": {"file": f"{Path(run_path).parent.name}/{Path(run_path).name}", "sha256": sha256(run_path),
                        "name": run.get("name"), "started": run.get("started"), "gpu": (run.get("environment") or {})
                        .get("gpu")},
                "testset": {"file": ts_path.name, "version": testset.version, "sha256": sha256(ts_path),
                            "in_repository": ts_path.resolve().is_relative_to(ROOT)}}
        table_ids = None
        if name in tables_in:
            src, ids = from_json(tables_in[name])
            table_ids = set(ids)
            info["table_questions"] = {"file": Path(src).name, "sha256": sha256(src), "key": tables_in[name]
                                       .partition(":")[2], "n": len(table_ids)}
        labels = {}
        if name in labels_in:
            src, lab = from_json(labels_in[name])
            prefix = args.labels_system + ":"
            labels = {k[len(prefix):]: v for k, v in lab.items() if k.startswith(prefix)}
            info["labels"] = {"file": Path(src).name, "sha256": sha256(src), "system": args.labels_system,
                              "n": len(labels)}
        rows += exporter.rows(name, run, testset, table_ids, labels)
        runs[name] = run
        sources["sets"][name] = info
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_question.csv", index=False)
    sc = scorecard(df, runs)
    sc.to_csv(out / "scorecard.csv", index=False)
    (out / "scorecard.md").write_text("# Scorecard: v3.4 on every question set\n\n" + md_table(sc) + "\n",
                                      encoding="utf-8")
    first = next(iter(runs.values()))
    inputs = {}
    for info in sources["sets"].values():
        for part in ("run", "testset", "table_questions", "labels"):
            if part in info:
                inputs[info[part]["file"]] = info[part]["sha256"]
    sources["inputs"] = [{"file": f, "sha256": h} for f, h in inputs.items()]
    sources.update({"exported_with": "scripts/export_final_evaluation.py", "config": args.config,
                    "chunk_sha256": corpus.fingerprint, "models": first.get("models"),
                    "checks": "src/policy_rag/evaluation of this repository (unchanged since commit 93b5f9d)"})
    (out / "source.json").write_text(json.dumps(sources, indent=1, ensure_ascii=False, default=str) + "\n",
                                     encoding="utf-8")
    print(f"wrote {out}/per_question.csv ({len(df)} rows), scorecard.csv, scorecard.md, source.json")


if __name__ == "__main__":
    main()
