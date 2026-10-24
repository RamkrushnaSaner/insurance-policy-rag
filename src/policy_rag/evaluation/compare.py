"""Compare runs: paired tests and the decision rule fixed in the config (``decision``).

Every run is re-scored with the current checks first, so all are judged by the same rules. Primary: "fully
correct", McNemar exact test on the questions both runs answered (Holm-adjusted when there are several candidates).
Secondary: paired bootstrap 95% intervals. Decisions only on the held-out test split.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .report import summarize
from .stats import holm, mcnemar_exact

SECONDARY = [("key_fact_recall", 1), ("critical_error", -1), ("groundedness", 1), ("hallucinated", -1),
             ("correct_abstention", 1), ("context_recall", 1), ("mrr10", 1)]


def per_question(run: dict) -> pd.DataFrame:
    rows = []
    for r in run["records"]:
        v, g, rt = r.get("verdict") or {}, r.get("grounding") or {}, r.get("retrieval") or {}
        rows.append({"id": r["id"], "split": r["split"], "answerable": r["answerable"],
                     "fully_correct": None if not v else float(bool(v["fully_correct"])),
                     "key_fact_recall": (v.get("fact_recall") or 0.0) if v and r["answerable"] else None,
                     "critical_error": None if not v else float(bool(v["critical"])),
                     "correct_abstention": None if not v or r["answerable"] else float(bool(v["fully_correct"])),
                     "groundedness": g.get("groundedness"), "hallucinated": None if not g else float(bool(g.get("hallucinated"))),
                     "context_recall": rt.get("context_recall"), "mrr10": rt.get("mrr10")})
    return pd.DataFrame(rows).set_index("id")


def compare(base: dict, cand: dict, split: str, n_bootstrap: int = 10_000, seed: int = 0) -> dict:
    a, b = per_question(base), per_question(cand)
    a, b = a[a.split == split], b[b.split == split]
    ids = a.index.intersection(b.index)
    rng = np.random.default_rng(seed)
    out = {"split": split, "n": len(ids), "rows": []}
    pa, pb = a.loc[ids, "fully_correct"], b.loc[ids, "fully_correct"]
    ok = pa.notna() & pb.notna()
    pa, pb = pa[ok], pb[ok]
    if len(pa):
        bb, cc = int(((pa == 1) & (pb == 0)).sum()), int(((pa == 0) & (pb == 1)).sum())
        d = (pb - pa).to_numpy()
        boots = d[rng.integers(0, len(d), (n_bootstrap, len(d)))].mean(axis=1)
        out.update(primary={"base": pa.mean(), "cand": pb.mean(), "diff": d.mean(), "ci": tuple(np.percentile(boots, [2.5, 97.5])),
                            "fixed": cc, "broken": bb, "p": mcnemar_exact(bb, cc), "n": len(d)},
                   fixed_ids=list(pa.index[(pa == 0) & (pb == 1)]), broken_ids=list(pa.index[(pa == 1) & (pb == 0)]))
    for m, sign in SECONDARY:
        x, y = a.loc[ids, m], b.loc[ids, m]
        k = x.notna() & y.notna()
        if not k.any():
            continue
        d = (y[k] - x[k]).to_numpy(float)
        boots = d[rng.integers(0, len(d), (n_bootstrap, len(d)))].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        g_lo, g_hi = sorted((sign * lo, sign * hi))
        out["rows"].append({"metric": m, "n": int(k.sum()), "base": round(x[k].mean(), 3), "candidate": round(y[k].mean(), 3),
                            "diff": round(d.mean(), 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                            "verdict": "better" if g_lo > 0 else "worse" if g_hi < 0 else "no clear change"})
    return out


def decide(res: dict, p_adj: float, cost: dict, decision) -> str:
    if res["split"] != "test":
        return "dev only: guides debugging, no decision"
    pr = res.get("primary")
    if not pr:
        return "no paired answers"
    worse = {r["metric"] for r in res["rows"] if r["verdict"] == "worse"}
    reasons = []
    if not (p_adj < decision.alpha and pr["diff"] > 0):
        reasons.append(f"fully correct not significantly better (p = {p_adj:.3f})")
    for m in ("groundedness", "hallucinated", "correct_abstention"):
        if m in worse:
            reasons.append(f"{m} significantly worse")
    if cost["median_seconds"] and cost["median_seconds"] > decision.max_median_seconds:
        reasons.append(f"median {cost['median_seconds']:.0f}s per answer > {decision.max_median_seconds:g}s")
    if cost["peak_gpu_gb"] and cost["peak_gpu_gb"] > decision.max_peak_gpu_gb:
        reasons.append(f"peak GPU {cost['peak_gpu_gb']} GB > {decision.max_peak_gpu_gb} GB")
    return "ADOPT" if not reasons else "KEEP THE BASELINE: " + "; ".join(reasons)


def blind_sheet(base: dict, cand: dict, by_id: dict, out_dir: Path, seed: int = 7) -> Path:
    """Answers that differ, side by side as A/B in random order per question (run names hidden) for a human read;
    the key is saved separately (blind_key.json)."""
    rng, a, b = random.Random(seed), {r["id"]: r for r in base["records"]}, {r["id"]: r for r in cand["records"]}
    rows, key = [], {}
    for qid in [i for i in b if i in a and a[i].get("answer") and b[i].get("answer") and a[i]["answer"] != b[i]["answer"]]:
        flip = rng.random() < 0.5
        x, y = (b[qid], a[qid]) if flip else (a[qid], b[qid])
        key[qid] = {"A": (cand if flip else base)["name"], "B": (base if flip else cand)["name"]}
        rows.append({"id": qid, "question": by_id[qid]["question"], "reference": by_id[qid]["reference"],
                     "answer_A": x["answer"], "answer_B": y["answer"], "better (A/B/same)": "", "notes": ""})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "blind_review.csv"
    pd.DataFrame(rows, columns=["id", "question", "reference", "answer_A", "answer_B", "better (A/B/same)", "notes"]).to_csv(p, index=False)
    (out_dir / "blind_key.json").write_text(json.dumps(key, indent=1))
    return p


def compare_runs(base: dict, cands: list, splits, decision, log=print) -> list:
    """Rows of the comparison table (one per candidate and split), printed as they are computed."""
    rows = []
    for split in splits:
        res = {c["name"]: compare(base, c, split, decision.n_bootstrap) for c in cands}
        p_adj = holm({k: r["primary"]["p"] for k, r in res.items() if r.get("primary")})
        for c in cands:
            r = res[c["name"]]
            if not r.get("primary"):
                continue
            s = summarize(c["records"])
            pr = r["primary"]
            verdict = decide(r, p_adj[c["name"]], {"median_seconds": s.get("median_seconds"),
                                                     "peak_gpu_gb": c.get("peak_gpu_gb")}, decision)
            rows.append({"split": split, "baseline": base["name"], "candidate": c["name"], "n": pr["n"],
                         "fully_correct_base": round(pr["base"], 3), "fully_correct_candidate": round(pr["cand"], 3),
                         "diff": round(pr["diff"], 3), "ci": f"{pr['ci'][0]:+.3f} to {pr['ci'][1]:+.3f}",
                         "fixed": pr["fixed"], "broken": pr["broken"], "p_mcnemar": round(pr["p"], 4),
                         "p_holm": round(p_adj[c["name"]], 4), "decision": verdict})
            log(f"\n=== [{split}] {base['name']} -> {c['name']}: fully correct {pr['base']:.3f} -> {pr['cand']:.3f} "
                f"(diff {pr['diff']:+.3f}, 95% CI {pr['ci'][0]:+.3f} to {pr['ci'][1]:+.3f}; fixed {pr['fixed']}, broken "
                f"{pr['broken']}; McNemar p {pr['p']:.4f}, Holm {p_adj[c['name']]:.4f}) | {verdict}")
            log(f"    now correct: {r['fixed_ids'] or 'none'} | now wrong: {r['broken_ids'] or 'none'}")
            if r["rows"]:
                log(pd.DataFrame(r["rows"]).to_string(index=False))
    return rows
