"""Check the checks: does the automatic verdict agree with a human reading? (Cohen's kappa)

1. Every full or rescore run writes ``labels_to_fill.csv``: 30 answers, stratified by verdict and answerability,
   without the automatic verdict. Mark ``human_fully_correct`` as 1 or 0 (all key facts, nothing false or missing).
2. ``policy-rag agreement runs/<name>/run.json labels.csv`` prints kappa and every disagreement. Target:
   kappa >= ``decision.kappa_target``; fix a check that disagrees for a clear reason.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from .stats import cohen_kappa


def label_sheet(run: dict, by_id: dict, path: Path, n: int = 30, seed: int = 11) -> Path:
    rng = random.Random(seed)
    recs = [r for r in run["records"] if r.get("verdict")]
    strata = {}
    for r in recs:
        strata.setdefault((bool(r["verdict"]["fully_correct"]), r["answerable"]), []).append(r)
    picked = []
    for key in sorted(strata, key=lambda k: -len(strata[k])):
        pool = sorted(strata[key], key=lambda r: (r["type"], r["id"]))
        share = max(2, round(n * len(pool) / max(1, len(recs))))
        picked += rng.sample(pool, min(share, len(pool)))
    picked = sorted(picked[:n], key=lambda r: r["id"])
    pd.DataFrame([{"id": r["id"], "split": r["split"], "type": r["type"], "question": r["question"],
                   "reference": by_id[r["id"]]["reference"], "answer": r["answer"], "human_fully_correct": "", "notes": ""}
                  for r in picked]).to_csv(path, index=False)
    return Path(path)


def read_labels(path) -> dict:
    lab = pd.read_csv(path)
    lab = lab[pd.to_numeric(lab["human_fully_correct"], errors="coerce").notna()]
    return dict(zip(lab["id"], lab["human_fully_correct"].astype(int)))


def agreement(run: dict, labels: dict) -> dict:
    recs = {r["id"]: r for r in run["records"] if r.get("verdict")}
    ids = [i for i in labels if i in recs]
    if not ids:
        return {"kappa": None, "n": 0, "disagreements": []}
    auto = [int(bool(recs[i]["verdict"]["fully_correct"])) for i in ids]
    human = [int(labels[i]) for i in ids]
    dis = [{"id": i, "human": h, "automatic": a, "issues": " | ".join(x["issue"] for x in recs[i]["verdict"]["issues"])}
           for i, h, a in zip(ids, human, auto) if h != a]
    return {"kappa": cohen_kappa(auto, human), "n": len(ids), "disagreements": dis}
