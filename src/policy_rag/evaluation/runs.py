"""Saved evaluation runs: one JSON file with the settings, the environment and one record per question.

Also reads the results zips and run JSON files written by the v3 notebooks, and the v2.6 compact format.
"""
from __future__ import annotations

import datetime
import io
import json
import zipfile
from pathlib import Path


def load_run(src) -> dict:
    p = Path(src)
    data = p.read_bytes()
    if p.suffix == ".zip":                     # the run is the JSON file that holds the per-question records
        z = zipfile.ZipFile(io.BytesIO(data))
        runs = [json.loads(z.read(n)) for n in z.namelist() if n.endswith(".json")]
        runs = [r for r in runs if isinstance(r, dict) and r.get("records")]
        if not runs:
            raise ValueError(f"no run with question records in {p.name}")
        run = runs[0]
    else:
        run = json.loads(data)
    if "verdict" not in (run.get("records") or [{}])[0]:          # v2.6 compact evaluation format
        run = {"name": run.get("name", p.stem), "notebook_version": run.get("notebook_version"), "chunk_sha256": None,
               "models": (run.get("environment") or {}).get("models", {}), "testset_version": run.get("testset_version"),
               "records": [{"id": r["id"], "question": r["question"], "answer": r.get("answer"), "context": r.get("context", ""),
                            "gate_route": r.get("gate_route"), "top_score": r.get("top_score"), "gen": r.get("gen"),
                            "error": r.get("error"), "context_tokens": r.get("context_tokens"), "retrieval": {}}
                           for r in run["records"]]}
    return run


def save_run(run: dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "run.json"
    p.write_text(json.dumps(run, ensure_ascii=False, default=str), encoding="utf-8")
    return p


def make_zip(out_dir: Path, name: str) -> Path:
    """All files of a run folder in one zip (handy for downloading from Colab). It contains policy text."""
    out_dir = Path(out_dir)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    z = out_dir.parent / f"rag_eval_{name}_{stamp}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        for p in sorted(out_dir.iterdir()):
            if p.is_file():
                f.write(p, f"{z.stem}/{p.name}")
    return z
