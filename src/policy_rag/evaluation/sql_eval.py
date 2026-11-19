"""Evaluation of the SQL step on its own question set (v3.5 step 5): does the answer model write SQL that returns the
right printed values, ask when a needed detail is missing, and decline when the tables do not hold the value?

The question set (JSON) gives each question a ``split`` (``dev`` shapes the prompt and the gate; ``held_out`` is run
once, after both are frozen), a ``type`` and what is expected:

- ``lookup``: values printed in one table (``facts``: value, table column, identifying columns). Pass: the query
  returns every expected value in a row whose identifying columns, when selected, match, in at most ``max_rows`` rows.
- ``calc``: a difference, a sum or a premium after a discount (``calc``: the computed value and its inputs). Pass: the
  computed value is in the rows, or among the calculations code made from them (``SqlOutcome.derived``), within
  Rs 1. Rows with the inputs only and no such calculation are not a pass.
- ``ask``: a detail with many choices is missing (an age). Pass: the model asks.
- ``not_available``: the tables do not hold the value. Pass: not_in_tables, no rows, or a question back.
- ``text_only``: table words, but the answer is in the wording. Pass: text_only, a query, or skipped by the gate.
- ``gate_control``: not about the tables. Pass: the gate skips it.

**Critical** (a wrong figure could reach the answer): rows returned for a lookup or calculation without the expected
values, rows for an ``ask`` question that pick one choice (at most 8 rows), any rows for a ``not_available`` question.
``loose`` marks questions worded without the gate's words (they measure the gate).

Targets are part of the question set, fixed before the held-out run. The run folder holds every reply and row, so
it contains policy figures: keep it private (``runs/`` is not committed).
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import re
import statistics
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Optional, Sequence

TYPES = ("lookup", "calc", "ask", "not_available", "text_only", "gate_control")
TABLE_TYPES = ("lookup", "calc", "ask", "not_available")         # questions the gate should pass
ASSUMED_ROWS = 8                                                  # an ask question answered in this many rows: assumed


def load_sql_testset(path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [q["id"] for q in data["questions"]]
    if len(set(ids)) != len(ids):
        raise ValueError("question ids must be unique")
    for q in data["questions"]:
        if q["type"] not in TYPES or q["split"] not in ("dev", "held_out"):
            raise ValueError(f"{q['id']}: unknown type {q['type']!r} or split {q['split']!r}")
    held = [q for q in data["questions"] if q["split"] == "held_out"]
    fp = hashlib.sha256(json.dumps(held, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if data.get("held_out_sha256") and data["held_out_sha256"] != fp:
        raise ValueError(f"the held-out questions changed after they were fixed (sha256 {fp[:12]}, expected "
                         f"{data['held_out_sha256'][:12]})")
    data["held_out_sha256_now"] = fp
    return data


# ---------------------------------------------------------------------------------------------------- grading
def _num(v) -> Optional[Decimal]:
    """A number the query returned; text (such as a printed age '40') never counts as a value."""
    if isinstance(v, bool) or not isinstance(v, (int, float, Decimal)):
        return None
    try:
        return Decimal(str(v))
    except (InvalidOperation, ValueError):
        return None


def fact_found(fact: dict, columns: Sequence[str], rows: Sequence[Sequence], tolerance: float = 0.0) -> bool:
    """The fact's value is a number in a row whose identifying columns (those the query selected) hold the fact's
    keys. Text cells are not values (a printed age '40' is not a 40% discount)."""
    want = Decimal(str(fact["value"]))
    for row in rows:
        if not any((x := _num(v)) is not None and abs(x - want) <= Decimal(str(tolerance)) for v in row):
            continue
        if all(str(v) == str(fact["keys"][c]) for c, v in zip(columns, row) if c in fact.get("keys", {})):
            return True
    return False


def grade(question: dict, outcome: dict) -> dict:
    """{verdict: pass | fail | critical | error, reason}. ``outcome``: ``SqlOutcome.to_dict()``."""
    t, e, status = question["type"], question["expect"], outcome["status"]
    columns, rows = outcome.get("columns") or [], outcome.get("rows") or []
    if status == "unavailable":
        return {"verdict": "error", "reason": "database unavailable: run again"}
    if t == "gate_control":
        return ({"verdict": "pass", "reason": "skipped by the gate"} if status == "skipped" else
                {"verdict": "fail", "reason": f"the gate passed it ({status})"})
    if t == "text_only":
        ok = status in ("text_only", "answered", "skipped")
        return {"verdict": "pass" if ok else "fail", "reason": status}
    if status == "skipped":
        return {"verdict": "fail", "reason": "the gate missed it"}
    if t == "ask":
        if status == "ask":
            return {"verdict": "pass", "reason": "asked"}
        if status == "answered":
            return ({"verdict": "critical", "reason": f"answered in {len(rows)} rows: assumed the missing detail"}
                    if len(rows) <= ASSUMED_ROWS else
                    {"verdict": "fail", "reason": f"listed {len(rows)} rows instead of asking"})
        return {"verdict": "fail", "reason": status}
    if t == "not_available":
        if status in ("not_in_tables", "no_rows", "ask"):
            return {"verdict": "pass", "reason": status}
        if status == "answered":
            return {"verdict": "critical", "reason": f"returned {len(rows)} rows for a value the tables do not hold"}
        return {"verdict": "fail", "reason": status}
    # lookup, calc
    if status != "answered":
        return {"verdict": "fail", "reason": status}
    if t == "lookup":
        missing = [f for f in e["facts"] if not fact_found(f, columns, rows)]
        if missing:
            return {"verdict": "critical", "reason": f"{len(missing)} of {len(e['facts'])} expected values not in the rows"}
        if len(rows) > e.get("max_rows", ASSUMED_ROWS):
            return {"verdict": "fail", "reason": f"right values, but {len(rows)} rows"}
        return {"verdict": "pass", "reason": "all values found"}
    c = e["calc"]
    # a difference is right whichever way round it was subtracted (the test set's order is arbitrary): its size counts
    wanted = [c["computed"]] + ([-c["computed"]] if c.get("how") == "a - b" and c["computed"] else [])
    if any(fact_found({"value": w, "keys": {}}, columns, rows, tolerance=1) for w in wanted):
        return {"verdict": "pass", "reason": "computed value found"}
    derived = [d["value"] for d in outcome.get("derived") or []]
    if any(fact_found({"value": w, "keys": {}}, ["derived"] * len(derived), [derived], tolerance=1) for w in wanted):
        return {"verdict": "pass", "reason": "computed by code from the rows"}
    if all(fact_found(f, columns, rows) for f in c["inputs"]):
        return {"verdict": "fail", "reason": "inputs only: the result was not computed in SQL"}
    return {"verdict": "critical", "reason": "neither the result nor its inputs in the rows"}


def summarize(records: List[dict], targets: Optional[dict] = None) -> dict:
    """Pass counts per type, critical errors, the gate, time; and the targets, when given, met or not."""
    out = {"questions": len(records), "by_type": {}, "critical": 0, "errors": 0}
    for t in TYPES:
        rs = [r for r in records if r["type"] == t]
        if rs:
            out["by_type"][t] = {"pass": sum(r["verdict"] == "pass" for r in rs), "of": len(rs),
                                 "critical": sum(r["verdict"] == "critical" for r in rs)}
    out["critical"] = sum(r["verdict"] == "critical" for r in records)
    out["errors"] = sum(r["verdict"] == "error" for r in records)
    table = [r for r in records if r["type"] in TABLE_TYPES]
    out["gate"] = {"table_questions_passed": sum(r["gated"] for r in table), "table_questions": len(table),
                   "loose_passed": sum(r["gated"] for r in table if r["loose"]),
                   "loose": sum(1 for r in table if r["loose"]),
                   "controls_skipped": sum(not r["gated"] for r in records if r["type"] == "gate_control"),
                   "controls": sum(1 for r in records if r["type"] == "gate_control")}
    secs = [r["seconds"] for r in records if r["gated"] and r["seconds"] is not None]
    out["median_seconds"] = round(statistics.median(secs), 1) if secs else None
    out["retries"] = sum(max(0, r["attempts"] - 1) for r in records)
    if targets:
        met = {}
        for t, tg in targets.items():
            if isinstance(tg, dict) and t in out["by_type"]:
                met[t] = out["by_type"][t]["pass"] >= tg["pass_at_least"]
        if "critical_at_most" in targets:
            met["critical"] = out["critical"] <= targets["critical_at_most"]
        if "median_seconds_at_most" in targets and out["median_seconds"] is not None:
            met["median_seconds"] = out["median_seconds"] <= targets["median_seconds_at_most"]
        met["no_errors"] = out["errors"] == 0
        out["targets_met"] = met
        out["all_targets_met"] = all(met.values())
    return out


def summary_markdown(name: str, split_summaries: Dict[str, dict], targets: dict) -> str:
    lines = [f"# SQL step evaluation: {name}", ""]
    for split, s in split_summaries.items():
        lines += [f"## {split} ({s['questions']} questions)", "",
                  "| Type | Pass | Target (held out) | Critical |", "|---|---|---|---|"]
        for t, v in s["by_type"].items():
            tg = targets.get(t)
            lines.append(f"| {t} | {v['pass']}/{v['of']} | " + (f"≥ {tg['pass_at_least']}/{tg['of']}" if tg else "–")
                         + f" | {v['critical']} |")
        g = s["gate"]
        lines += ["", f"- Critical: {s['critical']} (target ≤ {targets.get('critical_at_most', '–')}); errors: {s['errors']}",
                  f"- Gate: {g['table_questions_passed']}/{g['table_questions']} table questions reached the SQL step "
                  f"({g['loose_passed']}/{g['loose']} loosely worded); {g['controls_skipped']}/{g['controls']} "
                  "other questions skipped",
                  f"- Median seconds per SQL step: {s['median_seconds']} (target ≤ "
                  f"{targets.get('median_seconds_at_most', '–')}); corrected replies: {s['retries']}"]
        if "targets_met" in s:
            lines.append(f"- Targets met: {'**all**' if s['all_targets_met'] else '**not all**'} "
                         + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in s["targets_met"].items()))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------- the gate threshold
MAX_FALSE_ALARMS = 0.10       # share of questions not about the tables that may reach the SQL step (one extra call each)


def choose_gate_threshold(positives: Sequence[tuple], negatives: Sequence[tuple],
                          max_false_alarms: float = MAX_FALSE_ALARMS) -> dict:
    """The closeness threshold for ``SqlGate``, by a rule fixed before any run: the lowest threshold at which at most
    ``max_false_alarms`` of the negatives pass the whole gate (words or closeness). Lowest = the most table questions
    let through. ``positives`` / ``negatives``: (similarity, passed by words) per question. Threshold None: closeness
    is not used (the words alone already exceed the limit, or there is nothing to gain)."""
    def passes(items, t):
        return sum(1 for sim, words in items if words or (t is not None and sim is not None and sim >= t))
    best = None
    for t in sorted({sim for sim, _ in list(positives) + list(negatives) if sim is not None}):
        if passes(negatives, t) <= max_false_alarms * len(negatives):
            best = t
            break
    return {"threshold": best, "max_false_alarms": max_false_alarms,
            "positives": len(positives), "positives_passed": passes(positives, best),
            "positives_passed_by_words": passes(positives, None),
            "negatives": len(negatives), "negatives_passed": passes(negatives, best),
            "negatives_passed_by_words": passes(negatives, None)}


def gate_calibration_sets(step, sql_testset: dict, other_questions: Sequence[dict]) -> tuple:
    """Positives: the dev questions about the tables. Negatives: the dev gate controls and the other questions (a
    general question set) except those from the sections the tables come from or link to (they may be about the
    tables). Never the held-out half. Each item: (similarity, passed by words)."""
    from ..sql_step import gate as word_gate
    sections = {str(p.get("section_id")) for t in step.card.tables.values() for p in t["printed"]}
    for link in step.card.linked_text(list(step.card.tables)):
        sections.update(re.findall(r"^Section (\S+)", link.get("label") or ""))
    dev = [q for q in sql_testset["questions"] if q["split"] == "dev"]

    def score(text):
        return step.gate.similarity(text), bool(word_gate(text, step.names))
    pos = [score(q["question"]) for q in dev if q["type"] in TABLE_TYPES]
    neg = [score(q["question"]) for q in dev if q["type"] == "gate_control"]
    neg += [score(q["question"]) for q in other_questions if not {str(x) for x in q.get("sections", [])} & sections]
    return pos, neg


# ---------------------------------------------------------------------------------------------------- the run
class SqlEvaluation:
    """Runs ``SqlStep.run`` on every question of the chosen splits, grades it, and writes ``<output_dir>/<name>/``:
    ``sql_run.json`` (settings, every record with the model's replies and rows), ``per_question.csv`` and
    ``summary.md``. Targets are checked on the held-out split only."""

    def __init__(self, step, testset_path, name: str, splits: Sequence[str] = ("dev",), output_dir="runs",
                 settings: Optional[dict] = None, log=print):
        self.step, self.name, self.splits, self.log = step, name, list(splits), log
        self.testset_path = str(testset_path)
        self.testset = load_sql_testset(testset_path)
        self.out_dir = Path(output_dir) / name
        self.settings = settings or {}

    def run(self) -> dict:
        qs = [q for q in self.testset["questions"] if q["split"] in self.splits]
        records = []
        for i, q in enumerate(qs, 1):
            out = self.step.run(q["question"]).to_dict()
            g = grade(q, out)
            secs = [a.get("seconds") for a in out["attempts"] if a.get("seconds") is not None]
            replies = [a for a in out["attempts"] if not a.get("confirm")]     # the check before asking is no reply
            records.append({"id": q["id"], "split": q["split"], "type": q["type"], "loose": bool(q["expect"].get("loose")),
                            "question": q["question"], "expect": q["expect"], "gated": out["status"] != "skipped",
                            "status": out["status"], "verdict": g["verdict"], "reason": g["reason"],
                            "attempts": len(replies), "confirmed": [a.get("yes") for a in out["attempts"] if a.get("confirm")],
                            "seconds": round(sum(secs), 2) if secs else None,
                            "similarity": out.get("similarity"),
                            "outcome": out})
            self.log(f"[{i}/{len(qs)}] {q['id']} {q['type']}: {out['status']} -> {g['verdict']} ({g['reason']})")
        targets = self.testset.get("targets") or {}
        summaries = {s: summarize([r for r in records if r["split"] == s], targets if s == "held_out" else None)
                     for s in self.splits}
        run = {"name": self.name, "kind": "sql_step", "created": datetime.datetime.now().isoformat(timespec="seconds"),
               "testset": {"path": self.testset_path, "name": self.testset.get("name"),
                           "held_out_sha256": self.testset["held_out_sha256_now"]},
               "card_sha256": hashlib.sha256(self.step.card_text.encode()).hexdigest(),
               "settings": self.settings, "targets": targets, "summary": summaries, "records": records}
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "sql_run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1, default=str),
                                                   encoding="utf-8")
        with open(self.out_dir / "per_question.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "split", "type", "loose", "gated", "similarity", "status", "verdict", "reason", "attempts",
                        "seconds", "rows", "sql"])
            for r in records:
                w.writerow([r["id"], r["split"], r["type"], int(r["loose"]), int(r["gated"]), r.get("similarity"),
                            r["status"], r["verdict"], r["reason"], r["attempts"], r["seconds"], len(r["outcome"]["rows"]),
                            r["outcome"]["sql"]])
        md = summary_markdown(self.name, summaries, targets)
        (self.out_dir / "summary.md").write_text(md, encoding="utf-8")
        self.log("\n" + md)
        return run
