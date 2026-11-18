"""Grading and running the SQL step's question set, on an invented policy with a scripted model (no database)."""
import hashlib
import json

import pytest

from policy_rag.cli import build_parser
from policy_rag.config import SqlConfig
from policy_rag.evaluation.sql_eval import (SqlEvaluation, choose_gate_threshold, fact_found, gate_calibration_sets,
                                            grade, load_sql_testset, summarize)

pytest.importorskip("sqlglot")
from test_sql_step import FakeDb, ScriptedAnswerer, card_tables  # noqa: E402
from policy_rag.sql_step import SchemaCard, SqlStep  # noqa: E402


def prem(age, col, value, plan="Zenith Care", tier=1):
    return {"table": "premium_table", "keys": {"plan": plan, "tier": tier, "age": age}, "column": col, "value": value}


def q(qtype, **expect):
    return {"id": "Q", "split": "held_out", "type": qtype, "question": "?", "expect": expect}


def out(status, columns=(), rows=()):
    return {"status": status, "columns": list(columns), "rows": [list(r) for r in rows], "attempts": []}


COLS = ["plan", "tier", "age", "si_1000000"]


def test_a_fact_needs_its_value_in_a_row_with_matching_keys():
    f = prem("30", "si_1000000", 9100)
    assert fact_found(f, COLS, [["Zenith Care", 1, "30", 9100]])
    assert not fact_found(f, COLS, [["Zenith Care", 1, "31", 9100]])          # right value, wrong age
    assert fact_found(f, ["si_1000000"], [[9100]])                            # keys not selected: value is enough
    assert fact_found({"value": 12.5, "keys": {}}, ["pct"], [[12.50]])       # numbers compared as numbers
    assert not fact_found({"value": 40, "keys": {}}, ["age"], [["40"]])        # text is not a value
    assert fact_found({"value": 9660, "keys": {}}, ["x"], [[9660.4]], tolerance=1)


@pytest.mark.parametrize("qtype, expect, outcome, verdict", [
    ("lookup", {"facts": [prem("30", "si_1000000", 9100)], "max_rows": 8},
     out("answered", COLS, [["Zenith Care", 1, "30", 9100]]), "pass"),
    ("lookup", {"facts": [prem("30", "si_1000000", 9100)], "max_rows": 8},
     out("answered", COLS, [["Zenith Care", 1, "31", 9300]]), "critical"),
    ("lookup", {"facts": [prem("30", "si_1000000", 9100)], "max_rows": 1},
     out("answered", COLS, [["Zenith Care", 1, "30", 9100], ["Zenith Care", 1, "31", 9300]]), "fail"),
    ("lookup", {"facts": [prem("30", "si_1000000", 9100)]}, out("skipped"), "fail"),
    ("lookup", {"facts": [prem("30", "si_1000000", 9100)]}, out("ask"), "fail"),
    ("calc", {"calc": {"computed": 1300, "inputs": [prem("30", "si_1000000", 9100), prem("30", "si_1000000", 10400, "Zenith Care Plus")]}},
     out("answered", ["diff"], [[1300]]), "pass"),
    ("calc", {"calc": {"computed": 1300, "inputs": [prem("30", "si_1000000", 9100), prem("30", "si_1000000", 10400, "Zenith Care Plus")]}},
     out("answered", COLS, [["Zenith Care", 1, "30", 9100], ["Zenith Care Plus", 1, "30", 10400]]), "fail"),
    ("calc", {"calc": {"computed": 1300, "inputs": [prem("30", "si_1000000", 9100)]}},
     out("answered", ["diff"], [[1500]]), "critical"),
    ("calc", {"calc": {"computed": 1300, "inputs": [prem("30", "si_1000000", 9100), prem("30", "si_1000000", 10400, "Zenith Care Plus")]}},
     {**out("answered", COLS, [["Zenith Care", 1, "30", 9100], ["Zenith Care Plus", 1, "30", 10400]]),
      "derived": [{"kind": "difference", "value": 1300}, {"kind": "total", "value": 19500}]}, "pass"),
    ("calc", {"calc": {"computed": 1300, "how": "a - b", "inputs": [prem("30", "si_1000000", 10400)]}},
     out("answered", ["difference"], [[-1300]]), "pass"),                     # a difference: its size counts
    ("calc", {"calc": {"computed": 1300, "how": "a - b", "inputs": [prem("30", "si_1000000", 10400)]}},
     {**out("answered", ["difference"], [[-1300]]), "derived": [{"kind": "difference_by_query", "value": 1300}]}, "pass"),
    ("calc", {"calc": {"computed": 1300, "how": "a + b", "inputs": [prem("30", "si_1000000", 10400)]}},
     out("answered", ["total"], [[-1300]]), "critical"),                       # not for a total
    ("ask", {}, out("ask"), "pass"),
    ("ask", {}, out("answered", COLS, [["Zenith Care", 1, "30", 9100]]), "critical"),
    ("ask", {}, out("answered", ["age"], [[str(a)] for a in range(20)]), "fail"),
    ("ask", {}, out("not_in_tables"), "fail"),
    ("not_available", {}, out("not_in_tables"), "pass"),
    ("not_available", {}, out("no_rows"), "pass"),
    ("not_available", {}, out("ask"), "pass"),
    ("not_available", {}, out("answered", COLS, [["Zenith Care", 1, "30", 9100]]), "critical"),
    ("text_only", {}, out("text_only"), "pass"),
    ("text_only", {}, out("skipped"), "pass"),
    ("text_only", {}, out("answered", COLS, [["Zenith Care", 1, "30", 9100]]), "pass"),
    ("text_only", {}, out("not_in_tables"), "fail"),
    ("gate_control", {}, out("skipped"), "pass"),
    ("gate_control", {}, out("text_only"), "fail"),
    ("lookup", {"facts": []}, out("unavailable"), "error"),
])
def test_grading(qtype, expect, outcome, verdict):
    assert grade(q(qtype, **expect), outcome)["verdict"] == verdict


def test_summary_and_targets():
    recs = [{"type": "lookup", "verdict": "pass", "gated": True, "loose": False, "seconds": 30, "attempts": 1},
            {"type": "lookup", "verdict": "critical", "gated": True, "loose": True, "seconds": 50, "attempts": 3},
            {"type": "gate_control", "verdict": "pass", "gated": False, "loose": False, "seconds": None, "attempts": 0}]
    s = summarize(recs, {"lookup": {"pass_at_least": 1, "of": 2}, "critical_at_most": 0, "median_seconds_at_most": 60})
    assert s["by_type"]["lookup"] == {"pass": 1, "of": 2, "critical": 1} and s["critical"] == 1 and s["retries"] == 2
    assert s["gate"] == {"table_questions_passed": 2, "table_questions": 2, "loose_passed": 1, "loose": 1,
                         "controls_skipped": 1, "controls": 1}
    assert s["median_seconds"] == 40 and s["targets_met"] == {"lookup": True, "critical": False, "median_seconds": True,
                                                                "no_errors": True}
    assert s["all_targets_met"] is False


def write_set(tmp_path, questions, targets=None, fix=True):
    held = [x for x in questions if x["split"] == "held_out"]
    data = {"name": "invented", "targets": targets or {}, "questions": questions}
    if fix:
        data["held_out_sha256"] = hashlib.sha256(json.dumps(held, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    p = tmp_path / "set.json"
    p.write_text(json.dumps(data))
    return p


QUESTIONS = [
    {"id": "S1", "split": "dev", "type": "lookup",
     "question": "What is the annual premium for Zenith Care for a 30-year-old in Tier 1 with 10 lakh?",
     "expect": {"action": "query", "gate": True, "loose": False, "facts": [prem("30", "si_1000000", 9100)], "max_rows": 8}},
    {"id": "S2", "split": "dev", "type": "ask", "question": "What is the premium for Zenith Care with 10 lakh?",
     "expect": {"action": "ask", "gate": True, "loose": False}},
    {"id": "S3", "split": "held_out", "type": "gate_control", "question": "Is cataract surgery covered?",
     "expect": {"action": "skip", "gate": False, "loose": False}},
]


def test_the_held_out_half_cannot_change_after_it_is_fixed(tmp_path):
    p = write_set(tmp_path, QUESTIONS)
    assert load_sql_testset(p)["held_out_sha256_now"]
    data = json.loads(p.read_text())
    data["questions"][2]["question"] = "Is maternity covered?"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="held-out questions changed"):
        load_sql_testset(p)


def test_a_run_grades_writes_and_summarises(tmp_path):
    sql = "SELECT plan, tier, age, si_1000000 FROM premium_table WHERE age = '30' AND tier = 1"
    step = SqlStep(SqlConfig(enabled=True), ScriptedAnswerer([{"action": "query", "sql": sql},
                                                             {"action": "ask", "question": "How old?"}]),
                   db=FakeDb(lambda s: (["plan", "tier", "age", "si_1000000"], [("Zenith Care", 1, "30", 9100)])),
                   card=SchemaCard(card_tables()))
    logged = []
    ev = SqlEvaluation(step, write_set(tmp_path, QUESTIONS, {"gate_control": {"pass_at_least": 1, "of": 1},
                                                             "critical_at_most": 0}),
                       "t", splits=["dev", "held_out"], output_dir=tmp_path / "runs", log=logged.append)
    run = ev.run()
    assert [r["verdict"] for r in run["records"]] == ["pass", "pass", "pass"]
    assert run["summary"]["dev"]["by_type"] == {"lookup": {"pass": 1, "of": 1, "critical": 0},
                                                "ask": {"pass": 1, "of": 1, "critical": 0}}
    assert "targets_met" not in run["summary"]["dev"] and run["summary"]["held_out"]["all_targets_met"] is True
    folder = tmp_path / "runs" / "t"
    assert {p.name for p in folder.iterdir()} == {"sql_run.json", "per_question.csv", "summary.md"}
    saved = json.loads((folder / "sql_run.json").read_text())
    assert saved["records"][0]["outcome"]["rows"] == [["Zenith Care", 1, "30", 9100]] and len(saved["card_sha256"]) == 64
    assert "| lookup | 1/1 | – | 0 |" in (folder / "summary.md").read_text()
    assert logged[0].startswith("[1/3] S1 lookup: answered -> pass")


def test_the_command_line():
    args = build_parser().parse_args(["evaluate-sql", "--testset", "s.json", "--name", "sql_dev"])
    assert args.splits == ["dev"] and args.func.__name__ == "cmd_evaluate_sql"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["evaluate-sql", "--testset", "s.json", "--name", "x", "--splits", "test"])


def test_the_gate_threshold_rule():
    pos = [(0.9, False), (0.7, False), (0.5, False), (0.2, True)]
    neg = [(0.1, False)] * 16 + [(0.55, False), (0.6, False), (0.8, False), (0.3, True)]
    c = choose_gate_threshold(pos, neg, 0.10)                       # 20 negatives: at most 2 may pass
    assert c["threshold"] == 0.7 and c["negatives_passed"] == 2 and c["positives_passed"] == 3   # 0.6 lets 3 through
    assert c["positives_passed_by_words"] == 1 and c["negatives_passed_by_words"] == 1
    assert choose_gate_threshold(pos, [(0.1, True)] * 3, 0.10)["threshold"] is None     # words alone exceed the limit


def test_calibration_uses_dev_and_other_questions_never_the_held_out_half(tmp_path):
    from test_sql_step import WordVectors
    step = SqlStep(SqlConfig(enabled=True), ScriptedAnswerer([]), db=FakeDb(lambda s: None),
                   card=SchemaCard(card_tables()), embedder=WordVectors())
    data = load_sql_testset(write_set(tmp_path, QUESTIONS))
    other = [{"question": "Is cataract surgery covered?", "sections": ["10.1"]},
             {"question": "What is the premium tier?", "sections": ["Annexure-Z"]},         # a table's own section
             {"question": "What does the Sum Insured section say?", "sections": ["2"]}]      # linked: Section 2
    pos, neg = gate_calibration_sets(step, data, other)
    assert len(pos) == 2 and len(neg) == 1                           # dev: S1, S2 | cataract only (S3 is held out)
