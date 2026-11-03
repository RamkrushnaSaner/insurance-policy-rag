"""The headline numbers in README.md and docs/ recomputed from the published evidence in results/.

These files hold scores only (no policy text), so the numbers can be checked without the policy, the models or a GPU.
If a result file is rebuilt (scripts/export_results.py) and a number changes, this test fails until the documents
are updated too.
"""
import json

import pandas as pd
import pytest

from policy_rag.evaluation.stats import mcnemar_exact, wilson

from conftest import ROOT

R = ROOT / "results"


def flag(s):
    return s.astype(str).str.lower() == "true"


def counts(df, split=None):
    d = df if split is None else df[df.split == split]
    return int(flag(d.fully_correct).sum()), len(d)


def test_baseline_headline():
    pq = pd.read_csv(R / "baseline" / "per_question.csv")
    assert counts(pq) == (97, 113) and counts(pq, "test") == (79, 88) and counts(pq, "dev") == (18, 25)
    lo, hi = wilson(79, 88)
    assert (round(100 * lo), round(100 * hi)) == (82, 95)
    lo, hi = wilson(97, 113)
    assert (round(100 * lo), round(100 * hi)) == (78, 91)
    answerable = pq[flag(pq.answerable)]
    refused = answerable[answerable.gate.eq("no_answer") | answerable.issues.fillna("").str.contains("model refused")]
    assert len(answerable) == 94 and len(refused) == 4
    not_answerable = pq[~flag(pq.answerable)]
    assert (int(flag(not_answerable.fully_correct).sum()), len(not_answerable)) == (18, 19)
    assert int((pq.critical.fillna("") != "").sum()) == 2
    fails = pq[~flag(pq.fully_correct)]
    assert fails.first_failing_stage.value_counts().to_dict() == {"generation": 8, "context": 7, "retrieval": 1}
    ans = pq[flag(pq.answerable)]
    every, policy = ans[flag(ans.everyday_wording)], ans[~flag(ans.everyday_wording)]
    assert counts(every) == (20, 28) and counts(policy) == (59, 66)
    sc = pd.read_csv(R / "baseline" / "scorecard.csv").set_index("measure")
    assert sc.loc["groundedness (supported sentences, HHEM)", "all"] == "0.886"
    assert sc.loc["hallucination rate (answers with an unsupported sentence)", "all"] == "28.3%"
    assert sc.loc["seconds per answer, median", "test"] == "46.2"
    assert sc.loc["tokens per second / peak GPU memory (GB)", "all"] == "6.09 / 10.26"


def test_thinking_off():
    p = pd.read_csv(R / "thinking_off" / "paired.csv")
    t = p[p.split == "test"]
    lost = t[flag(t.fully_correct_baseline) & ~flag(t.fully_correct_candidate)].id.tolist()
    gained = t[~flag(t.fully_correct_baseline) & flag(t.fully_correct_candidate)].id.tolist()
    assert sorted(lost) == sorted(["P02", "C02", "K01", "K08", "D06", "D07", "PR03"]) and gained == []
    assert round(mcnemar_exact(len(lost), len(gained)), 4) == 0.0156
    assert counts(pd.read_csv(R / "thinking_off" / "per_question.csv"), "test") == (72, 88)


def test_v2_6_against_v3_1():
    p = pd.read_csv(R / "v2_6_vs_v3_1" / "paired.csv")
    assert len(p) == 24 and set(p.split) == {"dev"}
    assert int(flag(p.fully_correct_baseline).sum()) == 11 and int(flag(p.fully_correct_candidate).sum()) == 17
    fixed = int((~flag(p.fully_correct_baseline) & flag(p.fully_correct_candidate)).sum())
    broken = int((flag(p.fully_correct_baseline) & ~flag(p.fully_correct_candidate)).sum())
    assert (fixed, broken) == (6, 0) and round(mcnemar_exact(broken, fixed), 4) == 0.0312


@pytest.mark.parametrize("run, n, without, with_, gold, fixed, broken, p, harm, harm_broken", [
    ("run1", 16, 11, 12, 15, 2, 1, 1.0, 6, 0),
    ("run2", 28, 16, 21, 28, 6, 1, 0.125, 14, 1),
    ("run3", 32, 19, 21, 32, 5, 3, 0.7266, 12, 0),
])
def test_memory_runs(run, n, without, with_, gold, fixed, broken, p, harm, harm_broken):
    t = pd.read_csv(R / "memory" / run / "turns.csv")
    t = t[t.split == "test"]
    need, hc = t[t.kind.isin(["follow_up", "after_refusal"])], t[t.kind.isin(["self_contained", "topic_switch"])]
    nm, m, g = flag(need.correct_no_memory), flag(need.correct_memory), flag(need.correct_gold)
    assert (len(need), int(nm.sum()), int(m.sum()), int(g.sum())) == (n, without, with_, gold)
    assert (int((m & ~nm).sum()), int((~m & nm).sum())) == (fixed, broken)
    assert round(mcnemar_exact(broken, fixed), 4) == p
    assert (len(hc), int((flag(hc.correct_no_memory) & ~flag(hc.correct_memory)).sum())) == (harm, harm_broken)


def test_t4_sessions_reproduce_the_published_answers():
    s = pd.read_csv(R / "reproduction" / "t4_sessions.csv")
    checks = s[s.run != "no_thinking"]
    assert len(checks) == 3 and (checks.answers_generated == 0).all() and (checks.answers_identical == 113).all()
    assert (checks.contexts_identical == 113).all() and (checks.fully_correct == 97).all()
    for col in ("candidates_identical", "rerank_scores_identical", "contexts_identical", "gate_identical"):
        assert (s[col] == 113).all(), col                                      # all four T4 runs, thinking off too
    assert (checks.groundedness_mean == 0.886).all()
    assert int(s[s.run == "no_thinking"].contexts_identical.iloc[0]) == 113          # retrieval reproduced


def test_every_result_folder_names_its_sources():
    for f in R.rglob("*source.json"):
        info = json.loads(f.read_text())
        assert info["inputs"] and all(len(i["sha256"]) == 64 for i in info["inputs"]), f


TABLE_ONLY = ["G05", "G07", "G09", "L07", "C01", "C04", "X15", "N05", "T01", "PR01", "PR02", "AB02"]


def test_neon_tables_against_the_baseline():
    new = pd.read_csv(R / "neon_tables" / "per_question.csv").set_index("id")
    assert counts(new.reset_index()) == (100, 113) and counts(new.reset_index(), "test") == (78, 88)
    assert counts(new.reset_index(), "dev") == (22, 25)
    p = pd.read_csv(R / "neon_tables" / "paired.csv")
    t = p[p.split == "test"]
    fixed = sorted(t[~flag(t.fully_correct_baseline) & flag(t.fully_correct_candidate)].id)
    broken = sorted(t[flag(t.fully_correct_baseline) & ~flag(t.fully_correct_candidate)].id)
    assert fixed == ["AB01", "PR02", "X01", "X02", "X12"] and broken == ["C02", "D06", "F01", "L04", "L07", "M04"]
    assert mcnemar_exact(len(broken), len(fixed)) == 1.0
    tab = p[flag(p.table_only)]
    assert sorted(tab.id) == sorted(TABLE_ONLY)
    assert (int(flag(tab.fully_correct_baseline).sum()), int(flag(tab.fully_correct_candidate).sum())) == (8, 9)
    base = pd.read_csv(R / "baseline" / "per_question.csv").set_index("id")
    ids = base.index[base.split == "test"]
    g = pd.DataFrame({"a": base.loc[ids, "groundedness_hhem"], "b": new.loc[ids, "groundedness_hhem"]}).dropna()
    assert (len(g), round(g.a.mean(), 3), round(g.b.mean(), 3)) == (75, 0.895, 0.9)
    assert (round(100 * (g.a < 1).mean(), 1), round(100 * (g.b < 1).mean(), 1)) == (25.3, 24.0)
    six = pd.read_csv(R / "embedder_qwen3" / "per_question.csv").set_index("id")
    pop = six.index[(six.split == "test") & flag(six.answerable) & (six.evidence_in_text_file == 1)]
    assert len(pop) == 67 and int((base.loc[pop, "context_recall"] == 1).sum()) == 63
    assert int((new.loc[pop, "context_recall"] == 1).sum()) == 66
    ans = new[flag(new.answerable)]
    assert counts(ans[flag(ans.everyday_wording)].reset_index()) == (25, 28)
    assert counts(ans[~flag(ans.everyday_wording)].reset_index()) == (56, 66)
    sc = pd.read_csv(R / "neon_tables" / "scorecard.csv").set_index("measure")
    assert sc.loc["seconds per answer, median", "all"] == "49.2"
    assert sc.loc["tokens per second / peak GPU memory (GB)", "all"] == "6.39 / 12.53"
    assert sc.loc["groundedness (supported sentences, HHEM)", "all"] == "0.873"
    info = json.loads((R / "neon_tables" / "comparison_source.json").read_text())
    assert info["decision"] == "MERGE IN STEP 9: not worse than the baseline"
    assert info["decision_rule_fixed_before_the_run"] is True


def test_embedder_and_table_route_retrieval():
    six = pd.read_csv(R / "embedder_qwen3" / "per_question.csv")
    pop = six[(six.split == "test") & flag(six.answerable) & (six.evidence_in_text_file == 1)]
    assert (len(pop), int((pop.bge_context_recall == 1).sum()), int((pop.qwen3_context_recall == 1).sum())) == (67, 64, 64)
    assert (round(pop["bge_dense_recall@10"].mean(), 3), round(pop["qwen3_dense_recall@10"].mean(), 3)) == (0.955, 0.993)
    eight = pd.read_csv(R / "table_route_retrieval" / "per_question.csv").set_index("id")
    t = eight.loc[TABLE_ONLY]
    assert flag(t.table_in_context_on).all()
    assert (int((t.facts_in_context_off == 1).sum()), int((t.facts_in_context_on == 1).sum())) == (0, 11)
    p67 = eight.loc[pop.id]
    assert (int((p67.context_recall_off == 1).sum()), int((p67.context_recall_on == 1).sum())) == (64, 65)
    others = eight[flag(eight.answerable) & ~eight.index.isin(TABLE_ONLY)]
    assert not (others.context_recall_on < others.context_recall_off).any()
    assert not (others.facts_in_context_on < others.facts_in_context_off).any()
    assert not flag(eight[~flag(eight.answerable)].table_in_context_on).any()
