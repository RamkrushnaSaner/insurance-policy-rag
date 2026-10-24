"""The evaluation workflow end to end with the fake components and the sample test set."""
import json

from policy_rag.cli import main as cli
from policy_rag.evaluation.runner import Evaluation
from policy_rag.evaluation.testset import load_testset

from conftest import ROOT, fake_config


def test_the_published_test_set_is_unchanged():
    ts = load_testset(ROOT / "testsets" / "optima_secure_v3.json")
    assert ts.version == "7313dd7571"                     # the version behind the published results
    assert len(ts.items) == 113 and len(ts.select(["dev"])) == 25 and len(ts.select(["test"])) == 88
    assert sum(not t["answerable"] for t in ts.items) == 19 and sum(t["paraphrase"] for t in ts.items) == 32
    for t in ts.items:
        assert {"id", "split", "type", "answerable", "paraphrase", "question", "facts", "evidence", "reference"} <= set(t)


def test_full_retrieval_and_rescore_runs(tmp_path):
    cfg = fake_config(tmp_path)
    full = Evaluation(cfg, "full", "full", log=lambda *_: None).run()
    recs = {r["id"]: r for r in full["records"]}
    assert len(recs) == 6 and recs["S01"]["verdict"]["fully_correct"]
    assert recs["S05"]["gate_route"] == "no_answer" and recs["S05"]["verdict"]["fully_correct"]
    assert recs["S04"]["premium_routed"]
    assert full["retrieval_repeat"]["identical"] == 6 and full["repeat"]["identical"]
    out = tmp_path / "runs" / "full"
    for f in ("run.json", "report.md", "results.csv", "scorecard.csv", "labels_to_fill.csv", "environment.json"):
        assert (out / f).exists(), f

    retr = Evaluation(cfg, "retr", "retrieval", log=lambda *_: None).run()
    assert all(r["answer"] is None for r in retr["records"])
    assert [r["groups"] for r in retr["records"]] == [r["groups"] for r in full["records"]]

    again = Evaluation(cfg, "again", "rescore", rescore_from=str(out / "run.json"), log=lambda *_: None).run()
    assert [r["verdict"] for r in again["records"]] == [r["verdict"] for r in full["records"]]


def test_cli_evaluate_compare_and_agreement(tmp_path, capsys):
    cfg_file = tmp_path / "cfg.yaml"
    c = fake_config(tmp_path).to_dict()
    c.pop("source")
    cfg_file.write_text(json.dumps(c))                     # JSON is valid YAML
    cli(["evaluate", "--config", str(cfg_file), "--name", "a"])
    cli(["evaluate", "--config", str(cfg_file), "--name", "b", "--set", "context.token_budget=60"])
    run_a, run_b = tmp_path / "runs" / "a" / "run.json", tmp_path / "runs" / "b" / "run.json"
    cli(["compare", "--config", str(cfg_file), str(run_a), str(run_b)])
    assert (tmp_path / "runs" / "compare_a" / "comparison.csv").exists()
    labels = tmp_path / "runs" / "a" / "labels_to_fill.csv"
    import pandas as pd
    df = pd.read_csv(labels)
    df["human_fully_correct"] = 1
    df.to_csv(labels, index=False)
    cli(["agreement", "--config", str(cfg_file), str(run_a), str(labels)])
    assert "kappa" in capsys.readouterr().out


def test_evaluation_with_several_documents_searches_the_test_sets_policy(tmp_path):
    from conftest import DATA
    docs = [{"id": "sample", "name": "Sample Health Shield", "chunks": str(DATA / "sample_policy_chunks.json")},
            {"id": "second", "name": "Sample Family Care", "chunks": str(DATA / "second_policy_chunks.json")}]
    run = Evaluation(fake_config(tmp_path, documents=docs), "multi", "retrieval", log=lambda *_: None).run()
    assert run["scope"] == ["sample"]
    second = set(range(15, 18))                            # chunk ids of the second policy
    assert all(not second & set(r["candidates"]) for r in run["records"])


def test_evaluation_can_reuse_a_built_system(tmp_path, system):
    run = Evaluation(None, "reuse", "full", splits=["dev"], system=system, log=lambda *_: None).run()
    assert len(run["records"]) == 2 and system.generator.calls >= 2
