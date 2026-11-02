"""The table route inside the pipeline and the evaluation, with the fake components and the in-memory tables of
tests/table_fixtures.py: the table passage in the context, linked chunks as candidates, the gate, the answer check
and fallback, a failing route, records and retrieval metrics."""
import pytest

from policy_rag.config import ConfigError
from policy_rag.database import chunk_uid
from policy_rag.evaluation.checks import retrieval_metrics
from policy_rag.evaluation.runner import Evaluation, build_record
from policy_rag.evaluation.testset import CorpusText
from policy_rag.pipeline import RAGSystem
from policy_rag.prompts import NO_ANSWER_TEXT
from policy_rag.tables import TableRoute

from conftest import DATA, fake_config
from table_fixtures import MemoryTableData, dataset

LIST_Q = "Will I be reimbursed for attendant charges and the TV in my hospital room?"
PREMIUM_Q = "What is the annual premium for Zenith Care with a Rs. 10 lakh Sum Insured for a 30-year-old living in Mumbai?"


def _system(tmp_path, link_to_section="10.1", **sections):
    """The sample policy with the invented tables; Annexure B's table is linked to a chunk of the sample policy."""
    sections["tables"] = {"enabled": True, **sections.get("tables", {})}
    cfg = fake_config(tmp_path, **sections)
    probe = RAGSystem.from_config(fake_config(tmp_path), log=lambda *_: None)
    target = next(c for c in probe.corpus.chunks if c.section_id == link_to_section)
    ds = dataset(lists=True)
    ds["chunk_tables"].append(("p18-t1", chunk_uid(target.doc_id, target.embed_text), "refers_to"))
    logs = []
    rag = RAGSystem.from_config(cfg, log=logs.append, tables=TableRoute(MemoryTableData(ds)))
    return rag, target, logs


def test_table_passage_comes_first_and_linked_chunks_are_candidates(tmp_path):
    rag, target, _ = _system(tmp_path)
    st = rag.pipeline.search(LIST_Q)
    assert st["table"].answered and st["facts"].startswith("[Policy tables → Section Annexure-B]\nPolicy tables (exact")
    pos = rag.corpus.chunks.index(target)
    assert pos in st["candidates"] and st["table_linked"] == [pos]
    ctx = rag.pipeline.context_text(st["groups"], st["facts"])
    assert ctx.startswith(st["facts"] + "\n\n[")
    assert st["facts"] in rag.pipeline.prompt_for(LIST_Q, st["groups"], st["facts"])
    r = rag.ask(LIST_Q, show_sources=False)
    assert r.sources[0].startswith("[Policy tables] Annexure B, page 18 (table p18-t1)")


def test_the_tables_open_the_gate(tmp_path):
    rag, _, _ = _system(tmp_path, gate={"min_score": 1000.0})     # no text passage is relevant enough
    st = rag.pipeline.search(PREMIUM_Q)
    assert st["route"] == "answer" and st["table"].complete
    st = rag.pipeline.search("Which conditions have a 24-month waiting period?")      # a whole list
    assert st["route"] == "answer" and st["table"].fields["mode"] == "whole list"
    st = rag.pipeline.search(LIST_Q)                     # items on a list only add to the text: the text gate decides
    assert st["table"].answered and st["facts"] and st["route"] == "no_answer"
    st = rag.pipeline.search("Xylophone tuition reimbursement?")
    assert st["route"] == "no_answer"


def test_no_table_passage_when_the_route_has_no_answer(tmp_path):
    rag, _, _ = _system(tmp_path)
    st = rag.pipeline.search("What is the annual premium for Zenith Gold for a 30-year-old?")
    assert st["table"].status == "not_in_policy" and st["facts"] == "" and "table_linked" not in st
    st = rag.pipeline.search("Does the policy pay for an ambulance to take me to the hospital?")
    assert st["table"].status == "no_match" and st["facts"] == ""


def test_answer_check_and_fallback(tmp_path):
    rag, _, logs = _system(tmp_path)
    pipe = rag.pipeline
    t = pipe.table_answer(PREMIUM_Q)
    ctx = pipe.context_text([], pipe.table_passage(t))
    right = t.text.splitlines()[0]
    assert pipe.check_answer(PREMIUM_Q, right, t, ctx) == {"unsupported_figures": [], "refused": False, "fallback": False}
    check = pipe.check_answer(PREMIUM_Q, "It costs Rs. 99,999 a year.", t, ctx)
    assert check["fallback"] and check["answer"] == t.text and check["unsupported_figures"] == ["Rs 99,999"]
    assert "replaced by the answer written from the tables" in logs[-1]
    assert pipe.check_answer(PREMIUM_Q, NO_ANSWER_TEXT, t, ctx)["fallback"]      # refused although the tables answer
    lst = pipe.table_answer(LIST_Q)                                                # part of an answer: never replaced
    check = pipe.check_answer(LIST_Q, "Yes, up to Rs. 99,999.", lst, pipe.context_text([], pipe.table_passage(lst)))
    assert check["unsupported_figures"] == ["Rs 99,999"] and not check["fallback"]


def test_keep_setting_never_replaces(tmp_path):
    rag, _, _ = _system(tmp_path, tables={"enabled": True, "on_unsupported_figures": "keep"})
    t = rag.pipeline.table_answer(PREMIUM_Q)
    assert not rag.pipeline.check_answer(PREMIUM_Q, "Rs. 99,999.", t, "")["fallback"]


def test_the_whole_graph_records_the_check(tmp_path):
    rag, _, _ = _system(tmp_path)
    r = rag.ask(PREMIUM_Q, show_sources=False)               # the fake model copies a context sentence
    assert r.route == "answer" and "Rs. 1,450 a year (Zenith Care, Tier 1, age 30" in r.answer
    assert r.state["table_check"] == {"unsupported_figures": [], "refused": False, "fallback": False}


def test_a_failing_table_route_does_not_stop_the_answer(tmp_path):
    rag, _, logs = _system(tmp_path)

    def broken(*_):
        raise RuntimeError("database unreachable")
    rag.pipeline.tables.data.premiums = broken
    r = rag.ask(PREMIUM_Q, show_sources=False)
    assert "database unreachable" in r.state["table_error"] and "table" not in r.state
    assert any("table route failed" in m for m in logs)


def test_long_table_passages_are_shortened_and_say_so(tmp_path):
    rag, _, _ = _system(tmp_path, tables={"enabled": True, "max_tokens": 40})
    t = rag.pipeline.table_answer("Which conditions have a 24-month waiting period?")
    lines = rag.pipeline.table_passage(t).splitlines()
    assert lines[0] == "[Policy tables → Section 10.1]" and lines[1].endswith("The policy's list:")
    assert lines[2] == "(Shortened: 3 more lines of this table passage are not shown.)"
    assert lines[3].startswith("Source: Section 10.1, page 8 (table p8-t1)")


def test_record_carries_the_table_passage_and_evidence(tmp_path):
    rag, _, _ = _system(tmp_path)
    st = rag.pipeline.search(LIST_Q)
    item = {"id": "AB", "split": "test", "type": "list", "answerable": True, "paraphrase": True, "question": LIST_Q}
    rec = build_record(item, st, rag.context, rag.pipeline.gate)
    assert rec["context"].startswith(st["facts"]) and rec["facts"] == st["facts"]
    assert rec["table"]["intent"] == "list_lookup" and rec["table"]["in_context"] and rec["table"]["trace"] == []
    assert "Television Charges" in rec["table_evidence"] and rec["table_error"] is None


def test_table_text_counts_as_evidence(tmp_path):
    rag, _, _ = _system(tmp_path)
    text = CorpusText(rag.corpus, extra_text="Annexure B Item 3 Television Charges")
    assert "television charges" in text.norm and " television charges " in text.all_words
    item = {"answerable": True, "evidence": [["3 television charges"]], "facts": [["television charges"]]}
    plain = retrieval_metrics(item, [0], [0], [0], text, rag.context.block_tokens)
    with_table = retrieval_metrics(item, [0], [0], [0], text, rag.context.block_tokens, "3 Television Charges")
    assert plain["context_recall"] == 0.0 and with_table["context_recall"] == 1.0
    assert with_table["search_recall"] == 1.0 and with_table["facts_in_context"] == 1.0
    assert with_table["evidence_in_chunk_file"] == 1.0


def test_retrieval_evaluation_with_the_tables(tmp_path):
    rag, _, _ = _system(tmp_path)
    run = Evaluation(rag.cfg, "tables", mode="retrieval", system=rag, log=lambda *_: None).run()
    assert set(run["table_route"]) >= {"answered", "in_context", "fallbacks", "errors"}
    assert all("table" in r and "facts" in r for r in run["records"])
    assert run["models"]["tables"]["doc_id"] == "pytest-tables"


@pytest.mark.parametrize("tables, message", [
    ({"enabled": True, "on_unsupported_figures": "ignore"}, "on_unsupported_figures"),
    ({"enabled": True, "doc_id": "unknown"}, "is not a configured document"),
])
def test_tables_settings_are_checked(tmp_path, tables, message):
    with pytest.raises(ConfigError, match=message):
        fake_config(tmp_path, tables=tables)


def test_several_documents_need_a_tables_document(tmp_path):
    docs = [{"id": "sample", "name": "A", "chunks": str(DATA / "sample_policy_chunks.json")},
            {"id": "second", "name": "B", "chunks": str(DATA / "second_policy_chunks.json")}]
    with pytest.raises(ConfigError, match="tables.doc_id: required"):
        fake_config(tmp_path, documents=docs, tables={"enabled": True})
    assert fake_config(tmp_path, documents=docs, tables={"enabled": True, "doc_id": "second"}).tables.doc_id == "second"


def test_tables_are_off_by_default(system):
    assert system.cfg.tables.enabled is False and system.tables is None
    st = system.pipeline.search("Is cataract covered from day one?")
    assert "table" not in st and "facts" not in st


def test_a_fact_in_the_table_passage_is_not_a_corpus_gap(tmp_path):
    from policy_rag.evaluation.scoring import Scorer
    from policy_rag.evaluation.testset import load_testset
    rag, _, _ = _system(tmp_path)
    scorer = Scorer(load_testset(rag.cfg.evaluation.testset), rag.corpus, rag.context)
    item = {"id": "T", "answerable": True, "facts": [["rs 1,450 a year"]], "evidence": [], "question": "Premium?"}
    scorer.by_id = {"T": item}
    rec = {"id": "T", "answer": "I cannot say.", "context": "Policy tables (exact values ...). Rs. 1,450 a year",
           "facts": "Rs. 1,450 a year", "gate_route": "answer"}
    stages = {i["stage"] for i in scorer.score_record(rec, live=False)["verdict"]["issues"]}
    assert "corpus" not in stages and "generation" in stages
