"""The SQL step inside the pipeline (v3.5), with the fake components, a scripted SQL model and invented tables: the
rows as the first passage, the linked text pinned, the gate, asking back in a conversation, "not in the tables",
the figure check and its fallback, a failing step, and the evaluation record."""
import copy

import pytest

from policy_rag.config import ConfigError, SqlConfig
from policy_rag.evaluation.checks import score_answer
from policy_rag.evaluation.runner import build_record
from policy_rag.pipeline import RAGPipeline, RAGSystem
from policy_rag.prompts import NO_ANSWER_TEXT

from conftest import fake_config

pytest.importorskip("sqlglot")
from policy_rag.sql_step import SchemaCard, SqlStep  # noqa: E402
from test_sql_step import FakeDb, ScriptedAnswerer, card_tables  # noqa: E402

Q = "What is the annual premium for Zenith Care for a 30-year-old in Tier 1 with a 10 lakh Sum Insured?"
SQL = "SELECT plan, age, si_1000000 FROM premium_table WHERE plan = 'Zenith Care' AND tier = 1 AND age = '30'"
ROWS = (["plan", "age", "si_1000000"], [("Zenith Care", "30", 9100)])


def _system(tmp_path, replies, db_answer=lambda q: ROWS, link=True, **sections):
    """The sample policy and a SQL step over the invented card; its premium table is linked to a sample chunk."""
    probe = RAGSystem.from_config(fake_config(tmp_path), log=lambda *_: None)
    target = next(c for c in probe.corpus.chunks if c.section_id == "10.1")
    tables = copy.deepcopy(card_tables())
    tables["premium_table"]["printed"][0]["linked_text"] = (
        [{"relation": "read_with", "label": target.label, "text": target.text}] if link else [])
    step = SqlStep(SqlConfig(enabled=True), ScriptedAnswerer(replies), db=FakeDb(db_answer), card=SchemaCard(tables))
    logs = []
    sections["sql"] = {"enabled": True, **sections.get("sql", {})}
    rag = RAGSystem.from_config(fake_config(tmp_path, **sections), log=logs.append, sql=step)
    return rag, target, logs


def test_rows_come_first_and_the_linked_text_is_pinned(tmp_path):
    rag, target, logs = _system(tmp_path, [{"action": "query", "sql": SQL}] * 2, gate={"min_score": 1000.0})
    st = rag.pipeline.search(Q)
    assert st["sql"].answered and st["facts"].startswith("[Policy tables → Annexure Z, pages 40-41]\nPolicy tables: rows")
    pos = rag.corpus.chunks.index(target)
    assert pos in st["candidates"] and st["sql_pinned"] == [pos]
    assert any(pos in g for g in st["groups"])                      # pinned, whatever its score
    assert st["route"] == "answer"                                  # the rows open the gate (no text passage is enough)
    r = rag.ask(Q, show_sources=False)
    assert r.sources[0] == "[Policy tables] Annexure Z, pages 40-41" and r.route == "answer"
    assert "SQL step over 2 tables" in "".join(logs)


ASK = {"action": "ask", "question": "How old is the person to be insured?"}
NO_AGE = "What is the premium for Zenith Care in Tier 1 with a 10 lakh Sum Insured?"


def test_a_missing_detail_closes_the_text_answer_and_the_reply_completes_the_question(tmp_path):
    rag, _, _ = _system(tmp_path, [ASK, {"action": "query", "sql": SQL}], gate={"min_score": -1000.0})
    chat = rag.new_conversation()
    r = rag.ask(NO_AGE, conversation=chat, show_sources=False)
    assert r.route == "answer" and r.sources                       # the text answered: the step did not end the run
    assert r.answer.endswith("\n\nFor an exact figure from the policy's tables: How old is the person to be insured?")
    facts = r.state["facts"]
    assert facts.startswith("[Policy tables]\n") and '"How old is the person to be insured?"' in facts
    assert "Do not give a figure from the tables or from an illustration" in facts
    assert facts in rag.pipeline.prompt_for(r.state["query"], r.state["groups"], facts)   # the note reaches the model
    assert chat.pending == NO_AGE
    r = rag.ask("30 years", conversation=chat, show_sources=False)
    assert f"QUESTION: {NO_AGE} 30 years\n" in rag.sql.answerer.prompts[-1]
    assert r.route == "answer" and r.standalone.endswith("30 years") and chat.pending is None
    assert r.sources[0].startswith("[Policy tables]") and not r.answer.endswith("insured?")
    assert chat.to_dict()["pending"] is None and len(chat.turns) == 2


def test_a_new_question_after_the_closing_question_is_not_glued_to_it(tmp_path):
    rag, _, _ = _system(tmp_path, [ASK], gate={"min_score": -1000.0})
    chat = rag.new_conversation()
    rag.ask(NO_AGE, conversation=chat, show_sources=False)
    assert chat.pending == NO_AGE
    r = rag.ask("Is cataract surgery covered?", conversation=chat, show_sources=False)
    assert r.state["query"] == "Is cataract surgery covered?" and r.standalone is None and chat.pending is None


def test_a_refusal_keeps_the_closing_question_and_still_declines(tmp_path):
    rag, _, _ = _system(tmp_path, [ASK], gate={"min_score": 1000.0})
    r = rag.ask(NO_AGE, show_sources=False)
    assert r.route == "no_answer" and r.answer.startswith(NO_ANSWER_TEXT) and r.answer.endswith("insured?")
    from policy_rag.evaluation.checks import declines_at_start
    assert declines_at_start(r.answer)


def test_what_counts_as_the_reply_to_the_closing_question():
    from policy_rag.memory import replies_to_pending
    for m in ("30 years", "35", "Tier 1, Pune", "my father, 62", "individual policy with 10 lakh"):
        assert replies_to_pending(m), m
    for m in ("Is cataract surgery covered?", "What about maternity", "How long is the waiting period", "", "  "):
        assert not replies_to_pending(m), m


def test_not_in_the_tables_leaves_the_answer_to_the_text(tmp_path):
    rag, _, _ = _system(tmp_path, [{"action": "not_in_tables", "reason": "no 7 lakh column"}])
    st = rag.pipeline.search("What is the premium for Zenith Care with a 7 lakh Sum Insured for a 30-year-old?")
    assert st["sql"].status == "not_in_tables" and st["facts"] == RAGPipeline.NO_TABLE_VALUE
    assert st["route"] in ("answer", "no_answer") and "sql_pinned" not in st     # the text gate decides


def test_questions_the_gate_skips_and_a_failing_step(tmp_path):
    rag, _, _ = _system(tmp_path, [])
    st = rag.pipeline.search("Is cataract surgery covered?")
    assert st["sql"].status == "skipped" and not st.get("facts") and rag.sql.answerer.prompts == []

    def broken(q):
        raise ConnectionError("server closed the connection")
    rag, _, logs = _system(tmp_path, [{"action": "query", "sql": SQL}], db_answer=broken)
    st = rag.pipeline.search(Q)
    assert st["sql"].status == "unavailable" and not st.get("facts")         # answered from the text alone

    class Exploding:
        card = SchemaCard(card_tables())
        card_text = card.text

        def run(self, q):
            raise RuntimeError("boom")
    rag.pipeline.sql = Exploding()
    st = rag.pipeline.search(Q)
    assert st["sql_error"] == "sql: RuntimeError: boom" and "SQL step failed" in logs[-1]


def test_the_figure_check_and_its_fallback(tmp_path):
    rag, _, logs = _system(tmp_path, [{"action": "query", "sql": SQL}])
    pipe = rag.pipeline
    out = rag.sql.run(Q)
    ctx = pipe.context_text([], pipe.sql_passage(out))
    ok = pipe.check_sql_answer(Q, "The premium is Rs. 9,100 a year.", out, ctx)
    assert ok == {"unsupported_figures": [], "refused": False, "fallback": False}
    bad = pipe.check_sql_answer(Q, "The premium is Rs. 9,900 a year.", out, ctx)
    assert bad["fallback"] and bad["unsupported_figures"] == ["Rs 9,900"] and bad["model_answer"].endswith("9,900 a year.")
    assert bad["answer"].startswith("From the policy's tables (exact values as printed):\n- plan: Zenith Care; ")
    assert "Source: Zenith Care Premium - Tier 1 (Annexure Z, pages 40-41)" in bad["answer"]
    assert "replaced by the rows written out plainly" in logs[-1]
    assert not pipe.check_sql_answer(Q, NO_ANSWER_TEXT, out, ctx)["fallback"]           # a refusal is kept
    rag.cfg.sql.on_unsupported_figures = "keep"
    assert not pipe.check_sql_answer(Q, "The premium is Rs. 9,900 a year.", out, ctx)["fallback"]


def test_long_results_are_shortened(tmp_path):
    many = (["plan", "age", "si_1000000"], [("Zenith Care", str(a), 9000 + a) for a in range(20)])
    rag, _, _ = _system(tmp_path, [{"action": "query", "sql": SQL.replace(" AND age = '30'", "")}], db_answer=lambda q: many,
                        sql={"max_tokens": 120})
    out = rag.sql.run(Q)
    text = rag.pipeline.sql_passage(out)
    assert rag.context.count_tokens(text) <= 140 and "more rows of this result are not shown.)" in text
    assert text.count("\n- plan: Zenith Care") < 20


def test_the_evaluation_record_and_scoring(tmp_path):
    rag, _, _ = _system(tmp_path, [{"action": "query", "sql": SQL}])
    st = rag.pipeline.search(Q)
    item = {"id": "Z1", "split": "dev", "type": "fact", "answerable": True, "paraphrase": False, "question": Q,
            "facts": [["9,100"]]}
    rec = build_record(item, st, rag.context, rag.pipeline.gate)
    assert rec["sql"]["status"] == "answered" and rec["sql"]["rows"] == 1 and rec["context"].startswith("[Policy tables")
    v = score_answer(item, "How old is the person?", "", "", gate_route="ask")
    assert not v["fully_correct"] and any(i["kind"] == "asked" for i in v["issues"])
    na = dict(item, answerable=False, facts=[])
    assert score_answer(na, "How old is the person?", "", "", gate_route="ask")["fully_correct"]


def test_config_checks(tmp_path):
    with pytest.raises(ConfigError, match="on_unsupported_figures"):
        fake_config(tmp_path, sql={"on_unsupported_figures": "table_answer"})


def test_a_full_evaluation_run_with_the_sql_step(tmp_path):
    from policy_rag.evaluation.runner import Evaluation
    rag, _, _ = _system(tmp_path, [{"action": "query", "sql": SQL}] * 50)
    run = Evaluation(rag.cfg, "sqlrun", mode="full", splits=["dev", "test"], system=rag).run()
    recs = run["records"]
    assert recs and all("sql" in r and "sql_check" in r for r in recs)
    gated = [r for r in recs if r["sql"] and r["sql"]["status"] != "skipped"]
    assert all(r["sql"]["status"] == "answered" and r["context"].startswith("[Policy tables") for r in gated)
    assert run["models"]["sql"]["template"] == "sql_v2" and len(run["models"]["sql"]["card_sha256"]) == 64
