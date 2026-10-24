"""End to end with the fake components: retrieval, routing, context, gate, cache, several documents."""
from policy_rag.pipeline import RAGSystem
from policy_rag.prompts import NO_ANSWER_TEXT, TEMPLATES, build_prompt

from conftest import DATA, fake_config


def test_answer_comes_from_the_context(system):
    r = system.ask("What is the waiting period for kidney stones?", show_sources=False)
    assert r.route == "answer"
    assert "24 months" in r.answer
    assert any("10.1" in s for s in r.sources)


def test_gate_refuses_when_nothing_relevant_is_found(system):
    r = system.ask("Xylophone tuition reimbursement?", show_sources=False)
    assert r.route == "no_answer" and r.answer == NO_ANSWER_TEXT
    assert system.generator.calls == 0                     # the answer model is not called


def test_premium_tables_are_searched_only_for_cost_questions(system):
    prem = {c.idx for c in system.corpus.chunks if c.index == "premium"}
    st = system.pipeline.search("What is the annual premium for age 40?")
    assert st["routes"] == ["premium"] and prem & set(st["candidates"])
    st = system.pipeline.search("What is the waiting period for hernia?")
    assert st["routes"] == [] and not prem & set(st["candidates"])


def test_title_pinning_puts_the_named_section_first(system):
    st = system.pipeline.search("Explain the Wellness Bonus")
    first = system.corpus.chunks[st["groups"][0][0]]
    assert first.section_id == "4.1"


def test_pointer_brings_the_list_below(system):
    st = system.pipeline.search("Which conditions have a waiting period of 24 months?")
    ids = {i for g in st["groups"] for i in g}
    sids = [system.corpus.chunks[i].section_id for i in ids]
    assert sids.count("10.1") == 2                        # the "listed below" chunk and the list itself


def test_cache_reuses_answers(system):
    q = "How many days before admission are medical bills paid?"
    a1 = system.ask(q, show_sources=False)
    a2 = system.ask(q, show_sources=False)
    assert a1.answer == a2.answer and system.generator.calls == 1
    assert a2.state["gen"]["from_cache"] is True


def test_context_budget_is_respected(tmp_path):
    rag = RAGSystem.from_config(fake_config(tmp_path, context={"token_budget": 40}), log=lambda *_: None)
    st = rag.pipeline.search("hospitalization expenses room rent nursing")
    assert rag.context.tokens(st["groups"]) <= 40


def test_single_document_prompt_is_the_published_v2_3_prompt():
    p = build_prompt("Q?", "CTX")
    assert p == TEMPLATES["policy_analyst_v2_3"].format(context="CTX", query="Q?", extra_rules="")
    assert "different policies" not in p
    assert "different policies" in build_prompt("Q?", "CTX", multi_document=True)


def test_several_policies_with_a_document_filter(tmp_path):
    docs = [{"id": "sample", "name": "Sample Health Shield", "chunks": str(DATA / "sample_policy_chunks.json")},
            {"id": "second", "name": "Sample Family Care", "chunks": str(DATA / "second_policy_chunks.json")}]
    rag = RAGSystem.from_config(fake_config(tmp_path, documents=docs), log=lambda *_: None)
    q = "What are the hospitalization expenses covered?"
    both = rag.pipeline.search(q)
    assert {rag.corpus.chunks[i].doc_id for i in both["candidates"]} == {"sample", "second"}
    only = rag.pipeline.search(q, ["second"])
    assert {rag.corpus.chunks[i].doc_id for i in only["candidates"]} == {"second"}
    assert all(rag.corpus.chunks[i].doc_id == "second" for g in only["groups"] for i in g)
    r = rag.ask(q, documents=["second"], show_sources=False)
    assert "48 hours" in r.answer and all(s.startswith("[Sample Family Care |") for s in r.sources)
    prompt = rag.pipeline.prompt_for(q, only["groups"])
    assert "different policies" in prompt


def test_neighbours_never_cross_documents(tmp_path):
    docs = [{"id": "sample", "name": "A", "chunks": str(DATA / "sample_policy_chunks.json")},
            {"id": "second", "name": "B", "chunks": str(DATA / "second_policy_chunks.json")}]
    rag = RAGSystem.from_config(fake_config(tmp_path, documents=docs), log=lambda *_: None)
    last_of_first = max(c.idx for c in rag.corpus.chunks if c.doc_id == "sample" and c.index == "policy")
    for i in range(len(rag.corpus.chunks)):
        group = rag.context.group_for(i)
        assert len({rag.corpus.chunks[j].doc_id for j in group}) == 1, (i, group)
    assert last_of_first >= 0
