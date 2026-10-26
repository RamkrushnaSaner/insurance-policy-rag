"""Conversation memory: follow-up questions are rewritten into standalone questions; single questions unchanged."""
import json

import pytest

from policy_rag.memory import Conversation, check_rewrite
from policy_rag.pipeline import RAGSystem

from conftest import DATA, fake_config

WELLNESS = "What is the Wellness Bonus?"
FOLLOW = "How much can it grow at most?"
STANDALONE = "How much can the Wellness Bonus grow at most?"


def build(tmp_path, **sections):
    return RAGSystem.from_config(fake_config(tmp_path, **sections), log=lambda *_: None)


def _untimed(state):
    """The state without the generation's wall-clock fields, which differ from run to run."""
    gen = {k: v for k, v in (state.get("gen") or {}).items() if k not in ("seconds", "created")}
    return {**state, "gen": gen} if "gen" in state else state


def test_single_questions_take_the_same_path_with_memory_on_or_off(tmp_path):
    on, off = build(tmp_path / "a"), build(tmp_path / "b", memory={"enabled": False})
    for q in (WELLNESS, "What is the waiting period for kidney stones?", "Xylophone tuition reimbursement?"):
        a, b = _untimed(on.pipeline.run(q)), _untimed(off.pipeline.run(q))
        assert {k: a[k] for k in b} == b and "rewrite" not in a          # nothing added, nothing changed
        if a.get("groups"):
            assert on.answerer.key(on.pipeline.prompt_for(q, a["groups"])) == \
                off.answerer.key(off.pipeline.prompt_for(q, b["groups"]))  # same prompt, same cache key
    assert on.generator.rewrite_calls == 0


def test_first_turn_of_a_conversation_is_not_rewritten(system):
    chat = system.new_conversation()
    r = system.ask(WELLNESS, conversation=chat, show_sources=False)
    assert r.standalone is None and "rewrite" not in r.state and system.generator.rewrite_calls == 0
    assert len(chat.turns) == 1 and chat.turns[0].standalone == WELLNESS


def test_follow_up_is_rewritten_searched_and_stored(system):
    system.generator.rewrites[FOLLOW] = STANDALONE
    chat = system.new_conversation()
    system.ask(WELLNESS, conversation=chat, show_sources=False)
    r = system.ask(FOLLOW, conversation=chat, show_sources=False)
    assert r.standalone == STANDALONE and r.state["query"] == STANDALONE
    assert r.state["original_question"] == FOLLOW and r.state["rewrite"]["fallback"] is None
    first = system.corpus.chunks[r.state["groups"][0][0]]
    assert first.section_id == "4.1"                                       # title pinning saw "Wellness Bonus"
    assert "50%" in r.answer
    assert [t.question for t in chat.turns] == [WELLNESS, FOLLOW] and chat.turns[1].standalone == STANDALONE
    # the answer prompt holds the standalone question and no conversation
    prompt = system.pipeline.prompt_for(r.state["query"], r.state["groups"])
    assert STANDALONE in prompt and FOLLOW not in prompt and WELLNESS not in prompt.split("QUESTION:")[1]


def test_without_memory_the_follow_up_is_searched_as_typed(tmp_path):
    rag = build(tmp_path, memory={"enabled": False})
    chat = rag.new_conversation()
    rag.ask(WELLNESS, conversation=chat, show_sources=False)
    r = rag.ask(FOLLOW, conversation=chat, show_sources=False)
    assert r.standalone is None and r.state["query"] == FOLLOW and rag.generator.rewrite_calls == 0


def test_rewrites_are_cached(system):
    system.generator.rewrites[FOLLOW] = STANDALONE
    for _ in range(2):
        chat = system.new_conversation()
        system.ask(WELLNESS, conversation=chat, show_sources=False)
        r = system.ask(FOLLOW, conversation=chat, show_sources=False)
    assert system.generator.rewrite_calls == 1 and r.state["rewrite"]["from_cache"] is True


@pytest.mark.parametrize("reply, reason", [
    ("", "empty"),
    ("Yes, the bonus grows by 10% a year.", "answer"),
    ("How much can the Wellness Bonus grow after 7 years?", "figures"),
    ("x" * 500, "too long"),
])
def test_guards_fall_back_to_the_question_as_asked(system, reply, reason):
    system.generator.rewrites[FOLLOW] = reply
    chat = system.new_conversation()
    system.ask(WELLNESS, conversation=chat, show_sources=False)
    r = system.ask(FOLLOW, conversation=chat, show_sources=False)
    assert r.state["query"] == FOLLOW and reason in r.state["rewrite"]["fallback"]


def test_guard_accepts_figures_from_the_conversation():
    history = [{"question": "What is the waiting period for kidney stones?", "answer": "It is 24 months (Section 10.1)."}]
    q, why = check_rewrite('Standalone question: "Does the 24-month waiting period apply after an accident?"',
                           "Does it apply after an accident?", history)
    assert why is None and q == "Does the 24-month waiting period apply after an accident?"


def test_a_failing_rewrite_does_not_lose_the_turn(system, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model unavailable")
    chat = system.new_conversation()
    system.ask(WELLNESS, conversation=chat, show_sources=False)
    monkeypatch.setattr(system.rewriter, "rewrite", boom)
    r = system.ask(FOLLOW, conversation=chat, show_sources=False)
    assert r.state["query"] == FOLLOW and "rewrite failed" in r.state["rewrite"]["fallback"] and r.answer


def test_conversation_keeps_its_document_filter(tmp_path):
    docs = [{"id": "sample", "name": "Sample Health Shield", "chunks": str(DATA / "sample_policy_chunks.json")},
            {"id": "second", "name": "Sample Family Care", "chunks": str(DATA / "second_policy_chunks.json")}]
    rag = build(tmp_path, documents=docs)
    rag.generator.rewrites["And what is the waiting period for pre-existing diseases there?"] = \
        "What is the waiting period for pre-existing diseases?"
    chat = rag.new_conversation(["second"])
    rag.ask("What are the hospitalization expenses covered?", conversation=chat, show_sources=False)
    r = rag.ask("And what is the waiting period for pre-existing diseases there?", conversation=chat, show_sources=False)
    assert {rag.corpus.chunks[i].doc_id for i in r.state["candidates"]} == {"second"} and "48 months" in r.answer
    with pytest.raises(KeyError):
        rag.new_conversation(["missing"])


def test_window_answer_excerpts_and_save_load(tmp_path):
    chat = Conversation(documents=["sample"])
    for n in range(5):
        chat.add(f"q{n}", f"standalone {n}", "word " * 200, ["[Section 1]"], "answer")
    recent = chat.recent(3, 50)
    assert [h["question"] for h in recent] == ["standalone 2", "standalone 3", "standalone 4"]
    assert all(len(h["answer"]) <= 54 and h["answer"].endswith("...") for h in recent)
    back = Conversation.load(chat.save(tmp_path / "chat.json"))
    assert back.to_dict() == chat.to_dict()


def test_chat_loop_commands(system, tmp_path):
    system.generator.rewrites[FOLLOW] = STANDALONE
    inputs = iter([WELLNESS, FOLLOW, "/history", f"/save {tmp_path / 'c.json'}", "/doc sample", "/new",
                   "/doc nope", "/quit"])
    conv = system.chat(input_fn=lambda _: next(inputs))
    saved = json.loads((tmp_path / "c.json").read_text())
    assert [t["standalone"] for t in saved["turns"]] == [WELLNESS, STANDALONE]
    assert conv.turns == [] and conv.documents == ["sample"]                # /new kept the document filter


def test_dialogue_evaluation_end_to_end(tmp_path):
    from policy_rag.evaluation.dialogues import DialogueEvaluation
    rag = build(tmp_path)
    rag.generator.rewrites.update({FOLLOW: STANDALONE,
                                   "Is joint replacement after an accident subject to that?":
                                   "Is joint replacement after an accident subject to the waiting period?"})
    run = DialogueEvaluation(None, "dlg", system=rag, log=lambda *_: None).run()
    turns = {f"{t['dialogue']}.{t['turn']}": t for t in run["turns"]}
    assert set(turns) == {"SD1.2", "SD2.2", "SD3.2"}
    assert turns["SD1.2"]["rewrite"] == STANDALONE and turns["SD1.2"]["rewrite_ok"]
    assert turns["SD3.2"]["rewrite"] == "How many days before admission are medical bills paid?"   # topic switch kept
    assert set(turns["SD1.2"]["conditions"]) == {"no_memory", "memory", "gold"}
    assert turns["SD1.2"]["conditions"]["memory"]["fully_correct"]
    assert run["decision"].startswith(("ADOPT", "DO NOT ADOPT", "No decision"))
    for f in ("dialogues.json", "dialogue_report.md", "dialogues.csv"):
        assert (tmp_path / "runs" / "dlg" / f).exists()


def test_resume_from_a_saved_run_reuses_its_answers(tmp_path):
    from policy_rag.evaluation.runner import Evaluation
    first = Evaluation(fake_config(tmp_path / "one"), "base", "full", log=lambda *_: None).run()
    rag = build(tmp_path / "two")                                          # a fresh cache
    n = rag.cache.add_from(str(tmp_path / "one" / "runs" / "base" / "run.json"))
    assert n == sum(1 for r in first["records"] if r.get("gen"))
    again = Evaluation(None, "again", "full", system=rag, log=lambda *_: None).run()
    assert again["llm_calls"]["generated"] == 0
