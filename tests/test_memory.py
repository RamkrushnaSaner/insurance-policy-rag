"""Conversation memory: follow-up questions are rewritten into standalone questions; single questions unchanged."""
import json

import pytest

from policy_rag.memory import Conversation, check_rewrite
from policy_rag.pipeline import RAGSystem

from conftest import DATA, ROOT, fake_config

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


def test_figures_may_come_only_from_the_users_questions():
    history = [{"question": "Is there a 24-month waiting period for kidney stones?",
                "answer": "Yes. Kidney stones have a 24-month waiting period; hernia also has one (Section 10.1)."}]
    q, why = check_rewrite('Standalone question: "Does the 24-month waiting period apply after an accident?"',
                           "Does it apply after an accident?", history)
    assert why is None and q == "Does the 24-month waiting period apply after an accident?"
    history = [{"question": "What is the waiting period for kidney stones?", "answer": "It is 24 months (Section 10.1)."}]
    q, why = check_rewrite("Does the 24-month waiting period for kidney stones apply after an accident?",
                           "Does it apply after an accident?", history)
    assert q == "Does it apply after an accident?" and "not in the user's questions: 24" in why


def test_a_rewrite_may_take_a_name_but_not_sentences_from_an_answer():
    history = [{"question": "Do I get extra cover when I renew?",
                "answer": "Yes. The Plus Benefit increases the base sum insured automatically at each renewal, "
                          "irrespective of claims."}]
    q, why = check_rewrite("How does the Plus Benefit change if I lower my sum insured?",
                           "How does it change if I lower my sum insured?", history)
    assert why is None                                                       # a name from the answer is fine
    q, why = check_rewrite("How does the Plus Benefit, which increases the base sum insured automatically at each "
                           "renewal, change?", "How does it change?", history)
    assert q == "How does it change?" and "copies an earlier answer" in why
    q, why = check_rewrite("Do I get extra cover when I renew if I made claims?", "And if I made claims?", history)
    assert why is None                                                       # the user's own words are fine
    assert check_rewrite("How does the Plus Benefit, which increases the base sum insured automatically at each "
                         "renewal, change?", "How does it change?", history, copy_words=0)[1] is None


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
    assert set(turns) == {"SD1.2", "SD2.2", "SD3.2", "SD4.2"} and turns["SD4.2"]["kind"] == "self_contained"
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


def _row(n, kind, no_memory, memory, split="test"):
    return {"dialogue": f"T{n}", "turn": 2, "kind": kind, "split": split, "asked": "q", "rewrite": "q",
            "rewritten": False, "fallback": None, "rewrite_ok": True, "rewrite_seconds": 1.0, "rewrite_from_cache": False,
            "rewrite_think_tokens": 10, "conditions": {"no_memory": {"fully_correct": no_memory},
                                                       "memory": {"fully_correct": memory},
                                                       "gold": {"fully_correct": True}}}


def test_decision_counts_follow_ups_that_need_context_and_guards_the_rest(tmp_path):
    from policy_rag.evaluation.dialogues import DialogueEvaluation
    ev = DialogueEvaluation(fake_config(tmp_path), "x", splits=["dev", "test"])
    fixed = [_row(n, "follow_up", False, True) for n in range(8)]
    keep = [_row(10, "topic_switch", True, True), _row(11, "self_contained", True, True),
            _row(12, "follow_up", True, True, split="dev")]
    summary, verdict = ev._summarize(fixed + keep)
    assert verdict == "ADOPT MEMORY" and summary["primary"]["n"] == 8 and len(summary["primary"]["fixed"]) == 8
    _, verdict = ev._summarize(fixed + keep + [_row(13, "self_contained", True, False)])
    assert verdict.startswith("DO NOT ADOPT") and "harm-check" in verdict and "T13.2" in verdict
    _, verdict = ev._summarize([_row(n, "follow_up", False, True) for n in range(5)] + keep)
    assert "not significantly better" in verdict                       # 5 fixed, 0 broken: p = 0.0625


def test_follow_up_files_that_need_context_are_checked(tmp_path):
    from policy_rag.evaluation.dialogues import load_dialogues
    def write(turn):
        p = tmp_path / "d.json"
        p.write_text(json.dumps({"follow_ups_need_context": True, "dialogues": [
            {"id": "A", "turns": [{"question": WELLNESS, "item": "S03"}, {"item": "S03", **turn}]}]}))
        return p
    assert load_dialogues(write({"question": FOLLOW, "kind": "follow_up", "subject": ["wellness bonus"]}))
    with pytest.raises(ValueError, match="names its subject"):
        load_dialogues(write({"question": "How much can the Wellness Bonus grow?", "kind": "follow_up",
                              "subject": ["wellness bonus"]}))
    with pytest.raises(ValueError, match="needs a subject"):
        load_dialogues(write({"question": FOLLOW, "kind": "follow_up"}))
    assert load_dialogues(write({"question": "How much can the Wellness Bonus grow?", "kind": "self_contained"}))


def test_the_shipped_follow_up_set_follows_its_rules():
    from policy_rag.evaluation.dialogues import HARM_CHECKS, NEEDS_CONTEXT, load_dialogues
    from policy_rag.evaluation.testset import load_testset
    data = load_dialogues(ROOT / "testsets" / "optima_secure_followups.json")      # checks the subject rule
    by_id = load_testset(ROOT / "testsets" / "optima_secure_v3.json").by_id
    turns = [t for d in data["dialogues"] for t in d["turns"][1:]]
    assert data["follow_ups_need_context"] and all(by_id[t["item"]]["split"] == "test" for t in turns)
    assert sum(t["kind"] in NEEDS_CONTEXT for t in turns) >= 25 and sum(t["kind"] in HARM_CHECKS for t in turns) >= 10
    assert all(t["question"] == by_id[t["item"]]["question"] for t in turns if t["kind"] == "topic_switch")


@pytest.mark.parametrize("question, refers", [
    ("Is it covered?", True), ("Does that apply to day care too?", True), ("What does the first one exclude?", True),
    ("And for hernia?", True), ("What about my parents?", True), ("Is there one for NRIs?", True),
    ("Can they refuse to renew my policy?", True), ("If I do that, do I keep my waiting period?", True),
    ("How long is the free look period?", False), ("Is there a waiting period for cataract?", False),
    ("Does this policy cover treatment taken outside India?", False), ("Is cataract surgery covered from day one?", False),
    ("What is domiciliary hospitalization and when is it covered?", False),
    ("What is the 10% co-payment that applies to every claim?", False),
])
def test_needs_rewrite_finds_questions_that_point_back(question, refers):
    from policy_rag.memory import needs_rewrite
    assert needs_rewrite(question) is refers


def test_the_single_questions_of_the_test_set_are_all_complete():
    from policy_rag.memory import needs_rewrite
    from policy_rag.evaluation.testset import load_testset
    flagged = [i["id"] for i in load_testset(ROOT / "testsets" / "optima_secure_v3.json").items if needs_rewrite(i["question"])]
    assert flagged == []


def test_a_complete_question_in_a_conversation_is_not_rewritten(system):
    chat = system.new_conversation()
    system.ask(WELLNESS, conversation=chat, show_sources=False)
    r = system.ask("What is the waiting period for kidney stones?", conversation=chat, show_sources=False)
    assert system.generator.rewrite_calls == 0 and r.state["rewrite"]["skipped"] == "complete question"
    assert r.standalone is None and r.state["query"] == "What is the waiting period for kidney stones?"


def test_rewrites_must_keep_typed_figures_and_name_something_specific():
    history = [{"question": "What does the Wellness Bonus give for a 40-year-old?", "answer": "It adds 10% a year."}]
    q, why = check_rewrite("What does the Wellness Bonus give?", "And what does it give for a 40-year-old?", history)
    assert "drops figures the user typed: 40" in why
    q, why = check_rewrite("What does the policy not pay for?", "What does it not pay for?", history)
    assert q == "What does it not pay for?" and "names nothing specific" in why
    q, why = check_rewrite("What does the Wellness Bonus not pay for?", "What does it not pay for?", history)
    assert why is None


def test_the_rewrite_answer_is_short_and_keyed_separately(system):
    system.generator.rewrites[FOLLOW] = STANDALONE
    chat = system.new_conversation()
    system.ask(WELLNESS, conversation=chat, show_sources=False)
    r = system.ask(FOLLOW, conversation=chat, show_sources=False)
    assert system.generator.rewrite_max_new_tokens == system.cfg.memory.max_new_tokens == 64
    assert system.cfg.generator.max_new_tokens == 1000 and system.cfg.generator.think_budget == 1024   # unchanged
    from policy_rag.prompts import build_condense_prompt
    prompt = build_condense_prompt(chat.recent(3, 400)[:1], FOLLOW, system.cfg.memory.template)
    assert r.state["rewrite"]["key"] == system.answerer.key(prompt, 64) != system.answerer.key(prompt)


def test_a_rewrite_cut_off_at_the_length_limit_is_not_used(system, monkeypatch):
    real = system.generator.generate
    def cut(prompt, seed, max_new_tokens=None):
        out = real(prompt, seed, max_new_tokens)
        if prompt.rstrip().endswith("STANDALONE QUESTION:"):
            out.update(answer="How much can the Wellness Bonus grow at", hit_limit=True)
        return out
    monkeypatch.setattr(system.generator, "generate", cut)
    chat = system.new_conversation()
    system.ask(WELLNESS, conversation=chat, show_sources=False)
    r = system.ask(FOLLOW, conversation=chat, show_sources=False)
    assert r.state["query"] == FOLLOW and r.state["rewrite"]["fallback"] == "rewrite cut off at the length limit"
