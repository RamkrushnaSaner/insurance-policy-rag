"""The table route over the tables of migration 002 (printed lists, claim deadlines, grievance contacts, Sum Insured
examples), on the invented policy of tests/table_fixtures.py: routing, the precision rule for list items, whole
lists, the evidence behind the facts, linked text chunks and tracing to the printed cells. No database (the same
questions run against Postgres in test_tables_integration.py, opt-in)."""
import copy

import pytest

from policy_rag.tables import COMPLETE_INTENTS, INTENTS, TableRoute

from table_fixtures import MemoryTableData, dataset


LIST_QUESTIONS = [   # also asked of Postgres in test_tables_integration.py (same answers expected)
    "Will I be reimbursed for attendant charges and the TV in my hospital room?",
    "Is cataract covered from day one?",
    "How long do I have to wait before kidney stone treatment is covered?",
    "Does the policy pay for an ambulance to take me to the hospital?",
    "Will the policy pay for my wife's delivery?",
    "What does the AYUSH Treatment cover include?",
    "Which conditions have a 24-month waiting period?",
    "Which conditions have a 36-month waiting period?",
    "Which illnesses count as Major Medical Illnesses?",
    "Is kidney failure a major medical illness?",
    "Who can be covered under this policy?",
    "Can I cover my mother-in-law?",
    "Which benefits does the Global Health Cover (Emergency & Planned Treatments) include?",
    "Which benefits are included in the global health cover?",
    "How do I file a reimbursement claim?",
    "How do I raise a grievance?",
    "Can you show an example of how the sum insured is used across claims?",
    "Are gloves paid?",
]


@pytest.fixture(scope="module")
def route():
    return TableRoute(MemoryTableData(dataset(lists=True)))


def items(answer):
    return [r["item"] for r in answer.rows]


@pytest.mark.parametrize("question, intent", [
    ("How do I file a reimbursement claim? Give me the steps.", "claim_deadline"),
    ("What is the deadline to submit post-hospitalisation bills?", "claim_deadline"),
    ("How soon will I get paid after submitting my claim papers?", "list_lookup"),     # settlement, not the deadline
    ("How do I raise a grievance?", "grievance_contact"),
    ("Whom should I contact if I have a complaint about a claim?", "grievance_contact"),
    ("In what order is the Sum Insured used up when I make claims?", "si_utilisation"),
    ("Can you show an example of how the sum insured is used across claims?", "si_utilisation"),
    ("Is cataract covered from day one?", "list_lookup"),
    ("Is kidney failure a major medical illness?", "list_lookup"),
    ("What is the annual premium for Zenith Care for a 30-year-old?", "premium"),      # the earlier kinds come first
    ("Can I pay my premium every month?", "instalment"),
    ("What is the maximum entry age for a child?", None),
])
def test_routing(route, question, intent):
    assert route.route(question) == intent


def test_kinds_and_complete_answers():
    assert INTENTS[-4:] == ("claim_deadline", "grievance_contact", "si_utilisation", "list_lookup")
    assert set(COMPLETE_INTENTS) == {"premium", "deductible_discount", "instalment", "claim_limit", "checkup_limit",
                                     "city_tier"}


def test_items_named_in_the_question(route):
    a = route.answer("Will I be reimbursed for attendant charges and the TV in my hospital room?")
    assert a.answered and a.fields["mode"] == "items" and items(a) == ["Attendant Charges", "Television Charges"]
    assert "Annexure B – List I - Items not covered (Non-Medical Expenses) (Annexure B, page 18):" in a.text
    assert "- 2. Attendant Charges" in a.text and "- 3. Television Charges" in a.text
    assert not a.complete and a.trace() == [] and a.verify(a.text) == []


def test_waiting_list_item_brings_the_period_from_the_text(route):
    a = route.answer("Is cataract covered from day one?")
    assert items(a) == ["Cataract"]
    assert a.text.startswith("From the policy's lists:\nSection 10.1, Excl02 – 24-month waiting period. "
                             "The policy text (page 8) says: \"i. The listed conditions are excluded until 24 months")
    assert "Not applicable to claims due to an Accident." in a.text


def test_words_written_together_in_the_list(route):
    a = route.answer("How long do I have to wait before kidney stone treatment is covered?")
    assert items(a) == ["Kidneystone and bladder stone disorders"]


@pytest.mark.parametrize("question, why", [
    ("Does the policy pay for an ambulance to take me to the hospital?", "the only word shared is a section title's"),
    ("Will the policy pay for my wife's delivery?", "'delivery' is not the item's leading word"),
    ("What does the AYUSH Treatment cover include?", "AYUSH Treatment is a section of its own"),
    ("Are spectacles covered?", "not on any list"),
])
def test_weak_matches_are_not_answers(route, question, why):
    a = route.answer(question)
    assert a.intent == "list_lookup" and a.status == "no_match" and not a.rows, why
    assert a.facts_block() == ""


def test_whole_lists(route):
    a = route.answer("Which conditions have a 24-month waiting period?")
    assert a.fields["mode"] == "whole list" and len(a.rows) == 6
    assert "- Illnesses (Section 10.1, page 8, 3 listed): Kidneystone and bladder stone disorders; Cataract; " \
           "Varicose veins" in a.text
    a = route.answer("Which illnesses count as Major Medical Illnesses?")
    assert "Major Medical Illness (Section 4.8, page 5, 4 listed):" in a.text and "4. Major Organ Transplant" in a.text
    a = route.answer("Who can be covered under this policy?")
    assert a.text.splitlines()[2] == "i. Spouse; ii. Son; iii. Mother-in-law"


def test_whole_list_for_another_waiting_period_is_not_given(route):
    a = route.answer("Which conditions have a 36-month waiting period?")
    assert a.status == "no_match" and a.fields["skipped"]


def test_global_cover_named_in_the_question(route):
    a = route.answer("Which benefits does the Global Health Cover (Emergency & Planned Treatments) include?")
    assert {r["cover"] for r in a.rows} == {"Global Health Cover (Emergency & Planned Treatments)"}
    assert "B 3.4 AYUSH Treatment" in a.text
    a = route.answer("Which benefits are included in the global health cover?")
    assert len({r["cover"] for r in a.rows}) == 2                     # no cover named: both


def test_list_name_words_do_not_count(route):
    a = route.answer("Is kidney failure a major medical illness?")
    assert items(a) == ["Kidney failure requiring regular dialysis"]  # not "Major Organ Transplant"


def test_claim_deadlines(route):
    a = route.answer("How do I file a reimbursement claim?")
    assert a.text.splitlines()[1:3] == ["- Reimbursement of Hospitalization Expenses: Within 20 days of date of discharge.",
                                        "- Reimbursement of Post-Hospitalization Expenses: Within 10 days of end of treatment."]
    assert a.trace() == [] and a.verify(a.text) == []
    assert "Within 20 days of date of discharge." in a.evidence_text                   # the whole printed row
    assert a.linked_chunks and a.table_ids == ["p10-t1"]


def test_grievance_contacts(route):
    a = route.answer("How do I raise a grievance?")
    assert "- Write to us – Escalation level 1: escalate@zenith.example" in a.text and a.trace() == []


def test_si_example_amounts_are_traced_to_the_row(route):
    a = route.answer("Can you show an example of how the sum insured is used across claims?")
    assert "| 1 | Rs. 7 lakh | Rs. 5 lakh | - | - | Rs. 0 | Rs. 5 lakh | Base |" in a.text
    assert "Note: *A single claim cannot exceed the Sum Insured available." in a.text   # footnote mark in a cell
    assert a.trace() == []
    bad = copy.deepcopy(a)
    bad.rows[0]["admissible_inr"] = 600_000
    assert bad.trace() == ["p14-t1 row 1 col 0: amount 600000 not printed in the row"]


def test_trace_catches_an_item_that_differs_from_its_cell(route):
    a = route.answer("Are gloves paid?")
    assert items(a) == ["Gloves"] and a.trace() == []
    a.rows[0]["item"] = "Gloves and masks"
    assert a.trace() == ["p18-t1 row 4 col 1: item 'Gloves and masks' vs printed 'Gloves'"]


def test_evidence_is_the_cells_used(route):
    a = route.answer("Will I be reimbursed for attendant charges and the TV in my hospital room?")
    assert a.evidence_text.splitlines() == [
        "Annexure B – List I - Items not covered (Non-Medical Expenses) Attendant Charges",
        "Annexure B – List I - Items not covered (Non-Medical Expenses) Television Charges",
        "2 Attendant Charges", "3 Television Charges"]
    assert a.to_dict()["table_ids"] == ["p18-t1"] and a.to_dict()["linked_chunks"] == a.linked_chunks


def test_facts_block_for_the_answer_model(route):
    a = route.answer("Is cataract covered from day one?")
    assert a.facts_block().startswith("Policy tables (exact values read from the database). From the policy's lists:")
    assert a.facts_block().endswith("Source: Section 10.1, page 8 (table p8-t1) – Illnesses")


def test_without_the_lists_nothing_new_is_routed():
    plain = TableRoute(MemoryTableData(dataset()))
    assert not plain.has_lists
    for q in ("Is cataract covered from day one?", "How do I raise a grievance?", "How do I file a reimbursement claim?"):
        assert plain.route(q) is None
