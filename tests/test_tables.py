"""The table route on an invented policy (tests/table_fixtures.py): routing, fields, input checks, calculations,
the plain answer, number verification and tracing values to their printed cells. No database: the SQL itself is
tested against a real Postgres in test_tables_integration.py (opt-in)."""
import json
from decimal import Decimal

import pytest

from policy_rag.tables import TableRoute, group_inr, inr, printed_number

from table_fixtures import MemoryTableData, premium_value


@pytest.fixture(scope="module")
def route():
    return TableRoute(MemoryTableData())


@pytest.mark.parametrize("question, intent", [
    ("What is the annual premium for Zenith Care for a 30-year-old?", "premium"),
    ("How much more does Zenith Care Plus cost than Zenith Care for a 45 year old?", "premium"),
    ("How much more does a 40 year old pay in Mumbai than in Pune?", "premium"),
    ("Can I claim the cost of new spectacles?", None),                  # "cost of" is not about the premium
    ("How much does the policy pay for lost baggage?", None),
    ("Do the premium rates include GST?", None),
    ("Do I get my premium back as cashback?", None),
    ("Can I pay my premium every month?", "instalment"),
    ("How do the grace periods differ for monthly and yearly premium instalments?", "instalment"),
    ("What discount do I get with a Rs 50,000 deductible?", "deductible_discount"),
    ("What discount is given for buying online?", None),
    ("What is the maximum single-claim amount for Zenith Care in Year 2?", "claim_limit"),
    ("What is the difference between Zenith Care and Zenith Care Plus plans?", "claim_limit"),
    ("What is the maximum entry age for a child?", None),
    ("What is the Preventive Health Check-up benefit?", "checkup_limit"),
    ("Which cities are classified as Tier 1 for premium calculation?", "city_tier"),
    ("Is Pune a Tier 1 city?", "city_tier"),
    ("What is the waiting period for hernia?", None),
])
def test_routing(route, question, intent):
    assert route.route(question) == intent


def test_one_premium_with_its_cell(route):
    a = route.answer("What is the annual premium for Zenith Care with a Rs. 10 lakh Sum Insured for a 30-year-old "
                     "living in Mumbai?")
    v = premium_value("Zenith Care", 1, 30, 1_000_000)
    assert a.answered and a.intent == "premium" and len(a.rows) == 1
    assert a.text.startswith(f"Rs. {group_inr(v)} a year (Zenith Care, Tier 1, age 30, Sum Insured Rs. 10 lakh; "
                             "Annexure Z, excluding GST).")
    assert "Mumbai is listed as Tier 1 in Section 14 (page 15)." in a.text
    r = a.rows[0]
    assert (r["table_id"], r["page"], r["printed"]) == ("p20-t1", 20, group_inr(v))
    assert a.trace() == [] and a.verify(a.text) == []
    assert a.verify(f"It costs Rs. {group_inr(v + 100)} a year.") == [f"Rs {group_inr(v + 100)}"]   # a wrong figure
    assert v and a.verify(f"It costs Rs. {group_inr(v)} a year for Rs. 10 lakh cover.") == []
    assert json.loads(json.dumps(a.to_dict()))["status"] == "answered"
    assert a.facts_block().startswith("Policy tables (exact values read from the database; ")


def test_two_plans_give_the_difference_in_decimal(route):
    a = route.answer("How much more does Zenith Care Plus cost than Zenith Care for a 45-year-old living in Pune "
                     "with Rs. 5 lakh cover?")
    lo, hi = premium_value("Zenith Care", 2, 45, 500_000), premium_value("Zenith Care Plus", 2, 45, 500_000)
    assert a.calculations == [{"what": "Zenith Care Plus minus Zenith Care", "inputs": [hi, lo],
                               "result": Decimal(hi - lo), "unit": "inr"}]
    assert a.text.startswith(f"Zenith Care Plus costs Rs. {hi - lo} more a year than Zenith Care")
    assert "Pune is not named in the city list of Section 14 (page 15), so Tier 2 (Rest of India) applies." in a.text
    assert a.verify(f"Rs. {hi - lo} more") == []                   # the computed difference counts as supported


def test_places_aliases_and_two_tiers(route):
    a = route.answer("What premium for a 31 year old on Zenith Care, 5 lakh, Gurugram?")
    assert a.rows[0]["tier"] == 1 and "Gurugram counts as National Capital Region (NCR)" in a.text
    a = route.answer("How much more does a 31 year old pay in Mumbai than in Pune for Zenith Care 10 lakh?")
    assert {r["tier"] for r in a.rows} == {1, 2}
    assert a.text.startswith("Tier 2 (Pune) costs Rs. 100 more a year than Tier 1 (Mumbai)")
    a = route.answer("What is the annual premium for a 30-year-old for Zenith Care with 5 lakh cover?")
    assert {r["tier"] for r in a.rows} == {1, 2} and "both tiers are shown" in a.text


@pytest.mark.parametrize("question, status, says", [
    ("What is the annual premium for Zenith Lite for a 30-year-old?", "not_in_policy",
     "does not include a premium table for Zenith Lite: Annexure Z gives premiums only for Zenith Care and Zenith Care Plus"),
    ("What is the annual premium for Zenith Gold for a 30-year-old?", "not_in_policy",
     "does not include a plan called Zenith Gold"),
    ("What is the annual premium for a 30-year-old with a Rs. 7 lakh Sum Insured?", "not_in_policy",
     "does not include a Sum Insured of Rs. 7 lakh: it lists Rs. 5 lakh, Rs. 10 lakh"),
    ("What is the cost of the policy for a 125 year old?", "not_in_policy", "does not include age 125"),
    ("What is the annual premium for Zenith Care with Rs. 10 lakh cover?", "needs_input", "does not give one"),
    ("What is the annual premium for a 5-year-old on Zenith Care with 5 lakh cover?", "not_in_policy", "no row"),
])
def test_input_checks_explain_instead_of_guessing(route, question, status, says):
    a = route.answer(question)
    assert a.status == status and says in a.text and not a.rows
    assert a.facts_block() == ""


def test_product_name_means_every_priced_plan(route):
    a = route.answer("How much does my:Zenith Care cost for a 31 year old in Mumbai with 10 lakh cover?")
    assert [r["plan"] for r in a.rows] == ["Zenith Care", "Zenith Care Plus"]


def test_renewal_only_age_brings_its_footnote(route):
    a = route.answer("What premium for a 95 year old on Zenith Care, 5 lakh, in Mumbai?")
    assert "Note: Age 95 is printed as '>=90*'." in a.text and "Note: *Only for Renewal Purposes" in a.text


def test_claim_limits_table(route):
    a = route.answer("How does the maximum single-claim amount for Zenith Care change from Year 1 to Year 3?")
    assert a.fields["years"] == [1, 2, 3] and a.fields["items"] == ["max_single_claim"]
    assert "| Zenith Care – Maximum permissible amount for a single Hospitalization claim in a Policy Year | " \
           "Rs. 20 lakh | Rs. 25 lakh | Rs. 30 lakh |" in a.text
    a = route.answer("What is the difference between Zenith Care and Care Plus plans?")
    assert len(a.rows) == 2 * 6 * 3 and "| Zenith Care Plus – Secure Benefit | Rs. 15 lakh |" in a.text
    assert "Note: (Year 2) *Assuming no claim in year 1" in a.text and a.trace() == []
    a = route.answer("What is the maximum single claim under Zenith Lite in Year 1?")
    assert a.rows[0]["value"] == 1_000_000


def test_instalments_yes_and_no(route):
    a = route.answer("Can I pay my premium every month?")
    assert a.text.startswith("Yes. Premium can be paid in Monthly instalments (Section 9.1).")
    assert "- Monthly (Option 2): grace period 15 days" in a.text and a.trace() == []
    a = route.answer("Can I pay quarterly instalments?")
    assert a.text.startswith("No. The policy offers only Yearly and Monthly instalments (Section 9.1).")


def test_checkups_deductibles_and_tiers(route):
    a = route.answer("How much is the check-up limit for a family floater with 10 lakh sum insured?")
    assert [r["value"] for r in a.rows] == [5000] and "up to Rs. 5,000" in a.text and a.trace() == []
    a = route.answer("What discount do I get with a Rs 50,000 deductible on Zenith Care with 10 lakh sum insured?")
    assert [r["value"] for r in a.rows] == [Decimal("0.40")] and "40% discount" in a.text and a.trace() == []
    a = route.answer("Is Pune a Tier 1 city?")
    assert a.text.startswith("No. Pune: Tier 2.")
    a = route.answer("Which cities are classified as Tier 1 for premium calculation?")
    assert "- Tier 1: Mumbai, National Capital Region (NCR)" in a.text and "- Tier 2: Rest of India" in a.text


def test_not_routed_question_has_no_facts(route):
    a = route.answer("What is the waiting period for hernia?")
    assert a.intent is None and a.status == "not_routed" and a.text == "" and a.facts_block() == ""


def test_formatting_helpers():
    assert group_inr(1_000_000) == "10,00,000" and group_inr(338) == "338" and group_inr(128_750) == "1,28,750"
    assert inr(2_000_000, words=True) == "Rs. 20 lakh" and inr(20_000_000, words=True) == "Rs. 2 crore"
    assert inr(250_000, words=True) == "Rs. 2.5 lakh" and inr(13_500) == "Rs. 13,500"
    assert printed_number("3,50,00,000^") == 35_000_000 and printed_number("NIL") == 0
    assert printed_number("25%") == Decimal("0.25") and printed_number("15 Days") == 15 and printed_number("Age") is None
