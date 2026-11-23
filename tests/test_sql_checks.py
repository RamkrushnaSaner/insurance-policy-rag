"""The SQL step's check of a query against the question (sql_checks.py). The card is invented (plans, places, ages and
amounts); the patterns are the ones the saved sql_v2 runs showed: a column on the wrong side of an amount split, an
amount no column prints, a plan the table does not hold, a place in the wrong tier, a detail the question leaves out."""
import pytest

from policy_rag.config import SqlConfig
from policy_rag.sql_checks import NotInTables, QuestionCheck, ages_in, amounts_in, parse_value_heading, range_of
from policy_rag.sql_step import SchemaCard, SqlProblem, SqlStep, SqlValidator

from test_sql_step import ScriptedAnswerer

pytest.importorskip("sqlglot")

T1 = "Avalon, North Region, Port Avalon and Beta City & Gamma Town"
T2 = "Rest of Country"
AGES = [str(a) for a in range(0, 12)] + ["12*", "13*", ">=14*"]


def col(name, dtype, comment="", values=None):
    return {"name": name, "type": dtype, "comment": comment, "values": values}


def card():
    return SchemaCard({
        "premium_table": {
            "description": "Annexure Z: annual premium (Rs.) by age and Sum Insured.", "rows": 60,
            "columns": [col("plan", "text", "Plan named in the printed title", ["Zenith Care", "Zenith Care Plus"]),
                        col("tier", "smallint", "Premium tier", [1, 2]),
                        col("tier_cities", "text", "Places of the tier", [T1, T2]),
                        col("age", "text", "Printed heading: Age (as printed)", AGES)]
                       + [col(f"si_{n}", "integer", f"Printed heading: Sum Insured {n:,} (premium in Rs.)")
                          for n in (200000, 400000, 800000)]
                       + [col("page", "integer", "Page of the printed row")],
            "printed": [], "examples": []},
        "checkup_limit_table": {
            "description": "Section 5: check-up limit (Rs.) by Sum Insured.", "rows": 2,
            "columns": [col("policy_type", "text", "Which printed table", ["individual", "family_floater"]),
                        col("limit_si_2_lakhs", "integer", "Printed heading: 2 Lakhs (Limit of Cover in Rs.)"),
                        col("limit_si_4_8_lakhs", "integer", "Printed heading: 4 & 8 Lakhs (Limit of Cover in Rs.)"),
                        col("page", "integer", "Page of the printed table")],
            "printed": [], "examples": []},
        "deductible_discount_table": {
            "description": "Section 9: discount (%) with a deductible.", "rows": 2,
            "columns": [col("deductible_inr", "integer", "Printed heading: Deductible amount (INR)", [10000, 50000, 100000]),
                        col("pct_care_upto_4l", "numeric(5,2)",
                            "Printed heading: Zenith Care & Zenith Care Plus / Base SI <= 4 lakhs (discount %)"),
                        col("pct_care_over_4l", "numeric(5,2)",
                            "Printed heading: Zenith Care & Zenith Care Plus / Base SI > 4 lakhs (discount %)"),
                        col("pct_global", "numeric(5,2)", "Printed heading: Zenith Global (discount %)"),
                        col("page", "integer", "Page of the printed table")],
            "printed": [], "examples": []},
    })


class TierDb:
    """Answers the one query the check makes at start-up: the places of each tier."""

    def query(self, sql, params=None):
        assert sql.startswith("SELECT DISTINCT tier, tier_cities")
        return [(1, T1), (2, T2)]


@pytest.fixture(scope="module")
def qc():
    return QuestionCheck(card(), TierDb())


@pytest.fixture(scope="module")
def val():
    c = card()
    return SqlValidator(list(c.tables), 50, text_columns=c.text_columns())


def run(qc, val, question, sql):
    return qc.check(question, val.check(sql))


P = "SELECT si_400000 FROM premium_table WHERE plan = 'Zenith Care' AND tier = {t} AND age = '{a}'"


# ---------------------------------------------------------------------------------------------------- reading text
def test_amounts_ages_and_ranges_are_read_by_code():
    assert amounts_in("a 4 lakh cover, Rs 50,000 deductible, ₹8,00,000, 2L, 1 Cr, 25k") == [
        400000, 50000, 800000, 200000, 10000000, 25000]
    assert amounts_in("a 45-year-old") == []
    assert ages_in("my 9-year-old son with 4 lakh cover") == {9}
    assert ages_in("a 6-month-old baby, Rs 10,000 deductible") == {0}
    assert ages_in("ages 31 and 33, each with 2L") == {31, 33}
    assert range_of("from 2 lakh to 8 lakh", [200000, 800000]) == (200000, 800000)
    assert range_of("4 lakh upwards", [400000]) == (400000, None)
    assert range_of("4 lakh", [400000]) == (None, None)
    c = parse_value_heading("c", "Zenith Care & Zenith Care Plus / Base SI <= 4 lakhs (discount %)")
    assert c.plans == ["Zenith Care", "Zenith Care Plus"] and c.condition == ("<=", 400000)


def test_the_validator_records_every_column_read_and_every_literal_filter(val):
    c = val.check("SELECT p.si_800000 * (1 - d.pct_care_over_4l / 100.0) FROM premium_table AS p JOIN "
                  "deductible_discount_table AS d ON d.deductible_inr = 50000 WHERE p.plan = 'Zenith Care' "
                  "AND p.age IN ('9', '10') AND p.tier_cities LIKE '%Avalon%' AND 1 = p.tier")
    assert {"si_800000", "pct_care_over_4l", "deductible_inr", "plan", "age", "tier", "tier_cities"} <= set(c.read)
    assert c.filters == {"deductible_inr": [("=", 50000)], "plan": [("=", "Zenith Care")], "age": [("=", "9"), ("=", "10")],
                         "tier_cities": [("like", "%Avalon%")], "tier": [("=", 1)]}


# ---------------------------------------------------------------------------------------------------- amounts
def test_a_column_must_be_selected_by_the_amount_the_question_writes(qc, val):
    q = "Zenith Care premium for a 9-year-old in Avalon with 4 lakh cover?"
    assert run(qc, val, q, P.format(t=1, a="9")) == []
    with pytest.raises(SqlProblem, match="the question states 4 lakh"):
        run(qc, val, q, P.format(t=1, a="9").replace("si_400000", "si_800000"))


def test_an_amount_no_column_prints_is_not_in_the_tables(qc, val):
    with pytest.raises(NotInTables, match="prints no column for 3 lakh"):
        run(qc, val, "Zenith Care premium for a 9-year-old in Avalon with 3 lakh cover?", P.format(t=1, a="9"))
    with pytest.raises(NotInTables, match="checkup limit table prints no column for 6 lakh"):
        run(qc, val, "What is the family floater check-up limit for a 6 lakh Sum Insured?",
            "SELECT limit_si_4_8_lakhs FROM checkup_limit_table WHERE policy_type = 'family_floater'")


def test_the_amount_that_picks_a_row_is_not_used_again_for_a_column(qc, val):
    """A deductible picks the row; the Sum Insured picks the side of the 4 lakh split."""
    q = "Zenith Care, 8 lakh Sum Insured, Rs 10,000 deductible: what discount applies?"
    wrong = "SELECT pct_care_upto_4l FROM deductible_discount_table WHERE deductible_inr = 10000"
    with pytest.raises(SqlProblem, match="Base SI <= 4 lakhs .* but the question states 8 lakh"):
        run(qc, val, q, wrong)
    assert run(qc, val, q, wrong.replace("upto", "over")) == []
    with pytest.raises(SqlProblem, match="deductible_inr = 50000, but the question writes no such amount"):
        run(qc, val, q, wrong.replace("upto", "over").replace("10000", "50000"))


def test_with_no_amount_the_query_reads_every_column_it_could_mean(qc, val):
    q = "Zenith Care premium for a 9-year-old in Avalon?"
    with pytest.raises(SqlProblem, match="states no Sum Insured"):
        run(qc, val, q, P.format(t=1, a="9"))
    assert run(qc, val, q, P.format(t=1, a="9").replace("si_400000", "si_200000, si_400000, si_800000")) == []
    rng = "Zenith Care premiums for a 9-year-old in Avalon from 4 lakh upwards?"
    assert run(qc, val, rng, P.format(t=1, a="9").replace("si_400000", "si_400000, si_800000")) == []


# ---------------------------------------------------------------------------------------------------- plans
def test_a_plan_must_be_the_one_the_question_names(qc, val):
    q = "Zenith Care Plus premium for a 9-year-old in Avalon with 4 lakh cover?"
    with pytest.raises(SqlProblem, match="the question names 'Zenith Care Plus'"):
        run(qc, val, q, P.format(t=1, a="9"))
    assert run(qc, val, q, P.format(t=1, a="9").replace("'Zenith Care'", "'Zenith Care Plus'")) == []
    assert qc.named_plans("the Care Plus plan for my son") == ["Zenith Care Plus"]          # a short form


def test_a_plan_the_table_does_not_hold_is_not_in_the_tables(qc, val):
    with pytest.raises(NotInTables, match="premium table has no plan 'Zenith Global'"):
        run(qc, val, "Zenith Global premium for a 9-year-old in Avalon with 4 lakh cover?", P.format(t=1, a="9"))
    with pytest.raises(SqlProblem, match="which the question does not name"):        # an unknown name: the model decides
        run(qc, val, "Zenith Restore premium for a 9-year-old in Avalon with 4 lakh cover?", P.format(t=1, a="9"))
    with pytest.raises(SqlProblem, match="the question states 'Zenith Global'"):
        run(qc, val, "What discount does a Rs 50,000 deductible give on Zenith Global?",
            "SELECT pct_care_upto_4l FROM deductible_discount_table WHERE deductible_inr = 50000")
    assert run(qc, val, "What discount does a Rs 50,000 deductible give on Zenith Global?",
               "SELECT pct_global FROM deductible_discount_table WHERE deductible_inr = 50000") == []


# ---------------------------------------------------------------------------------------------------- ages
def test_an_age_must_be_a_number_the_question_writes(qc, val):
    q = "Zenith Care premium for my 13-year-old in Avalon with 4 lakh cover?"
    assert run(qc, val, q, P.format(t=1, a="13*")) == []                                  # the printed row, with its mark
    with pytest.raises(SqlProblem, match="the question's age reads the printed row '13\\*'"):
        run(qc, val, q, P.format(t=1, a="11"))
    assert run(qc, val, q.replace("13-year-old", "grandmother, aged 20,"), P.format(t=1, a=">=14*")) == []
    assert run(qc, val, q.replace("13-year-old", "6-month-old"), P.format(t=1, a="0")) == []
    with pytest.raises(SqlProblem, match="writes no age"):
        run(qc, val, "Zenith Care premium for my father in Avalon with 4 lakh cover?", P.format(t=1, a="9"))


# ---------------------------------------------------------------------------------------------------- places
def test_a_listed_place_or_a_stated_tier_gets_its_tier(qc, val):
    assert run(qc, val, "Zenith Care premium for a 9-year-old in Port Avalon with 4 lakh cover?", P.format(t=1, a="9")) == []
    with pytest.raises(SqlProblem, match="names Port Avalon \\(Tier 1\\), but the query reads only Tier 2"):
        run(qc, val, "Zenith Care premium for a 9-year-old in Port Avalon with 4 lakh cover?", P.format(t=2, a="9"))
    assert run(qc, val, "Zenith Care premium for a 9-year-old in a Tier 2 town with 4 lakh cover?",
               P.format(t=2, a="9")) == []
    with pytest.raises(SqlProblem, match="states Tier 2, but the query reads Tier 1"):
        run(qc, val, "Zenith Care premium for a 9-year-old in a Tier 2 town with 4 lakh cover?", P.format(t=1, a="9"))


def test_any_other_place_keeps_the_rows_and_a_note_says_which_tier_was_used(qc, val):
    q = "Zenith Care premium for a 9-year-old in Delta Village with 4 lakh cover?"
    assert run(qc, val, q, P.format(t=2, a="9")) == [
        "The question's place is not one the policy lists for Tier 1, so Tier 2 (Rest of Country) premiums were used."]
    note = run(qc, val, q, P.format(t=1, a="9").replace("tier = 1", "tier_cities LIKE '%Avalon%'"))
    assert note == ["Tier 1 premiums were used for a place the policy does not list by name: this assumes it lies in one "
                    f"of the places of Tier 1 ({T1})."]
    both = "Zenith Care, 9-year-old, 4 lakh: how much more in Port Avalon than in Delta Village?"
    sql = ("SELECT (SELECT si_400000 FROM premium_table WHERE plan = 'Zenith Care' AND tier = 1 AND age = '9') - "
           "(SELECT si_400000 FROM premium_table WHERE plan = 'Zenith Care' AND tier = 2 AND age = '9') AS d")
    assert run(qc, val, both, sql) == [
        "The question's place is not one the policy lists for Tier 1, so Tier 2 (Rest of Country) premiums were used."]


def test_other_listed_values_in_the_questions_words_or_noted(qc, val):
    sql = "SELECT limit_si_4_8_lakhs FROM checkup_limit_table WHERE policy_type = 'individual'"
    assert run(qc, val, "What is the check-up limit for an individual policy with 8 lakh cover?", sql) == []
    with pytest.raises(SqlProblem, match="the question says 'family floater'"):
        run(qc, val, "What is the check-up limit for a family floater with 8 lakh cover?", sql)
    assert run(qc, val, "What is the check-up limit for one person with 8 lakh cover?", sql) == [
        "The checkup limit table was read for policy type: individual, which the question states in other words."]


# ---------------------------------------------------------------------------------------------------- in the step
class RowsDb(TierDb):
    def query(self, sql, params=None):
        return [("plan",)] if sql.startswith("EXPLAIN ") else super().query(sql, params)

    def query_columns(self, sql, params=None):
        return ["v"], [(123,)]


def step(*replies, **cfg):
    return SqlStep(SqlConfig(enabled=True, **cfg), ScriptedAnswerer(list(replies)), db=RowsDb(), card=card())


Q = "Zenith Care, 8 lakh Sum Insured, Rs 10,000 deductible: what discount applies?"
WRONG = "SELECT pct_care_upto_4l FROM deductible_discount_table WHERE deductible_inr = 10000"


def test_a_query_that_disagrees_with_the_question_is_sent_back_with_the_reason(monkeypatch):
    s = step({"action": "query", "sql": WRONG}, {"action": "query", "sql": WRONG.replace("upto", "over")})
    monkeypatch.setattr(s.gate, "check", lambda q: (["deductible"], None))
    out = s.run(Q)
    assert out.status == "answered" and len(out.attempts) == 2
    assert "but the question states 8 lakh" in out.attempts[0]["problem"] and "8 lakh" in s.answerer.prompts[1]
    assert out.attempts[0]["question_check"] == out.attempts[0]["problem"] and "question_check" not in out.attempts[1]
    off = step({"action": "query", "sql": WRONG}, check_question=False)
    monkeypatch.setattr(off.gate, "check", lambda q: (["deductible"], None))
    assert off.run(Q).status == "answered"                                           # the switch turns the check off


def test_not_in_the_tables_is_decided_by_code_and_notes_reach_the_block_and_the_record(monkeypatch):
    s = step({"action": "query", "sql": P.format(t=1, a="9")})
    monkeypatch.setattr(s.gate, "check", lambda q: (["premium"], None))
    out = s.run("Zenith Care premium for a 9-year-old in Avalon with 3 lakh cover?")
    assert out.status == "not_in_tables" and "prints no column for 3 lakh" in out.reason and len(out.attempts) == 1
    s = step({"action": "query", "sql": P.format(t=2, a="9")})
    monkeypatch.setattr(s.gate, "check", lambda q: (["premium"], None))
    out = s.run("Zenith Care premium for a 9-year-old in Delta Village with 4 lakh cover?")
    note = "The question's place is not one the policy lists for Tier 1, so Tier 2 (Rest of Country) premiums were used."
    assert out.status == "answered" and out.notes == [note]
    assert f"Note: {note}" in out.block() and out.to_dict()["notes"] == [note]
    from policy_rag.pipeline import RAGPipeline
    assert RAGPipeline.with_table_notes("The premium is Rs 123.", out) == f"The premium is Rs 123.\n\n{note}"
    assert RAGPipeline.with_table_notes(f"Rs 123. {note}", out) == f"Rs 123. {note}"     # not repeated
    assert SqlConfig().check_question is True
