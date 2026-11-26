"""The reading check (sql_reading.py, v3.6): the model states which words of the question each value comes from; code
verifies the reading. The card is invented (test_sql_checks.card). The first group are question patterns the v3.5
check refused although the query was right (a misspelt plan, number words, a place mentioned but not lived in, a range
of ages next to 'Tier 1', a budget); the second group are mistakes the v3.5 check caught and this one must still catch
(a column on the wrong side of an amount split, another plan's column, one plan column when no plan is named, an
amount no column prints, an amount the question does not state)."""
import pytest

from policy_rag.config import SqlConfig
from policy_rag.sql_checks import NotInTables, QuestionCheck
from policy_rag.sql_reading import (ReadingCheck, ages_read, amounts_read, digits_for_words, names_value, parse_reading)
from policy_rag.sql_step import SchemaCard, SqlProblem, SqlStep, SqlValidator

from test_sql_checks import T1, TierDb, card
from test_sql_step import ScriptedAnswerer

pytest.importorskip("sqlglot")
pytest.importorskip("rapidfuzz")
pytest.importorskip("text_to_num")


@pytest.fixture(scope="module")
def rc():
    return ReadingCheck(card(), TierDb())


@pytest.fixture(scope="module")
def qc():
    return QuestionCheck(card(), TierDb())


@pytest.fixture(scope="module")
def val():
    c = card()
    return SqlValidator(list(c.tables), 50, text_columns=c.text_columns())


def R(column, value, words):
    return {"column": column, "value": value, "words": words}


def run(rc, val, question, sql, reading):
    return rc.check_reading(question, val.check(sql), parse_reading(reading))


P = "SELECT si_400000 FROM premium_table WHERE plan = 'Zenith Care' AND tier = {t} AND age = '{a}'"


# ---------------------------------------------------------------------------------------------------- reading words
def test_numbers_are_read_from_digits_and_words():
    assert digits_for_words("a thirty-three-year-old") == "a 33 year old"
    assert ages_read("thirty-three-year-old") == ({33}, False)
    assert ages_read("9 yr old with 4 lakh cover") == ({9}, False)           # the amount is not an age
    assert ages_read("30 months old") == ({2, 3}, True)                       # completed years or the nearest year
    assert ages_read("umur sembilan") == (set(), False)                       # not read: no evidence either way
    assert amounts_read("four lakh") == [400000] and amounts_read("Rs 50,000 deductible") == [50000]
    assert amounts_read("two crore") == [20000000] and amounts_read("4 lac") == [400000]


def test_names_are_matched_against_the_listed_names_only():
    plans = {"Zenith Care": ["Zenith Care"], "Zenith Care Plus": ["Zenith Care Plus", "care plus"],
             "Zenith Global": ["Zenith Global", "global"]}
    assert names_value("zenit car", plans)[:2] == ("named", "Zenith Care")
    assert names_value("the Care Plus plan", plans)[:2] == ("named", "Zenith Care Plus")
    assert names_value("zenith", plans)[0] in ("ambiguous", "none")          # names no one plan
    assert names_value("a 9 year old", plans)[0] == "none"


@pytest.mark.parametrize("name", ["Zenith Care", "Zenith Care Plus", "Zenith Global", "Avalon", "Port Avalon",
                                  "Beta City", "Gamma Town", "North Region"])
def test_misspellings_generated_from_the_listed_names_are_matched(rc, name):
    """Variants made from the tables' own names (a letter dropped, two letters swapped, the last letter dropped),
    never from a test question."""
    f = rc.tables["premium_table"]
    names = rc._name_forms(rc.plans, "plan") if name.startswith("Zenith") else {p: [p] for p in f.places}
    w = name.lower()
    variants = {w[:i] + w[i + 1:] for i in range(1, len(w)) if w[i] != " " and w[i - 1] != " "}
    variants |= {w[:i] + w[i + 1] + w[i] + w[i + 2:] for i in range(1, len(w) - 1) if " " not in w[i:i + 2]}
    variants.add(" ".join(x[:-1] if len(x) > 4 else x for x in w.split()))
    wrong = {v: names_value(v, names)[:2] for v in variants if names_value(v, names)[:2] != ("named", name)}
    assert not wrong, wrong


# ---------------------------------------------------------------------------------------------------- refused before, right now
def test_a_misspelt_plan_in_shorthand(rc, qc, val):
    q = "premum for zenit car 9 yr old avalon 4 lac??"
    reading = [R("plan", "Zenith Care", "zenit car"), R("age", "9", "9 yr old"), R("tier", 1, "avalon"),
               R("si_400000", 400000, "4 lac")]
    assert run(rc, val, q, P.format(t=1, a="9"), reading) == []
    with pytest.raises(SqlProblem, match="which the question does not name"):
        qc.check(q, val.check(P.format(t=1, a="9")))                           # the v3.5 check refused it


def test_an_age_and_an_amount_in_words(rc, qc, val):
    q = "What is the yearly premium for a nine-year-old in Gamma Town on Zenith Care with four lakh cover?"
    reading = [R("plan", "Zenith Care", "Zenith Care"), R("age", "9", "nine-year-old"), R("tier", 1, "Gamma Town"),
               R("si_400000", 400000, "four lakh")]
    assert run(rc, val, q, P.format(t=1, a="9"), reading) == []
    with pytest.raises(SqlProblem, match="writes no age"):
        qc.check(q, val.check(P.format(t=1, a="9")))


def test_a_place_mentioned_is_not_where_the_person_lives(rc, qc, val):
    q = "I live in Delta Village, right next to Avalon. What is Zenith Care with 4 lakh cover for a 9-year-old?"
    reading = [R("plan", "Zenith Care", "Zenith Care"), R("age", "9", "9-year-old"), R("tier", 2, "Delta Village"),
               R("si_400000", 400000, "4 lakh")]
    notes = run(rc, val, q, P.format(t=2, a="9"), reading)
    assert notes == ["Delta Village is not one of the places the policy lists for Tier 1, so Tier 2 (Rest of Country) "
                     "premiums were used."]
    with pytest.raises(SqlProblem, match="names Avalon"):
        qc.check(q, val.check(P.format(t=2, a="9")))
    with pytest.raises(SqlProblem, match="name Avalon .Tier 1., but the query reads Tier 2"):   # the wrong words
        run(rc, val, q, P.format(t=2, a="9"), reading[:2] + [R("tier", 2, "Avalon")] + reading[3:])


def test_a_range_of_ages_next_to_a_tier_number(rc, qc, val):
    q = "List the Zenith Care premiums for a 4 lakh Sum Insured in Tier 1 for every age from 6 to 9."
    sql = "SELECT age, si_400000 FROM premium_table WHERE plan = 'Zenith Care' AND tier = 1 AND age IN ('6', '7', '8', '9')"
    reading = [R("plan", "Zenith Care", "Zenith Care"), R("age", ["6", "7", "8", "9"], "every age from 6 to 9"),
               R("tier", 1, "Tier 1"), R("si_400000", 400000, "4 lakh")]
    assert run(rc, val, q, sql, reading) == []
    with pytest.raises(SqlProblem, match="the question's age reads"):
        qc.check(q, val.check(sql))
    by_number = sql.replace("age IN ('6', '7', '8', '9')", "age_from BETWEEN 6 AND 9")
    assert run(rc, val, q, by_number, reading) == []                           # migration 004's numbers
    with pytest.raises(SqlProblem, match="compares the age with 11"):
        run(rc, val, q, by_number.replace("AND 9", "AND 11"), reading)
    with pytest.raises(SqlProblem, match="reads age = '10'"):
        run(rc, val, q, sql.replace("'9')", "'9', '10')"), reading)


def test_a_budget_chooses_no_column(rc, qc, val):
    q = "I'm 9 and live in Avalon. With a budget of Rs 7,000 a year, what is the highest Zenith Care Sum Insured I can buy?"
    sql = ("SELECT si_200000, si_400000, si_800000 FROM premium_table WHERE plan = 'Zenith Care' AND tier = 1 "
           "AND age = '9'")
    reading = [R("plan", "Zenith Care", "Zenith Care"), R("age", "9", "I'm 9"), R("tier", 1, "Avalon")]
    assert run(rc, val, q, sql, reading) == []
    with pytest.raises(NotInTables, match="prints no column for Rs 7,000"):
        qc.check(q, val.check(sql))


# ---------------------------------------------------------------------------------------------------- still caught
def test_a_column_on_the_wrong_side_of_an_amount_split(rc, val):
    q = "Zenith Care, 8 lakh Sum Insured, Rs 10,000 deductible: what discount applies?"
    sql = "SELECT pct_care_upto_4l FROM deductible_discount_table WHERE deductible_inr = 10000"
    reading = [R("deductible_inr", 10000, "Rs 10,000 deductible"), R("pct_care_upto_4l", None, "8 lakh"),
               R("pct_care_upto_4l", None, "Zenith Care")]
    with pytest.raises(SqlProblem, match="Base SI <= 4 lakhs.*words read give 8 lakh"):
        run(rc, val, q, sql, reading)
    fixed = [R("deductible_inr", 10000, "Rs 10,000 deductible"), R("pct_care_over_4l", None, "8 lakh"),
             R("pct_care_over_4l", None, "Zenith Care")]
    assert run(rc, val, q, sql.replace("upto", "over"), fixed) == []


def test_another_plans_column(rc, val):
    q = "We have Zenith Care with 2 lakh cover and want a Rs 50,000 deductible: what discount do we get?"
    sql = "SELECT pct_global FROM deductible_discount_table WHERE deductible_inr = 50000"
    reading = [R("deductible_inr", 50000, "Rs 50,000 deductible"), R("pct_global", None, "Zenith Care")]
    with pytest.raises(SqlProblem, match="Zenith Global.*words read give 'Zenith Care'"):
        run(rc, val, q, sql, reading)


def test_one_plan_column_when_no_plan_is_named(rc, val):
    q = "Which plan gets the smallest discount for a Rs 50,000 deductible?"
    sql = "SELECT pct_global FROM deductible_discount_table WHERE deductible_inr = 50000"
    with pytest.raises(SqlProblem, match="no words of the question for the plan"):
        run(rc, val, q, sql, [R("deductible_inr", 50000, "Rs 50,000 deductible")])
    every = sql.replace("pct_global", "pct_care_upto_4l, pct_care_over_4l, pct_global")
    assert run(rc, val, q, every, [R("deductible_inr", 50000, "Rs 50,000 deductible")]) == []


def test_an_amount_no_column_prints(rc, val):
    q = "Zenith Care premium for a 9-year-old in Avalon with 3 lakh cover?"
    reading = [R("plan", "Zenith Care", "Zenith Care"), R("age", "9", "9-year-old"), R("tier", 1, "Avalon"),
               R("si_400000", 400000, "3 lakh")]
    with pytest.raises(NotInTables, match="prints no column for 3 lakh"):
        run(rc, val, q, P.format(t=1, a="9"), reading)


def test_an_amount_the_question_does_not_state(rc, val):
    q = "If I opt for the deductible on Zenith Care with 4 lakh Sum Insured, what discount do I get?"
    sql = "SELECT pct_care_upto_4l FROM deductible_discount_table WHERE deductible_inr = 50000"
    reading = [R("pct_care_upto_4l", None, "Zenith Care"), R("pct_care_upto_4l", None, "4 lakh")]
    with pytest.raises(SqlProblem, match="picks deductible_inr = 50000, but no words"):
        run(rc, val, q, sql, reading)
    with pytest.raises(SqlProblem, match="picks deductible_inr = 50000, but no words"):    # words of another amount
        run(rc, val, q, sql, reading + [R("deductible_inr", 50000, "4 lakh")])


def test_a_sum_insured_band_the_question_does_not_state(rc, val):
    q = "What discount does a Rs 50,000 deductible give on Zenith Care?"
    sql = "SELECT pct_care_upto_4l FROM deductible_discount_table WHERE deductible_inr = 50000"
    reading = [R("deductible_inr", 50000, "Rs 50,000 deductible"), R("pct_care_upto_4l", None, "Zenith Care")]
    with pytest.raises(SqlProblem, match="no words of the question for the Sum Insured"):
        run(rc, val, q, sql, reading)
    both = sql.replace("pct_care_upto_4l", "pct_care_upto_4l, pct_care_over_4l")
    assert run(rc, val, q, both, reading) == []                                # both readings: listed, not asked


# ---------------------------------------------------------------------------------------------------- the reading itself
def test_words_the_question_does_not_write_are_refused(rc, val):
    q = "Zenith Care premium for a 9-year-old in Avalon with 4 lakh cover?"
    reading = [R("plan", "Zenith Care", "Zenith Care"), R("age", "9", "aged nine"), R("tier", 1, "Avalon"),
               R("si_400000", 400000, "4 lakh")]
    with pytest.raises(SqlProblem, match="does not write them"):
        run(rc, val, q, P.format(t=1, a="9"), reading)


def test_words_that_name_another_plan(rc, val):
    q = "Zenith Care premium for a 9-year-old in Avalon with 4 lakh cover?"
    reading = [R("plan", "Zenith Care Plus", "Zenith Care"), R("age", "9", "9-year-old"), R("tier", 1, "Avalon"),
               R("si_400000", 400000, "4 lakh")]
    with pytest.raises(SqlProblem, match="reads plan = 'Zenith Care Plus', but the words read for it name 'Zenith Care'"):
        run(rc, val, q, P.format(t=1, a="9").replace("'Zenith Care'", "'Zenith Care Plus'"), reading)


def test_a_value_the_query_uses_needs_words(rc, val):
    q = "Zenith Care premium for a 9-year-old in Avalon with 4 lakh cover?"
    reading = [R("plan", "Zenith Care", "Zenith Care"), R("tier", 1, "Avalon"), R("si_400000", 400000, "4 lakh")]
    with pytest.raises(SqlProblem, match="chooses rows by age, but the reading gives no words"):
        run(rc, val, q, P.format(t=1, a="9"), reading)
    with pytest.raises(SqlProblem, match="needs \"column\", \"value\" and \"words\""):
        parse_reading([{"column": "age", "value": "9"}])


def test_words_code_cannot_read_are_noted_not_refused(rc, val):
    q = "Zenith Care premium, umur sembilan, Avalon, 4 lakh?"
    reading = [R("plan", "Zenith Care", "Zenith Care"), R("age", "9", "umur sembilan"), R("tier", 1, "Avalon"),
               R("si_400000", 400000, "4 lakh")]
    assert run(rc, val, q, P.format(t=1, a="9"), reading) == [
        "The age was read from 'umur sembilan' as 9 (not checked by code)."]


def test_an_age_in_months(rc, val):
    q = "My son is 30 months old and we live in Avalon. What is Zenith Care with 4 lakh cover for him?"
    reading = [R("plan", "Zenith Care", "Zenith Care"), R("age", "2", "30 months old"), R("tier", 1, "Avalon"),
               R("si_400000", 400000, "4 lakh")]
    notes = run(rc, val, q, P.format(t=1, a="2"), reading)
    assert notes and "does not say whether age counts completed years" in notes[0]
    assert run(rc, val, q, P.format(t=1, a="3"), reading)                      # the nearest year is accepted too
    with pytest.raises(SqlProblem, match="reads age = '5'"):
        run(rc, val, q, P.format(t=1, a="5"), reading)


def test_an_age_the_tables_do_not_print(rc, val):
    q = "Zenith Care premium for a 9-year-old in Avalon with 4 lakh cover?"
    reading = [R("plan", "Zenith Care", "Zenith Care"), R("age", "9", "9-year-old"), R("tier", 1, "Avalon"),
               R("si_400000", 400000, "4 lakh")]
    assert run(rc, val, q, P.format(t=1, a="9"), reading) == []
    old = [R("plan", "Zenith Care", "Zenith Care"), R("age", ">=14*", "aged 20"), R("tier", 1, "Avalon"),
           R("si_400000", 400000, "4 lakh")]
    assert run(rc, val, "Zenith Care premium aged 20 in Avalon with 4 lakh cover?", P.format(t=1, a=">=14*"), old) == []


# ---------------------------------------------------------------------------------------------------- in the step
class FakeDb:
    def __init__(self, rows):
        self.rows, self.ran = rows, []

    def query(self, sql, params=None):
        if sql.startswith("SELECT DISTINCT tier, tier_cities"):
            return [(1, T1), (2, "Rest of Country")]
        assert sql.startswith("EXPLAIN ")
        return [("plan",)]

    def query_columns(self, sql, params=None):
        self.ran.append(sql)
        return self.rows


def test_the_step_checks_the_reading_and_retries_with_the_reason():
    q = "premium for zenit car 9 yr old avalon 4 lac??"
    good = {"action": "query", "sql": P.format(t=1, a="9"),
            "reading": [R("plan", "Zenith Care", "zenit car"), R("age", "9", "9 yr old"), R("tier", 1, "avalon"),
                        R("si_400000", 400000, "4 lac")]}
    guessed = dict(good, reading=good["reading"][:1] + [R("age", "9", "nine years")] + good["reading"][2:])
    step = SqlStep(SqlConfig(enabled=True, template="sql_v3", reading_check=True), ScriptedAnswerer([guessed, good]),
                   db=FakeDb((["si_400000"], [(5900,)])), card=SchemaCard(card().tables))
    assert step.question_check is None and step.reading_check is not None
    out = step.run(q)
    assert out.status == "answered" and len(out.attempts) == 2, out.attempts
    assert "does not write them" in out.attempts[0]["problem"] and out.attempts[1]["reading"] == good["reading"]
    assert '"reading"' in step.answerer.prompts[0] and "copy the words of the QUESTION" in step.answerer.prompts[0]
