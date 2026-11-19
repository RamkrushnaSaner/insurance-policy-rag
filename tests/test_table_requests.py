"""Table requests (sql.mode: request, v3.5 Critical 3) without a model: the request is scripted, and the built queries
run on an in-memory SQLite copy of an invented card. Plans, places, ages and amounts are invented (no policy text)."""
import json
import sqlite3

import pytest

from policy_rag.config import ConfigError, SqlConfig, load_config
from policy_rag.prompts import build_request_prompt
from policy_rag.sql_step import SchemaCard, SqlStep
from policy_rag.table_requests import (NotInTables, RequestResolver, age_in, amounts_in, ask_text, parse_request,
                                       parse_value_heading, parse_yes_no, range_of, table_models, tables_text)

pytest.importorskip("sqlglot")

T1 = "Avalon, North Region, Port Avalon and Beta City & Gamma Town"
T2 = "Rest of Country"


# ---------------------------------------------------------------------------------------------------- an invented card
def col(name, dtype, comment="", values=None):
    return {"name": name, "type": dtype, "comment": comment, "values": values}


AGES = [str(a) for a in range(0, 12)] + ["12*", "13*", ">=14*"]
SI = {"si_200000": 2, "si_400000": 4, "si_800000": 8}


def card_tables():
    return {
        "premium_table": {
            "description": "Annexure Z: annual premium (Rs.) by age and Sum Insured.", "rows": 2 * 2 * len(AGES),
            "columns": [col("plan", "text", "Plan named in the printed title", ["Zenith Care", "Zenith Care Plus"]),
                        col("tier", "smallint", "Premium tier", [1, 2]),
                        col("tier_cities", "text", "Places of the tier", [T1, T2]),
                        col("age", "text", "Printed heading: Age (as printed)", AGES)]
                       + [col(c, "integer", f"Printed heading: Sum Insured {n * 100000:,} (premium in Rs.)")
                          for c, n in SI.items()]
                       + [col("page", "integer", "Page of the printed row")],
            "printed": [{"printed_table": "p40-t1", "title": "Zenith Care Premium", "section_id": "Annexure-Z",
                         "pages": [40], "introduced_by": "", "footnotes": ["*Renewal only (invented)"],
                         "linked_text": []}],
            "examples": []},
        "checkup_limit_table": {
            "description": "Section 5: check-up limit (Rs.) by Sum Insured.", "rows": 2,
            "columns": [col("policy_type", "text", "Which printed table", ["individual", "family_floater"]),
                        col("applies", "text", "The printed sentence above the table",
                            ["For each person under an individual policy, every policy year, as printed."] * 1),
                        col("limit_si_2_lakhs", "integer", "Printed heading: 2 Lakhs (Limit of Cover in Rs.)"),
                        col("limit_si_4_8_lakhs", "integer", "Printed heading: 4 & 8 Lakhs (Limit of Cover in Rs.)"),
                        col("page", "integer", "Page of the printed table")],
            "printed": [{"printed_table": "p5-t1", "title": "", "section_id": "5", "pages": [5], "introduced_by": "",
                         "footnotes": [], "linked_text": []}],
            "examples": []},
        "deductible_discount_table": {
            "description": "Section 9: discount (%) on the premium with a deductible.", "rows": 2,
            "columns": [col("deductible_inr", "integer", "Printed heading: Deductible amount (INR)", [10000, 50000]),
                        col("pct_care_upto_4l", "numeric(5,2)",
                            "Printed heading: Zenith Care & Zenith Care Plus / Base SI <= 4 lakhs (discount %)"),
                        col("pct_care_over_4l", "numeric(5,2)",
                            "Printed heading: Zenith Care & Zenith Care Plus / Base SI > 4 lakhs (discount %)"),
                        col("pct_global", "numeric(5,2)", "Printed heading: Zenith Global (discount %)"),
                        col("page", "integer", "Page of the printed table")],
            "printed": [{"printed_table": "p9-t1", "title": "", "section_id": "9", "pages": [9], "introduced_by": "",
                         "footnotes": [], "linked_text": []}],
            "examples": []},
    }


def premium(plan, tier, age, si):
    """An invented premium: grows with plan, tier, age and Sum Insured."""
    a = int("".join(ch for ch in age if ch.isdigit()))
    return 1000 * si + 100 * a + (500 if plan.endswith("Plus") else 0) + (300 if tier == 1 else 0)


class SqliteDb:
    """The invented card's tables in SQLite, under schema policy_sql (the built queries run as written)."""

    def __init__(self):
        self.con = sqlite3.connect(":memory:")
        self.con.execute("ATTACH ':memory:' AS policy_sql")
        self.ran = []
        c = self.con
        c.execute('CREATE TABLE policy_sql.premium_table (plan, tier, tier_cities, age, si_200000, si_400000, si_800000, '
                  'page)')
        for plan in ("Zenith Care", "Zenith Care Plus"):
            for tier, places in ((1, T1), (2, T2)):
                for age in AGES:
                    c.execute("INSERT INTO policy_sql.premium_table VALUES (?,?,?,?,?,?,?,?)",
                              (plan, tier, places, age, *[premium(plan, tier, age, n) for n in SI.values()], 40))
        c.execute("CREATE TABLE policy_sql.checkup_limit_table (policy_type, applies, limit_si_2_lakhs, "
                  "limit_si_4_8_lakhs, page)")
        c.executemany("INSERT INTO policy_sql.checkup_limit_table VALUES (?,?,?,?,?)",
                      [("individual", "x", 700, 900, 5), ("family_floater", "y", 1100, 1500, 5)])
        c.execute("CREATE TABLE policy_sql.deductible_discount_table (deductible_inr, pct_care_upto_4l, "
                  "pct_care_over_4l, pct_global, page)")
        c.executemany("INSERT INTO policy_sql.deductible_discount_table VALUES (?,?,?,?,?)",
                      [(10000, 12.5, 6, 3, 9), (50000, 30, 20, 9, 9)])

    def query(self, sql, params=None):
        return self.con.execute(sql).fetchall()

    def query_columns(self, sql, params=None):
        self.ran.append(sql)
        cur = self.con.execute(sql)
        return [d[0] for d in cur.description], cur.fetchall()


class Script:
    """Replies in order; keeps (label, max_new_tokens, prompt) of every call."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def answer(self, prompt, use_cache=True, label="answer", max_new_tokens=None):
        self.calls.append((label, max_new_tokens, prompt))
        r = self.replies.pop(0)
        return {"key": f"k{len(self.calls)}", "answer": r if isinstance(r, str) else json.dumps(r), "seconds": 0.1}


def step(*replies, **cfg):
    db = SqliteDb()
    s = SqlStep(SqlConfig(enabled=True, mode="request", **cfg), Script(replies), db=db,
                card=SchemaCard(card_tables()))
    s.gate.check = lambda q: (["premium"], None)                 # the gate is tested elsewhere
    return s


def lookup(table="premium_table", **values):
    return {"table": table, "values": {k: {"stated": s, "value": v} for k, (s, v) in values.items()}}


def request(*lookups, asks="premium", combine=None):
    return {"action": "lookup", "asks": asks, "lookups": list(lookups), "combine": combine}


# ---------------------------------------------------------------------------------------------------- reading words
def test_amounts_ranges_and_ages_are_read_by_code():
    assert amounts_in("a 4 lakh Sum Insured") == [400000]
    assert amounts_in("₹8,00,000 cover, Rs 50,000 deductible, 2L, 1.5 lakh, 1 Cr, 25k") == [
        800000, 50000, 200000, 150000, 10000000, 25000]
    assert amounts_in("a 45-year-old") == [] and amounts_in("45", bare_from=1) == [45]
    assert range_of("from 2 lakh to 8 lakh") == (200000, 800000)
    assert range_of("4 lakh upwards") == (400000, None) and range_of("up to 4 lakh") == (None, 400000)
    assert range_of("4 lakh") == (400000, 400000)
    assert [age_in(x) for x in ("a 45-year-old", "aged 93", "6-month-old baby", "18 months", "no age")] == [
        45, 93, 0, 1, None]


def test_value_headings_say_what_selects_each_column():
    c = parse_value_heading("a", "Sum Insured 400,000 (premium in Rs.)")
    assert c.amounts == [400000] and c.plans == [] and c.condition is None
    assert parse_value_heading("b", "4 & 8 Lakhs (Limit of Cover in Rs.)").amounts == [400000, 800000]
    c = parse_value_heading("c", "Zenith Care & Zenith Care Plus / Base SI <= 4 lakhs (discount %)")
    assert c.plans == ["Zenith Care", "Zenith Care Plus"] and c.condition == ("<=", 400000)
    assert c.covers(400000, 400000) and not c.covers(800000, 800000) and c.covers(200000, 800000)
    assert parse_value_heading("d", "Zenith Global (discount %)").plans == ["Zenith Global"]


def test_the_card_gives_each_table_its_fields():
    models = table_models(SchemaCard(card_tables()), SqliteDb())
    p = models["premium_table"]
    assert set(p.fields) == {"plan", "place", "age", "sum_insured"} and p.key_columns == ["plan", "tier", "age"]
    assert p.places == {"Avalon": 1, "North Region": 1, "Port Avalon": 1, "Beta City": 1, "Gamma Town": 1}
    assert p.catch_all == 2
    assert set(models["checkup_limit_table"].fields) == {"policy_type", "sum_insured"}   # long texts are no field
    d = models["deductible_discount_table"]
    assert set(d.fields) == {"deductible", "sum_insured", "plan"} and d.fields["plan"].kind == "plan_column"
    text = tables_text(models, SchemaCard(card_tables()))
    assert "Tier 1: Avalon, North Region, Port Avalon, Beta City, Gamma Town" in text
    assert "printed ages: 0 to 11; 12* to 13*; >=14*" in text and "footnote: *Renewal only (invented)" in text


def test_reply_parsing():
    assert parse_request('<think>x</think>{"action": "text_only", "reason": "a rule"}') == {
        "action": "text_only", "reason": "a rule"}
    r = parse_request(json.dumps(request(lookup(age=("45", 45)), combine="null")))
    assert r["combine"] is None and r["lookups"][0]["table"] == "premium_table"
    for bad in ('{"action": "query", "sql": "SELECT 1"}', '{"action": "lookup", "lookups": []}',
                '{"action": "lookup", "lookups": [{}], "combine": "ratio"}', "no json"):
        with pytest.raises(Exception):
            parse_request(bad)


# ---------------------------------------------------------------------------------------------------- one value
Q = "What is the annual premium for Zenith Care for a 9-year-old in Port Avalon with a 4 lakh Sum Insured?"
FULL = request(lookup(plan=("Zenith Care", "Zenith Care"), age=("9-year-old", 9), place=("Port Avalon", "Port Avalon"),
                      sum_insured=("4 lakh", 400000)), asks="annual premium")


def test_every_value_is_resolved_by_code_and_the_query_holds_only_the_cards_values():
    s = step(FULL)
    out = s.run(Q)
    assert out.status == "answered" and out.columns == ["plan", "tier", "age", "si_400000", "page"]
    assert out.rows == [("Zenith Care", 1, "9", premium("Zenith Care", 1, "9", 4), 40)]
    assert "'Zenith Care'" in out.sql and "\"tier\" = 1" in out.sql and "Port Avalon" not in out.sql
    assert out.notes == ["Port Avalon is listed by the policy in Tier 1, so Tier 1 premiums apply."]
    assert "Note: Port Avalon is listed" in out.block() and out.derived == []
    assert [c[0] for c in s.answerer.calls] == ["sql"] and out.to_dict()["request"]["asks"] == "annual premium"


def test_a_place_the_policy_lists_is_resolved_by_code_whatever_the_model_says():
    wrong = request(lookup(plan=("Zenith Care", "Zenith Care"), age=("9-year-old", 9), place=("Port Avalon", "none"),
                           sum_insured=("4 lakh", 400000)), asks="annual premium")
    assert step(wrong).run(Q).rows[0][1] == 1


@pytest.mark.parametrize("words, choice, tier, note", [
    ("Delta Village", "none", 2, "Delta Village is not one of the places the policy lists for Tier 1, so Tier 2 (Rest of "
                                 "Country) premiums apply."),
    ("Elm Hamlet", "North Region", 1, "Elm Hamlet is taken to be in North Region, which the policy lists in Tier 1, "
                                      "so Tier 1 premiums apply."),
    ("Old Avalon", "none", 1, "Old Avalon is listed by the policy in Tier 1, so Tier 1 premiums apply."),  # code: names it
    ("a Tier 2 town", "none", 2, None),
])
def test_other_places_take_the_models_choice_from_the_list_and_the_answer_says_so(words, choice, tier, note):
    q = f"Zenith Care premium for a 9-year-old in {words} with 4 lakh cover?"
    r = request(lookup(plan=("Zenith Care", "Zenith Care"), age=("9-year-old", 9), place=(words, choice),
                       sum_insured=("4 lakh", 400000)))
    out = step(r).run(q)
    assert out.status == "answered" and out.rows[0][1] == tier and out.notes == ([note] if note else [])


def test_a_place_outside_the_list_must_be_corrected():
    q = "Zenith Care premium for a 9-year-old in Delta Village with 4 lakh cover?"
    bad = request(lookup(plan=("Zenith Care", "Zenith Care"), age=("9-year-old", 9), place=("Delta Village", "Delta"),
                         sum_insured=("4 lakh", 400000)))
    s = step(bad, bad)
    out = s.run(q)
    assert out.status == "failed" and "not a listed place" in out.reason and len(out.attempts) == 2
    assert "not a listed place" in s.answerer.calls[1][2]                         # the reason is shown to the model


def test_quoted_words_must_be_in_the_question_and_one_correction_is_allowed():
    bad = request(lookup(plan=("Zenith Care", "Zenith Care"), age=("nine years", 9), place=("Port Avalon", "Port Avalon"),
                         sum_insured=("4 lakh", 400000)), asks="annual premium")
    s = step(bad, FULL)
    out = s.run(Q)
    assert out.status == "answered" and len(out.attempts) == 2 and "'nine years' is not in the question" in \
        out.attempts[0]["problem"]
    no_asks = dict(FULL, asks="the cost")
    assert step(no_asks, no_asks).run(Q).status == "failed"


@pytest.mark.parametrize("q, values, reason", [
    ("Zenith Care premium for a 9-year-old in Avalon with 3 lakh cover?",
     {"sum_insured": ("3 lakh", 300000)}, "prints no column for a Sum Insured of 3 lakh"),
    ("Zenith Care premium for a 9-year-old in Avalon with 16 lakh cover?",
     {"sum_insured": ("16 lakh", 1600000)}, "prints no column for a Sum Insured of 16 lakh"),
    ("Zenith Global premium for a 9-year-old in Avalon with 4 lakh cover?",
     {"plan": ("Zenith Global", "Zenith Care")}, "has no plan 'Zenith Global'"),          # a name the card knows elsewhere
    ("Zenith Restore premium for a 9-year-old in Avalon with 4 lakh cover?",
     {"plan": ("Zenith Restore", "Zenith Care")}, "has no plan 'Zenith Restore'"),       # shares a word with the plans
])
def test_values_the_tables_do_not_hold_are_not_in_the_tables(q, values, reason):
    base = {"plan": ("Zenith Care", "Zenith Care"), "age": ("9-year-old", 9), "place": ("Avalon", "Avalon"),
            "sum_insured": ("4 lakh", 400000)}
    base.update(values)
    out = step(request(lookup(**base))).run(q)
    assert out.status == "not_in_tables" and reason in out.reason and not out.rows


def test_ages_short_names_and_meanings():
    q = "Premium on Care Plus for my 13-year-old in Avalon, 2 lakh cover?"
    r = request(lookup(plan=("Care Plus", "Zenith Care Plus"), age=("13-year-old", 13), place=("Avalon", "Avalon"),
                       sum_insured=("2 lakh", 200000)), asks="Premium")
    out = step(r).run(q)
    assert out.rows[0][:3] == ("Zenith Care Plus", 1, "13*")                     # a printed age with its mark
    q = "Premium on Zenith Care for my grandmother, aged 20, in Avalon, 2 lakh cover?"
    r = request(lookup(plan=("Zenith Care", "Zenith Care"), age=("aged 20", 20), place=("Avalon", "Avalon"),
                       sum_insured=("2 lakh", 200000)), asks="Premium")
    assert step(r).run(q).rows[0][2] == ">=14*"                                  # an open-ended printed row
    q = "What is the check-up limit for one person with 8 lakh cover?"
    r = request(lookup("checkup_limit_table", policy_type=("one person", "individual"), sum_insured=("8 lakh", 800000)),
                asks="check-up limit")
    out = step(r).run(q)                                                         # a meaning: the model's choice
    assert out.columns == ["policy_type", "limit_si_4_8_lakhs", "page"] and out.rows == [("individual", 900, 5)]


def test_a_deductible_column_is_chosen_by_plan_and_by_the_sum_insureds_printed_condition():
    q = "What discount does a Rs 50,000 deductible give on Zenith Care with 8 lakh cover?"
    r = request(lookup("deductible_discount_table", deductible=("Rs 50,000", 50000), plan=("Zenith Care", "Zenith Care"),
                       sum_insured=("8 lakh", 800000)), asks="discount")
    out = step(r).run(q)
    assert out.columns == ["deductible_inr", "pct_care_over_4l", "page"] and out.rows[0][1] == 20
    q = "What discount does a Rs 50,000 deductible give on Zenith Care?"
    r = request(lookup("deductible_discount_table", deductible=("Rs 50,000", 50000), plan=("Zenith Care", "Zenith Care")),
                asks="discount")
    assert step(r).run(q).columns == ["deductible_inr", "pct_care_upto_4l", "pct_care_over_4l", "page"]
    q = "What discount does a Rs 20,000 deductible give on Zenith Care?"
    r = request(lookup("deductible_discount_table", deductible=("Rs 20,000", 20000), plan=("Zenith Care", "Zenith Care")),
                asks="discount")
    out = step(r).run(q)
    assert out.status == "not_in_tables" and "no deductible of Rs 20,000" in out.reason


def test_a_range_of_sum_insured_reads_the_columns_inside_it():
    q = "Zenith Care premiums for a 9-year-old in Avalon from 4 lakh upwards?"
    r = request(lookup(plan=("Zenith Care", "Zenith Care"), age=("9-year-old", 9), place=("Avalon", "Avalon"),
                       sum_insured=("from 4 lakh upwards", [400000, None])), asks="premiums")
    assert step(r).run(q).columns == ["plan", "tier", "age", "si_400000", "si_800000", "page"]


# ---------------------------------------------------------------------------------------------------- open fields
def test_up_to_eight_rows_for_open_fields_are_returned():
    q = "What is the premium for a 9-year-old in Avalon with 4 lakh cover?"
    r = request(lookup(age=("9-year-old", 9), place=("Avalon", "Avalon"), sum_insured=("4 lakh", 400000)))
    out = step(r).run(q)
    assert out.status == "answered" and [row[0] for row in out.rows] == ["Zenith Care", "Zenith Care Plus"]
    q = "What is the preventive health check-up benefit?"
    out = step(request(lookup("checkup_limit_table"), asks="check-up benefit")).run(q)
    assert out.status == "answered" and len(out.rows) == 2 and len(out.columns) == 4


def test_more_rows_make_the_step_ask_after_the_question_is_confirmed():
    q = "What is the premium for Zenith Care in Avalon with 4 lakh cover?"
    r = request(lookup(plan=("Zenith Care", "Zenith Care"), place=("Avalon", "Avalon"), sum_insured=("4 lakh", 400000)))
    s = step(r, "<think>a premium is asked</think>yes")
    out = s.run(q)
    assert out.status == "ask" and out.clarification == "How old is the person to be insured?"
    assert s.answerer.calls[1][:2] == ("sql", 20) and "Is the user asking for a value from this table?" in \
        s.answerer.calls[1][2] and "annual premium (Rs.) by age" in s.answerer.calls[1][2]
    assert out.attempts[-1]["confirm"] and out.attempts[-1]["yes"] is True
    assert step(r, "maybe").run(q).status == "ask"                             # no clear reply: ask as before
    out = step(r, "No.").run(q)
    assert out.status == "text_only" and "does not ask for a value" in out.reason
    assert step(r, confirm_ask=False).run(q).status == "ask"


def test_the_question_asks_for_the_field_with_most_choices():
    q = "Zenith Care premium with 4 lakh cover?"
    r = request(lookup(plan=("Zenith Care", "Zenith Care"), sum_insured=("4 lakh", 400000)), asks="premium")
    out = step(r, "yes").run(q)                                               # 2 tiers x 15 printed ages
    assert out.status == "ask" and out.clarification == "How old is the person to be insured?"
    assert out.attempts[0]["too_many_rows"] == {"rows": 30, "open": {"place": 2, "age": 15}}
    q = "Zenith Care premium for a 9-year-old with 4 lakh cover?"
    r = request(lookup(plan=("Zenith Care", "Zenith Care"), age=("9-year-old", 9), sum_insured=("4 lakh", 400000)))
    out = step(r, "yes", ask_rows=1).run(q)
    assert out.status == "ask" and out.clarification == "Which city or town do you live in? The premium depends on it."
    models = table_models(SchemaCard(card_tables()), SqliteDb())
    assert ask_text(models["premium_table"], "plan") == "Which plan: Zenith Care or Zenith Care Plus?"
    assert ask_text(models["deductible_discount_table"], "deductible") == \
        "Which deductible amount? The policy prints Rs 10,000, Rs 50,000."
    assert parse_yes_no("<think>no idea</think> Yes.") is True and parse_yes_no("") is None


# ---------------------------------------------------------------------------------------------------- calculations
def two(plan_a, plan_b, combine):
    return request(lookup(plan=(plan_a, plan_a), age=("9-year-old", 9), place=("Avalon", "Avalon"),
                          sum_insured=("4 lakh", 400000)),
                   lookup(plan=(plan_b, plan_b), age=("9-year-old", 9), place=("Avalon", "Avalon"),
                          sum_insured=("4 lakh", 400000)), combine=combine, asks="How much more")


def test_differences_and_sums_are_computed_by_code():
    q = "How much more does Zenith Care Plus cost than Zenith Care for a 9-year-old in Avalon with 4 lakh cover?"
    out = step(two("Zenith Care", "Zenith Care Plus", "difference")).run(q)
    d = out.derived[0]
    assert out.status == "answered" and d["kind"] == "difference" and d["value"] == 500
    assert d["text"].startswith("Difference: plan Zenith Care Plus, tier 1, age 9") and "= Rs 500" in d["text"]
    out = step(two("Zenith Care", "Zenith Care Plus", "sum")).run(q)
    assert out.derived[0]["kind"] == "total" and out.derived[0]["value"] == 2 * premium("Zenith Care", 1, "9", 4) + 500


def test_an_amount_after_a_discount_reads_two_tables():
    q = "What is the Zenith Care premium for a 9-year-old in Avalon with 4 lakh cover after a Rs 10,000 deductible discount?"
    r = request(lookup(plan=("Zenith Care", "Zenith Care"), age=("9-year-old", 9), place=("Avalon", "Avalon"),
                       sum_insured=("4 lakh", 400000)),
                lookup("deductible_discount_table", deductible=("Rs 10,000", 10000), plan=("Zenith Care", "Zenith Care"),
                       sum_insured=("4 lakh", 400000)), combine="after_discount")
    out = step(r).run(q)
    p = premium("Zenith Care", 1, "9", 4)
    assert out.status == "answered" and out.tables == ["premium_table", "deductible_discount_table"]
    assert out.derived[0]["kind"] == "after_percent" and float(out.derived[0]["value"]) == p * 0.875
    assert len(out.rows) == 2 and all(None in row for row in out.rows)
    assert "(empty)" not in out.block()                                       # the other table's cells are not shown


def test_a_combination_needs_one_value_per_lookup():
    q = "How much more does Zenith Care Plus cost than Zenith Care for a 9-year-old in Avalon?"
    r = request(lookup(plan=("Zenith Care", "Zenith Care"), age=("9-year-old", 9), place=("Avalon", "Avalon")),
                lookup(plan=("Zenith Care Plus", "Zenith Care Plus"), age=("9-year-old", 9), place=("Avalon", "Avalon")),
                combine="difference", asks="How much more")
    out = step(r, r).run(q)
    assert out.status == "failed" and "each lookup must give one value" in out.reason


def test_a_column_can_be_listed():
    q = "Which places are Tier 1?"
    r = request({"table": "premium_table", "list": "tier_cities",
                 "values": {"place": {"stated": "Tier 1", "value": "Tier 1"}}}, asks="Which places")
    out = step(r).run(q)
    assert out.status == "answered" and out.rows == [(T1,)] and "SELECT DISTINCT" in out.sql


def test_text_only_and_bad_tables():
    assert step({"action": "text_only", "reason": "a rule"}).run("Is the check-up yearly?").status == "text_only"
    bad = request(lookup("price_table"))
    out = step(bad, bad).run(Q)
    assert out.status == "failed" and "unknown table" in out.reason


# ---------------------------------------------------------------------------------------------------- config, prompt
def test_config_and_prompt():
    cfg = load_config(None, ["sql.mode=request"])
    assert cfg.sql.mode == "request" and cfg.sql.ask_rows == 8 and cfg.sql.confirm_ask is True
    for bad in ("sql.mode=free", "sql.ask_rows=0", "sql.ask_rows=60"):
        with pytest.raises(ConfigError):
            load_config(None, [bad])
    p = build_request_prompt("Q?", "TABLES-TEXT", attempts=[("r", "why")])
    assert "TABLES-TEXT" in p and "QUESTION: Q?" in p and "Problem: why" in p and '"action": "text_only"' in p


def test_resolver_works_alone():
    models = table_models(SchemaCard(card_tables()), SqliteDb())
    res = RequestResolver(models).resolve(FULL, Q)
    assert res[0].filters == {"plan": "Zenith Care", "age": "9", "tier": 1} and res[0].value_columns == ["si_400000"]
    with pytest.raises(NotInTables):
        RequestResolver(models).resolve(request(lookup(sum_insured=("3 lakh", 300000))), "premium with 3 lakh?")
    from policy_rag.table_requests import _age_row
    assert _age_row(["0", "1", "12*"], 12) == "12*" and _age_row(["0", "1", "12*"], 30) is None


# ---------------------------------------------------------------------------------------------------- in the pipeline
def test_the_answer_states_the_tier_code_used(tmp_path):
    from conftest import fake_config
    from policy_rag.pipeline import RAGPipeline, RAGSystem
    from policy_rag.sql_step import SqlOutcome
    q = "Zenith Care premium for a 9-year-old in Delta Village with 4 lakh cover?"
    r = request(lookup(plan=("Zenith Care", "Zenith Care"), age=("9-year-old", 9), place=("Delta Village", "none"),
                       sum_insured=("4 lakh", 400000)))
    rag = RAGSystem.from_config(fake_config(tmp_path, sql={"enabled": True, "mode": "request"}, gate={"min_score": -1e9}),
                                log=lambda *_: None, sql=step(r))
    out = rag.ask(q, show_sources=False)
    note = "Delta Village is not one of the places the policy lists for Tier 1, so Tier 2 (Rest of Country) premiums apply."
    assert out.route == "answer" and out.answer.endswith("\n\n" + note)
    assert out.state["facts"].startswith("[Policy tables") and f"Note: {note}" in out.state["facts"]
    assert RAGPipeline.with_table_notes("x", SqlOutcome(question="q", status="ask", notes=["n"])) == "x"
    assert RAGPipeline.with_table_notes("x", None) == "x"


def test_every_value_code_can_see_in_the_question_must_be_in_the_request():
    """A plan name the card knows, a listed place or an amount with a unit, left out of the request, would read rows
    for a case the question did not ask about: the request goes back to the model with the reason."""
    q = "What is the Zenith Global premium for a 9-year-old in Avalon with 4 lakh cover?"
    left_out = request(lookup(age=("9-year-old", 9), place=("Avalon", "Avalon"), sum_insured=("4 lakh", 400000)))
    stated = request(lookup(plan=("Zenith Global", "Zenith Global"), age=("9-year-old", 9), place=("Avalon", "Avalon"),
                            sum_insured=("4 lakh", 400000)))
    s = step(left_out, stated)
    out = s.run(q)
    assert out.status == "not_in_tables" and "has no plan 'Zenith Global'" in out.reason
    assert "the question states plan 'Zenith Global' in the lookup on premium_table" in out.attempts[0]["problem"]
    q = "Zenith Care premium for a 9-year-old in Port Avalon with 16 lakh cover?"
    r = request(lookup(plan=("Zenith Care", "Zenith Care"), age=("9-year-old", 9)))
    out = step(r, r).run(q)
    assert out.status == "failed" and "the amount 16 lakh" in out.reason and "place 'Port Avalon' in the lookup" in out.reason
    q = "Discount for a Rs 50,000 deductible on Zenith Care Plus with 8 lakh cover?"
    r = request(lookup("deductible_discount_table", deductible=("Rs 50,000", 50000), sum_insured=("8 lakh", 800000)),
                asks="Discount")
    out = step(r, r).run(q)
    assert out.status == "failed" and "plan 'Zenith Care Plus' in the lookup on deductible_discount_table" in out.reason


def test_quoted_amounts_must_be_written_in_the_question_and_names_are_compared_by_kind():
    q = "Zenith Care premium for a 4-year-old in Avalon with 8 lakh cover?"
    made_up = request(lookup(plan=("Zenith Care", "Zenith Care"), age=("4-year-old", 4), place=("Avalon", "Avalon"),
                             sum_insured=("4 lakh", 400000)))            # its words are all in the question, not as one
    out = step(made_up, made_up).run(q)
    assert out.status == "failed" and "'4 lakh' is not in the question" in out.reason
    q = "What is the check-up limit for a Zenith Care individual policy with 8 lakh cover?"
    r = request(lookup("checkup_limit_table", policy_type=("Zenith Care individual policy", "individual"),
                       sum_insured=("8 lakh", 800000)), asks="check-up limit")
    assert step(r).run(q).rows == [("individual", 900, 5)]                # a plan name is no policy type
    q = "Zenith Care premium for a senior citizen in Avalon with 4 lakh cover?"
    guessed = request(lookup(plan=("Zenith Care", "Zenith Care"), age=("senior citizen", 12), place=("Avalon", "Avalon"),
                             sum_insured=("4 lakh", 400000)))
    left_open = request(lookup(plan=("Zenith Care", "Zenith Care"), place=("Avalon", "Avalon"),
                               sum_insured=("4 lakh", 400000)))
    out = step(guessed, left_open, "yes").run(q)
    assert out.status == "ask" and "gives no age in years or months" in out.attempts[0]["problem"]
