"""The SQL step without a database or a model: the gate, the schema card, the reply parser, the SQL checks, the retries
and the result block. The tables, values and replies are invented (no policy text)."""
import json
from decimal import Decimal

import pytest

from policy_rag.config import ConfigError, SqlConfig, load_config
from policy_rag.prompts import build_sql_prompt
from policy_rag.sql_step import (SchemaCard, SqlGate, SqlOutcome, SqlProblem, SqlStep, compact_values, gate, group_inr,
                                 indian_amount, parse_action, table_profiles, with_lakh)

sqlglot = pytest.importorskip("sqlglot")
from policy_rag.sql_step import SqlValidator  # noqa: E402

TABLES = ["premium_table", "checkup_limit_table", "deductible_discount_table"]


# ---------------------------------------------------------------------------------------------------- gate
@pytest.mark.parametrize("q, kinds", [
    ("What is the annual premium for Zenith Care with a 10 lakh Sum Insured for a 30-year-old?", ["premium"]),
    ("How much does the policy cost for a 45 year old in Pune?", ["premium"]),
    ("What is the preventive health check-up limit for a family floater with 10 lakh?", ["checkup"]),
    ("What discount do I get with a Rs 50,000 deductible?", ["deductible"]),
    ("Can I opt for an aggregate deductible on the premium?", ["premium", "deductible"]),
    ("What is the grace period for paying the premium?", []),
    ("Can I pay the premium in monthly instalments?", []),
    ("Is the premium refunded if I cancel the policy?", []),
    ("How much will the policy pay for an ambulance?", []),
    ("Is cataract surgery covered?", []),
    ("Is the cost of an ambulance covered?", []),                             # a money word, but no plan or age
    ("What would the plan cost per year for a 5-year-old?", ["premium"]),       # a money word with an age
    ("How much does the policy pay for an annual health screening?", []),     # other wordings: left to closeness
    ("If I pay the first Rs 50,000 of claims myself, what do I save?", []),
])
def test_gate(q, kinds):
    assert gate(q) == kinds


def test_gate_knows_the_names_in_the_tables():
    q = "How much more does Zenith Care Plus cost than Zenith Care?"
    assert gate(q) == [] and gate(q, names=["Zenith Care", "Zenith Care Plus"]) == ["premium"]
    q = "What does Zenith Care cost per year?"                                 # a money word with a plan name
    assert gate(q) == [] and gate(q, names=["Zenith Care"]) == ["premium"]
    assert SchemaCard(card_tables()).names() == ["Zenith Care", "Zenith Care Plus"]


# ---------------------------------------------------------------------------------------------------- formatting
def test_values_are_shown_short():
    ages = [str(a) for a in range(0, 66)] + [f"{a}*" for a in range(66, 90)] + [">=90*"]
    assert compact_values(sorted(ages)) == "'0', '1', ... '65', '66*', '67*', ... '89*', '>=90*'"   # each a value
    assert compact_values(["Zenith Care", "Zenith Care Plus"]) == "'Zenith Care', 'Zenith Care Plus'"
    assert compact_values([1, 2]) == "1, 2"
    assert compact_values(["10", "12"]) == "'10', '12'"                 # no run: listed as they are
    assert compact_values(["item 1", "item 2", "item 3"]) == "'item 1', 'item 2', 'item 3'"
    assert compact_values([25000, 100000, 150000, 20000000]) == "25000, 100000 (1 lakh), 150000 (1.5 lakh), 20000000 (2 crore)"
    assert compact_values([c * 2 for c in "abcdefghij"], limit=3) == "'aa', 'bb', 'cc', ... (10 values)"
    assert group_inr(1000000) == "10,00,000" and group_inr(16100) == "16,100" and group_inr(999) == "999"
    assert [indian_amount(n) for n in (99999, 100000, 500000, 5000000, 10000000, 25000000)] == [
        "", "1 lakh", "5 lakh", "50 lakh", "1 crore", "2.5 crore"]
    assert with_lakh("Sum Insured 500,000 (premium in Rs.)") == "Sum Insured 500,000 (5 lakh) (premium in Rs.)"
    assert with_lakh("Base SI 5,00,000 or Rs. 1,500 or 20 lakhs") == "Base SI 5,00,000 (5 lakh) or Rs. 1,500 or 20 lakhs"


# ---------------------------------------------------------------------------------------------------- the card
def card_tables():
    """An invented card, as SchemaCard.load would read it."""
    def col(name, dtype, comment="", values=None):
        return {"name": name, "type": dtype, "comment": comment, "values": values}
    return {
        "premium_table": {
            "description": "Annexure Z: annual premium (Rs.) by age and Sum Insured.", "rows": 8,
            "columns": [col("plan", "text", "Plan named in the printed title", ["Zenith Care", "Zenith Care Plus"]),
                        col("tier", "smallint", "Premium tier", [1, 2]),
                        col("age", "text", "Printed heading: Age", ["30", "31", "66*", ">=90*"]),
                        col("si_500000", "integer", "Printed heading: Sum Insured 500,000 (premium in Rs.)"),
                        col("si_1000000", "integer", "Printed heading: Sum Insured 1,000,000 (premium in Rs.)"),
                        col("page", "integer", "Page of the printed row")],
            "printed": [{"printed_table": "p40-t1", "title": "Zenith Care Premium - Tier 1", "section_id": "Annexure-Z",
                         "pages": [40, 41], "introduced_by": "Refer Annexure Z for the premium.",
                         "footnotes": ["*Renewal only (invented)"],
                         "linked_text": [{"relation": "read_with", "label": "Section 2 - Sum Insured options",
                                          "text": "Sum Insured options are 5 and 10 lakh."}]}],
            "examples": [{"plan": "Zenith Care", "tier": 1, "age": "30", "si_500000": 7100, "si_1000000": 9100,
                          "page": 40}]},
        "deductible_discount_table": {
            "description": "Section 9: discount (%) with a deductible.", "rows": 2,
            "columns": [col("deductible_inr", "integer", "Printed heading: Deductible amount (INR)"),
                        col("pct_zenith", "numeric(5,2)", "Printed heading: Zenith Care (discount %)"),
                        col("page", "integer", "Page of the printed table")],
            "printed": [{"printed_table": "p9-t1", "title": "", "section_id": "9", "pages": [9], "introduced_by": "",
                         "footnotes": [], "linked_text": []}],
            "examples": []},
    }


def test_card_text_shows_tables_columns_values_and_notes():
    text = SchemaCard(card_tables()).text
    for part in ["policy_sql.premium_table (8 rows): Annexure Z: annual premium",
                 "Printed table: Zenith Care Premium - Tier 1 (Annexure Z, pages 40-41)",
                 "Introduced by: Refer Annexure Z for the premium.", "Footnote: *Renewal only (invented)",
                 "age (text): Printed heading: Age. Values: '30', '31', '66*', '>=90*'",
                 "tier (smallint): Premium tier. Values: 1, 2",
                 "si_1000000 (integer): Printed heading: Sum Insured 1,000,000 (10 lakh) (premium in Rs.)",
                 "plan='Zenith Care', tier=1, age='30', si_500000=7100",
                 "policy_sql.deductible_discount_table (2 rows)", "Printed table without a title (Section 9, page 9)"]:
        assert part in text, part


# ---------------------------------------------------------------------------------------------------- the reply
def test_reply_parsing():
    assert parse_action('{"action": "query", "sql": "SELECT 1"}')["sql"] == "SELECT 1"
    fenced = 'Here it is:\n```json\n{"action": "query",\n "sql": "SELECT plan\n  FROM premium_table"}\n```'
    assert parse_action(fenced)["sql"] == "SELECT plan\n  FROM premium_table"      # newlines inside the string
    assert parse_action('<think>{"action": "text_only"}</think>{"action": "ask", "question": "Your age?"}') == {
        "action": "ask", "sql": "", "question": "Your age?", "reason": ""}
    assert parse_action('{"note": 1} {"action": "NOT_IN_TABLES", "reason": "no 7 lakh column"}')["action"] == "not_in_tables"
    for bad, msg in [("SELECT 1", "not one JSON object"), ('{"action": "guess"}', "unknown action"),
                     ('{"action": "query"}', "needs the SQL"), ('{"action": "ask"}', "needs the question")]:
        with pytest.raises(SqlProblem, match=msg):
            parse_action(bad)


def test_prompt():
    p = build_sql_prompt("Premium for 30?", "CARD", 50)
    assert "1 lakh = 100,000" in p and "AS difference" in p and "Never compare text with BETWEEN" in p   # sql_v2
    assert "1 lakh" not in build_sql_prompt("Premium for 30?", "CARD", 50, template="sql_v1")
    assert "TABLES\nCARD\n" in p and "QUESTION: Premium for 30?\nJSON:" in p and "at most 50 rows" in p
    assert '{"action": "query", "sql": "<one SELECT statement>"}' in p and "EARLIER" not in p
    p2 = build_sql_prompt("Premium for 30?", "CARD", 50, attempts=[("SELECT 1", "not JSON")])
    assert "EARLIER REPLIES (they could not be used):\nReply 1: SELECT 1\nProblem: not JSON\nWrite a corrected reply." in p2
    assert p2.endswith("JSON:")
    with pytest.raises(KeyError, match="unknown SQL template"):
        build_sql_prompt("q", "c", template="nope")


# ---------------------------------------------------------------------------------------------------- the SQL checks
@pytest.fixture
def validator():
    return SqlValidator(TABLES, max_rows=50, text_columns=["plan", "age", "tier_cities", "policy_type"])


def test_a_plain_query_is_qualified_and_limited(validator):
    c = validator.check("SELECT plan, si_1000000 FROM premium_table WHERE age = '30' AND tier = 1;")
    assert c.sql == ("SELECT plan, si_1000000 FROM policy_sql.premium_table WHERE age = '30' AND tier = 1 LIMIT 51")
    assert c.tables == ["premium_table"] and c.fetch_limit == 51


def test_limits(validator):
    assert validator.check("SELECT plan FROM premium_table LIMIT 5").fetch_limit is None
    assert validator.check("SELECT plan FROM premium_table LIMIT 500").sql.endswith("LIMIT 51")
    with pytest.raises(SqlProblem, match="LIMIT must be a whole number"):
        validator.check("SELECT plan FROM premium_table LIMIT (SELECT 5)")


def test_allowed_forms(validator):
    ok = [
        "SELECT plan, si_500000 FROM premium_table WHERE tier_cities LIKE '%Alpha%' AND tier >= 1 AND si_500000 < 9000",
        "WITH a AS (SELECT * FROM policy_sql.premium_table WHERE tier = 1) SELECT plan, age, si_500000 FROM a",
        "SELECT a.si_1000000 - b.si_1000000 AS diff, a.age FROM premium_table a JOIN premium_table b ON a.age = b.age "
        "WHERE a.plan = 'Zenith Care Plus' AND b.plan = 'Zenith Care'",
        "SELECT count(*), max(si_500000) FROM premium_table WHERE age ~ '^[0-9]+$' AND plan ILIKE '%care%'",
        "SELECT plan FROM premium_table UNION SELECT CAST(deductible_inr AS text) FROM deductible_discount_table",
        "SELECT CASE WHEN tier = 1 THEN plan END, round(si_500000 * 1.1, 0), regexp_replace(age, '[*]', '') "
        "FROM premium_table WHERE EXISTS (SELECT 1 FROM checkup_limit_table)",
    ]
    for sql in ok:
        assert validator.check(sql).sql.count("policy_sql.") >= 1, sql
    c = validator.check(ok[1])
    assert c.aliases == {} and c.tables == ["premium_table"]
    assert validator.check("SELECT si_500000 AS zc FROM premium_table").aliases == {"zc": "si_500000"}


@pytest.mark.parametrize("sql, msg", [
    ("DELETE FROM premium_table", "only a SELECT"),
    ("UPDATE premium_table SET si_500000 = 1", "only a SELECT"),
    ("DROP TABLE premium_table", "only a SELECT"),
    ("SELECT 1; SELECT 2", "exactly one statement"),
    ("SELECT * FROM rag.chunks", "unknown table"),
    ("SELECT * FROM chunks", "unknown table"),
    ("SELECT * FROM pg_catalog.pg_roles", "unknown table"),
    ("SELECT plan FROM other.policy_sql.premium_table", "unknown table"),
    ("SELECT pg_sleep(10), plan FROM premium_table", "pg_sleep"),
    ("SELECT plan, current_setting('role') FROM premium_table", "current_setting"),
    ("SELECT * FROM generate_series(1, 10)", "generate_series|policy tables"),
    ("SELECT 7100 AS si_500000 FROM premium_table", "constant"),
    ("SELECT 'Zenith' AS plan, si_500000 FROM premium_table", "constant"),
    ("SELECT plan FROM premium_table FOR UPDATE", "LOCK"),
    ("SELECT plan INTO stolen FROM premium_table", "INTO"),
    ("SELECT plan FROM premium_table WHERE age = $1", "PARAMETER"),
    ("SELECT plan FROM premium_table WHERE age = %s", "PLACEHOLDER"),
    ("WITH d AS (DELETE FROM premium_table RETURNING *) SELECT plan FROM d", "DELETE"),
    ("SELECT 1", "constant|none of the policy tables"),
    ("SELECT plan FROM premium_table WHERE", "not valid SQL"),
    ("SELECT age, si_500000 FROM premium_table WHERE age BETWEEN '66*' AND '89*'", "age is text"),
    ("SELECT age, si_500000 FROM premium_table WHERE CAST(age AS int) > 60", "age is text"),
])
def test_refused(validator, sql, msg):
    with pytest.raises(SqlProblem, match=msg):
        validator.check(sql)


# ---------------------------------------------------------------------------------------------------- the step
class ScriptedAnswerer:
    """Returns the given replies in order and keeps the prompts."""

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def answer(self, prompt, use_cache=True, label="answer", max_new_tokens=None):
        assert label == "sql" and max_new_tokens == 400
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        return {"key": f"k{len(self.prompts)}", "answer": reply if isinstance(reply, str) else json.dumps(reply),
                "think_tokens": 10, "answer_tokens": 5, "seconds": 0.1, "from_cache": False}


class DbError(Exception):
    def __init__(self, msg, sqlstate):
        super().__init__(msg)
        self.sqlstate = sqlstate


class FakeDb:
    """Answers model SQL from a function of the SQL text: (columns, rows) or an exception."""

    def __init__(self, answer):
        self.answer, self.ran = answer, []

    def query(self, sql, params=()):
        assert sql.startswith("EXPLAIN ") and params is None
        out = self.answer(sql[len("EXPLAIN "):])
        if isinstance(out, Exception):
            raise out
        return [("plan",)]

    def query_columns(self, sql, params=()):
        assert params is None
        self.ran.append(sql)
        out = self.answer(sql)
        if isinstance(out, Exception):
            raise out
        return out


ROWS = (["plan", "age", "si_1000000", "page"], [("Zenith Care", "30", 9100, 40), ("Zenith Care Plus", "30", 10400, 40)])
Q = "What is the annual premium for a 30-year-old with a 10 lakh Sum Insured in Tier 1?"
SQL = "SELECT plan, age, si_1000000, page FROM premium_table WHERE age = '30' AND tier = 1"


def step(replies, db_answer, **cfg):
    return SqlStep(SqlConfig(enabled=True, **cfg), ScriptedAnswerer(replies), db=FakeDb(db_answer),
                   card=SchemaCard(card_tables()))


def test_answered_rows_become_a_block_with_printed_headings():
    s = step([{"action": "query", "sql": SQL}], lambda sql: ROWS)
    out = s.run(Q)
    assert out.status == "answered" and out.answered and out.tables == ["premium_table"] and len(out.attempts) == 1
    assert out.sql == SQL.replace("FROM premium_table", "FROM policy_sql.premium_table") + " LIMIT 51"
    assert s.db.ran == [out.sql]
    assert out.labels == ["plan", "Age", "Sum Insured 1,000,000 (premium in Rs.)", "page"]
    block = out.block()
    assert "Printed table: Zenith Care Premium - Tier 1 (Annexure Z, pages 40-41)" in block
    assert "- plan: Zenith Care; Age: 30; Sum Insured 1,000,000 (premium in Rs.): 9,100; page: 40" in block
    assert "- plan: Zenith Care Plus; Age: 30; Sum Insured 1,000,000 (premium in Rs.): 10,400; page: 40" in block
    assert "Footnote: *Renewal only (invented)" in block and f"Query: {out.sql}" in block
    assert out.linked_text == [{"relation": "read_with", "label": "Section 2 - Sum Insured options",
                                "text": "Sum Insured options are 5 and 10 lakh."}]
    # the figure check: values of the rows (and the question) pass, others are reported
    assert out.verify("The premium is Rs. 9,100 for Zenith Care and Rs 10,400 for Zenith Care Plus (10 lakh).") == []
    assert out.verify("The premium is Rs. 9,300.") == ["Rs 9,300"]
    d = out.to_dict()
    assert d["rows"][0] == ["Zenith Care", "30", 9100, 40] and d["status"] == "answered" and d["block"] == block


def test_percent_and_renamed_columns():
    sql = "SELECT deductible_inr, pct_zenith AS zc FROM deductible_discount_table WHERE deductible_inr = 50000"
    s = step([{"action": "query", "sql": sql}],
             lambda q: (["deductible_inr", "zc"], [(50000, Decimal("12.50"))]))
    out = s.run("What discount do I get with a 50,000 deductible?")
    assert out.labels == ["Deductible amount (INR)", "zc (Zenith Care (discount %))"]
    assert "- Deductible amount (INR): 50,000; zc (Zenith Care (discount %)): 12.5%" in out.block()
    assert out.verify("You get a 12.5% discount.") == [] and out.verify("You get a 15% discount.") == ["15%"]


def test_questions_the_gate_does_not_match_cost_nothing():
    s = step([], lambda sql: ROWS)
    out = s.run("Is cataract surgery covered?")
    assert out.status == "skipped" and out.attempts == [] and out.block() == "" and s.answerer.prompts == []


@pytest.mark.parametrize("reply, status, field, value", [
    ({"action": "ask", "question": "How old is the person to be insured?"}, "ask", "clarification",
     "How old is the person to be insured?"),
    ({"action": "not_in_tables", "reason": "no 7 lakh column"}, "not_in_tables", "reason", "no 7 lakh column"),
    ({"action": "text_only", "reason": "about the rules"}, "text_only", "reason", "about the rules"),
])
def test_the_models_decisions(reply, status, field, value):
    s = step([reply], lambda sql: pytest.fail("no query should run"))
    out = s.run(Q)
    assert out.status == status and getattr(out, field) == value and out.block() == "" and len(out.attempts) == 1


def test_unusable_replies_are_shown_to_the_model_and_corrected():
    replies = ["SELECT plan FROM premium_table",                                   # not JSON
               {"action": "query", "sql": "SELECT * FROM rag.chunks"},             # refused table
               {"action": "query", "sql": SQL}]
    s = step(replies, lambda sql: ROWS)
    out = s.run(Q)
    assert out.status == "answered" and len(out.attempts) == 3
    assert [a.get("problem", "")[:20] for a in out.attempts] == ['the reply is not one', "unknown table 'rag.c", ""]
    p2, p3 = s.answerer.prompts[1], s.answerer.prompts[2]
    assert "Reply 1: SELECT plan FROM premium_table\nProblem: the reply is not one JSON object" in p2
    assert "Reply 2: " in p3 and "Problem: unknown table 'rag.chunks'" in p3
    assert len(s.db.ran) == 1                                                      # only the valid query ran


def test_database_errors_are_shown_to_the_model():
    calls = []

    def db(sql):
        calls.append(sql)
        return DbError('column "si_700000" does not exist', "42703") if "si_700000" in sql else ROWS
    s = step([{"action": "query", "sql": SQL.replace("si_1000000", "si_700000")}, {"action": "query", "sql": SQL}], db)
    out = s.run(Q)
    assert out.status == "answered" and "the database refused the query: column \"si_700000\" does not exist" in \
        out.attempts[0]["problem"]
    timeout = step([{"action": "query", "sql": SQL}] * 3,
                   lambda sql: DbError("canceling statement due to statement timeout", "57014")).run(Q)
    assert timeout.status == "failed" and "took too long" in timeout.reason and len(timeout.attempts) == 3


def test_no_rows_and_too_many_rows():
    empty = step([{"action": "query", "sql": SQL}] * 3, lambda sql: (ROWS[0], [])).run(Q)
    assert empty.status == "no_rows" and len(empty.attempts) == 3 and "not_in_tables" in empty.reason
    fixed = step([{"action": "query", "sql": SQL.replace("'30'", "'30 years'")},
                  {"action": "not_in_tables", "reason": "no such age"}],
                 lambda sql: (ROWS[0], [])).run(Q)
    assert fixed.status == "not_in_tables" and len(fixed.attempts) == 2
    many = step([{"action": "query", "sql": "SELECT plan, age FROM premium_table"}],
                lambda sql: (["plan", "age"], [("Zenith Care", str(a)) for a in range(51)]), max_retries=0).run(Q)
    assert many.status == "failed" and "more than 50 rows" in many.reason and len(many.attempts) == 1


def test_a_lost_connection_makes_the_step_unavailable():
    out = step([{"action": "query", "sql": SQL}], lambda sql: ConnectionError("server closed the connection")).run(Q)
    assert out.status == "unavailable" and len(out.attempts) == 1 and out.block() == ""


def test_outcome_without_rows_has_no_block():
    assert SqlOutcome(question="q", status="failed").block() == ""


# ---------------------------------------------------------------------------------------------------- config
def test_config():
    cfg = load_config(data={"sql": {"enabled": True}})
    assert cfg.sql.url_env == "SQL_DATABASE_URL" and cfg.sql.max_rows == 50 and cfg.sql.statement_timeout_ms == 5000
    with pytest.raises(ConfigError, match="cannot both be on"):
        load_config(data={"sql": {"enabled": True}, "tables": {"enabled": True},
                          "documents": [{"id": "x", "chunks": "x.json"}]})
    with pytest.raises(ConfigError, match="max_rows"):
        load_config(data={"sql": {"max_rows": 0}})
    with pytest.raises(ConfigError, match="gate_similarity"):
        load_config(data={"sql": {"gate_similarity": 1.5}})
    with pytest.raises(ConfigError, match="unknown setting"):
        load_config(data={"sql": {"max_row": 5}})


# ---------------------------------------------------------------------------------------------------- the gate's closeness
class WordVectors:
    """A stand-in embedder: one dimension per word of a small vocabulary (so closeness = shared words)."""
    name = "word-vectors"
    VOCAB = ["premium", "annual", "price", "yearly", "age", "sum", "insured", "health", "exam", "limit", "discount",
             "deductible", "cataract", "surgery", "covered", "zenith"]

    def _vec(self, text):
        words = text.lower().replace("?", " ").replace(",", " ").split()
        return [float(sum(w.startswith(v) for w in words)) for v in self.VOCAB]

    def encode_documents(self, texts):
        return [self._vec(t) for t in texts]

    def encode_query(self, text):
        return [self._vec(text)]


def test_profiles_come_from_the_card():
    prof = table_profiles(SchemaCard(card_tables()))
    assert "Annexure Z: annual premium (Rs.) by age and Sum Insured." in prof and "Age" in prof
    assert "Sum Insured 1,000,000 (premium in Rs.)" in prof and "Zenith Care (discount %)" in prof
    assert "Refer Annexure Z for the premium." in prof and len(prof) == len(set(prof))


def test_closeness_lets_other_wordings_through_only_with_a_threshold():
    card = SchemaCard(card_tables())
    loose = "What would Zenith set me back annually for a 25 lakh sum insured?"   # no gate word
    off = SqlGate(card.names(), WordVectors(), table_profiles(card), threshold=None)
    kinds, sim = off.check(loose)
    assert kinds == [] and sim > 0.3                                  # recorded, not used
    on = SqlGate(card.names(), WordVectors(), table_profiles(card), threshold=0.3)
    assert on.check(loose)[0] == ["similar"]
    assert on.check("Is cataract surgery covered?") == ([], 0.0)
    assert SqlGate(card.names()).check("Is cataract surgery covered?") == ([], None)    # no embedder: words only


def test_the_step_records_closeness():
    s = SqlStep(SqlConfig(enabled=True, gate_similarity=0.3), ScriptedAnswerer([{"action": "text_only", "reason": "x"}]),
                db=FakeDb(lambda q: ROWS), card=SchemaCard(card_tables()), embedder=WordVectors())
    out = s.run("What would Zenith set me back annually for a 25 lakh sum insured?")
    assert out.kinds == ["similar"] and out.similarity > 0.3 and out.to_dict()["similarity"] == out.similarity
