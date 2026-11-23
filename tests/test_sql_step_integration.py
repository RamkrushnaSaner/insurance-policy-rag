"""The SQL step against a real Postgres at migration 003 (opt-in): the schema card read from schema policy_sql, model SQL
run as a login in group rag_sql, its permissions, the read-only session and the statement timeout.

Runs only when POLICY_RAG_TEST_DATABASE_URL_V35 is set to a development database (never production) with migrations
001, 002 and 003 applied, and a login that may write, delete and create roles there. The policy, its tables and
values are invented (document ``pytest-sql``); everything added (rows and the login ``pytest_sql_login``) is removed
before and after. The test is skipped when policy_sql already shows another document's rows.
"""
import json
import os

import pytest

from policy_rag.config import SqlConfig
from policy_rag.database import Database, chunk_uid
from policy_rag.sql_step import SchemaCard, SqlProblem, SqlRunner, SqlStep, SqlValidator

from test_sql_step import ScriptedAnswerer

ENV = "POLICY_RAG_TEST_DATABASE_URL_V35"
LOGIN_ENV = "POLICY_RAG_TEST_SQL_LOGIN_URL"
DOC, LOGIN, PASSWORD = "pytest-sql", "pytest_sql_login", "pytest-only-password"
pytestmark = pytest.mark.skipif(not os.environ.get(ENV), reason=f"{ENV} is not set (opt-in database test)")

SI = [500000, 1000000, 1500000, 2000000, 2500000, 5000000, 10000000, 20000000]
AGES = ["30", "31", "66*", ">=90*"]
PLANS = {  # (plan, tier) -> printed parts [(table_id, page, ages)]
    ("Zenith Care", 1): [("p40-t1", 40, AGES[:2]), ("p41-t1", 41, AGES[2:])],
    ("Zenith Care", 2): [("p42-t1", 42, AGES)],
    ("Zenith Care Plus", 1): [("p43-t1", 43, AGES)],
    ("Zenith Care Plus", 2): [("p44-t1", 44, AGES)],
}
CITIES = {1: "Alpha City, Beta Town", 2: "Rest of Zland"}


def premium(plan, tier, age, si):
    """The invented premium of one cell."""
    n = int(age.strip(">=*"))
    return 5000 + n * 100 + SI.index(si) * 1500 + (700 if plan.endswith("Plus") else 0) + (400 if tier == 1 else 0)


def _cleanup(con):
    con.execute("DELETE FROM rag.policy_versions WHERE doc_id = %s", (DOC,))     # tables, chunks, links, rows go with it
    con.execute("DELETE FROM rag.documents WHERE doc_id = %s", (DOC,))
    con.execute(f"DROP ROLE IF EXISTS {LOGIN}")


def _load(con):
    con.execute("INSERT INTO rag.documents (doc_id, title, insurer, metadata) VALUES (%s, 'Zenith (invented)', 'Test', '{}')",
                (DOC,))
    vid = con.execute("INSERT INTO rag.policy_versions (doc_id, version_label, source_file, source_sha256, pages, "
                      "pipeline, status, activated_at) VALUES (%s, 'v1', 'invented', %s, 50, '{}', 'active', now()) "
                      "RETURNING version_id", (DOC, "0" * 64)).fetchone()[0]

    def table(tid, page, title, section, first_of=None, footnotes=(), reference=""):
        con.execute("INSERT INTO rag.tables (version_id, table_id, record) VALUES (%s, %s, %s)",
                    (vid, tid, json.dumps({"table_id": tid, "pages": [page], "title": title, "section_id": section})))
        if first_of is not None:
            con.execute("INSERT INTO rag.table_refs (version_id, table_id, record) VALUES (%s, %s, %s)", (vid, tid, json.dumps({
                "table_id": tid, "parts": first_of, "section_id": section, "title": title,
                "footnotes": [{"text": f} for f in footnotes], "reference": {"text": reference, "method": "pointer"},
                "review": {"status": "ok"}})))

    for (plan, tier), parts in PLANS.items():
        title = f"{plan} Plan Premium (Excl. GST) - Tier {tier} ({CITIES[tier]})"
        for n, (tid, page, _) in enumerate(parts):
            table(tid, page, title if n == 0 else None, "Annexure-Z", [p[0] for p in parts] if n == 0 else None,
                  ["*Renewal only (invented)"], "Refer Annexure Z - Premium Table (invented)")
    table("p9-t1", 9, "9. Health Check (invented)", "9", ["p9-t1"], (), "For Individual Policies (invented):")
    table("p9-t2", 9, None, "9", ["p9-t2"], (), "For Family Floater Policies (invented):")
    table("p9-t3", 9, None, "10", ["p9-t3"], (), "Deductible discount (invented):")
    for (plan, tier), parts in PLANS.items():
        for tid, _page, ages in parts:
            for row, age in enumerate(ages, 1):
                con.execute("INSERT INTO rag.premium_table VALUES (%s, %s, %s, %s, %s, " + ", ".join(["%s"] * 8)
                            + ", %s, %s)", (vid, plan, tier, CITIES[tier], age,
                                            *[premium(plan, tier, age, si) for si in SI], tid, row))
    con.execute("INSERT INTO rag.checkup_limit_table VALUES (%s, 'individual', 'For Individual Policies (invented):', "
                "1500, 2500, 3500, 4500, 5500, 'p9-t1', 2), (%s, 'family_floater', 'For Family Floater Policies "
                "(invented):', 3000, 5000, 7000, 9000, 11000, 'p9-t2', 2)", (vid, vid))
    con.execute("INSERT INTO rag.deductible_discount_table VALUES (%s, 25000, 21.5, 11, 10, 3, 'p9-t3', 2), "
                "(%s, 50000, 31, 21, 19.5, 6, 'p9-t3', 3), (%s, 100000, 41, 31, 29, 9, 'p9-t3', 4)", (vid, vid, vid))
    for idx, (label, text, links) in enumerate([
            ("Section 2 - Sum Insured options (invented)", "The Sum Insured options are 5 and 10 lakh.",
             [("p40-t1", "read_with"), ("p42-t1", "read_with")]),
            ("Section 9 - Health Check (invented)", "A health check is paid up to the limits below.",
             [("p9-t1", "introduced_by")])]):
        embed = f"{label}: {text}"
        uid = chunk_uid(DOC, embed)
        con.execute("INSERT INTO rag.chunks (version_id, chunk_uid, record) VALUES (%s, %s, %s)", (vid, uid, json.dumps({
            "idx": idx, "kind": "text", "index": "policy", "pages": [2 + 7 * idx], "text": text, "label": label,
            "embed_text": embed, "section_id": label.split()[1]})))
        for tid, relation in links:
            con.execute("INSERT INTO rag.chunk_tables VALUES (%s, %s, %s, %s)", (vid, tid, uid, relation))
    con.execute(f"CREATE ROLE {LOGIN} LOGIN PASSWORD '{PASSWORD}' IN ROLE rag_sql")


@pytest.fixture(scope="module")
def login_url():
    psycopg = pytest.importorskip("psycopg")
    pytest.importorskip("sqlglot")
    owner = os.environ[ENV]
    with psycopg.connect(owner, autocommit=True) as con:
        if con.execute("SELECT to_regnamespace('policy_sql') IS NULL").fetchone()[0]:
            pytest.skip("schema policy_sql is missing: apply migration 003")
        _cleanup(con)
        other = con.execute("SELECT count(*) FROM policy_sql.premium_table").fetchone()[0]
        if other:
            pytest.skip(f"policy_sql already shows {other} premium rows of another document")
        _load(con)
    url = psycopg.conninfo.make_conninfo(owner, user=LOGIN, password=PASSWORD)
    os.environ[LOGIN_ENV] = url
    yield url
    os.environ.pop(LOGIN_ENV, None)
    with psycopg.connect(owner, autocommit=True) as con:
        con.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE usename = %s", (LOGIN,))
        _cleanup(con)


def sql_db(**cfg):
    return Database(SqlConfig(url_env=LOGIN_ENV, **cfg))


def test_the_card_is_read_from_the_database(login_url):
    db = sql_db()
    card = SchemaCard.load(db)
    assert sorted(card.tables) == ["checkup_limit_table", "deductible_discount_table", "premium_table"]
    text = card.text
    for part in [
            "policy_sql.premium_table (16 rows): Annexure C: gross annual premium",
            "Printed table: Zenith Care Plan Premium (Excl. GST) - Tier 1 (Alpha City, Beta Town) (Annexure Z, pages 40-41)",
            "Printed table: Zenith Care Plus Plan Premium (Excl. GST) - Tier 2 (Rest of Zland) (Annexure Z, page 44)",
            "Introduced by: Refer Annexure Z - Premium Table (invented)", "Footnote: *Renewal only (invented)",
            "plan (text): Plan named in the printed title. Values: 'Zenith Care', 'Zenith Care Plus'",
            "tier (smallint): Premium tier named in the printed title (1 or 2). Values: 1, 2",
            "tier_cities (text): Places of the tier, as printed in the title. Values: 'Alpha City, Beta Town', 'Rest of Zland'",
            "age (text): Printed heading: Age (as printed; * marks the printed footnote). Values: '30', '31', '66*', '>=90*'",
            "si_1000000 (integer): Printed heading: Sum Insured 1,000,000 (10 lakh) (premium in Rs.)",
            "deductible_inr (integer): Printed heading: Deductible amount (INR). Values: 25000, 50000, 100000 (1 lakh)",
            "limit_si_5_lakhs (integer): Printed heading: 5 Lakhs (Limit of Cover in Rs.). Values: 1500, 3000",
            "policy_sql.checkup_limit_table (2 rows)", "Printed table: 9. Health Check (invented) (Section 9, page 9)",
            "Printed table without a title (Section 10, page 9)",
            "pct_secure_global (numeric(5,2)): Printed heading: "]:
        assert part in text, part
    assert text.count("Example rows:") == 3
    assert {"Zenith Care", "Zenith Care Plus", "Rest of Zland"} <= set(card.names())
    db.close()


def test_the_step_answers_from_the_tables(login_url):
    sql = ("SELECT plan, age, si_1000000, page FROM premium_table WHERE tier = 2 AND age IN ('31', '>=90*') "
           "ORDER BY plan, age")
    step = SqlStep(SqlConfig(enabled=True, url_env=LOGIN_ENV), ScriptedAnswerer([{"action": "query", "sql": sql}]))
    out = step.run("What is the premium in Zland for a 31-year-old and someone over 90, with a 10 lakh Sum Insured?")
    assert out.status == "answered", out.attempts
    assert out.rows == [(p, a, premium(p, 2, a, 1000000), pg) for p, pg in [("Zenith Care", 42), ("Zenith Care Plus", 44)]
                        for a in ["31", ">=90*"]]
    block = out.block()
    assert f"- plan: Zenith Care; Age (as printed; * marks the printed footnote): 31; Sum Insured 1,000,000 (premium in " \
           f"Rs.): {premium('Zenith Care', 2, '31', 1000000):,}; page: 42" in block
    assert "Footnote: *Renewal only (invented)" in block and "Query: SELECT plan" in block
    assert [l["label"] for l in out.linked_text] == ["Section 2 - Sum Insured options (invented)"]
    assert out.verify(f"Rs. {premium('Zenith Care Plus', 2, '31', 1000000):,}") == []
    step.db.close()


def test_the_sql_login_reads_policy_sql_only(login_url):
    db = sql_db()
    runner, validator = SqlRunner(db), SqlValidator(["premium_table"])
    assert db.query("SELECT count(*) FROM policy_sql.deductible_discount_table") == [(3,)]
    for sql in ["SELECT * FROM rag.chunks", "SELECT * FROM rag.premium_table", "SELECT * FROM rag.policy_versions"]:
        with pytest.raises(SqlProblem, match="permission denied"):
            runner.run(sql)                    # the database refuses even SQL the validator would have refused
        with pytest.raises(SqlProblem, match="unknown table"):
            validator.check(sql)
    psycopg = pytest.importorskip("psycopg")
    for sql in ["CREATE TABLE policy_sql.x (a int)", "INSERT INTO rag.documents (doc_id) VALUES ('x')",
                "UPDATE policy_sql.premium_table SET si_500000 = 1"]:
        with pytest.raises(psycopg.Error) as e:
            db.query(sql)                      # read-only session, and no rights to write
        assert e.value.sqlstate in ("25006", "42501", "55000"), (sql, e.value.sqlstate, str(e.value))
    assert db.reconnects == 0
    db.close()


def test_a_slow_query_is_cancelled_and_the_connection_kept(login_url):
    db = sql_db(statement_timeout_ms=200)
    runner = SqlRunner(db)
    slow = SqlValidator(["premium_table"]).check(
        "SELECT count(*) FROM premium_table a, premium_table b, premium_table c, premium_table d, premium_table e, "
        "premium_table f, premium_table g")
    with pytest.raises(SqlProblem, match="took too long"):
        runner.run(slow.sql)
    assert runner.run("SELECT count(*) FROM policy_sql.premium_table") == (["count"], [(16,)])
    assert db.reconnects == 0
    db.close()


def test_the_question_check_reads_the_places_of_each_tier_from_the_database(login_url):
    from policy_rag.sql_checks import QuestionCheck
    db = sql_db()
    qc = QuestionCheck(SchemaCard.load(db), db)
    f = qc.tables["premium_table"]
    assert f.places == {"Alpha City": 1, "Beta Town": 1} and f.catch_all == 2 and f.tier_text[2] == "Rest of Zland"
    q = "What is the Zenith Care premium for a 31-year-old in Beta Town with a 10 lakh Sum Insured?"
    wrong = "SELECT plan, age, si_1000000 FROM premium_table WHERE plan = 'Zenith Care' AND tier = 2 AND age = '31'"
    step = SqlStep(SqlConfig(enabled=True, url_env=LOGIN_ENV), ScriptedAnswerer(
        [{"action": "query", "sql": wrong}, {"action": "query", "sql": wrong.replace("tier = 2", "tier = 1")}]))
    out = step.run(q)
    assert out.status == "answered" and out.rows == [("Zenith Care", "31", premium("Zenith Care", 1, "31", 1000000))]
    assert "names Beta Town (Tier 1), but the query reads only Tier 2" in out.attempts[0]["problem"] and out.notes == []
    step.db.close()
    db.close()
