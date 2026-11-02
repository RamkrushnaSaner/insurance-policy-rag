"""The table route's SQL against a real Postgres (opt-in): the invented policy of tests/table_fixtures.py is written to
the ``rag`` schema, then every question must get the same answer from SQL (``TableData``) as from the in-memory
stand-in the unit tests use.

Runs only when POLICY_RAG_TEST_DATABASE_URL is set to a development database with migration 001 applied and a login
that may write and delete (never production). Everything added is removed before and after.
"""
import json
import os

import pytest

from policy_rag.config import DatabaseConfig
from policy_rag.database import Database
from policy_rag.tables import TableData, TableRoute

from table_fixtures import DOC, MemoryTableData, dataset

ENV = "POLICY_RAG_TEST_DATABASE_URL"
pytestmark = pytest.mark.skipif(not os.environ.get(ENV), reason=f"{ENV} is not set (opt-in database test)")

QUESTIONS = [
    "What is the annual premium for Zenith Care with a Rs. 10 lakh Sum Insured for a 30-year-old living in Mumbai?",
    "How much more does Zenith Care Plus cost than Zenith Care for a 45-year-old living in Pune with Rs. 5 lakh cover?",
    "What is the annual premium for a 30-year-old in a Tier 2 city?",
    "How much more does a 31 year old pay in Mumbai than in Pune for Zenith Care 10 lakh?",
    "What premium for a 95 year old on Zenith Care, 5 lakh, Gurugram?",
    "What is the annual premium for Zenith Lite for a 30-year-old?",
    "What is the annual premium for a 30-year-old with a Rs. 7 lakh Sum Insured?",
    "How does the maximum single-claim amount for Zenith Care change from Year 1 to Year 3?",
    "What is the difference between Zenith Care and Care Plus plans?",
    "Can I pay my premium every month?",
    "Can I pay quarterly instalments?",
    "How much is the check-up limit for a family floater with 10 lakh sum insured?",
    "What is the Preventive Health Check-up benefit?",
    "What discount do I get with a Rs 50,000 deductible on Zenith Care with 10 lakh sum insured?",
    "What discount do I get with a Rs 25,000 deductible?",
    "Is Pune a Tier 1 city?",
    "Which cities are classified as Tier 1 for premium calculation?",
]


def _cleanup(con):
    con.execute("DELETE FROM rag.policy_versions WHERE doc_id = %s", (DOC,))     # cells, typed rows, blocks go with it
    con.execute("DELETE FROM rag.documents WHERE doc_id = %s", (DOC,))


def _load(con, ds):
    con.execute("INSERT INTO rag.documents (doc_id, title, insurer, metadata) VALUES (%s, %s, 'Test', '{}')",
                (DOC, ds["title"]))
    vid = con.execute("INSERT INTO rag.policy_versions (doc_id, version_label, source_file, source_sha256, pages, "
                      "pipeline, status, activated_at) VALUES (%s, 'v1', 'invented', %s, 30, '{}', 'active', now()) "
                      "RETURNING version_id", (DOC, "0" * 64)).fetchone()[0]
    for n, b in enumerate(ds["blocks"]):
        rec = {"id": b["block_id"], "order": n, "page": b["page"], "type": "list_item", "text": b["text"],
               "section": b["section"]}
        con.execute("INSERT INTO rag.text_blocks (version_id, block_id, record) VALUES (%s, %s, %s)",
                    (vid, b["block_id"], json.dumps(rec)))
    for tid, t in ds["tables"].items():
        rows = 1 + max(r for r, _ in t["cells"])
        cols = 1 + max(c for _, c in t["cells"])
        rec = {"table_id": tid, "caption": t["caption"], "title": t["title"], "section_id": t["section_id"],
               "pages": [t["page"]], "n_rows": rows, "n_cols": cols, "continues": None}
        con.execute("INSERT INTO rag.tables (version_id, table_id, record) VALUES (%s, %s, %s)", (vid, tid, json.dumps(rec)))
        with con.cursor() as cur:
            cur.executemany("INSERT INTO rag.table_cells (version_id, table_id, row, col, record) VALUES (%s, %s, %s, %s, %s)",
                            [(vid, tid, r, c, json.dumps({"table_id": tid, "row": r, "col": c, "text": text,
                                                          "header_path": [], "row_path": []}))
                             for (r, c), text in t["cells"].items()])
    for f in ds["refs"]:
        t = ds["tables"][f["table_id"]]
        rec = {"table_id": f["table_id"], "parts": f["parts"], "section_id": t["section_id"], "caption": t["caption"],
               "title": t["title"], "reference": {"text": t["caption"] or t["title"] or "", "method": "heading"},
               "review": {"status": "heading-only"}, "notes": f["notes"], "footnotes": f["footnotes"]}
        con.execute("INSERT INTO rag.table_refs (version_id, table_id, record) VALUES (%s, %s, %s)",
                    (vid, f["table_id"], json.dumps(rec)))
    for name, rows in list(ds["typed"].items()) + [("city_tiers", ds["places"])]:
        cols = list(rows[0])
        with con.cursor() as cur:
            cur.executemany(f"INSERT INTO rag.{name} (version_id, {', '.join(cols)}) "
                            f"VALUES (%s, {', '.join(['%s'] * len(cols))})", [(vid, *[r[c] for c in cols]) for r in rows])


@pytest.fixture(scope="module")
def routes():
    psycopg = pytest.importorskip("psycopg")
    ds = dataset()
    with psycopg.connect(os.environ[ENV], autocommit=True) as con:
        if not con.execute("SELECT to_regclass('rag.premium_rates') IS NOT NULL").fetchone()[0]:
            pytest.skip("the rag schema (migration 001) is not applied in this database")
        _cleanup(con)
        with con.transaction():
            _load(con, ds)
    db = Database(DatabaseConfig(url_env=ENV))
    yield TableRoute(TableData(db, DOC)), TableRoute(MemoryTableData(ds))
    db.close()
    with psycopg.connect(os.environ[ENV], autocommit=True) as con:
        _cleanup(con)


def test_the_questions_cover_every_kind(routes):
    answers = [routes[0].answer(q) for q in QUESTIONS]
    assert {a.intent for a in answers if a.answered} == {"premium", "claim_limit", "instalment", "checkup_limit",
                                                          "deductible_discount", "city_tier"}
    assert sum(a.answered for a in answers) == 15 and sum(a.status == "not_in_policy" for a in answers) == 2


def test_catalog_from_sql_matches(routes):
    sql, mem = routes
    for key in ("title", "plans", "places"):
        assert sql.cat[key] == mem.cat[key], key
    assert sql.cat["premium"] == mem.cat["premium"]
    assert sorted(sql.cat["illustration"]["plans"]) == mem.cat["illustration"]["plans"]


@pytest.mark.parametrize("question", QUESTIONS)
def test_sql_and_memory_give_the_same_answer(routes, question):
    sql, mem = routes
    a, b = sql.answer(question), mem.answer(question)
    assert (a.intent, a.status) == (b.intent, b.status)
    assert a.text == b.text
    assert len(a.rows) == len(b.rows) and a.trace() == [] and a.verify(a.text) == []
