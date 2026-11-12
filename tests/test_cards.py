"""Table cards (release candidate R1) on invented tables: roles, footnote scope, header labels, grids, chunk file."""
import json

import pytest

from policy_rag.cards import CardSettings, build_cards, role_report, table_roles, write_chunk_file
from policy_rag.config import DocumentConfig
from policy_rag.corpus import load_corpus


def cell(tid, row, col, text, header=False, hp=(), rp=(), group=False):
    return {"table_id": tid, "row": row, "col": col, "text": text, "is_header": header, "header_path": list(hp),
            "row_path": list(rp), "group_row": group}


def benefits_table():
    """A schedule of benefits: a header row, three data rows, footnotes '*' (one row) and '**' (one row)."""
    t = {"table_id": "p2-t1", "section_id": "A", "section_title": "Schedule of Benefits", "caption": "Schedule",
         "title": None, "page": 2}
    cells = [cell("p2-t1", 0, 0, "Benefit", header=True), cell("p2-t1", 0, 1, "Plan One", header=True),
             cell("p2-t1", 0, 2, "Plan Two", header=True)]
    for r, (name, a, b) in enumerate([("Room Rent", "At actuals", "At actuals"), ("Bonus*", "10%", "20%"),
                                      ("**Deductible", "25k", "50k")], start=1):
        label = name.strip("*")
        cells += [cell("p2-t1", r, 0, name, rp=[name]), cell("p2-t1", r, 1, a, hp=["Plan One"], rp=[name]),
                  cell("p2-t1", r, 2, b, hp=["Plan Two"], rp=[name])]
    ref = {"parts": ["p2-t1"], "caption": "Schedule", "title": None, "section_title": "Schedule of Benefits",
           "reference_text": "The benefits of each plan are below.",
           "footnotes": [{"marker": "*", "text": "*Only after one claim-free year"},
                         {"marker": "**", "text": "**Optional, at an extra premium"},
                         {"marker": None, "text": "All amounts per policy year"}]}
    return t, cells, ref


def example_table():
    """A worked example with a long two-level header and a footnote whose marker sits in a header cell."""
    long_head = "Coverage opted on an individual basis covering every member of the family separately at one time"
    t = {"table_id": "p3-t1", "section_id": "9", "section_title": "Premium Computation", "caption": "Illustration 1",
         "title": None, "page": 3}
    cells = [cell("p3-t1", 0, 1, long_head + "^", header=True), cell("p3-t1", 1, 0, "Age", header=True),
             cell("p3-t1", 1, 1, "Premium", header=True)]
    for r, age in enumerate([25, 35], start=2):
        cells += [cell("p3-t1", r, 0, str(age), rp=[str(age)]),
                  cell("p3-t1", r, 1, f"{age}00", hp=[long_head + "^", "Premium"], rp=[str(age)]),
                  cell("p3-t1", r, 2, "5", hp=["Sum Insured in Lakhs"], rp=[str(age)])]
    ref = {"parts": ["p3-t1"], "caption": "Illustration 1", "title": None, "section_title": "Premium Computation",
           "reference_text": "Plan One, 1 year, Tier 1", "footnotes": [{"marker": "^", "text": "^Before taxes"}]}
    return t, cells, ref, long_head


def premium_grid():
    """Two parts of one premium table (ages 0-44 and 45-89, 45 rows each: a grid); the renewal-only marker is on the
    rows of the second part from age 66."""
    title = "Plan One Gross Premium (Excl. GST) - Tier 1 (City A, City B, City C and City D)"
    tables, cells = [], []
    for part, ages in (("p8-t1", range(0, 45)), ("p9-t1", range(45, 90))):
        tables.append({"table_id": part, "section_id": "C", "section_title": "Premium Table", "caption": None,
                       "title": title, "page": int(part[1])})
        for r, age in enumerate(ages):
            label = f"{age}*" if age >= 66 else str(age)
            cells.append(cell(part, r, 0, label, rp=[label]))
            for col, si in enumerate(("500,000", "1,000,000"), start=1):
                cells.append(cell(part, r, col, f"{10_000 + age * 10 + col:,}", hp=[title, f"Sum Insured / {si}"],
                                  rp=[label]))
    ref = {"parts": ["p8-t1", "p9-t1"], "caption": None, "title": title, "section_title": "Premium Table",
           "reference_text": "Premiums exclude GST", "footnotes": [{"marker": "*", "text": "*Only for Renewal Purposes"}]}
    return tables, cells, ref


def by_row(cards, tid):
    return {c["row"]: c for c in cards if c["table_id"] == tid}


def test_roles_come_from_the_tables_own_words():
    t1, _, r1 = benefits_table()
    t2, _, r2, _ = example_table()
    roles = table_roles([t1, t2], [r1, r2])
    assert roles["p2-t1"]["role"] == "rule" and roles["p3-t1"]["role"] == "example"
    assert roles["p3-t1"]["reason"] == 'caption says "Illustration"'
    over = table_roles([t1, t2], [r1, r2], overrides={"p3-t1": "rule"})
    assert over["p3-t1"] == {"role": "rule", "reason": 'caption says "Illustration"', "table": "p3-t1", "overridden": True}
    with pytest.raises(ValueError):
        table_roles([t1], [r1], overrides={"p2-t1": "maybe"})
    rep = role_report([t1, t2], [r1, r2])
    assert [(r["table"], r["role"]) for r in rep] == [("p2-t1", "rule"), ("p3-t1", "example")]


def test_footnotes_go_only_on_the_rows_that_carry_their_marker():
    t, cells, ref = benefits_table()
    rows = by_row(build_cards([t], cells, [ref]), "p2-t1")
    assert "Only after one claim-free year" in rows[2]["text"] and "Only after" not in rows[1]["text"]
    assert "Optional, at an extra premium" in rows[3]["text"] and "Optional" not in rows[2]["text"]   # "**" is not "*"
    assert all("All amounts per policy year" in c["text"] for c in rows.values())                     # no marker: all


def test_a_marker_in_a_header_keeps_the_footnote_on_every_row():
    t, cells, ref, _ = example_table()
    assert all("Before taxes" in c["text"] for c in build_cards([t], cells, [ref]))


def test_a_footnote_of_a_two_part_table_stays_off_the_part_without_its_marker():
    tables, cells, ref = premium_grid()
    cards = build_cards(tables, cells, [ref])
    first, second = by_row(cards, "p8-t1"), by_row(cards, "p9-t1")
    assert not any("Renewal" in c["text"] for c in first.values())
    assert [r for r, c in second.items() if "Renewal" in c["text"]] == list(range(21, 45))           # ages 66-89


def test_headers_labels_and_heading():
    t, cells, ref, long_head = example_table()
    card = by_row(build_cards([t], cells, [ref]), "p3-t1")[2]
    assert card["role"] == "example" and card["text"].startswith("Worked example: Illustration 1")
    assert card["label"] == "Section 9 - Premium Computation - Illustration 1 (table, page 3)"
    line = card["text"].split("\n")[1]
    cut = " ".join(long_head.split()[:12]) + " …"
    assert cut + " / Premium: 2500" in line and long_head not in line                                # cut to 12 words
    tables, gcells, gref = premium_grid()
    grid = by_row(build_cards(tables, gcells, [gref]), "p8-t1")[5]
    assert grid["text"].split("\n")[0].endswith("Tier 1 (City A, City B, City C and City D)")        # shared: whole
    assert "Sum Insured / 500,000: 10,051" in grid["text"]


def test_grids_get_their_own_index_after_the_ordinary_tables():
    t, cells, ref = benefits_table()
    tables, gcells, gref = premium_grid()
    cards = build_cards(tables + [t], gcells + cells, [gref, ref])
    kinds = [c["index"] for c in cards]
    assert kinds == ["table"] * 3 + ["grid"] * 90
    assert CardSettings().header_words == 12


def test_chunk_file_appends_cards_after_the_text(tmp_path):
    text = {"document": {"id": "x"}, "chunks": [
        {"idx": 0, "section_id": "1", "section_title": "Cover", "label": "Section 1 - Cover", "text": "Covered.",
         "pages": [1], "kind": "text", "index": "policy"}]}
    src = tmp_path / "text.json"
    src.write_text(json.dumps(text))
    t, cells, ref = benefits_table()
    out = write_chunk_file(src, build_cards([t], cells, [ref]), tmp_path / "cards.json")
    rows = json.loads(out.read_text())["chunks"]
    assert [r["idx"] for r in rows] == [0, 1, 2, 3]
    corpus = load_corpus([DocumentConfig("x", str(out), "x")])
    assert [c.index for c in corpus.chunks] == ["policy", "table", "table", "table"]
    assert all(c.is_table for c in corpus.chunks[1:])
