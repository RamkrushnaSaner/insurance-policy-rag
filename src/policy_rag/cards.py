"""Table cards: one searchable passage per data row of every printed table (release candidate R1).

The typed tables answer a fixed set of questions with SQL (``tables.py``); cards make every printed table searchable
like the policy text, with no question patterns. A card holds one data row:

    <heading: "Worked example: " for an example table, then the table's caption or title and the header shared by
     every value of the row>
    <row label>: <header>: <value>; <header>: <value>; ...
    Introduced in the policy as: <the sentence that introduces the table, at most 40 words>
    Notes: <the footnotes that apply to this row>

Meaning that is fixed when the cards are built, not guessed later:

* **role**: a table is a worked ``example`` when its caption, title, section title or introducing sentence calls it an
  illustration or example; otherwise it is a ``rule``. The owner reviews the list (``role_report``) and can override
  any table.
* **footnotes**: a footnote goes only on the rows that carry its marker ("66*" carries "*", "**Aggregate Deductible"
  carries "**"). A marker in a header cell, a footnote without a marker, or a marker that no cell carries keeps the
  footnote on the whole table.
* **headers**: each header label of a value is cut to at most 12 words. The header shared by every value of a row (a
  premium table's "... Gross Premium (Excl. GST) - Tier 1 (...)") goes once into the heading and is not cut.
* **index**: large numeric grids (at least 40 data rows, at least 80% numeric values: the premium tables) go to the
  ``grid`` index, every other table to the ``table`` index; the label starts with the section and its title, like
  the text passages.

Cards never go into the database's ``rag.chunks`` (tables are stored as tables there): they are written to a chunk
file together with the text chunks (``write_chunk_file``) and embedded at start-up.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

EXAMPLE_WORDS = re.compile(r"\billustrat\w*|\bexamples?\b|\be\.g\.|\bfor instance\b|\bsample\b", re.I)
_MARKER = re.compile(r"[*^#†‡]+")


@dataclass(frozen=True)
class CardSettings:
    grid_min_rows: int = 40
    grid_min_numeric: float = 0.8
    header_words: int = 12
    title_words: int = 12
    intro_words: int = 40
    notes_words: int = 40


def is_number(s: str) -> bool:
    return bool(re.fullmatch(r"[\s\d,.%*^₹-]+", s or "")) and any(ch.isdigit() for ch in s)


def first_words(text: str, n: int) -> str:
    w = (text or "").split()
    return " ".join(w[:n]) + (" …" if len(w) > n else "")


def _common_prefix(paths: Sequence[Sequence[str]]) -> List[str]:
    out = []
    for parts in zip(*paths):
        if all(p == parts[0] for p in parts):
            out.append(parts[0])
        else:
            break
    return out


def markers_in(text: str) -> set:
    """Footnote markers written in a cell ("66*" -> {"*"}, "**Aggregate Deductible" -> {"**"})."""
    return set(_MARKER.findall(text or ""))


# ---------------------------------------------------------------------------------------------------- roles
def table_roles(tables: Iterable[dict], refs: Iterable[dict], overrides: Optional[Dict[str, str]] = None) -> Dict[str, dict]:
    """Role of every table, keyed by every part's table id: {"role", "reason", "table" (first part), "overridden"}.
    ``overrides``: {first part's table id: "rule" | "example"}, after the owner's review."""
    parts_of = {r["parts"][0]: list(r["parts"]) for r in refs}
    by_id = {t["table_id"]: t for t in tables}
    for tid in by_id:                                   # a part without a reference row is its own table
        if not any(tid in p for p in parts_of.values()):
            parts_of[tid] = [tid]
    ref_of = {r["parts"][0]: r for r in refs}
    out = {}
    for first, parts in parts_of.items():
        t, r = by_id.get(first, {}), ref_of.get(first, {})
        fields = [("caption", r.get("caption") or t.get("caption")), ("title", r.get("title") or t.get("title")),
                  ("section title", r.get("section_title") or t.get("section_title")),
                  ("introducing sentence", r.get("reference_text"))]
        hit = next(((name, m.group(0)) for name, v in fields if v for m in [EXAMPLE_WORDS.search(v)] if m), None)
        role = "example" if hit else "rule"
        reason = f'{hit[0]} says "{hit[1]}"' if hit else "no field calls it an illustration or example"
        overridden = bool(overrides and first in overrides and overrides[first] != role)
        if overrides and first in overrides:
            if overrides[first] not in ("rule", "example"):
                raise ValueError(f"role override for {first}: expected 'rule' or 'example', got {overrides[first]!r}")
            role = overrides[first]
        for p in parts:
            out[p] = {"role": role, "reason": reason, "table": first, "overridden": overridden}
    return out


def role_report(tables: Iterable[dict], refs: Iterable[dict], overrides: Optional[Dict[str, str]] = None) -> List[dict]:
    """One row per table for the owner's review: first part, parts, section, caption or title, role, why."""
    tables = list(tables)
    by_id = {t["table_id"]: t for t in tables}
    roles = table_roles(tables, refs, overrides)
    rows, seen = [], set()
    for t in sorted(tables, key=lambda t: (t.get("page") or 0, t["table_id"])):
        r = roles[t["table_id"]]
        if r["table"] in seen:
            continue
        seen.add(r["table"])
        first = by_id.get(r["table"], t)
        parts = sorted(p for p, x in roles.items() if x["table"] == r["table"])
        rows.append({"table": r["table"], "parts": parts, "section": first.get("section_id"),
                     "section_title": first.get("section_title"),
                     "caption_or_title": first.get("caption") or first.get("title") or "", "role": r["role"],
                     "why": r["reason"], "overridden": r["overridden"]})
    return rows


# ---------------------------------------------------------------------------------------------------- footnotes
def scope_footnotes(ref: dict, rows: Dict[str, Dict[int, List[dict]]]) -> tuple:
    """(notes for the whole table, {(part, row): notes for that row only}) for one table and its parts.
    ``rows``: {part: {row: [cells]}} with every cell of the parts (header cells included)."""
    whole, per_row = [], {}
    for f in ref.get("footnotes") or []:
        text = (f.get("text") or "").strip()
        if not text:
            continue
        m = (f.get("marker") or "").strip()
        carriers, in_header = [], False
        if m:
            for part, by_row in rows.items():
                for r, cells in by_row.items():
                    hits = [c for c in cells if m in markers_in(c.get("text"))]
                    if any(c.get("is_header") for c in hits):
                        in_header = True
                    elif hits:
                        carriers.append((part, r))
        if not m or in_header or not carriers:
            whole.append(text)
        else:
            for key in carriers:
                per_row.setdefault(key, []).append(text)
    return whole, per_row


# ---------------------------------------------------------------------------------------------------- cards
def build_cards(tables: Iterable[dict], cells: Iterable[dict], refs: Iterable[dict],
                settings: CardSettings = CardSettings(), overrides: Optional[Dict[str, str]] = None) -> List[dict]:
    """Cards (chunk-file rows without ``idx``) for every data row of every table part.

    ``tables``: {table_id, section_id, section_title, caption, title, page}; ``cells``: {table_id, row, col, text,
    is_header, header_path, row_path, group_row}; ``refs``: {parts, caption, title, section_title, reference_text,
    footnotes: [{marker, text}]} (one per table, keyed by its first part)."""
    tables, refs = list(tables), list(refs)
    roles = table_roles(tables, refs, overrides)
    ref_of_part = {p: r for r in refs for p in r["parts"]}
    by_row: Dict[str, Dict[int, List[dict]]] = {}
    for c in cells:
        text = (c.get("text") or "").strip()
        if text:
            by_row.setdefault(c["table_id"], {}).setdefault(int(c["row"]), []).append(
                {**c, "text": text, "hp": list(c.get("header_path") or []), "rp": list(c.get("row_path") or [])})
    notes_of = {}
    for r in refs:
        notes_of[r["parts"][0]] = scope_footnotes(r, {p: by_row.get(p, {}) for p in r["parts"]})
    cards = []
    for t in sorted(tables, key=lambda t: (t.get("page") or 0, t["table_id"])):
        tid, sid, page = t["table_id"], t.get("section_id") or "", t.get("page")
        role = roles[tid]
        ref = ref_of_part.get(tid, {})
        whole, per_row = notes_of.get(role["table"], ([], {}))
        rows = []
        for r, cs in sorted(by_row.get(tid, {}).items()):
            if all(c.get("is_header") for c in cs) or all(c.get("group_row") for c in cs):
                continue                                    # heading rows and group labels: carried by the paths
            rows.append((r, cs))
        values = [c["text"] for _, cs in rows for c in cs if c["hp"] and c["rp"]]
        grid = (len(rows) >= settings.grid_min_rows and bool(values)
                and sum(map(is_number, values)) / len(values) >= settings.grid_min_numeric)
        section_title = (t.get("section_title") or ref.get("section_title") or "").strip()
        title = (t.get("caption") or t.get("title") or ref.get("caption") or ref.get("title") or section_title or
                 f"table {tid}").strip()
        short = first_words(title, settings.title_words).rstrip(" …")
        intro = (ref.get("reference_text") or "").strip()
        intro = "" if not intro or intro == title else first_words(intro, settings.intro_words)
        example = role["role"] == "example"
        label = (f"Section {sid} - {section_title} - {short} (table, page {page})"
                 if section_title and short != section_title else f"Section {sid} - {short} (table, page {page})")
        for r, cs in rows:
            paths = [c["hp"] for c in cs if c["hp"]]
            shared = _common_prefix(paths) if len(paths) > 1 else []
            row_label = " > ".join(max((c["rp"] for c in cs), key=len))
            parts = []
            for c in cs:
                head = " / ".join(first_words(h, settings.header_words) for h in c["hp"][len(shared):])
                parts.append(f"{head}: {c['text']}" if head else c["text"])
            heading = ("Worked example: " if example else "") + title + (
                " — " + " / ".join(shared) if shared and " / ".join(shared) != title else "")
            notes = per_row.get((tid, r), []) + whole
            text = heading + "\n" + (f"{row_label}: " if row_label else "") + "; ".join(parts)
            text += (f"\nIntroduced in the policy as: {intro}" if intro else "")
            text += (f"\nNotes: {first_words('; '.join(notes), settings.notes_words)}" if notes else "")
            cards.append({"section_id": f"{sid}#{tid}", "section_title": short, "clause": None, "label": label,
                          "path": "", "group": tid, "index": "grid" if grid else "table", "kind": "table_row",
                          "is_table": True, "pages": [page], "block_ids": [], "text": text,
                          "embed_text": f"{label}: {text}", "table_id": tid, "row": r, "role": role["role"],
                          "probe": f"{row_label or parts[0]} {short}"})
    return [c for c in cards if c["index"] == "table"] + [c for c in cards if c["index"] == "grid"]


# ---------------------------------------------------------------------------------------------------- database, files
_ACTIVE = "JOIN rag.active_versions v ON v.version_id = {a}.version_id AND v.doc_id = %(doc)s"


def read_tables(db, doc_id: str) -> tuple:
    """(tables, cells, refs) of the active version of a document, read-only (``db``: ``database.Database``)."""
    tables = db.query_dicts(f"""
        SELECT t.table_id, t.section_id, t.record->>'section_title' AS section_title, t.caption, t.title,
               t.pages[1] AS page
          FROM rag.tables t {_ACTIVE.format(a='t')} ORDER BY t.pages[1], t.table_id""", {"doc": doc_id})
    cells = db.query_dicts(f"""
        SELECT c.table_id, c.row, c.col, coalesce(c.text, '') AS text, coalesce(c.is_header, false) AS is_header,
               coalesce(c.header_path, '{{}}') AS header_path, coalesce(c.row_path, '{{}}') AS row_path,
               coalesce((c.record->>'group_row')::boolean, false) AS group_row
          FROM rag.table_cells c {_ACTIVE.format(a='c')} ORDER BY c.table_id, c.row, c.col""", {"doc": doc_id})
    refs = db.query_dicts(f"""
        SELECT r.parts, r.caption, r.title, r.record->>'section_title' AS section_title, r.reference_text,
               coalesce(r.record->'footnotes', '[]'::jsonb) AS footnotes
          FROM rag.table_refs r {_ACTIVE.format(a='r')}""", {"doc": doc_id})
    for r in refs:
        if isinstance(r["footnotes"], str):
            r["footnotes"] = json.loads(r["footnotes"])
    return tables, cells, refs


def write_chunk_file(text_chunk_file, cards: Sequence[dict], out_path) -> Path:
    """The text chunk file with the cards appended (``idx`` continues after the text chunks)."""
    doc = json.loads(Path(text_chunk_file).read_text(encoding="utf-8"))
    n = len(doc["chunks"])
    rows = list(doc["chunks"]) + [dict(c, idx=n + i) for i, c in enumerate(cards)]
    out = Path(out_path)
    out.write_text(json.dumps({**doc, "chunks": rows}, ensure_ascii=False), encoding="utf-8")
    return out
