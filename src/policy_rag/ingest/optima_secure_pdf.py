"""my:Optima Secure prospectus PDF -> chunk file (structure-aware ingestion; specific to this document's layout).

Usage (CPU only, a few seconds; needs the ``ingest`` extra: pdfplumber, tiktoken):
    policy-rag chunk-pdf policy.pdf data/policy_chunks.json [--csv data/policy_chunks.csv]

1. Extraction  - two-column reading order, running header/footer removed, tables taken out of the
                 text flow and kept as tables, bold numbered lines recognised as headings.
2. Structure   - real section numbers from the PDF; lettered clauses (a., b., ...) inside sections;
                 exclusion codes (Excl01-Excl18).
3. Chunking    - one chunk per section when it is short; long sections split at clause boundaries;
                 one chunk per standard exclusion; tables turned into self-describing lines
                 (plan tables pivoted per plan, premium tables per plan / tier / age band).
4. Export      - one record per chunk with the fields the system reads (see FIELDS and policy_rag/corpus.py).

The page-layout constants and patterns below are tuned to this one PDF. Another policy needs its own chunker that
writes the same fields (docs/adding_a_policy.md); the rest of the system does not change.
"""
import re, json, collections
import pdfplumber

HEADER_MAX_TOP, FOOTER_MIN_TOP, MID = 100, 785, 297.6
HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s*([A-Z][^\n]*)$")
CLAUSE = re.compile(r"^([a-z])\.\s+(?=\S)")
EXCL = re.compile(r"Excl(\d{2})")


def _inside(w, bbox, pad=1.5):
    x0, top, x1, bottom = bbox
    return w["x0"] >= x0 - pad and w["x1"] <= x1 + pad and w["top"] >= top - pad and w["bottom"] <= bottom + pad


def _lines(words):
    """Group words into lines (same baseline), left to right."""
    rows = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if rows and abs(sum(x["top"] for x in rows[-1]) / len(rows[-1]) - w["top"]) <= 2.5:
            rows[-1].append(w)
        else:
            rows.append([w])
    out = []
    for r in rows:
        r.sort(key=lambda w: w["x0"])
        bold = sum("Bold" in w["fontname"] for w in r) / len(r)
        first_bold = "Bold" in r[0]["fontname"]
        out.append({"kind": "line", "text": " ".join(w["text"] for w in r), "top": r[0]["top"], "x0": r[0]["x0"],
                    "bold": bold, "first_bold": first_bold, "size": max(w["size"] for w in r)})
    return out


def read_pdf(path):
    """Elements in reading order: lines and tables, with page numbers. Each page is cut into
    horizontal bands that are either full-width or two-column; two-column bands are read left
    column first, then right column."""
    elements = []
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            words = [w for w in page.extract_words(extra_attrs=["fontname", "size"])
                     if HEADER_MAX_TOP < w["top"] < FOOTER_MIN_TOP]
            tables = [t for t in page.find_tables() if t.bbox[1] > HEADER_MAX_TOP - 5]
            words = [w for w in words if not any(_inside(w, t.bbox) for t in tables)]
            # rows that touch the gutter are full-width (F); the rows between two F rows form a
            # two-column band (C) only if both columns have text there, otherwise they are full-width too
            rows = []
            for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
                if rows and abs(rows[-1][0]["top"] - w["top"]) <= 2.5:
                    rows[-1].append(w)
                else:
                    rows.append([w])
            hit = [any(w["x0"] < MID + 6 and w["x1"] > MID - 6 for w in r) for r in rows]
            marks, k = [None] * len(rows), 0
            while k < len(rows):
                if hit[k]:
                    marks[k] = "F"; k += 1; continue
                j = k
                while j < len(rows) and not hit[j]:
                    j += 1
                seg = [w for r in rows[k:j] for w in r]
                both = any((w["x0"] + w["x1"]) / 2 < MID for w in seg) and any((w["x0"] + w["x1"]) / 2 >= MID for w in seg)
                for q in range(k, j):
                    marks[q] = "C" if both else "F"
                k = j
            items, groups = [], []
            for r, m in zip(rows, marks):                     # consecutive rows of one type = one band
                if groups and groups[-1][0] == m:
                    groups[-1][1].extend(r)
                else:
                    groups.append([m, list(r)])
            for m, ws in groups:
                parts = [ws] if m == "F" else [[w for w in ws if (w["x0"] + w["x1"]) / 2 < MID],
                                               [w for w in ws if (w["x0"] + w["x1"]) / 2 >= MID]]
                for part in parts:
                    for ln in _lines(part):
                        ln["band"] = m
                        ln["col"] = 0 if m == "F" else (0 if ln["x0"] < MID else 1)
                        items.append(ln)
            for t in tables:
                full = t.bbox[0] < MID - 10 and t.bbox[2] > MID + 10
                items.append({"kind": "table", "rows": t.extract(), "top": t.bbox[1], "x0": t.bbox[0],
                              "band": "F" if full else "C", "col": 0 if full or t.bbox[0] < MID else 1})
            items.sort(key=lambda i: i["top"])
            # group into bands of the same type, in vertical order
            bands = []
            for it in items:
                if bands and bands[-1][0] == it["band"]:
                    bands[-1][1].append(it)
                else:
                    bands.append([it["band"], [it]])
            for kind, its in bands:
                ordered = its if kind == "F" else sorted(its, key=lambda i: (i["col"], i["top"]))
                for it in ordered:
                    it["page"] = pno
                    elements.append(it)
    return elements


def _clean(cell):
    return re.sub(r"\s+", " ", (cell or "").replace("\n", " ")).strip()


def table_to_text(rows, caption=""):
    """A table as self-describing lines. Header row + data rows -> 'Header: value; ...' per row;
    header-less lists (e.g. the Excl02 disease tables) -> a plain list."""
    rows = [[_clean(c) for c in r] for r in rows if any(_clean(c) for c in r)]
    if not rows:
        return ""
    head = rows[0]
    numeric = lambda s: bool(re.fullmatch(r"[\d,.\s%]+|NIL|Covered|Not Covered|At Actuals", s or "", re.I))
    titled = (len(rows) > 1 and len(head) >= 2 and all(head) and all(len(c.split()) <= 6 and not c.endswith(".") for c in head)
              and all(len(r) == len(head) for r in rows))
    has_header = len(rows) > 1 and ((sum(numeric(c) for c in head) == 0 and sum(numeric(c) for r in rows[1:] for c in r) > 0) or titled)
    if not has_header:
        items = [c for r in rows for c in r if c]
        return (caption + ": " if caption else "") + "; ".join(items) + "."
    lines = [caption] if caption else []
    for r in rows[1:]:
        label = r[0]
        pairs = [f"{h or '-'}: {v}" for h, v in zip(head[1:], r[1:]) if v]
        lines.append(f"{label} — " + "; ".join(pairs))
    return "\n".join(lines)


def build_sections(elements):
    """Split the element stream at bold numbered headings (and Annexure headings)."""
    sections = [{"id": "0", "title": "Introduction", "pages": set(), "items": []}]
    last_num = ()
    for el in elements:
        if el["kind"] == "table" and "Gross Premium" in (el["rows"][0][0] or "") and sections[-1]["id"] != "Annexure-C":
            sections.append({"id": "Annexure-C", "title": "Premium Table (Gross Premium, excl. GST)", "pages": {el["page"]}, "items": []})
        if el["kind"] == "line" and el["first_bold"]:
            m = HEADING.match(el["text"])
            anx = re.match(r"^(Annexure|ANNEXURE)\s+([A-Z])\b\s*[-–]?\s*(.*)$", el["text"])
            if m:
                num = tuple(int(x) for x in m.group(1).split("."))
                if num > last_num and len(m.group(1).split(".")[0]) <= 2 and int(m.group(1).split(".")[0]) <= 15:
                    last_num = num
                    title = re.sub(r"\s*:\s*$", "", m.group(2)).strip()
                    sections.append({"id": m.group(1), "title": title, "pages": {el["page"]}, "items": []})
                    continue
            if anx:
                sections.append({"id": f"Annexure-{anx.group(2)}", "title": anx.group(3).strip() or f"Annexure {anx.group(2)}",
                                 "pages": {el["page"]}, "items": []})
                continue
        sections[-1]["items"].append(el)
        sections[-1]["pages"].add(el["page"])
    return sections


def section_text_units(sec):
    """Units inside a section, in order: ("text"|"table", clause_label, payload).
    Clauses start at lettered items in sequence (a., b., c. ...), so roman numerals inside a clause
    (i., ii.) are not mistaken for clause starts. Tables keep their caption (the short bold line just
    above) and absorb a continuation table that follows directly (same number of columns)."""
    units, cur, label, expect, caption = [], [], None, "a", ""
    def flush():
        nonlocal cur
        if cur:
            units.append(["text", label, " ".join(cur)])
            cur = []
    for el in sec["items"]:
        if el["kind"] == "table":
            flush()
            rows = [r for r in el["rows"]]
            prev = units[-1] if units else None
            if prev and prev[0] == "table" and not caption and len(prev[2]["rows"][-1]) == len(rows[0]):
                prev[2]["rows"] += rows                    # continuation of the previous table
            else:
                units.append(["table", label, {"caption": caption, "rows": rows}])
            caption = ""
            continue
        t = el["text"].strip()
        m = CLAUSE.match(t)
        if m and m.group(1) == expect:
            flush()
            code = EXCL.search(t)
            name = re.split(r"\s*[:–-]\s*Code|:\s", CLAUSE.sub("", t), maxsplit=1)[0].strip(" :–-")
            label = (f"Excl{code.group(1)} " if code else f"({m.group(1)}) ") + name[:70]
            expect = chr(ord(expect) + 1)
        elif el["first_bold"] and el["bold"] > 0.9 and len(t) < 60 and not re.match(r"^[a-z]\.|^[ivx]+\.", t):
            caption = t                                    # e.g. "Illnesses", "Surgical Procedures", "Year 2"
            continue
        cur.append(t)
    flush()
    # a clause whose code appears only at its end ("... Code – Excl12.") gets the code in its label
    for u in units:
        if u[0] == "text" and u[1] and u[1].startswith("("):
            c = EXCL.findall(u[2])
            if c:
                u[1] = f"Excl{c[0]} " + u[1].split(") ", 1)[1]
    return units

def code_in_label(label):
    return bool(EXCL.search(label or ""))


# ---------------------------------------------------------------- tables -> self-describing text
def _money(s):
    """'10,00,000' / '500,000' / '1,00,00,000' -> '₹10 lakh' / '₹5 lakh' / '₹1 crore' (other text unchanged)."""
    v = re.sub(r"[,\s]", "", s or "")
    if not re.fullmatch(r"\d{5,}", v):
        return _clean(s)
    n = int(v)
    if n >= 10**7:
        return f"₹{n / 10**7:g} crore"
    if n >= 10**5:
        return f"₹{n / 10**5:g} lakh"
    return f"₹{n:,}"

def _plan_table(caption, rows):
    head = [_clean(c) for c in rows[0]]
    plan_cols = [j for j, h in enumerate(head) if h.startswith("Optima")]
    first = plan_cols[0]
    out = []
    for j in plan_cols:
        vals, group = [], ""
        for r in rows[1:]:
            label = " ".join(_clean(c) for c in r[:first] if _clean(c)).strip(" ^*")
            cells = [_clean(c) for c in r[first:] if _clean(c)]
            if label and not cells:
                group = label                                  # a sub-heading row such as "Base Coverages"
                continue
            if j < len(r) and _clean(r[j]):
                vals.append(f"{label}: {_money(r[j])}")
        out.append(f"{caption + ' — ' if caption else ''}{head[j]}: " + "; ".join(vals) + ".")
    return out

def _pair_table(rows):
    items = []
    for r in rows[1:]:
        for k in range(0, len(r) - 1, 2):
            if _clean(r[k]) and _clean(r[k + 1]):
                items.append((int(re.sub(r"\D", "", _clean(r[k])) or 0), _clean(r[k + 1])))
    return [f"{n}. {name}" for n, name in sorted(items)]

def _premium_tables(rows):
    """[(plan, tier, [(age, {SI: premium})])] from the merged rows of one premium table."""
    cap = _clean(rows[0][0])
    m = re.search(r"Optima (Secure|Super Secure) Plan Gross Premium \(Excl\. GST\) - (Tier \d)\s*\(?([^)]*)\)?", cap)
    plan, tier, where = (f"Optima {m.group(1)}", m.group(2), m.group(3)) if m else (cap, "", "")
    si_row = next(r for r in rows if sum(bool(re.fullmatch(r"[\d,]{7,}", _clean(c))) for c in r) >= 5)
    sis = [_money(c) for c in si_row[1:]]
    data = []
    for r in rows[rows.index(si_row) + 1:]:
        age = _clean(r[0])
        if re.fullmatch(r"(>=)?\d{1,3}\*?", age):
            data.append((age, [_clean(c) for c in r[1:]]))
    return plan, f"{tier} ({where})" if where else tier, sis, data

def table_lines(caption, rows):
    rows = [r for r in rows if any(_clean(c) for c in r)]
    if not rows:
        return []
    head = [_clean(c) for c in rows[0]]
    if sum(h.startswith("Optima") for h in head) >= 3:
        return _plan_table(caption, rows)
    if [h.replace(" ", "") for h in head[:4]] == ["S.No.", "Item", "S.No.", "Item"]:
        return _pair_table(rows)
    numbered = [r for r in rows if len(r) >= 2 and re.fullmatch(r"\d{1,3}", _clean(r[0]) or "x")]
    if len(numbered) >= len(rows) - 1 and len(rows[0]) % 2 == 0:      # [no, item, no, item] without a header
        items = []
        for r in rows:
            for k in range(0, len(r) - 1, 2):
                if re.fullmatch(r"\d{1,3}", _clean(r[k]) or "x") and _clean(r[k + 1]):
                    items.append((int(_clean(r[k])), _clean(r[k + 1])))
        return ([caption] if caption else []) + [f"{n}. {name}" for n, name in sorted(items)]
    return [table_to_text(rows, caption)]


# ---------------------------------------------------------------- chunks
_enc = None

def tok(s):
    """Token count (tiktoken cl100k_base; its vocabulary downloads once on first use)."""
    global _enc
    if _enc is None:
        import tiktoken
        _enc = tiktoken.get_encoding("cl100k_base")
    return len(_enc.encode(s))

MAX_TOK, HARD_TOK = 320, 450

def cover_of(sid):
    if sid.startswith(("4.9", "4.10", "4.11")):
        return "global_overseas"
    if sid.startswith("4"):
        return "optional_cover"
    if sid.startswith("3"):
        return "base_cover"
    if sid.startswith("10"):
        return "exclusions"
    if sid.startswith("11"):
        return "claims"
    if sid.startswith(("2", "Annexure-A")):
        return "plans"
    if sid.startswith(("8", "14", "15", "Annexure-C")):
        return "premium_discounts"
    return "general"

def _sentences(t):
    return [p for p in re.split(r"(?<=[.;])\s+(?=[A-Z0-9(•])", t) if p.strip()]

def build_chunks(sections):
    by_id = {s["id"]: s for s in sections}
    def path(sid):
        parts, cur = [], sid
        while cur:
            if cur in by_id and by_id[cur]["items"] is not None:
                parts.append(f"{cur} {by_id[cur]['title']}" if not cur.startswith("Annexure") else f"{cur.replace('-', ' ')} – {by_id[cur]['title']}")
            cur = cur.rsplit(".", 1)[0] if "." in cur else None
        return " › ".join(reversed(parts))
    chunks = []
    def emit(sec, text, clause=None, kind="text", group=None, label_extra=""):
        p = path(sec["id"]) + (f" › {label_extra}" if label_extra else "")
        chunks.append({"idx": len(chunks), "section_id": sec["id"], "section_title": sec["title"],
                       "path": p, "clause": clause, "kind": kind, "cover": cover_of(sec["id"]),
                       "group": group or sec["id"], "pages": sorted(sec["pages"]),
                       "text": text.strip(), "token_count": tok(text), "embed_text": f"{p}: {text.strip()}",
                       "_extra": label_extra})
    for sec in sections:
        units = section_text_units(sec)
        if not units:
            continue                                          # heading-only section (11, 12, 13): title lives in the paths
        if sec["id"] == "Annexure-C":
            for _, _, tab in units:
                plan, tier, sis, data = _premium_tables(tab["rows"])
                for k in range(0, len(data), 5):
                    band = data[k:k + 5]
                    head_line = (f"Annual gross premium (₹, excl. GST), {plan}, {tier}, by Sum Insured "
                                 f"({' / '.join(sis)}):")
                    lines = [f"Age {a}: " + " / ".join(vals) for a, vals in band]
                    note = "\n* Only for renewal purposes." if any("*" in a for a, _ in band) else ""
                    emit(sec, head_line + "\n" + "\n".join(lines) + note, kind="table", group=f"Annexure-C/{plan}/{tier}",
                         label_extra=f"{plan}, {tier}, ages {band[0][0]}–{band[-1][0]}")
            continue
        excl_mode = any(u[1] and u[1].startswith("Excl") for u in units)
        if excl_mode:                                         # one chunk per standard exclusion (+ its lists)
            for kind, label, pay in units:
                if kind == "table":
                    txt = "\n".join(table_lines(pay["caption"], pay["rows"]))
                    emit(sec, txt, clause=label.split()[0] if label else None, kind="table",
                         group=f"{sec['id']}/{label.split()[0] if label else ''}", label_extra=(label or "") + " – list")
                else:
                    emit(sec, pay, clause=label.split()[0] if label else None,
                         group=f"{sec['id']}/{label.split()[0] if label else 'intro'}", label_extra=label or "")
            continue
        buf, buf_tok, n = [], 0, 0
        def flush(kind="text"):
            nonlocal buf, buf_tok, n
            if buf:
                n += 1
                emit(sec, "\n".join(buf), kind=kind, label_extra="(continued)" if n > 1 else "")
                buf, buf_tok = [], 0
        for kind, label, pay in units:
            pieces = table_lines(pay["caption"], pay["rows"]) if kind == "table" else [pay]
            if kind == "table" and tok("\n".join(pieces)) <= HARD_TOK:
                pieces = ["\n".join(pieces)]
                if buf and buf_tok + tok(pieces[0]) > HARD_TOK:
                    flush()
            for piece in pieces:
                t = tok(piece)
                if t > HARD_TOK:                              # one long paragraph: split at sentences
                    flush()
                    cur, ct = [], 0
                    for snt in _sentences(piece):
                        st = tok(snt)
                        if cur and ct + st > MAX_TOK:
                            buf, buf_tok = [" ".join(cur)], ct
                            flush()
                            cur, ct = [], 0
                        cur.append(snt); ct += st
                    buf, buf_tok = [" ".join(cur)], ct
                    continue
                if buf and buf_tok + t > MAX_TOK and not (t < 40 and buf_tok + t <= HARD_TOK):
                    flush()
                buf.append(piece); buf_tok += t
        flush()
    for c in chunks:                                          # number the pieces of each section
        c["subchunk_id"] = sum(1 for d in chunks[:c["idx"] + 1] if d["section_id"] == c["section_id"])
        c["is_table"] = c["kind"] == "table"
    return chunks


# ---------------------------------------------------------------- export
# Field order of the chunk file. The notebook needs text and section_id, and uses section_title, label, embed_text,
# clause, index ("policy" / "premium"), group and kind / is_table when present.
FIELDS = ["idx", "section_id", "section_title", "path", "clause", "kind", "cover", "group", "pages", "text",
          "token_count", "embed_text", "subchunk_id", "is_table", "label", "index"]
CSV_FIELDS = ["idx", "index", "section_id", "label", "kind", "cover", "group", "pages", "tokens", "text"]

# Fingerprints of the files behind the published results (README, "Results")
REFERENCE_PDF_SHA256 = "6a2f8822f1569a94687935275151c8ac385634d18e96fb37675f479e295979a7"
REFERENCE_JSON_SHA256 = "74aa551cf3df001813443c3d5c83279b36b57dd25a6f7092e185537725376e56"


def to_records(chunks):
    """Final chunk records. The label names the section (and the clause or table part); the embedding text is
    "label: text". Premium-table chunks (Annexure C) form their own index, searched only for cost questions."""
    out = []
    for c in chunks:
        label = f"Section {c['section_id']} - {c['section_title']}"
        if c["_extra"] and c["_extra"] != "(continued)":
            label += f" - {c['_extra']}"
        rec = dict(c, label=label, embed_text=f"{label}: {c['text']}",
                   index="premium" if c["section_id"] == "Annexure-C" else "policy")
        out.append({k: rec[k] for k in FIELDS})
    return out


def chunk_pdf(path):
    """PDF path -> list of chunk records."""
    return to_records(build_chunks(build_sections(read_pdf(path))))


def main(argv=None):
    import argparse, csv, hashlib
    from pathlib import Path
    ap = argparse.ArgumentParser(description="Chunk the my:Optima Secure PDF into a chunk file.")
    ap.add_argument("pdf", help="the policy PDF")
    ap.add_argument("out", help="output chunk file (.json)")
    ap.add_argument("--csv", help="also write a .csv copy (same chunks, fewer fields)")
    a = ap.parse_args(argv)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    recs = chunk_pdf(a.pdf)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(CSV_FIELDS)
            for r in recs:
                w.writerow([r["idx"], r["index"], r["section_id"], r["label"], r["kind"], r["cover"], r["group"],
                            ",".join(map(str, r["pages"])), r["token_count"], r["text"]])
    n_prem = sum(r["index"] == "premium" for r in recs)
    print(f"{len(recs)} chunks ({len(recs) - n_prem} policy, {n_prem} premium-table) in "
          f"{len({r['section_id'] for r in recs})} sections -> {a.out}")
    pdf_sha, out_sha = sha(a.pdf), sha(a.out)
    print(f"PDF    sha256 {pdf_sha}  ({'the PDF used for the published results' if pdf_sha == REFERENCE_PDF_SHA256 else 'a different PDF from the one used for the published results'})")
    print(f"chunks sha256 {out_sha}  ({'identical to the chunk file used for the published results' if out_sha == REFERENCE_JSON_SHA256 else 'differs from the chunk file used for the published results'})")
    if pdf_sha == REFERENCE_PDF_SHA256 and out_sha != REFERENCE_JSON_SHA256:
        print("  Same PDF, different chunks: check the pdfplumber and tiktoken versions (requirements-lock.txt).")


if __name__ == "__main__":
    main()
