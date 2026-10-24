"""Plan-table check for the my:Optima Secure policy (evaluation only; specific to this document).

Reads every (plan, row, year, amount) statement in an answer, from running text and markdown tables, and compares
it with the plan illustration (Section 2.1): Base Sum Insured Rs. 10 lakh, no claims, values in lakh. The plan
names and rows below are this policy's; another policy needs its own table checker (see docs/extending.md).
"""
import re

# Plan illustration (Section 2.1): Base Sum Insured Rs. 10 lakh, no claims; values in lakh
PLAN_TABLE = {}
for _plan, _single, _all, _sb, _plus in [("suraksha", (10, 11, 12), (20, 21, 22), 0, (0, 1, 2)),
                                         ("secure", (20, 25, 30), (30, 35, 40), 10, (0, 5, 10)),
                                         ("super", (30, 35, 40), (40, 45, 50), 20, (0, 5, 10))]:
    for _y in (1, 2, 3):
        PLAN_TABLE.update({(_plan, "single", _y): _single[_y - 1], (_plan, "all", _y): _all[_y - 1],
                           (_plan, "secure_benefit", _y): _sb, (_plan, "plus", _y): _plus[_y - 1],
                           (_plan, "base", _y): 10, (_plan, "restore", _y): 10})


# ---------------------------------------------------------------- plan-table values
_PLAN_RES = [("global_plus", r"(?:optima\s+)?secure\s+global\s+plus|global\s+plus"),
             ("global", r"(?:optima\s+)?secure\s+global|\bglobal\b(?!\s+(?:health|cover))"),
             ("super", r"(?:optima\s+)?super[\s-]*secure"),
             ("suraksha", r"(?:optima\s+)?suraksha"),
             ("secure", r"optima\s+secure(?!\s+(?:global|benefit|policy))|\bsecure\s+plan|(?<=for )secure\b(?!\s+benefit)"
                        r"|(?<=for the )secure\b(?!\s+benefit)|(?<=, )secure\b(?!\s+benefit)|(?<=and )secure\b(?!\s+benefit)"
                        r"|(?m:^[ \t\-*•]*(?:\d+\.\s*)?secure\b(?!\s+benefit))")]
_ROW_RES = [("secure_benefit", r"secure\s+benefit"), ("plus", r"plus\s+benefit|cumulative\s+bonus"),
            ("restore", r"restore"), ("base", r"base\s+sum\s+insured"),
            ("single", r"single[\s-]+(?:hospitali[sz]ation[\s-]+)?claim|per[\s-]+claim|any\s+one\s+claim|one\s+claim"),
            ("all", r"all\s+(?:the\s+)?(?:hospitali[sz]ation\s+)?claims|aggregate|total\s+(?:of\s+all\s+)?claims"
                    r"|(?:annual|yearly)\s+(?:limit|maximum)")]
_YEAR_RES = [("all", r"\b(?:in\s+)?(?:all|each\s+of\s+the)\s+(?:three\s+|3\s+)?(?:policy\s+)?years\b|\bacross\s+(?:all\s+)?(?:the\s+)?(?:three\s+)?years\b"),
             ("num", r"\byear\s*[-‐]?\s*(1|2|3|one|two|three)\b|\b(first|second|third|1st|2nd|3rd)\s+(?:policy\s+)?year\b")]
_YEAR_WORD = {"1": 1, "one": 1, "first": 1, "1st": 1, "2": 2, "two": 2, "second": 2, "2nd": 2,
              "3": 3, "three": 3, "third": 3, "3rd": 3}
_AMOUNT_RES = [r"(?:(?:₹|\brs\.?|\binr)\s*)?(\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?)\b",
               r"(?:₹|\brs\.?|\binr)\s*(\d{1,3}(?:,\d{2,3}){2,}|\d{6,})(?![\d,])"]
_PLAN_NAME = {"suraksha": "Optima Suraksha", "secure": "Optima Secure", "super": "Optima Super Secure",
              "global": "Optima Secure Global", "global_plus": "Optima Secure Global Plus"}
_ROW_NAME = {"single": "single claim", "all": "all claims in a year", "secure_benefit": "Secure Benefit",
             "plus": "Plus Benefit / Cumulative Bonus", "base": "Base Sum Insured", "restore": "Restore Benefit"}

def _events(text: str):
    """Plan, row, year and amount mentions with their positions (longer plan names masked first)."""
    ev, work = [], text
    for name, pat in _PLAN_RES:
        for m in re.finditer(pat, work, re.I):
            ev.append((m.start(), m.end(), "plan", name))
        work = re.sub(pat, lambda m: " " * len(m.group(0)), work, flags=re.I)
    for name, pat in _ROW_RES:
        ev += [(m.start(), m.end(), "row", name) for m in re.finditer(pat, text, re.I)]
    for kind, pat in _YEAR_RES:
        for m in re.finditer(pat, text, re.I):
            if kind == "all":
                ev.append((m.start(), m.end(), "year", "all"))
            else:
                ev.append((m.start(), m.end(), "year", _YEAR_WORD[(m.group(1) or m.group(2)).lower()]))
    taken = []
    for k, pat in enumerate(_AMOUNT_RES):
        for m in re.finditer(pat, text, re.I):
            if any(a <= m.start() < b for a, b in taken):
                continue
            num = float(m.group(1).replace(",", ""))
            val = (num * 100 if m.group(2).lower().startswith("cr") else num) if k == 0 else num / 1e5
            taken.append((m.start(), m.end()))
            ev.append((m.start(), m.end(), "amount", round(val, 3)))
    return sorted(ev)

def _first(text: str, kind: str):
    return next((v for _, _, k, v in _events(text) if k == kind), None)

def _prose_claims(text: str, state: dict):
    """Claims (plan, row, year, lakh) from running text; state carries plan/row/year headings."""
    ev = _events(text)
    breaks = [m.end() for m in re.finditer(r"\n|;|(?<=[.!?])\s+", text)]
    seg_of = lambda pos: sum(b <= pos for b in breaks)
    amounts = [e for e in ev if e[2] == "amount"]
    claims = []
    seg_has_amount = {seg_of(a[0]) for a in amounts}
    headings = [(x[0], x[3]) for x in ev if x[2] == "year" and seg_of(x[0]) not in seg_has_amount]   # "Year 2:" lines
    for n, (s, e, _, val) in enumerate(amounts):
        nxt = amounts[n + 1][0] if n + 1 < len(amounts) else len(text)
        if re.match(r"\s*(?:more|extra|additional|higher|lower|less|greater)\b", text[e:e + 20], re.I) or re.search(
                r"\b(?:additional|extra|difference of|more than|less than|by|increase of)\s*(?:a\s+)?(?:₹|rs\.?|inr)?\s*$",
                text[max(0, s - 30):s], re.I):
            continue                                   # a difference ("Rs. 10 lakh more"), not a table value
        seg_end = min([b for b in breaks if b > e] + [len(text)])
        stop = min(nxt, seg_end)
        after = [x for x in ev if e <= x[0] < stop and x[2] != "amount"]
        lead = re.match(r"[\s,(]*(?:for|under|in|of)\s+(?:the\s+)?", text[e:stop], re.I)
        # plan: "for <plan>[, <plan> and <plan>]" right after the amount, else the latest plan before it
        plans = []
        if lead:
            pos = e + lead.end()
            for x in after:
                if x[2] == "plan" and x[0] <= pos + 2:
                    plans.append(x[3])
                    tail = re.match(r"\s*(?:,|and|&)?\s*(?:the\s+)?", text[x[1]:stop])
                    pos = x[1] + (tail.end() if tail else 0)
        if not plans:
            before = [x for x in ev if x[2] == "plan" and x[1] <= s]
            plans = [before[-1][3]] if before else ([state["plan"]] if state.get("plan") else [])
        # row: "for a single claim" right after the amount, else the latest row mention before it
        row = None
        rlead = re.match(r"\s*(?:for|per|on|in)\s+(?:a\s+|the\s+|any\s+|each\s+)?", text[e:stop], re.I)
        if rlead:
            row = next((x[3] for x in after if x[2] == "row" and x[0] <= e + rlead.end() + 1), None)
        if row is None:                          # same line first, then a heading line ("Secure Benefit:") above it
            line_start = text.rfind("\n", 0, s) + 1
            same_line = [x for x in ev if x[2] == "row" and line_start <= x[0] and x[1] <= s]
            head_rows = [x for x in ev if x[2] == "row" and x[1] <= s and not any(
                a[0] >= text.rfind("\n", 0, x[0]) + 1 and a[0] < (text.find("\n", x[0]) % (len(text) + 1)) for a in amounts)]
            row = same_line[-1][3] if same_line else (head_rows[-1][3] if head_rows else state.get("row"))
        # year: "in Year 2" / "(Year 2)" / "in all three years" right after, else earlier in the sentence,
        # else a heading such as "Year 2:"
        year = None
        ylead = re.match(r"[\s,(\-]*(?:in|for|during|of|from)?\s*(?:the\s+)?", text[e:stop], re.I)
        for x in after:
            if x[2] == "year" and x[0] <= e + ylead.end() + 1 and not re.match(r"\s*[:\-–|]", text[x[1]:x[1] + 3]):
                year = x[3]
                break
        if year is None:
            same = [x for x in ev if x[2] == "year" and x[1] <= s and seg_of(x[0]) == seg_of(s)]
            heads = [y for pos, y in headings if pos < s]
            year = same[-1][3] if same else (heads[-1] if heads else state.get("heading_year"))
        claims += [(p, row, year, val) for p in plans]
    plans_seen = [x for x in ev if x[2] == "plan"]
    rows_seen = [x for x in ev if x[2] == "row"]
    if plans_seen:
        state["plan"] = plans_seen[-1][3]
    if rows_seen:
        state["row"] = rows_seen[-1][3]
    if headings:
        state["heading_year"] = headings[-1][1]
    return claims

def _markdown_table_claims(lines, state: dict):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r if c)]
    if len(rows) < 2:
        return []
    header, claims = rows[0], []
    cols = [{"plan": _first(c, "plan"), "row": _first(c, "row"), "year": _first(c, "year")} for c in header]
    for r in rows[1:]:
        label = {"plan": _first(r[0], "plan"), "row": _first(r[0], "row"), "year": _first(r[0], "year")} if r else {}
        for j, cell in enumerate(r[1:], start=1):
            col = cols[j] if j < len(cols) else {}
            pick = lambda k: label.get(k) or col.get(k) or state.get("heading_year" if k == "year" else k)
            for _, _, kind, val in _events(cell):
                if kind == "amount" and pick("plan"):
                    claims.append((pick("plan"), pick("row"), pick("year"), val))
    return claims

def question_state(question: str) -> dict:
    """Defaults taken from the question for answers that do not repeat them ("Rs. 45 lakh."): its plan when it
    names exactly one, its row, and its year when it names one year."""
    ev = _events(question or "")
    plans = {v for _, _, k, v in ev if k == "plan"}
    years = [v for _, _, k, v in ev if k == "year" and v != "all"]
    st = {"row": _first(question or "", "row")} if _first(question or "", "row") else {}
    if len(plans) == 1:
        st["plan"] = next(iter(plans))
    if len(set(years)) == 1:
        st["heading_year"] = years[0]
    return st

def parse_table_claims(answer: str, question: str = ""):
    """All (plan, row, year, value in lakh) statements in an answer: running text and markdown tables."""
    state, claims, block, table = question_state(question), [], [], []
    for line in (answer or "").splitlines() + [""]:
        if line.strip().startswith("|"):
            if block:
                claims += _prose_claims("\n".join(block), state)
                block = []
            table.append(line)
        else:
            if table:
                claims += _markdown_table_claims(table, state)
                table = []
            block.append(line)
    if block:
        claims += _prose_claims("\n".join(block), state)
    return claims

def table_check(answer: str, cells, question: str = "") -> dict:
    """Compare stated plan-table values with Sections 2.1-2.3. A value stated without a year counts
    as correct if it matches that plan and row in any year (reported as 'year not named')."""
    expected = {tuple(c) for c in cells}
    correct, wrong_cells, wrong, unnamed = set(), set(), [], 0
    for plan, row, year, val in parse_table_claims(answer, question):
        if plan not in ("suraksha", "secure", "super") or row is None:
            continue
        truth = {y: PLAN_TABLE.get((plan, row, y)) for y in (1, 2, 3)}
        if year is None:
            ys = [y for y, t in truth.items() if t is not None and abs(t - val) < 1e-6]
            if ys:
                unnamed += 1
                correct.update((plan, row, y) for y in ys)
            else:
                wrong.append(f"{_PLAN_NAME[plan]}, {_ROW_NAME[row]} (year not named): answer {val:g} lakh, "
                             f"policy {'/'.join(f'{t:g}' for t in truth.values())} lakh for years 1/2/3")
            continue
        for y in ((1, 2, 3) if year == "all" else (year,)):
            t = truth.get(y)
            if t is None:
                continue
            if abs(t - val) < 1e-6:
                correct.add((plan, row, y))
            else:
                wrong_cells.add((plan, row, y))
                wrong.append(f"{_PLAN_NAME[plan]}, {_ROW_NAME[row]}, Year {y}: answer {val:g} lakh, policy {t:g} lakh")
    return {"expected": len(expected), "correct": len(expected & correct), "wrong": wrong,
            "missing": [f"{_PLAN_NAME[p]}, {_ROW_NAME[r]}, Year {y}" for p, r, y in sorted(expected - correct - wrong_cells)],
            "year_not_named": unnamed}
