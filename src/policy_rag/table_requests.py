"""Table requests (v3.5, Critical 3): the model states what a question asks of the numeric tables; code checks every
value, finds the rows and does the arithmetic. The model never writes SQL (``sql.mode: request``).

The split follows one rule: the model judges meaning, code checks what can be checked exactly, and the answer shows
any assumption.

- **The model** reads the question and replies with a request (``prompts.REQUEST_TEMPLATES``): the table, the words
  that ask for the value (``asks``), and for each field the question states, the words it uses (``stated``) and the
  value they mean (``value``): a listed plan or policy type, an age, an amount, or the listed place the question's
  place lies in ("none" for a place the policy does not list). It can ask for two lookups and their difference, sum,
  or an amount after a percentage; or list a column. Or it replies ``text_only``.
- **Code** (``RequestResolver``) checks that every quoted phrase is in the question, and resolves each field against
  the schema card: a listed value, a printed age row, the column whose printed heading names an amount (or whose
  printed condition covers it), the column whose heading names a plan, the row of a printed amount. A value the
  tables do not hold gives "not in the tables". A place the policy lists is resolved by code; any other place takes
  the model's choice from the listed places, and the answer says which tier was used and why.
- **Rows for a field the question leaves open** are counted: up to ``sql.ask_rows`` are returned, more make the step
  ask for that field, with a question written by code, after one short check that the question really asks for a
  value of that table (``sql.confirm_ask``).
- **Code builds the query** (values are the card's own values, never words from the question), the SQL validator
  checks it as before, and the read-only runner runs it. Differences, sums and amounts after a discount are computed
  in Decimal.

A request that fails a check is shown to the model with the reason and corrected once (``sql.request_retries``);
after that the step reports "failed" and the text passages answer alone.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Sequence, Tuple

LAKH, CRORE = 100_000, 10_000_000
COMBINES = ("difference", "sum", "after_discount")


class NotInTables(Exception):
    """A value the tables do not hold (decided by code; no retry)."""


# ---------------------------------------------------------------------------------------------------- reading words
def norm(text: str) -> str:
    """Lower case; ₹ / Rs. / INR written alike; dashes and spaces collapsed."""
    t = (text or "").lower().replace("₹", " rs ")
    t = re.sub(r"\brs\.?(?=\s|\d|$)", " rs ", t)
    t = re.sub(r"[‐‑‒–—-]", "-", t)
    return re.sub(r"\s+", " ", t).strip(" .,;:?!")


def words_of(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", norm(text))


def in_question(stated: str, question: str) -> bool:
    """The quoted words are in the question: as a phrase, or (when the model dropped or reordered a small word) every
    word, and then every amount or number they hold must be one the question writes ("5 lakh" is not in "a 5-year-old
    with 25 lakh cover")."""
    s, q = norm(stated), norm(question)
    if s and s in q:
        return True
    ws = words_of(stated)
    if not ws or not set(ws) <= set(words_of(question)):
        return False
    return set(amounts_in(stated, bare_from=1)) <= set(amounts_in(question, bare_from=1))


_UNIT = {"lakh": LAKH, "lakhs": LAKH, "lac": LAKH, "lacs": LAKH, "l": LAKH, "crore": CRORE, "crores": CRORE,
         "cr": CRORE, "k": 1000, "thousand": 1000}
_AMOUNT = re.compile(r"(rs\s*)?(\d+(?:,\d{2,3})*(?:\.\d+)?)\s*(lakhs?|lacs?|l|crores?|cr|k|thousand)?(?![a-z])")


def amounts_in(text: str, bare_from: int = 1000) -> List[int]:
    """Amounts in rupees written in ``text``: 10 lakh, 5L, ₹50,00,000, Rs 25,000, 1.5 lakh, 2 Cr, 25k. A bare number
    counts only from ``bare_from`` (so an age is not an amount)."""
    out = []
    for m in _AMOUNT.finditer(norm(text)):
        cur, num, unit = m.group(1), m.group(2), m.group(3)
        value = Decimal(num.replace(",", ""))
        if unit:
            value *= _UNIT[unit]
        elif not cur and value < bare_from:
            continue
        if value == value.to_integral_value():
            out.append(int(value))
    return out


def marked_amounts(text: str) -> List[int]:
    """Amounts written with a unit (lakh, crore, L, Cr, k) or a rupee mark, so they are surely amounts."""
    out = []
    for m in _AMOUNT.finditer(norm(text)):
        if m.group(1) or m.group(3):
            out += amounts_in(m.group(0), bare_from=1)
    return list(dict.fromkeys(out))


def range_of(text: str) -> Tuple[Optional[int], Optional[int]]:
    """(low, high) of an amount range in ``text``: two amounts, or one with 'upwards' / 'up to' (None: open end)."""
    found = amounts_in(text)
    t = norm(text)
    if len(found) >= 2:
        return min(found), max(found)
    if len(found) == 1:
        if re.search(r"\b(upwards?|onwards?|above|or more|and more|more than|at least|over|beyond)\b", t):
            return found[0], None
        if re.search(r"\b(up to|upto|below|under|less than|at most)\b", t):
            return None, found[0]
        return found[0], found[0]
    return None, None


def age_in(text: str) -> Optional[int]:
    """An age in whole years: '45-year-old' -> 45, 'aged 93' -> 93, '6-month-old' -> 0, '18 months' -> 1."""
    t = norm(text)
    m = re.search(r"(\d{1,3})\s*-?\s*(months?|mos?)\b", t)
    if m:
        return int(m.group(1)) // 12
    m = re.search(r"\d{1,3}", t)
    return int(m.group(0)) if m else None


def amount_words(n: Optional[int]) -> str:
    if n is None:
        return ""
    if n >= CRORE and n % (CRORE // 100) == 0:
        return f"{_dec(Decimal(n) / CRORE)} crore"
    if n >= LAKH and n % 1000 == 0:
        return f"{_dec(Decimal(n) / LAKH)} lakh"
    from .sql_step import group_inr
    return "Rs " + group_inr(n)


def _dec(d: Decimal) -> str:
    s = format(d.normalize(), "f")
    return s.rstrip("0").rstrip(".") if "." in s else s


# ---------------------------------------------------------------------------------------------------- the tables
_VALUE_HEADING = re.compile(r"\(([^()]*\bin rs\.?|[^()]*%)\)\s*$", re.I)
_CONDITION = re.compile(r"(<=|>=|<|>)\s*(\d[\d,]*(?:\.\d+)?)\s*(lakhs?|crores?)?", re.I)
_CATCH_ALL = re.compile(r"\brest of\b|\ball other|\bothers?\b", re.I)


@dataclass
class ValueColumn:
    """A column that holds the values asked for (its heading ends with '(... in Rs.)' or '(... %)'), with what its
    printed heading says selects it: amounts it names, a condition on an amount, plans it names."""
    name: str
    heading: str
    amounts: List[int] = field(default_factory=list)
    condition: Optional[Tuple[str, int]] = None
    plans: List[str] = field(default_factory=list)

    def covers(self, low: Optional[int], high: Optional[int]) -> bool:
        """Whether an amount (low == high) or a range is selected by this column's heading."""
        if self.amounts:
            return any((low is None or a >= low) and (high is None or a <= high) for a in self.amounts)
        if self.condition:
            op, v = self.condition
            if op in ("<=", "<"):                        # the range reaches below the limit
                return low is None or (low <= v if op == "<=" else low < v)
            return high is None or (high >= v if op == ">=" else high > v)
        return True


def parse_value_heading(name: str, heading: str) -> ValueColumn:
    """'Sum Insured 500,000 (premium in Rs.)' -> amounts [500000]; '20, 25 & 50 Lakhs (Limit of Cover in Rs.)' ->
    [2000000, 2500000, 5000000]; 'Optima A & Optima B / Base SI <= 20 lakhs (discount %)' -> plans [A, B], condition
    ('<=', 2000000); 'Optima C (discount %)' -> plans [C]."""
    sel = _VALUE_HEADING.sub("", heading).strip()
    col = ValueColumn(name, heading)
    parts = [p.strip() for p in sel.split("/")]
    for part in parts:
        m = _CONDITION.search(part)
        if m:
            unit = LAKH if (m.group(3) or "").lower().startswith("lakh") else CRORE if (m.group(3) or "").lower().startswith("crore") else 1
            col.condition = (m.group(1), int(Decimal(m.group(2).replace(",", "")) * unit))
        elif re.search(r"\d", part):
            unit = LAKH if re.search(r"\blakhs?\b", part, re.I) else CRORE if re.search(r"\bcrores?\b", part, re.I) else 1
            col.amounts = [int(Decimal(n.replace(",", "")) * unit) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", part)]
        else:
            col.plans = [p.strip() for p in re.split(r",\s*|\s*&\s*|\s+and\s+", part) if p.strip()]
    return col


@dataclass
class Field:
    """One thing a question can state about a table. kind: choice (a listed text value), place (a listed place, its
    tier), age (a printed age row), row_amount (a printed amount that picks a row), amount (picks value columns by
    their printed amounts or conditions), plan_column (picks value columns by the plans their headings name)."""
    name: str
    kind: str
    column: Optional[str] = None
    values: list = field(default_factory=list)


@dataclass
class TableModel:
    name: str
    description: str
    fields: Dict[str, Field]
    value_columns: List[ValueColumn]
    key_columns: List[str]                     # selected with the values, so every value can be traced to its row
    places: Dict[str, int] = field(default_factory=dict)      # listed place -> tier
    catch_all: Optional[int] = None                           # the tier for every place not listed
    tier_text: Dict[int, str] = field(default_factory=dict)   # tier -> its places as printed

    def listable(self) -> List[str]:
        return [c for c in self.key_columns] + (["tier_cities"] if self.tier_text else [])


def _heading(c: dict) -> Optional[str]:
    com = c.get("comment") or ""
    return com.split(":", 1)[1].strip() if com.lower().startswith("printed heading:") else None


def table_models(card, db) -> Dict[str, TableModel]:
    """What each table lets a question state, read from the schema card (and, for the places of each tier, one query)."""
    out = {}
    for name, t in card.tables.items():
        cols = t["columns"]
        values, keys, fields_ = [], [], {}
        for c in cols:
            h = _heading(c)
            if h and _VALUE_HEADING.search(h):
                values.append(parse_value_heading(c["name"], h))
        value_names = {v.name for v in values}
        names = {c["name"] for c in cols}
        model = TableModel(name, t.get("description") or "", fields_, values, keys)
        for c in cols:
            n, h = c["name"], _heading(c)
            if n in value_names or n == "page":
                continue
            if n in ("tier", "tier_cities") and {"tier", "tier_cities"} <= names:
                if n == "tier":
                    keys.append(n)
                continue
            vals = c.get("values")
            if n == "age" or (h or "").lower().startswith("age"):
                fields_["age"] = Field("age", "age", n, list(vals or []))
                keys.append(n)
            elif c["type"] in ("text", "character varying") and vals and all(len(str(v)) <= 40 for v in vals):
                fields_[n] = Field(n, "choice", n, list(vals))
                keys.append(n)
            elif c["type"] in ("integer", "bigint", "smallint", "numeric") and vals:
                fname = re.sub(r"_(inr|rs|amount)$", "", n)
                fields_[fname] = Field(fname, "row_amount", n, [int(v) for v in vals])
                keys.append(n)
        if {"tier", "tier_cities"} <= names:
            for tier, text in db.query(f'SELECT DISTINCT tier, tier_cities FROM policy_sql."{name}" ORDER BY 1', None):
                model.tier_text[int(tier)] = text
                if _CATCH_ALL.search(text or ""):
                    model.catch_all = int(tier)
                else:
                    for p in re.split(r",\s*|\s+and\s+|\s*&\s*", text or ""):
                        if p.strip():
                            model.places[p.strip()] = int(tier)
            fields_["place"] = Field("place", "place", "tier", sorted(model.places))
        if any(v.amounts or v.condition for v in values):
            fields_["sum_insured"] = Field("sum_insured", "amount")
        plans = list(dict.fromkeys(p for v in values for p in v.plans))
        if plans and "plan" not in fields_:
            fields_["plan"] = Field("plan", "plan_column", None, plans)
        out[name] = model
    return out


def tables_text(models: Dict[str, TableModel], card) -> str:
    """The TABLES part of the request prompt: what each table holds and the fields a question can state."""
    lines = []
    for m in models.values():
        lines.append(f"{m.name}: {m.description}")
        for f in m.fields.values():
            if f.kind == "choice":
                lines.append(f"  - {f.name}: one of " + ", ".join(f"'{v}'" for v in f.values))
            elif f.kind == "place":
                tiers = "; ".join(f"Tier {t}: " + ", ".join(p for p, tt in m.places.items() if tt == t)
                                  for t in sorted(set(m.places.values())))
                lines.append(f"  - place: the listed place the question's place lies in ({tiers}), or \"none\" for a "
                             f"place in none of them (Tier {m.catch_all}: {m.tier_text.get(m.catch_all, '')}). A tier "
                             "stated as such (\"Tier 2\") is a place too.")
            elif f.kind == "age":
                lines.append(f"  - age: in whole years (printed ages: {_age_span(f.values)})")
            elif f.kind == "row_amount":
                lines.append(f"  - {f.name}: an amount in rupees (printed: "
                             + ", ".join(amount_words(v) for v in f.values) + ")")
            elif f.kind == "amount":
                printed = []
                for v in m.value_columns:
                    if v.amounts:
                        printed.append(" / ".join(amount_words(a) for a in v.amounts))
                    elif v.condition:
                        printed.append(f"{v.condition[0]} {amount_words(v.condition[1])}"
                                       + (f" (for {', '.join(v.plans)})" if v.plans else ""))
                lines.append("  - sum_insured: an amount in rupees, or a range (printed: "
                             + ", ".join(dict.fromkeys(printed)) + ")")
            elif f.kind == "plan_column":
                lines.append("  - plan: one of " + ", ".join(f"'{v}'" for v in f.values))
        lister = [c for c in m.listable() if c not in ("tier",)]
        if lister:
            lines.append("  columns that can be listed: " + ", ".join(lister))
        notes = list(dict.fromkeys(n for p in card.tables[m.name].get("printed", []) for n in p.get("footnotes") or [] if n))
        for n in notes:
            lines.append(f"  footnote: {n}")
    return "\n".join(lines)


def _age_span(values: Sequence[str]) -> str:
    """'0 to 65; 66* to 89*; >=90*': plain ages, ages printed with a mark, and open-ended rows."""
    plain = sorted(int(v) for v in values if re.fullmatch(r"\d+", str(v)))
    marked = sorted((int(re.search(r"\d+", str(v)).group(0)), str(v)) for v in values
                    if re.fullmatch(r"\d+\D+", str(v)))
    other = [str(v) for v in values if not re.fullmatch(r"\d+\D*", str(v))]
    parts = ([f"{plain[0]} to {plain[-1]}"] if plain else []) + \
            ([f"{marked[0][1]} to {marked[-1][1]}"] if marked else []) + other
    return "; ".join(parts)


# ---------------------------------------------------------------------------------------------------- the reply
def parse_request(reply: str) -> dict:
    """The first JSON object with an "action" in the reply, checked for shape (not yet against the question)."""
    from .sql_step import SqlProblem
    text = re.sub(r"<think>.*?</think>|</?think>", "", reply or "", flags=re.S)
    decoder = json.JSONDecoder(strict=False)
    obj = None
    for m in re.finditer(r"\{", text):
        try:
            cand, _ = decoder.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(cand, dict) and "action" in cand:
            obj = cand
            break
    if obj is None:
        raise SqlProblem('the reply is not one JSON object with an "action"')
    action = str(obj.get("action") or "").strip().lower()
    if action == "text_only":
        return {"action": "text_only", "reason": str(obj.get("reason") or "").strip()}
    if action != "lookup":
        raise SqlProblem(f'unknown action {action!r}: use "lookup" or "text_only"')
    lookups = obj.get("lookups")
    if not isinstance(lookups, list) or not lookups or not all(isinstance(x, dict) for x in lookups):
        raise SqlProblem('a "lookup" reply needs "lookups": a list of at least one {"table": ..., "values": {...}}')
    combine = obj.get("combine")
    combine = str(combine).strip().lower() if combine not in (None, "", "null", "none") else None
    if combine is not None and combine not in COMBINES:
        raise SqlProblem(f'unknown combine {combine!r}: use null, ' + ", ".join(f'"{c}"' for c in COMBINES))
    return {"action": "lookup", "asks": str(obj.get("asks") or "").strip(), "lookups": lookups, "combine": combine}


# ---------------------------------------------------------------------------------------------------- resolving
@dataclass
class Resolved:
    """One lookup, checked: the table, the row filters, the value columns, what it lists, the fields left open."""
    table: str
    filters: Dict[str, object] = field(default_factory=dict)
    value_columns: List[str] = field(default_factory=list)
    list_column: Optional[str] = None
    open_fields: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    amount: Optional[Tuple[Optional[int], Optional[int], str]] = None     # (low, high, words) picking value columns
    plan: Optional[Tuple[str, str]] = None                                # (plan, words) picking value columns


_FILLER = frozenset("a an the my our your his her their of for on in with under plan plans policy policies cover "
                    "option type".split())


def _content(text) -> set:
    return {w for w in words_of(str(text).replace("_", " ")) if w not in _FILLER}


def _phrase_in(name: str, text: str) -> bool:
    return re.search(r"\b" + re.escape(norm(str(name).replace("_", " "))) + r"\b", norm(str(text).replace("_", " "))) is not None


def _stated_words(lk: dict) -> str:
    vals = lk.get("values") or {}
    return " | ".join(str(v.get("stated") if isinstance(v, dict) else v) for v in vals.values() if v not in (None, "", {}))


def _in_stated_range(amount: int, lookups: list) -> bool:
    for lk in lookups:
        for v in (lk.get("values") or {}).values():
            if isinstance(v, dict) and v.get("stated"):
                low, high = range_of(str(v["stated"]))
                if low != high and (low is None or amount >= low) and (high is None or amount <= high):
                    return True
    return False


def _field_words(f: Field) -> str:
    return f.name.replace("_", " ")


class RequestResolver:
    def __init__(self, models: Dict[str, TableModel]):
        self.models = models
        # every name of the same kind the card knows: listed values and the plans the headings name
        self.known_names = sorted({str(v) for m in models.values() for f in m.fields.values()
                                   if f.kind in ("choice", "plan_column") for v in f.values}, key=len)
        self.plan_names = sorted({str(v) for m in models.values() for f in m.fields.values() if f.name == "plan"
                                  for v in f.values}, key=len)
        self.names_by_field: Dict[str, List[str]] = {}
        for m in models.values():
            for f in m.fields.values():
                if f.kind in ("choice", "plan_column"):
                    self.names_by_field.setdefault(f.name, [])
                    self.names_by_field[f.name] += [str(v) for v in f.values
                                                    if str(v) not in self.names_by_field[f.name]]

    def resolve(self, request: dict, question: str) -> List[Resolved]:
        """Every lookup checked against the question and the card. SqlProblem: the model must correct the request;
        NotInTables: the tables do not hold the value."""
        from .sql_step import SqlProblem
        asks = request.get("asks") or ""
        if not asks:
            raise SqlProblem('"asks" must quote the words of the question that ask for the value')
        if not in_question(asks, question):
            raise SqlProblem(f'"asks" ({asks!r}) is not in the question: copy the words exactly')
        out = [self._lookup(lk, question) for lk in request["lookups"]]
        self._accounted_for(request, question)
        combine = request.get("combine")
        if combine and len(out) != 2:
            raise SqlProblem(f'"{combine}" needs exactly two lookups (found {len(out)})')
        return out

    def _accounted_for(self, request: dict, question: str):
        """Every value the question states that code can see must be in the request: a plan (or policy type) name the
        card knows, a place the policy lists, an amount written with lakh, crore or a rupee mark. A request that
        leaves one out would read rows for a case the question did not ask about (a plan the tables do not hold read
        as the plans they do hold)."""
        from .sql_step import SqlProblem
        missing = []
        lookups = request["lookups"]
        stated_all = " | ".join(_stated_words(lk) for lk in lookups)
        for amount in marked_amounts(question):
            if amount not in amounts_in(stated_all, bare_from=1) and not _in_stated_range(amount, lookups):
                missing.append(f"the amount {amount_words(amount)} (state it in the field it belongs to)")
        for lk in lookups:
            m = self.models[str(lk.get("table")).split(".")[-1]]
            stated = lk.get("values") or {}
            for f in m.fields.values():
                if f.name in stated and stated[f.name] not in (None, "", {}):
                    continue
                if f.kind in ("choice", "plan_column"):
                    names = list(f.values) + ([n for n in self.plan_names] if f.name == "plan" else [])
                    found = [x for x in names if _phrase_in(x, question)]
                    found = [x for x in found if not any(x != y and _phrase_in(x, y) for y in found)]
                    if found:
                        missing.append(f"{f.name} {', '.join(repr(str(x)) for x in dict.fromkeys(found))} in the lookup "
                                       f"on {m.name}")
                elif f.kind == "place":
                    found = [p for p in m.places if _phrase_in(p, question)]
                    found = [p for p in found if not any(p != q and _phrase_in(p, q) for q in found)]
                    if found:
                        missing.append(f"place {', '.join(repr(p) for p in found)} in the lookup on {m.name}")
        if missing:
            raise SqlProblem("the question states " + "; ".join(missing) + ". Give every value the question states, "
                             "even one the tables may not hold: code decides that")

    def _lookup(self, lk: dict, question: str) -> Resolved:
        from .sql_step import SqlProblem
        table = str(lk.get("table") or "").strip()
        if table.startswith("policy_sql."):
            table = table.split(".", 1)[1]
        if table not in self.models:
            raise SqlProblem(f"unknown table {table!r}: use one of {', '.join(self.models)}")
        m = self.models[table]
        values = lk.get("values") or {}
        if not isinstance(values, dict):
            raise SqlProblem('"values" must be an object of fields')
        unknown = [k for k in values if k not in m.fields]
        if unknown:
            raise SqlProblem(f"{table} has no field {', '.join(map(repr, unknown))}: use " + ", ".join(m.fields))
        r = Resolved(table)
        stated = {}
        for name, v in values.items():
            if v in (None, "", {}):
                continue
            if not isinstance(v, dict):
                v = {"stated": str(v), "value": v}
            words = str(v.get("stated") or "").strip()
            if not words:
                raise SqlProblem(f'{name}: give "stated", the words of the question for it')
            if not in_question(words, question):
                raise SqlProblem(f'{name}: {words!r} is not in the question: copy the words exactly')
            stated[name] = (words, v.get("value"))
        for name, f in m.fields.items():
            if name not in stated:
                if f.kind in ("choice", "place", "age", "row_amount"):
                    r.open_fields.append(name)
                continue
            words, value = stated[name]
            getattr(self, "_" + f.kind)(m, f, words, value, r)
        r.value_columns = self._value_columns(m, r)
        lst = lk.get("list")
        if lst:
            if lst not in m.listable():
                raise SqlProblem(f"{table}: {lst!r} cannot be listed; use one of {', '.join(m.listable())}")
            r.list_column, r.value_columns = lst, []
        return r

    # each field kind
    def _choice(self, m, f, words, value, r):
        r.filters[f.column] = self._named(m, f, words, value)

    def _named(self, m, f, words, value):
        """A listed name (a plan, a policy type) for the question's words. Code first: the longest name the card
        knows anywhere that the words contain (so "Optima Secure Global" is not "Optima Secure"), then a short form
        ("Super Secure"). Otherwise the model's value, when listed, unless the words name something else of the same
        kind (they share a word with the known names but match none: "Optima Restore")."""
        said = _content(words)
        names = self.names_by_field.get(f.name, [str(v) for v in f.values])     # plans with plans, types with types
        known = [k for k in names if _phrase_in(k, words)]
        if known:
            best = max(known, key=lambda k: len(_content(k)))
            listed = [v for v in f.values if _content(v) == _content(best)]
            if listed:
                return listed[0]
            raise NotInTables(f"The {_table_words(m)} has no {_field_words(f)} {words!r}: it lists "
                              + " and ".join(f"'{v}'" for v in f.values) + ".")
        short = [v for v in f.values if said and said <= _content(v)]
        if len(short) == 1:
            return short[0]
        chosen = [v for v in (short or f.values) if _content(v) == _content(str(value or ""))]
        overlaps = said & set().union(*[_content(k) for k in names]) if names else set()
        if chosen and not (overlaps and not short):
            return chosen[0]
        raise NotInTables(f"The {_table_words(m)} has no {_field_words(f)} {words!r}: it lists "
                          + " and ".join(f"'{v}'" for v in f.values) + ".")

    def _place(self, m, f, words, value, r):
        from .sql_step import SqlProblem
        t = norm(words)
        tier = re.search(r"\btier\s*-?\s*(\d+)\b", t)
        if tier and int(tier.group(1)) in m.tier_text:
            r.filters["tier"] = int(tier.group(1))
            return
        if _CATCH_ALL.search(t) and m.catch_all is not None:
            r.filters["tier"] = m.catch_all
            return
        listed = [p for p in m.places if re.search(r"\b" + re.escape(norm(p)) + r"\b", t)]
        if listed:
            p = max(listed, key=len)
            r.filters["tier"] = m.places[p]
            r.notes.append(f"{words} is listed by the policy in Tier {m.places[p]}, so Tier {m.places[p]} premiums apply.")
            return
        choice = str(value or "").strip()
        if choice.lower() in ("none", "no", "other", "") or (m.catch_all is not None and _CATCH_ALL.search(choice)):
            if m.catch_all is None:
                raise NotInTables(f"The {_table_words(m)} lists no tier for {words!r}.")
            r.filters["tier"] = m.catch_all
            r.notes.append(f"{words} is not one of the places the policy lists for "
                           + ", ".join(f"Tier {t}" for t in sorted(set(m.places.values())))
                           + f", so Tier {m.catch_all} ({m.tier_text.get(m.catch_all, '')}) premiums apply.")
            return
        named = [p for p in m.places if norm(p) == norm(choice)]
        if not named:
            raise SqlProblem(f'place: {choice!r} is not a listed place: give the listed place {words!r} lies in, or '
                             '"none"')
        p = named[0]
        r.filters["tier"] = m.places[p]
        r.notes.append(f"{words} is taken to be in {p}, which the policy lists in Tier {m.places[p]}, so Tier "
                       f"{m.places[p]} premiums apply.")

    def _age(self, m, f, words, value, r):
        from .sql_step import SqlProblem
        n = age_in(words)
        if n is None:                    # "senior citizen", "my parents": a range or several people, not one age
            raise SqlProblem(f"age: {words!r} gives no age in years or months: leave age out (code asks for it)")
        row = _age_row(f.values, n)
        if row is None:
            raise NotInTables(f"The {_table_words(m)} prints no row for age {n}.")
        r.filters[f.column] = row

    def _row_amount(self, m, f, words, value, r):
        from .sql_step import SqlProblem
        found = amounts_in(words, bare_from=1)
        n = found[0] if found else None
        if n is None:
            try:
                n = int(value)
            except (TypeError, ValueError):
                raise SqlProblem(f"{f.name}: give the amount in rupees for {words!r}") from None
        if n not in f.values:
            raise NotInTables(f"The {_table_words(m)} prints no {f.name} of {amount_words(n)} (printed: "
                              + ", ".join(amount_words(v) for v in f.values) + ").")
        r.filters[f.column] = n

    def _amount(self, m, f, words, value, r):
        from .sql_step import SqlProblem
        low, high = range_of(words)
        if low is None and high is None:
            if isinstance(value, list) and len(value) == 2:
                low, high = [None if v is None else int(v) for v in value]
            else:
                try:
                    low = high = int(value)
                except (TypeError, ValueError):
                    raise SqlProblem(f"sum_insured: give the amount in rupees for {words!r}") from None
        r.amount = (low, high, words)

    def _plan_column(self, m, f, words, value, r):
        r.plan = (self._named(m, f, words, value), words)

    def _value_columns(self, m: TableModel, r: Resolved) -> List[str]:
        cols = list(m.value_columns)
        plan, si = r.plan, r.amount
        if plan:
            cols = [c for c in cols if not c.plans or any(norm(p) == norm(plan[0]) for p in c.plans)]
            cols = [c for c in cols if c.plans] or cols
        if si:
            low, high, words = si
            kept = [c for c in cols if c.covers(low, high)]
            selective = [c for c in cols if c.amounts or c.condition]
            if selective and not any(c.amounts or c.condition for c in kept):
                what = amount_words(low) if low == high else words
                raise NotInTables(f"The {_table_words(m)} prints no column for a Sum Insured of {what}.")
            cols = kept
        if not cols:
            raise NotInTables(f"The {_table_words(m)} prints no column for this.")
        return [c.name for c in cols]


def _age_row(values: Sequence[str], n: int) -> Optional[str]:
    """The printed age row for an age: '45', '72*' (printed with a mark), '>=90*' (a row for every age from 90)."""
    for v in values:
        if re.fullmatch(r"\D*" + str(n) + r"\D*", str(v)) and not re.match(r"\s*[<>]", str(v)):
            return v
    for v in values:
        m = re.match(r"\s*(>=|>)\s*(\d+)", str(v))
        if m and (n >= int(m.group(2)) if m.group(1) == ">=" else n > int(m.group(2))):
            return v
    return None


def _table_words(m: TableModel) -> str:
    return {"premium_table": "premium table", "checkup_limit_table": "check-up limit table",
            "deductible_discount_table": "deductible discount table"}.get(m.name, m.name.replace("_", " "))


# ---------------------------------------------------------------------------------------------------- queries
def build_select(m: TableModel, r: Resolved, limit: int) -> str:
    """The query for one lookup. Every value in it is one of the card's own values (never the question's words)."""
    from .sql_step import literal
    where = " AND ".join(f'"{c}" = {literal(v)}' for c, v in r.filters.items())
    if r.list_column:
        col = f'"{r.list_column}"'
        return (f'SELECT DISTINCT {col} FROM policy_sql."{m.name}"' + (f" WHERE {where}" if where else "")
                + f" ORDER BY {col} LIMIT {int(limit)}")
    cols = list(dict.fromkeys(m.key_columns + r.value_columns + ["page"]))
    order = ", ".join(f'"{c}"' for c in m.key_columns) or '"page"'
    return (f"SELECT {', '.join(f'{chr(34)}{c}{chr(34)}' for c in cols)} FROM policy_sql.\"{m.name}\""
            + (f" WHERE {where}" if where else "") + f" ORDER BY {order} LIMIT {int(limit)}")


def build_count(m: TableModel, r: Resolved) -> str:
    """Rows and distinct values of each open field, for choosing what to ask."""
    from .sql_step import literal
    where = " AND ".join(f'"{c}" = {literal(v)}' for c, v in r.filters.items())
    counts = ["count(*)"] + [f'count(DISTINCT "{m.fields[f].column}")' for f in r.open_fields]
    return f'SELECT {", ".join(counts)} FROM policy_sql."{m.name}"' + (f" WHERE {where}" if where else "")


ASK_TEXT = {"age": "How old is the person to be insured?",
            "place": "Which city or town do you live in? The premium depends on it."}


def ask_text(m: TableModel, field_name: str) -> str:
    f = m.fields[field_name]
    if field_name in ASK_TEXT:
        return ASK_TEXT[field_name]
    if f.kind == "row_amount":
        return f"Which {f.name} amount? The policy prints " + ", ".join(amount_words(v) for v in f.values) + "."
    names = [str(v).replace("_", " ") for v in f.values]
    return f"Which {f.name.replace('_', ' ')}: " + (", ".join(names[:-1]) + " or " + names[-1] if len(names) > 1
                                                       else names[0]) + "?"


def confirm_prompt(question: str, descriptions: Sequence[str]) -> str:
    tables = "\n".join(f"- {d}" for d in descriptions)
    return (f"A user of a health insurance policy asked:\n{question.strip()}\n\nThe policy has this table:\n{tables}\n\n"
            "Is the user asking for a value from this table? Reply with one word: yes or no.")


def parse_yes_no(reply: str) -> Optional[bool]:
    text = re.sub(r"<think>.*?</think>|</?think>", "", reply or "", flags=re.S).strip().lower()
    m = re.search(r"\b(yes|no)\b", text)
    return None if m is None else m.group(1) == "yes"


def combine_values(combine: str, parts: List[Tuple[str, Decimal, str]]) -> dict:
    """``parts``: (label, value, kind) for the two lookups, in request order. Returns a calculation for the block."""
    from .sql_step import money, _plain
    (la, a, ka), (lb, b, kb) = parts
    if combine == "difference":
        hi, lo = ((la, a), (lb, b)) if a >= b else ((lb, b), (la, a))
        show = money if ka == kb == "amount" else _plain
        return {"kind": "difference", "value": hi[1] - lo[1],
                "text": f"Difference: {hi[0]} ({show(hi[1])}) minus {lo[0]} ({show(lo[1])}) = {show(hi[1] - lo[1])}"}
    if combine == "sum":
        show = money if ka == kb == "amount" else _plain
        return {"kind": "total", "value": a + b, "text": f"Total: {la} ({show(a)}) plus {lb} ({show(b)}) = {show(a + b)}"}
    amount, pct = ((la, a), (lb, b)) if ka == "amount" else ((lb, b), (la, a))
    off = amount[1] * pct[1] / 100
    return {"kind": "after_percent", "value": amount[1] - off,
            "text": f"{amount[0]} {money(amount[1])} less {_plain(pct[1])}% ({pct[0]}): {money(amount[1])} x "
                    f"(1 - {_plain(pct[1])}/100) = {money(amount[1] - off)} (taken off: {money(off)})"}
