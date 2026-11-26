"""The SQL step's check of a query against the question (v3.5, P2 of the roadmap).

The answer model writes the query, as before (prompt ``sql_v2``). After the query runs, code compares the query's
choices with what the question states, using only the schema card: printed column headings, listed values, printed
ages and the places of each tier. No word lists: a value code cannot read exactly is never guessed.

- **Amounts.** Every value column the query reads must be selected by an amount the question writes (10 lakh,
  Rs 2 crore, ₹50,00,000, 25,000): the column's printed heading names it, or its printed condition (<= 20 lakh)
  covers it. An amount the query uses to pick a row (a deductible) must be written in the question, and is not used
  again for a column. With no amount in the question, the query must read every column it could mean, not choose one.
- **Plans.** A plan the query reads (a row or a column) must be the plan the question names; the longest name the card
  knows wins (Optima Secure Global is not Optima Secure). With no plan named, the query must not choose one.
- **Ages.** The age row the query reads must be the printed row of a number the question writes (months count as
  years: 6 months is age 0). With no number, the query must not choose an age.
- **Places.** A place the policy lists, or a tier the question states, must get its tier. Any other place is the
  model's judgement: the rows are kept, and a note written by code says which tier was used and why.
- **Other listed values** (individual or family floater): the value's words, or the model's reading of other words,
  which a note then states.

A disagreement the model can correct raises ``SqlProblem`` (the step's existing correction loop shows it the reason);
a value the tables do not print raises ``NotInTables``. Nothing here calls the model or the database after start-up.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Sequence, Set, Tuple

LAKH, CRORE = 100_000, 10_000_000


class NotInTables(Exception):
    """The question asks for a value the tables do not print (decided by code)."""


# ---------------------------------------------------------------------------------------------------- reading text
def norm(text: str) -> str:
    """Lower case; ₹ / Rs. / INR written alike; dashes and spaces collapsed."""
    t = (text or "").lower().replace("₹", " rs ")
    t = re.sub(r"\b(rs|inr)\.?(?=\s|\d|$)", " rs ", t)
    t = re.sub(r"[‐‑‒–—-]", "-", t)
    return re.sub(r"\s+", " ", t).strip(" .,;:?!")


def words_of(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", norm(str(text).replace("_", " ")))


_UNIT = {"lakh": LAKH, "lakhs": LAKH, "lac": LAKH, "lacs": LAKH, "l": LAKH, "crore": CRORE, "crores": CRORE,
         "cr": CRORE, "k": 1000, "thousand": 1000}
_NUMBER = re.compile(r"(rs\s*)?(\d+(?:,\d{2,3})*(?:\.\d+)?)\s*(lakhs?|lacs?|l|crores?|cr|k|thousand)?(?![a-z])")


def amounts_in(text: str) -> List[int]:
    """Amounts in rupees the text writes: 10 lakh, 5L, ₹50,00,000, Rs 25,000, 1.5 lakh, 2 Cr, 25k. A bare number counts
    from 1,000 (so an age is not an amount)."""
    out = []
    for m in _NUMBER.finditer(norm(text)):
        cur, num, unit = m.group(1), m.group(2), m.group(3)
        value = Decimal(num.replace(",", ""))
        if unit:
            value *= _UNIT[unit]
        elif not cur and value < 1000:
            continue
        if value == value.to_integral_value():
            out.append(int(value))
    return list(dict.fromkeys(out))


def range_of(text: str, amounts: Sequence[int]) -> Tuple[Optional[int], Optional[int]]:
    """(low, high) the stated amounts span: two or more amounts, or one with 'upwards' / 'up to' (None: open end)."""
    if len(amounts) >= 2:
        return min(amounts), max(amounts)
    if len(amounts) == 1:
        t = norm(text)
        if re.search(r"\b(upwards?|onwards?|and above|or more|and more|at least)\b", t):
            return amounts[0], None
        if re.search(r"\b(up to|upto|or less|at most)\b", t):
            return None, amounts[0]
    return None, None


def ages_in(text: str) -> Set[int]:
    """Every number the text writes that is not an amount, as whole years (a number of months becomes its years)."""
    out = set()
    t = norm(text)
    for m in _NUMBER.finditer(t):
        cur, num, unit = m.group(1), m.group(2), m.group(3)
        if cur or unit or "," in num or "." in num or int(num) >= 1000:
            continue
        months = re.match(r"\s*-?\s*months?\b", t[m.end():])
        out.add(int(num) // 12 if months else int(num))
    return out


def amount_words(n: Optional[int]) -> str:
    if n is None:
        return ""
    if n >= CRORE and n % (CRORE // 100) == 0:
        return f"{_dec(Decimal(n) / CRORE)} crore"
    if n >= LAKH and n % 1000 == 0:
        return f"{_dec(Decimal(n) / LAKH)} lakh"
    return f"Rs {n:,}"


def _dec(d: Decimal) -> str:
    s = format(d.normalize(), "f")
    return s.rstrip("0").rstrip(".") if "." in s else s


# ---------------------------------------------------------------------------------------------------- the card
_VALUE_HEADING = re.compile(r"\(([^()]*\bin rs\.?|[^()]*%)\)\s*$", re.I)
_CONDITION = re.compile(r"(<=|>=|<|>)\s*(\d[\d,]*(?:\.\d+)?)\s*(lakhs?|crores?)?", re.I)
_CATCH_ALL = re.compile(r"\brest of\b|\ball other|\bothers?\b", re.I)


@dataclass
class ValueColumn:
    """A column holding the values asked for (heading ends '(... in Rs.)' or '(... %)'), with what its printed
    heading says selects it: amounts it names, a condition on an amount, plans it names."""
    name: str
    heading: str
    amounts: List[int] = field(default_factory=list)
    condition: Optional[Tuple[str, int]] = None
    plans: List[str] = field(default_factory=list)

    @property
    def by_amount(self) -> bool:
        return bool(self.amounts or self.condition)

    def covers(self, low: Optional[int], high: Optional[int]) -> bool:
        """Whether an amount (low == high) or a range is selected by this column's heading."""
        if self.amounts:
            return any((low is None or a >= low) and (high is None or a <= high) for a in self.amounts)
        if self.condition:
            op, v = self.condition
            if op in ("<=", "<"):
                return low is None or (low <= v if op == "<=" else low < v)
            return high is None or (high >= v if op == ">=" else high > v)
        return True


def parse_value_heading(name: str, heading: str) -> ValueColumn:
    """'Sum Insured 500,000 (premium in Rs.)' -> amounts [500000]; '20, 25 & 50 Lakhs (Limit of Cover in Rs.)' ->
    [2000000, 2500000, 5000000]; 'Optima A & Optima B / Base SI <= 20 lakhs (discount %)' -> plans [A, B], condition
    ('<=', 2000000); 'Optima C (discount %)' -> plans [C]."""
    sel = _VALUE_HEADING.sub("", heading).strip()
    col = ValueColumn(name, heading)
    for part in (p.strip() for p in sel.split("/")):
        m = _CONDITION.search(part)
        if m:
            unit = (LAKH if (m.group(3) or "").lower().startswith("lakh") else
                    CRORE if (m.group(3) or "").lower().startswith("crore") else 1)
            col.condition = (m.group(1), int(Decimal(m.group(2).replace(",", "")) * unit))
        elif re.search(r"\d", part):
            unit = LAKH if re.search(r"\blakhs?\b", part, re.I) else CRORE if re.search(r"\bcrores?\b", part, re.I) else 1
            col.amounts = [int(Decimal(n.replace(",", "")) * unit) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", part)]
        else:
            col.plans = [p.strip() for p in re.split(r",\s*|\s*&\s*|\s+and\s+", part) if p.strip()]
    return col


def _heading(c: dict) -> Optional[str]:
    com = c.get("comment") or ""
    return com.split(":", 1)[1].strip() if com.lower().startswith("printed heading:") else None


def age_row(values: Sequence[str], n: int) -> Optional[str]:
    """The printed age row for an age: '45', '72*' (printed with a mark), '>=90*' (every age from 90)."""
    for v in values:
        if re.fullmatch(r"\D*" + str(n) + r"\D*", str(v)) and not re.match(r"\s*[<>]", str(v)):
            return v
    for v in values:
        m = re.match(r"\s*(>=|>)\s*(\d+)", str(v))
        if m and (n >= int(m.group(2)) if m.group(1) == ">=" else n > int(m.group(2))):
            return v
    return None


@dataclass
class TableFacts:
    name: str
    words: str                                                  # how notes name the table
    values: Dict[str, ValueColumn] = field(default_factory=dict)
    choices: Dict[str, List[str]] = field(default_factory=dict)  # short text columns with listed values (plan, ...)
    ages: Dict[str, List[str]] = field(default_factory=dict)     # age column -> printed rows
    row_amounts: Dict[str, List[int]] = field(default_factory=dict)  # integer columns that pick a row (deductible)
    places: Dict[str, int] = field(default_factory=dict)         # listed place -> tier
    catch_all: Optional[int] = None                              # the tier of every place not listed
    tier_text: Dict[int, str] = field(default_factory=dict)      # tier -> its places as printed


# ---------------------------------------------------------------------------------------------------- the check
_FILLER = frozenset("a an the my our your his her their of for on in with under plan plans policy policies".split())


def _phrase_spans(name: str, text: str) -> List[Tuple[int, int]]:
    pat = r"\b" + r"\s+".join(re.escape(w) for w in words_of(name)) + r"\b"
    return [m.span() for m in re.finditer(pat, " ".join(words_of(text)))]


class QuestionCheck:
    def __init__(self, card, db=None):
        """``card``: the SQL step's ``SchemaCard``; ``db``: read once for the places of each tier."""
        self.tables: Dict[str, TableFacts] = {}
        for name, t in card.tables.items():
            f = TableFacts(name, re.sub(r"_table$", " table", name).replace("_", " "))
            names = {c["name"] for c in t["columns"]}
            for c in t["columns"]:
                n, h, vals = c["name"], _heading(c), c.get("values")
                if h and _VALUE_HEADING.search(h):
                    f.values[n] = parse_value_heading(n, h)
                elif n == "age" or (h or "").lower().startswith("age"):
                    f.ages[n] = [str(v) for v in vals or []]
                elif n in ("tier", "tier_cities", "page"):
                    continue
                elif c["type"] in ("text", "character varying") and vals and all(len(str(v)) <= 40 for v in vals):
                    f.choices[n] = [str(v) for v in vals]
                elif c["type"] in ("integer", "bigint", "smallint", "numeric") and vals and h:
                    f.row_amounts[n] = [int(v) for v in vals if v is not None]   # a printed column (a deductible)
            if db is not None and {"tier", "tier_cities"} <= names:
                for tier, text in db.query(f'SELECT DISTINCT tier, tier_cities FROM {_schema()}."{name}" ORDER BY 1', None):
                    f.tier_text[int(tier)] = text
                    if _CATCH_ALL.search(text or ""):
                        f.catch_all = int(tier)
                    else:
                        for p in re.split(r",\s*|\s+and\s+|\s*&\s*", text or ""):
                            if p.strip():
                                f.places[p.strip()] = int(tier)
            self.tables[name] = f
        plans = [v for f in self.tables.values() for c, vals in f.choices.items() if c == "plan" for v in vals]
        plans += [p for f in self.tables.values() for v in f.values.values() for p in v.plans]
        self.plans = list(dict.fromkeys(plans))
        firsts = {words_of(p)[0] for p in self.plans if words_of(p)}
        self.brand = next(iter(firsts)) if len(firsts) == 1 else None   # a word every plan name starts with

    # ------------------------------------------------------------------ what the question states
    def short_forms(self) -> Dict[str, str]:
        """Each plan's name without the word every plan name starts with ("Super Secure"), when no other plan's name
        holds it ("Secure" is in several names, so it names none of them)."""
        out = {}
        for p in self.plans:
            w = words_of(p)
            if not (self.brand and len(w) > 1 and w[0] == self.brand):
                continue
            form = " ".join(w[1:])
            if not any(q != p and _phrase_spans(form, q) for q in self.plans):
                out[form] = p
        return out

    def named_plans(self, question: str) -> List[str]:
        """The plans the question names, longest names first: a full name, or a short form that names one plan. A name
        inside a longer one that is also found does not count."""
        found = []
        for form, p in [(p, p) for p in self.plans] + list(self.short_forms().items()):
            found += [(s, e, p) for s, e in _phrase_spans(form, question)]
        keep = [x for x in found if not any(y != x and y[0] <= x[0] and x[1] <= y[1] and y[1] - y[0] > x[1] - x[0]
                                             for y in found)]
        return list(dict.fromkeys(p for _, _, p in sorted(keep)))

    def listed_places(self, f: TableFacts, question: str) -> List[str]:
        found = [(s, e, p) for p in f.places for s, e in _phrase_spans(p, question)]
        keep = [x for x in found if not any(y != x and y[0] <= x[0] and x[1] <= y[1] for y in found)]
        return list(dict.fromkeys(p for _, _, p in keep))

    # ------------------------------------------------------------------ the check
    def check(self, question: str, checked) -> List[str]:
        """Notes for the answer (assumptions code could not check exactly). SqlProblem: the query disagrees with the
        question; NotInTables: the question asks for a value the tables do not print."""
        from .sql_step import SqlProblem
        notes: List[str] = []
        filters, read = checked.filters, set(checked.read)
        stated = amounts_in(question)
        used = set()
        named = self.named_plans(question)
        for t in checked.tables:
            f = self.tables.get(t)
            if f is None:
                continue
            # rows picked by an amount (a deductible): the amount must be written in the question
            for col in f.row_amounts:
                for _, v in filters.get(col, []):
                    if v in stated:
                        used.add(v)
                    elif v in f.row_amounts[col]:
                        raise SqlProblem(f"the query picks {col} = {v}, but the question writes no such amount")
            # listed text values: the plan named; other values in the question's words, or noted
            for col, listed in f.choices.items():
                for _, v in filters.get(col, []):
                    if col == "plan" or str(v) in self.plans:
                        self._plan_row(f, col, str(v), named)
                    else:
                        notes += self._choice(f, col, str(v), listed, question)
            # ages
            for col, rows in f.ages.items():
                for _, v in filters.get(col, []):
                    self._age(f, col, str(v), rows, question)
            # places
            notes += self._tiers(f, filters, question)
        remaining = [a for a in stated if a not in used]
        for t in checked.tables:
            f = self.tables.get(t)
            if f is not None:
                self._value_columns(f, read & set(f.values), remaining, named, question)
        return list(dict.fromkeys(notes))

    def _plan_row(self, f: TableFacts, col: str, v: str, named: List[str]):
        from .sql_step import SqlProblem
        if not named:
            raise SqlProblem(f"the query reads {col} = '{v}', which the question does not name. If the question names "
                             "a plan the table does not list, reply not_in_tables; if it names no plan, read every "
                             "plan or reply ask")
        if v in named:
            return
        missing = [n for n in named if n not in f.choices.get(col, [])]
        if missing and not any(n in f.choices.get(col, []) for n in named):
            raise NotInTables(f"The {f.words} has no plan " + " or ".join(f"'{n}'" for n in missing) + ": it lists "
                              + " and ".join(f"'{x}'" for x in f.choices.get(col, [])) + ".")
        raise SqlProblem(f"the query reads {col} = '{v}', but the question names " + " and ".join(f"'{n}'" for n in named))

    def _choice(self, f: TableFacts, col: str, v: str, listed: List[str], question: str) -> List[str]:
        from .sql_step import SqlProblem
        q = set(words_of(question))
        said = lambda x: set(words_of(x)) <= q
        if said(v):
            return []
        others = [x for x in listed if x != v and said(x)]
        if others:
            raise SqlProblem(f"the query reads {col} = '{v}', but the question says '{others[0].replace('_', ' ')}'")
        return [f"The {f.words} was read for {col.replace('_', ' ')}: {v.replace('_', ' ')}, which the question "
                "states in other words."]

    def _age(self, f: TableFacts, col: str, v: str, rows: List[str], question: str):
        from .sql_step import SqlProblem
        ages = ages_in(question)
        if not ages:
            raise SqlProblem(f"the question writes no age, but the query reads {col} = '{v}': reply ask for the age")
        printed = {a: age_row(rows, a) for a in ages}
        if v in printed.values():
            return
        if not any(printed.values()):
            raise NotInTables(f"The {f.words} prints no row for age " + " or ".join(str(a) for a in sorted(ages)) + ".")
        raise SqlProblem(f"the query reads {col} = '{v}', but the question's age reads the printed row "
                         + " or ".join(f"'{r}'" for r in printed.values() if r))

    def _tiers(self, f: TableFacts, filters, question: str) -> List[str]:
        from .sql_step import SqlProblem
        if not f.tier_text:
            return []
        chosen: Optional[Set[int]] = None
        for op, v in filters.get("tier", []):
            chosen = (chosen or set()) | {int(v)} if str(v).lstrip("-").isdigit() else chosen
        by_text: Set[int] = set()
        for op, v in filters.get("tier_cities", []):
            pat = norm(str(v)).replace("%", "").strip() if op == "like" else None
            by_text |= {t for t, text in f.tier_text.items()
                        if (pat is not None and pat in norm(text)) or (pat is None and norm(text) == norm(str(v)))}
        if filters.get("tier_cities"):
            chosen = by_text if chosen is None else (chosen | by_text)
        if not chosen:
            return []
        stated = {int(m.group(1)) for m in re.finditer(r"\btier\s*-?\s*(\d+)\b", norm(question))}
        listed = self.listed_places(f, question)
        listed_tiers = {f.places[p] for p in listed}
        if listed and not (chosen & (listed_tiers | stated)):
            raise SqlProblem("the question names " + ", ".join(f"{p} (Tier {f.places[p]})" for p in listed)
                             + f", but the query reads only Tier {', '.join(map(str, sorted(chosen)))}")
        notes = []
        for t in sorted(chosen):
            if t in stated or t in listed_tiers:
                continue
            if stated and not listed and t not in stated:
                raise SqlProblem(f"the question states Tier {', '.join(map(str, sorted(stated)))}, but the query "
                                 f"reads Tier {t}")
            if t == f.catch_all:
                notes.append("The question's place is not one the policy lists for "
                             + ", ".join(f"Tier {x}" for x in sorted(set(f.places.values())))
                             + f", so Tier {t} ({f.tier_text.get(t, '')}) premiums were used.")
            else:
                notes.append(f"Tier {t} premiums were used for a place the policy does not list by name: this assumes "
                             f"it lies in one of the places of Tier {t} ({f.tier_text.get(t, '')}).")
        return notes

    def _value_columns(self, f: TableFacts, read: Set[str], amounts: List[int], named: List[str], question: str):
        from .sql_step import SqlProblem
        if not read:
            return
        low, high = range_of(question, amounts)
        span = low is not None or high is not None

        def by_amount_ok(c: ValueColumn) -> bool:
            if not c.by_amount or not amounts:
                return True
            return any(c.covers(a, a) for a in amounts) or (span and c.covers(low, high))

        plan_named = [n for n in named if any(n in c.plans for c in f.values.values())]

        def by_plan_ok(c: ValueColumn) -> bool:
            return not c.plans or not plan_named or any(n in c.plans for n in plan_named)

        meant = {n for n, c in f.values.items() if by_amount_ok(c) and by_plan_ok(c)}
        wrong = sorted(read - meant)
        if wrong:
            if not meant:
                what = (" or ".join(amount_words(a) for a in amounts) if amounts and any(
                    f.values[w].by_amount for w in wrong) else " or ".join(plan_named))
                raise NotInTables(f"The {f.words} prints no column for {what}.")
            c = f.values[wrong[0]]
            stated = ([amount_words(a) for a in amounts] if not by_amount_ok(c) else []) + \
                     ([f"'{n}'" for n in plan_named] if not by_plan_ok(c) else [])
            raise SqlProblem(f"the query reads the column '{c.heading}', but the question states " + " and ".join(stated)
                             + ": read the column whose printed heading covers it")
        open_amount = not amounts and any(f.values[n].by_amount for n in read)
        open_plan = not plan_named and any(f.values[n].plans for n in read)
        if (open_amount or open_plan) and read < meant:
            left = "an amount" if open_amount else "a plan"
            raise SqlProblem(f"the question states no {'Sum Insured' if open_amount else 'plan'}, but the query reads "
                             f"only some of the {f.words}'s columns (choosing {left}): read every column it could "
                             "mean, or reply ask")


def _schema() -> str:
    from .sql_step import SCHEMA
    return SCHEMA
