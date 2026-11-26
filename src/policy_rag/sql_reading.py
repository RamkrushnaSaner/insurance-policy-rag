"""The SQL step's reading check (v3.6): the model states its reading of the question, code verifies it.

Before (``sql_checks.QuestionCheck``, v3.5), code re-read the question on its own with patterns: ages and amounts in
digits, plan names spelled exactly, any listed place anywhere in the question taken as where the person lives, every
amount taken as a Sum Insured. When those patterns misread a question, a correct query was refused.

Now the model's reply carries a ``reading``: for every value the query uses to choose rows or columns, the column,
the value and the exact words of the question it comes from::

    {"column": "plan", "value": "Optima Secure", "words": "optma secur"}
    {"column": "age", "value": "46", "words": "46 yr old"}
    {"column": "si_1000000", "value": 1000000, "words": "10 lac"}

Code checks only that reading, with the tables' own values (nothing written for one question):

1. The words are in the question (a value with no words behind it is a guess).
2. The words mean the value: a name is matched (``rapidfuzz``) against the names the tables list (plans, places,
   policy types), and must match the value clearly better than any other name; numbers are read from digits or
   number words (``text2num``), with lakh and crore as units and months turned into years.
3. The query uses the values read, and every value it uses to choose rows or columns is read from the question;
   a column chosen by an amount or a plan is checked against the amount or plan read (its printed heading).
4. A value the tables do not print (an age row, a Sum Insured column) is reported as not in the tables.

A check blocks a query only on evidence that the reading and the question, or the reading and the query, disagree.
When code cannot read the words (e.g. a number written in another language), it does not refuse: it records the
assumption as a note, and the answer shows it.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .sql_checks import NotInTables, QuestionCheck, TableFacts, ValueColumn, age_row, amount_words, amounts_in, norm

NAME_FLOOR = 80.0          # rapidfuzz ratio (0-100) a name must reach to count as named by the words
NAME_MARGIN = 8.0          # ... and by how much it must beat every other name (word for word, word_match)
WORD_FLOOR = 70.0          # two words count as the same word when this close (a typo or two)
QUOTE_FLOOR = 90.0         # partial ratio for words copied from the question with a slip ("46-yr-old" for "46 yr old")
NUMBER_SLACK = 1           # a bound may be one off a number read ("after 85": age_from > 85 or age_from >= 86)


def _fuzz():
    try:
        from rapidfuzz import fuzz
    except ImportError as e:                                       # pragma: no cover
        raise ImportError('the reading check needs rapidfuzz: pip install "policy-rag[db]"') from e
    return fuzz


def plain(text) -> str:
    """Lower case letters and digits, one space between words ('Family_Floater' -> 'family floater')."""
    return " ".join(re.findall(r"[a-z0-9]+", norm(str(text).replace("_", " "))))


def digits_for_words(text: str) -> str:
    """Number words as digits ('thirty-three' -> '33', 'ten lakh' -> '10 lakh'); text unchanged without text2num."""
    t = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", (text or "").lower())
    try:
        from text_to_num import alpha2digit
    except ImportError:                                            # pragma: no cover
        return t
    try:
        return alpha2digit(t, "en", threshold=0)
    except Exception:                                              # pragma: no cover - a parser failure is no evidence
        return t


_NUM = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)(\s*-?\s*months?\b)?")


def ages_read(words: str) -> Tuple[Set[int], bool]:
    """Whole years the words state, and whether they were given in months ('30 months' -> {2} ... see below).
    Numbers that are amounts (with lakh, crore, Rs ...) are not ages. A number of months gives both the completed
    years and the nearest year (the policy does not say which), e.g. 30 months -> {2, 3}."""
    t = digits_for_words(words)
    money = {m.span() for m in re.finditer(r"(rs\.?|₹|inr)\s*\d[\d,.]*|\d[\d,.]*\s*(lakhs?|lacs?|crores?|cr\b|k\b|l\b)",
                                           t)}
    out, months = set(), False
    for m in _NUM.finditer(t):
        if any(a <= m.start() < b for a, b in money) or "." in m.group(1):
            continue
        n = int(m.group(1))
        if m.group(2):
            months = True
            out |= {n // 12, int(n / 12 + 0.5)}
        elif n < 150:
            out.add(n)
    return out, months


def amounts_read(words: str) -> List[int]:
    """Rupee amounts the words state, from digits or number words ('ten lakh' -> 1000000)."""
    return amounts_in(digits_for_words(words))


def numbers_read(words: str) -> Set[int]:
    """Whole numbers the words write, in digits or words ('Tier one' -> {1})."""
    return {int(n) for n in re.findall(r"(?<![\d.])\d+(?![\d.])", digits_for_words(words))}


def _best_runs(words: str, names: Dict[str, Sequence[str]]) -> Tuple[List[str], Dict[str, Tuple[float, int, int]]]:
    """For each name, its best score against any run of consecutive words, and that run (first, last + 1)."""
    fuzz = _fuzz()
    toks = plain(words).split()
    out = {}
    for name, forms in names.items():
        best = (0.0, 0, 0)
        for i in range(len(toks)):
            for j in range(i + 1, min(len(toks), i + 6) + 1):
                g = " ".join(toks[i:j])
                for form in forms:
                    sc = fuzz.ratio(g, plain(form)) if plain(form) else 0.0
                    if sc > best[0]:
                        best = (sc, i, j)
        out[name] = best
    return toks, out


def word_match(span: str, form: str) -> float:
    """How well the words of ``span`` and of a name's form match, word for word (0-100): each word of the name is
    paired with at most one word of the span when they are close (``WORD_FLOOR``: a typo or two), and the score is
    the F1 of the two sides. So 'prot avalon' fits 'Port Avalon' better than 'Avalon', and 'optima secure plan' fits
    'Optima Secure' better than 'Optima Secure Global'. A span of one word written without spaces is compared whole."""
    fuzz = _fuzz()
    a, b = plain(span).split(), plain(form).split()
    if not a or not b:
        return 0.0
    if len(a) == 1 and len(b) > 1:
        return fuzz.ratio(a[0], "".join(b))
    pairs = sorted(((fuzz.ratio(x, y), i, j) for i, x in enumerate(a) for j, y in enumerate(b)), reverse=True)
    used_a, used_b, total = set(), set(), 0.0
    for sc, i, j in pairs:
        if sc < WORD_FLOOR:
            break
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            total += sc
    if not total:
        return 0.0
    recall, precision = total / len(b), total / len(a)
    return 2 * recall * precision / (recall + precision)


def names_value(words: str, names: Dict[str, Sequence[str]]) -> Tuple[str, Optional[str], List[str]]:
    """Which listed name the words name: ('named', name, [name]), ('ambiguous', None, the close names) or ('none',
    None, []). First, each name is matched (rapidfuzz ratio) against every run of consecutive words, so extra words do
    not hurt; the names that reach ``NAME_FLOOR`` are found. Then those names are compared word for word
    (``word_match``) with the whole run they cover together, so a name inside a longer one does not win when the
    words fit the longer one; the best must beat every other by ``NAME_MARGIN``."""
    toks, runs = _best_runs(words, names)
    found = {n: r for n, r in runs.items() if r[0] >= NAME_FLOOR}
    if not found:
        return "none", None, []
    span = " ".join(toks[min(r[1] for r in found.values()):max(r[2] for r in found.values())])
    final = sorted(((max(word_match(span, f) for f in names[n]), n) for n in found), reverse=True)
    top, name = final[0]
    second = final[1][0] if len(final) > 1 else 0.0
    if top - second < NAME_MARGIN:
        return "ambiguous", None, [n for sc, n in final if sc >= top - NAME_MARGIN]
    return "named", name, [name]


def parse_reading(raw) -> List[dict]:
    """The reading as a list of {column, value, words}; anything else is a reply the model must correct."""
    from .sql_step import SqlProblem
    if raw in (None, "", []):
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise SqlProblem('"reading" must be a list of {"column", "value", "words"}')
    out = []
    for item in raw:
        if not isinstance(item, dict) or not str(item.get("column") or "").strip() or not str(item.get("words") or "").strip():
            raise SqlProblem('each "reading" entry needs "column", "value" and "words" (the question\'s own words)')
        out.append({"column": str(item["column"]).strip(), "value": item.get("value"), "words": str(item["words"]).strip()})
    return out


class ReadingCheck(QuestionCheck):
    """Checks the model's reading against the question and the query (module docstring). Uses the table facts of
    ``QuestionCheck`` (listed values, printed age rows, column headings, the places of each tier)."""

    def check_reading(self, question: str, checked, reading: Sequence[dict]) -> List[str]:
        """Notes for the answer (assumptions code could not check). Raises SqlProblem when the reading disagrees with
        the question or the query, NotInTables when it asks for a value the tables do not print."""
        from .sql_step import SqlProblem
        notes: List[str] = []
        q_plain = plain(question)
        tables = [self.tables[t] for t in checked.tables if t in self.tables]
        known = {c for f in tables for c in set(f.choices) | set(f.ages) | set(f.row_amounts) | set(f.values)}
        known |= {"tier", "tier_cities", "age_from", "age_to"}
        by_col: Dict[str, List[dict]] = {}
        for item in reading:
            col = item["column"]
            if col not in known:
                raise SqlProblem(f'the reading names the column {col!r}, which the tables read do not have')
            if not self._quoted(item["words"], q_plain):
                raise SqlProblem(f"the reading gives the words {item['words']!r} for {col}, but the question does not "
                                 "write them: copy the question's own words (spelling as written)")
            by_col.setdefault("age" if col in ("age_from", "age_to") else
                              "tier" if col == "tier_cities" else col, []).append(item)
        for f in tables:
            notes += self._plans_and_choices(f, checked, by_col)
            notes += self._ages(f, checked, by_col)
            notes += self._tiers_read(f, checked, by_col)
            self._row_amounts(f, checked, by_col)
            notes += self._value_columns_read(f, checked, by_col)
        return list(dict.fromkeys(notes))

    # ------------------------------------------------------------------ 1. the words are the question's
    @staticmethod
    def _quoted(words: str, q_plain: str) -> bool:
        w = plain(words)
        if not w:
            return False
        if w in q_plain:
            return True
        return _fuzz().partial_ratio(w, q_plain) >= QUOTE_FLOOR

    # ------------------------------------------------------------------ 2-3. names: plans and other listed values
    def _name_forms(self, values: Iterable[str], col: str) -> Dict[str, List[str]]:
        short = {p: f for f, p in self.short_forms().items()} if col == "plan" else {}
        return {v: [v] + ([short[v]] if v in short else []) for v in values}

    def _plans_and_choices(self, f: TableFacts, checked, by_col) -> List[str]:
        from .sql_step import SqlProblem
        notes = []
        for col, listed in f.choices.items():
            if col == "tier_cities":
                continue
            used = [str(v) for op, v in checked.filters.get(col, []) if op == "="]
            if not used:
                continue
            values = self.plans if col == "plan" else listed
            items = by_col.get(col, [])
            if not items:
                raise SqlProblem(f"the query chooses {col} = " + ", ".join(f"'{v}'" for v in used) + ", but the reading "
                                 f"gives no words of the question for it: add them to \"reading\"; if the question does "
                                 f"not state the {col.replace('_', ' ')}, read every {col.replace('_', ' ')} or reply ask")
            named: Set[str] = set()
            for item in items:
                status, name, close = names_value(item["words"], self._name_forms(values, col))
                if status == "named":
                    named.add(name)
                elif status == "ambiguous":
                    raise SqlProblem(f"the words {item['words']!r} fit several values of {col} (" + ", ".join(
                        f"'{c}'" for c in close) + "): read every one they could mean, or reply ask")
                else:
                    raise SqlProblem(f"the words {item['words']!r} name none of the {col.replace('_', ' ')} values the "
                                     "tables list (" + ", ".join(f"'{v}'" for v in values) + "): if the question names "
                                     "one the tables do not list, reply not_in_tables; if it names none, read every "
                                     "one or reply ask")
            missing = [n for n in named if n not in listed]
            if missing and col == "plan" and not any(n in listed for n in named):
                raise NotInTables(f"The {f.words} has no plan " + " or ".join(f"'{n}'" for n in missing) + ": it lists "
                                  + " and ".join(f"'{x}'" for x in listed) + ".")
            wrong = [v for v in used if v not in named]
            if wrong:
                raise SqlProblem(f"the query reads {col} = '{wrong[0]}', but the words read for it name "
                                 + " and ".join(f"'{n}'" for n in sorted(named)))
        return notes

    # ------------------------------------------------------------------ ages
    def _ages(self, f: TableFacts, checked, by_col) -> List[str]:
        from .sql_step import SqlProblem
        notes = []
        for col, rows in f.ages.items():
            printed = [str(v) for op, v in checked.filters.get(col, []) if op == "="]
            bounds = [v for c in ("age_from", "age_to") for op, v in getattr(checked, "bounds", {}).get(c, [])]
            bounds += [v for c in ("age_from", "age_to") for op, v in checked.filters.get(c, []) if op == "="]
            if not printed and not bounds:
                continue
            items = by_col.get("age", [])
            if not items:
                raise SqlProblem("the query chooses rows by age, but the reading gives no words of the question for "
                                 "the age: add them to \"reading\"; if the question does not state the age, reply ask")
            ages: Set[int] = set()
            unread = []
            for item in items:
                got, months = ages_read(item["words"])
                if not got:
                    unread.append(item)
                    continue
                ages |= got
                if months:
                    notes.append(f"The age was read from {item['words']!r} as {min(got)} or {max(got)} (in years); the "
                                 "policy does not say whether age counts completed years." if len(got) > 1 else
                                 f"The age was read from {item['words']!r} as {min(got)}.")
            if unread and not ages:
                notes += [f"The age was read from {i['words']!r} as {i['value']} (not checked by code)." for i in unread]
                continue
            lo, hi = min(ages), max(ages)
            allowed = {age_row(rows, a) for a in (range(lo, hi + 1) if len(ages) >= 2 else ages)} - {None}
            if printed and not allowed:
                raise NotInTables(f"The {f.words} prints no row for age " + " or ".join(map(str, sorted(ages))) + ".")
            for v in printed:
                if v not in allowed:
                    raise SqlProblem(f"the query reads {col} = '{v}', but the words read for the age give "
                                     + (f"ages {lo} to {hi}" if len(ages) >= 2 else f"age {lo}")
                                     + (": the printed row" + ("s are " if len(allowed) > 1 else " is ")
                                        + ", ".join(f"'{r}'" for r in sorted(allowed, key=_age_key)) if allowed else ""))
            for b in bounds:
                if isinstance(b, (int, float)) and not any(abs(b - a) <= NUMBER_SLACK for a in ages):
                    raise SqlProblem(f"the query compares the age with {b}, but the words read for the age give "
                                     + ", ".join(map(str, sorted(ages))))
        return notes

    # ------------------------------------------------------------------ places and tiers
    def _tiers_read(self, f: TableFacts, checked, by_col) -> List[str]:
        from .sql_step import SqlProblem
        if not f.tier_text:
            return []
        chosen = self._chosen_tiers(f, checked)
        if not chosen:
            return []
        items = by_col.get("tier", [])
        if not items:
            raise SqlProblem("the query chooses the premium tier, but the reading gives no words of the question for "
                             "it (the place or the tier): add them to \"reading\"; if the question does not state the "
                             "place, read every tier or reply ask")
        places = {p: [p] for p in f.places}
        notes = []
        for item in items:
            status, place, close = names_value(item["words"], places) if places else ("none", None, [])
            stated = {n for n in numbers_read(item["words"]) if n in f.tier_text} \
                if re.search(r"\btier\b", plain(item["words"])) else set()
            if status == "named":
                if f.places[place] not in chosen:
                    raise SqlProblem(f"the words {item['words']!r} name {place} (Tier {f.places[place]}), but the query "
                                     f"reads Tier {', '.join(map(str, sorted(chosen)))}")
            elif stated:
                if not stated & chosen:
                    raise SqlProblem(f"the words {item['words']!r} state Tier {', '.join(map(str, sorted(stated)))}, but "
                                     f"the query reads Tier {', '.join(map(str, sorted(chosen)))}")
            elif status == "ambiguous":
                raise SqlProblem(f"the words {item['words']!r} fit several listed places (" + ", ".join(close)
                                 + "): give the words that name where the person lives")
            else:
                for t in sorted(chosen):
                    if t == f.catch_all:
                        notes.append(f"{item['words']} is not one of the places the policy lists for "
                                     + ", ".join(f"Tier {x}" for x in sorted(set(f.places.values())))
                                     + f", so Tier {t} ({f.tier_text.get(t, '')}) premiums were used.")
                    else:
                        notes.append(f"Tier {t} premiums were used for {item['words']}, which the policy does not list "
                                     f"by name: this assumes it lies in one of the places of Tier {t} "
                                     f"({f.tier_text.get(t, '')}).")
        return notes

    @staticmethod
    def _chosen_tiers(f: TableFacts, checked) -> Set[int]:
        chosen: Set[int] = {int(v) for op, v in checked.filters.get("tier", []) if str(v).lstrip("-").isdigit()}
        for op, v in checked.filters.get("tier_cities", []):
            pat = norm(str(v)).replace("%", "").strip() if op == "like" else None
            chosen |= {t for t, text in f.tier_text.items()
                       if (pat is not None and pat in norm(text)) or (pat is None and norm(text) == norm(str(v)))}
        return chosen

    # ------------------------------------------------------------------ amounts that pick a row (a deductible)
    def _row_amounts(self, f: TableFacts, checked, by_col):
        from .sql_step import SqlProblem
        for col in f.row_amounts:
            used = [v for op, v in checked.filters.get(col, []) if op == "="]
            if not used:
                continue
            stated = {a for item in by_col.get(col, []) for a in amounts_read(item["words"])}
            for v in used:
                if v not in stated:
                    raise SqlProblem(f"the query picks {col} = {v}, but no words of the question read for {col} state "
                                     "that amount: if the question does not state it, read every row or reply ask")

    # ------------------------------------------------------------------ columns chosen by an amount or a plan
    def _value_columns_read(self, f: TableFacts, checked, by_col) -> List[str]:
        """The checks of ``QuestionCheck._value_columns`` (a column chosen by an amount or a plan must be the one its
        printed heading gives; with none stated, every column it could mean is read), fed with the amounts and plans
        of the reading instead of patterns over the whole question."""
        from .sql_step import SqlProblem
        read = set(checked.read) & set(f.values)
        if not read:
            return []
        all_plans = list(dict.fromkeys([p for c in f.values.values() for p in c.plans] + list(self.plans)))
        forms = self._name_forms(all_plans, "plan")
        items = [i for c in read for i in by_col.get(c, [])] + by_col.get("plan", [])
        amounts: List[int] = []
        named_plans: Set[str] = set()
        unread = []
        for item in items:
            got = amounts_read(item["words"])
            status, name, _ = names_value(item["words"], forms)
            amounts += got
            if status == "named":
                named_plans.add(name)
            elif not got and item["column"] in f.values and f.values[item["column"]].by_amount:
                unread.append(item)
        amounts = list(dict.fromkeys(amounts))

        def by_amount_ok(c: ValueColumn) -> bool:
            return not c.by_amount or not amounts or any(c.covers(a, a) for a in amounts)

        def by_plan_ok(c: ValueColumn) -> bool:
            return not c.plans or not named_plans or bool(named_plans & set(c.plans))

        meant = {n for n, c in f.values.items() if by_amount_ok(c) and by_plan_ok(c)}
        wrong = sorted(read - meant)
        if wrong:
            if not meant:
                what = (" or ".join(amount_words(a) for a in amounts) if amounts else
                        " or ".join(f"'{n}'" for n in sorted(named_plans)))
                raise NotInTables(f"The {f.words} prints no column for {what}.")
            c = f.values[wrong[0]]
            stated = ([amount_words(a) for a in amounts] if not by_amount_ok(c) else []) + \
                     ([f"'{n}'" for n in sorted(named_plans)] if not by_plan_ok(c) else [])
            raise SqlProblem(f"the query reads the column '{c.heading}', but the words read give " + " and ".join(stated)
                             + ": read the column whose printed heading covers it")
        open_amount = not amounts and not unread and any(f.values[n].by_amount for n in read)
        open_plan = not named_plans and any(f.values[n].plans for n in read)
        if (open_amount or open_plan) and read < meant:
            what = "the Sum Insured" if open_amount else "the plan"
            raise SqlProblem(f"the reading gives no words of the question for {what}, but the query reads only some of "
                             f"the {f.words}'s columns (choosing {'an amount' if open_amount else 'a plan'}): if the "
                             "question states it, add its words to \"reading\" with the column; if not, read every "
                             "column it could mean, or reply ask")
        return [f"The Sum Insured was read from {i['words']!r} for the column '{f.values[i['column']].heading}' (not "
                "checked by code)." for i in unread]


def _age_key(v: str) -> int:
    m = re.search(r"\d+", v)
    return int(m.group(0)) if m else 0
