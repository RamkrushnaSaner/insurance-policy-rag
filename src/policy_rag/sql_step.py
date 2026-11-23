"""The SQL step (v3.5): questions about the policy's numeric tables are answered from SQL that the answer model writes.

The numeric tables (premiums, preventive health check-up limits, aggregate deductible discounts) are stored in their
printed layout (migration 003) and shown, for the active version only, as the views of schema ``policy_sql``. Every
other printed table is text chunks, searched like the rest of the policy.

1. ``SqlGate``: a quick check. Questions whose words name a premium or cost, a check-up or a deductible, or whose
   embedding is close to the tables' reference text (``sql.gate_similarity``), get the SQL step (one more call of
   the answer model); every other question goes straight to the text search.
2. ``SchemaCard``: what the model is shown, read from the database at start-up: each table's description, printed
   titles, pages, introducing sentence and footnotes, and each column's type, printed heading and (text columns)
   the values it holds, plus a few example rows. Nothing in it is written by hand.
3. The prompt (``prompts.SQL_TEMPLATES``) asks for ONE JSON object: ``query`` (a SELECT), ``ask`` (a detail needed
   to choose the rows is missing: one question for the user), ``not_in_tables`` or ``text_only``.
4. ``SqlValidator``: one SELECT (or WITH ... SELECT); only the policy tables (unqualified names are qualified); only
   listed functions; every selected value comes from a column; no locking, no INTO, no parameters; at most
   ``max_rows`` rows.
5. ``SqlRunner``: EXPLAIN, then the query, in a read-only session with a statement timeout, as a login that can read
   schema ``policy_sql`` and nothing else (group ``rag_sql``).
6. A reply that cannot be used (not JSON, refused SQL, a database error, no rows, too many rows) is shown to the
   model with the reason, and a corrected reply asked for, up to ``max_retries`` times.
7. ``derive``: differences, totals and amounts after a discount, computed by code in Decimal from the returned
   rows, so the model never has to do the arithmetic; ``explain_differences``: when the query subtracts itself, the
   size of the result and which side is larger, so no answer depends on the sign.
8. ``SqlOutcome.block()``: the rows for the answer model's context, each value under its printed heading, with the
   query, the calculations and the tables' printed titles, pages, introducing sentences and footnotes;
   ``linked_text``: the text chunks to read with the tables; ``verify(answer)``: figures in an answer that are in
   neither the block nor the question.

Nothing is invented here: values come only from rows a query returned, a missing detail is asked for, and a value the
tables do not hold is reported as such.

Needs ``psycopg`` and ``sqlglot`` (``pip install "policy-rag[db]"``).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Sequence, Tuple

from .sql_checks import NotInTables
from .prompts import build_sql_prompt

SCHEMA = "policy_sql"
ACTIONS = ("query", "ask", "not_in_tables", "text_only")
# skipped: the gate did not match | answered: rows returned | ask, not_in_tables, text_only: the model's decision |
# no_rows: valid queries returned nothing | failed: no usable reply within the retries | unavailable: database down
STATUSES = ("skipped", "answered", "ask", "not_in_tables", "text_only", "no_rows", "failed", "unavailable")

# ---------------------------------------------------------------------------------------------------- 1. gate
_PREMIUM = re.compile(r"\bpremiums?\b", re.I)
_COST = re.compile(r"\bhow much\b[^?]*\bcosts?\b|\bhow much (more |less |extra )?(do|does|would|will|should)\s+"
                   r"(i|we|you|a|an|my|someone|one|\d+)\b[^?]*\bpay\b|\b(cost|price)s? (of|for)\b|\bcosts? (more|less)\b"
                   r"|\b(cheaper|costlier|expensive)\b", re.I)
_INSURANCE = re.compile(r"\b(policy|plan|cover|insurance|premium)\b", re.I)
_AGE = re.compile(r"\b\d{1,3}\s*-?\s*(years?|yrs?)\b|\baged?\s*\d", re.I)
_PREMIUM_NOT = re.compile(r"instal+ment|grace period|refund|deduction|income tax|\bgst\b|\btax\b|discount|loading"
                          r"|cash ?back|\bmonthly\b", re.I)
# common money words: they count when the question also names a plan the tables hold, or an age
_MONEY = re.compile(r"\b(costs?|prices?|priced|pricing|charges?|rates?|pay|paying|spend)\b", re.I)
_CHECKUP = re.compile(r"\bcheck[\s-]?ups?\b", re.I)
_DEDUCTIBLE = re.compile(r"\bdeductibles?\b", re.I)


def gate(question: str, names: Sequence[str] = ()) -> List[str]:
    """What the question is about, among premium, checkup and deductible ([] = no SQL step). ``names``: names the
    tables hold (plans, ...); a cost question counts when it names one, a policy word or an age, and a common money
    word ("cost", "price", "pay") when it names one or an age. Other wordings are left to ``SqlGate``'s closeness to
    the tables' reference text."""
    q = question or ""
    kinds = []
    named = any(re.search(r"\b" + re.escape(n) + r"\b", q, re.I) for n in names)
    cost = _COST.search(q) and (_INSURANCE.search(q) or _AGE.search(q) or named)
    money = _MONEY.search(q) and (named or _AGE.search(q))
    if (_PREMIUM.search(q) or cost or money) and not _PREMIUM_NOT.search(q):
        kinds.append("premium")
    if _CHECKUP.search(q):
        kinds.append("checkup")
    if _DEDUCTIBLE.search(q):
        kinds.append("deductible")
    return kinds


def table_profiles(card: "SchemaCard") -> List[str]:
    """Short texts that say what the tables hold, read from the card: each table's description, printed titles,
    introducing sentences and column headings. A question close to one of them is about the tables."""
    out = []
    for t in card.tables.values():
        texts = [t["description"]] + [p["title"] for p in t["printed"]] + [p["introduced_by"] for p in t["printed"]]
        texts += [re.sub(r"(?i)^printed heading:\s*", "", c["comment"]) for c in t["columns"]
                  if c["comment"].lower().startswith("printed heading:")]
        out += [x.strip() for x in texts if x and x.strip() and x.strip() not in out]
    return out


class SqlGate:
    """Words (``gate``) or closeness: a question passes when its words name a premium, a check-up or a deductible, or
    when its embedding is at least ``threshold`` (cosine) from one of the tables' reference texts. The closeness is
    computed whenever an embedder is given (it is recorded for every question), and used only when ``threshold`` is
    set: it lets wordings through that no word list foresees ("What's the yearly price tag of ...")."""

    def __init__(self, names: Sequence[str] = (), embedder=None, profiles: Sequence[str] = (),
                 threshold: Optional[float] = None):
        self.names, self.embedder, self.threshold = list(names), embedder, threshold
        self.profiles = list(profiles)
        self._profile_vecs = None
        if embedder is not None and self.profiles:
            self._profile_vecs = _unit(embedder.encode_documents(self.profiles))

    def similarity(self, question: str) -> Optional[float]:
        if self._profile_vecs is None:
            return None
        import numpy as np
        q = _unit(np.asarray(self.embedder.encode_query(question), dtype="float32").reshape(1, -1))[0]
        return round(float((self._profile_vecs @ q).max()), 4)

    def check(self, question: str) -> Tuple[List[str], Optional[float]]:
        kinds, sim = gate(question, self.names), self.similarity(question)
        if not kinds and self.threshold is not None and sim is not None and sim >= self.threshold:
            kinds = ["similar"]
        return kinds, sim


def _unit(m):
    import numpy as np
    m = np.asarray(m, dtype="float32")
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n == 0, 1, n)


# ---------------------------------------------------------------------------------------------------- formatting
def group_inr(n: int) -> str:
    """Indian digit grouping, as the policy prints amounts: 1000000 -> '10,00,000'."""
    s, sign = str(abs(int(n))), "-" if int(n) < 0 else ""
    if len(s) <= 3:
        return sign + s
    head, parts = s[:-3], []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    return sign + ",".join(([head] if head else []) + parts) + "," + s[-3:]


def where(section_id) -> str:
    """'Annexure-C' -> 'Annexure C'; '5' -> 'Section 5'."""
    s = str(section_id or "")
    return s.replace("-", " ") if s.lower().startswith(("annexure", "schedule", "appendix")) else f"Section {s}"


def pages_text(pages: Sequence[int]) -> str:
    p = sorted({int(x) for x in pages or []})
    if not p:
        return ""
    if p == list(range(p[0], p[-1] + 1)) and len(p) > 1:
        return f"pages {p[0]}-{p[-1]}"
    return ("page " if len(p) == 1 else "pages ") + ", ".join(map(str, p))


def printed_line(p: dict) -> str:
    """'Printed table: <title> (Annexure C, pages 19-20)'."""
    where_ = ", ".join(x for x in (where(p["section_id"]) if p.get("section_id") else "", pages_text(p.get("pages")))
                       if x)
    head = f"Printed table: {p['title']}" if p.get("title") else "Printed table without a title"
    return head + (f" ({where_})" if where_ else "")


def literal(v) -> str:
    """A value as SQL would write it: text quoted, numbers plain."""
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    if isinstance(v, Decimal):
        return _plain(v)
    return str(v)


def _plain(d: Decimal) -> str:
    s = format(d.normalize(), "f")
    return s.rstrip("0").rstrip(".") if "." in s else s


_PRINTED_VALUE = re.compile(r"^(\D*?)(\d+)(\D*)$")
_GROUPED = re.compile(r"(?<![\d,])\d{1,3}(?:,\d{2,3})+(?![\d,])")


def indian_amount(n) -> str:
    """500000 -> '5 lakh', 5000000 -> '50 lakh', 10000000 -> '1 crore', 150000 -> '1.5 lakh'; '' below 1 lakh."""
    d = Decimal(str(n))
    if d >= 10_000_000:
        return f"{_plain(d / 10_000_000)} crore"
    if d >= 100_000:
        return f"{_plain(d / 100_000)} lakh"
    return ""


def with_lakh(text: str) -> str:
    """Amounts of a lakh or more written with their lakh/crore value: 'Sum Insured 5,000,000' -> '... 5,000,000 (50 lakh)'."""
    def add(m):
        words = indian_amount(m.group(0).replace(",", ""))
        return f"{m.group(0)} ({words})" if words else m.group(0)
    return _GROUPED.sub(add, text or "")


def value_text(v) -> str:
    """A listed value: text quoted, numbers plain, amounts of a lakh or more with their lakh/crore value."""
    if isinstance(v, (int, Decimal)) and not isinstance(v, bool) and indian_amount(v):
        return f"{literal(v)} ({indian_amount(v)})"
    return literal(v)


def compact_values(values: Sequence, limit: int = 40) -> str:
    """The values a column holds, short. Runs of printed numbers are shortened but stay separate values:
    "'0', '1', ... '65', '66*', '67*', ... '89*', '>=90*'"."""
    vals = list(values)
    if vals and all(isinstance(v, str) and _PRINTED_VALUE.match(v) for v in vals):
        parsed = sorted((int(m.group(2)), m.group(1), m.group(3), v) for v in vals for m in [_PRINTED_VALUE.match(v)])
        runs: List[List[tuple]] = []
        for p in parsed:
            last = runs[-1][-1] if runs else None
            if last and (last[1], last[2]) == (p[1], p[2]) and p[0] == last[0] + 1:
                runs[-1].append(p)
            else:
                runs.append([p])
        if len(runs) < len(vals):
            return ", ".join(", ".join(literal(x[3]) for x in r) if len(r) <= 3 else
                             f"{literal(r[0][3])}, {literal(r[1][3])}, ... {literal(r[-1][3])}" for r in runs)
    shown = ", ".join(value_text(v) for v in vals[:limit])
    return shown + (f", ... ({len(vals)} values)" if len(vals) > limit else "")


def _short(v, n: int = 60):
    if isinstance(v, str) and len(v) > n:
        return v[:n - 1].rstrip() + "…"
    return v


# ---------------------------------------------------------------------------------------------------- 2. schema card
LISTED_TYPES = ("text", "smallint", "character varying", "boolean")   # columns whose values are always shown
FEW_VALUES = 12                                                       # other columns: shown when they hold this few
_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")


@dataclass
class SchemaCard:
    """The tables the model may query. ``tables[name]``: description, rows, columns [{name, type, heading, values}],
    printed (one entry per printed table: title, section, pages, introduced_by, footnotes, linked_text), examples."""
    tables: Dict[str, dict]

    @classmethod
    def load(cls, db, examples: int = 2) -> "SchemaCard":
        """Read the card from the database (the views of schema policy_sql)."""
        tables: Dict[str, dict] = {}
        for table, _pos, name, dtype, comment in db.query(
                f"SELECT sql_table, position, column_name, data_type, printed_heading FROM {SCHEMA}.column_reference "
                "ORDER BY sql_table, position"):
            t = tables.setdefault(table, {"description": "", "rows": 0, "columns": [], "printed": [], "examples": []})
            t["columns"].append({"name": name, "type": dtype, "comment": comment or "", "values": None})
        for table, description in db.query(
                "SELECT c.relname, obj_description(c.oid, 'pg_class') FROM pg_class c "
                f"WHERE c.relnamespace = '{SCHEMA}'::regnamespace AND c.relkind = 'v'"):
            if table in tables:
                tables[table]["description"] = description or ""
        for table, printed_table, title, section_id, pages, introduced_by, footnotes, linked in db.query(
                f"SELECT sql_table, printed_table, printed_title, section_id, pages, introduced_by, footnotes, "
                f"linked_text FROM {SCHEMA}.table_reference ORDER BY sql_table, pages[1], printed_table"):
            if table in tables:
                tables[table]["printed"].append({
                    "printed_table": printed_table, "title": title or "", "section_id": section_id,
                    "pages": list(pages or []), "introduced_by": introduced_by or "",
                    "footnotes": list(footnotes or []), "linked_text": list(linked or [])})
        for table, t in tables.items():
            if not _NAME.match(table) or not all(_NAME.match(c["name"]) for c in t["columns"]):
                raise ValueError(f"unexpected table or column name in {SCHEMA}.{table}")
            t["rows"] = int(db.query(f'SELECT count(*) FROM {SCHEMA}."{table}"')[0][0])
            for c in t["columns"]:
                few = c["type"] not in LISTED_TYPES and db.query(
                    f'SELECT count(DISTINCT "{c["name"]}") FROM {SCHEMA}."{table}"')[0][0] <= FEW_VALUES
                if c["type"] in LISTED_TYPES or few:
                    c["values"] = [r[0] for r in db.query(
                        f'SELECT DISTINCT "{c["name"]}" FROM {SCHEMA}."{table}" ORDER BY 1')]
            if examples:
                order = ", ".join(str(i + 1) for i in range(len(t["columns"])))
                names, rows = db.query_columns(f'SELECT * FROM {SCHEMA}."{table}" ORDER BY {order} LIMIT {int(examples)}')
                t["examples"] = [dict(zip(names, r)) for r in rows]
        if not tables:
            raise ValueError(f"no tables in {SCHEMA}.column_reference: is the database at migration 003?")
        return cls(tables)

    # ------------------------------------------------------------------ lookups
    def heading(self, table: str, column: str) -> Optional[str]:
        """The printed heading of a column ('Sum Insured 500,000 (premium in Rs.)'), None for other columns."""
        for c in self.tables.get(table, {}).get("columns", []):
            if c["name"] == column and c["comment"].lower().startswith("printed heading:"):
                return c["comment"].split(":", 1)[1].strip()
        return None

    def label(self, tables: Sequence[str], column: str) -> str:
        """How a result column is shown: its printed heading when it has one in one of ``tables``, else its name."""
        for t in tables:
            h = self.heading(t, column)
            if h:
                return h
        return column

    def table_of(self, column: str, tables: Sequence[str]) -> Optional[str]:
        """The first of ``tables`` that has ``column`` (None for a computed column)."""
        for t in tables:
            if any(c["name"] == column for c in self.tables.get(t, {}).get("columns", [])):
                return t
        return None

    def text_columns(self) -> List[str]:
        """Columns of text type (compared only with = or IN to listed values)."""
        return sorted({c["name"] for t in self.tables.values() for c in t["columns"]
                       if c["type"] in ("text", "character varying")})

    def names(self) -> List[str]:
        """Short text values the tables hold (plan names, ...): a question naming one is about the tables."""
        out = []
        for t in self.tables.values():
            for c in t["columns"]:
                out += [v for v in c["values"] or [] if isinstance(v, str) and 3 <= len(v) <= 40
                        and re.search(r"[A-Za-z]{3}", v) and v not in out]
        return out

    def printed(self, tables: Sequence[str]) -> List[dict]:
        return [p for t in tables for p in self.tables.get(t, {}).get("printed", [])]

    def linked_text(self, tables: Sequence[str]) -> List[dict]:
        """Text chunks linked to these tables ({relation, label, text}), each once, in order."""
        out, seen = [], set()
        for p in self.printed(tables):
            for link in p["linked_text"]:
                key = (link.get("label"), link.get("text"))
                if key not in seen:
                    seen.add(key)
                    out.append(link)
        return out

    # ------------------------------------------------------------------ the text the model reads
    @property
    def text(self) -> str:
        out = []
        for name, t in self.tables.items():
            out.append(f"{SCHEMA}.{name} ({t['rows']} rows): {t['description']}".rstrip(": "))
            for p in t["printed"]:
                out.append("  " + printed_line(p))
            for line in dict.fromkeys(p["introduced_by"] for p in t["printed"] if p["introduced_by"]):
                out.append(f"  Introduced by: {line}")
            for note in dict.fromkeys(n for p in t["printed"] for n in p["footnotes"] if n):
                out.append(f"  Footnote: {note}")
            out.append("  Columns:")
            for c in t["columns"]:
                line = f"    {c['name']} ({c['type']})" + (f": {with_lakh(c['comment'])}" if c["comment"] else "")
                if c["values"] is not None:
                    line += f". Values: {compact_values(c['values'])}"
                out.append(line)
            if t["examples"]:
                out.append("  Example rows:")
                for row in t["examples"]:
                    out.append("    " + ", ".join(f"{k}={literal(_short(v))}" for k, v in row.items()))
            out.append("")
        return "\n".join(out).strip()


# ---------------------------------------------------------------------------------------------------- 3. the reply
class SqlProblem(ValueError):
    """A reply that cannot be used; the message is shown to the model."""

    def __init__(self, message: str, no_rows: bool = False):
        super().__init__(message)
        self.no_rows = no_rows


def parse_action(reply: str) -> dict:
    """The first JSON object with an "action" in the reply: {action, sql, question, reason}."""
    text = re.sub(r"<think>.*?</think>|</?think>", "", reply or "", flags=re.S)
    decoder = json.JSONDecoder(strict=False)              # SQL written over several lines inside the string
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
    if action not in ACTIONS:
        raise SqlProblem(f"unknown action {action!r}: use one of {', '.join(ACTIONS)}")
    out = {"action": action, "sql": "", "question": "", "reason": str(obj.get("reason") or "").strip()}
    if action == "query":
        out["sql"] = str(obj.get("sql") or "").strip()
        if not out["sql"]:
            raise SqlProblem('a "query" reply needs the SQL in "sql"')
    if action == "ask":
        out["question"] = str(obj.get("question") or "").strip()
        if not out["question"]:
            raise SqlProblem('an "ask" reply needs the question for the user in "question"')
    return out


# ---------------------------------------------------------------------------------------------------- 4. validator
# Functions a query may call (sqlglot's names and PostgreSQL's own; operators such as AND, ~ or || are not functions)
FUNCTIONS = frozenset("""
    count sum min max avg round abs floor ceil ceiling trunc coalesce nullif greatest least cast try_cast case if
    lower upper initcap trim btrim ltrim rtrim length char_length substring substr replace regexp_replace concat
    string_agg group_concat array_agg array row_number rank dense_rank left right split_part strpos str_position
    position exists""".split())


@dataclass
class CheckedSql:
    sql: str                              # the query as it will run (tables qualified, row limit set)
    tables: List[str]                     # the policy tables it reads
    fetch_limit: Optional[int]            # the LIMIT added or lowered here (max_rows + 1: more rows = too many)
    aliases: Dict[str, str] = field(default_factory=dict)   # output name -> table column it shows unchanged
    keys: List[str] = field(default_factory=list)           # columns that choose rows (= or IN against values)
    # output name -> its two operands when the query itself subtracts (``a - b AS name``): (column or None, the
    # operand's WHERE condition or its SQL text) for each side
    subtractions: Dict[str, Tuple[Tuple[Optional[str], str], Tuple[Optional[str], str]]] = field(default_factory=dict)
    read: List[str] = field(default_factory=list)           # every column the query reads, anywhere (subqueries too)
    # column -> the literals it is compared with anywhere in the query: ("=", value) for = and IN, ("like", pattern)
    filters: Dict[str, List[Tuple[str, object]]] = field(default_factory=dict)


class SqlValidator:
    def __init__(self, tables: Sequence[str], max_rows: int = 50, schema: str = SCHEMA,
                 text_columns: Sequence[str] = ()):
        try:
            import sqlglot  # noqa: F401
        except ImportError as e:
            raise ImportError('the SQL step needs sqlglot: pip install "policy-rag[db]"') from e
        self.allowed, self.max_rows, self.schema = set(tables), max_rows, schema
        self.text_columns = set(text_columns)

    def check(self, sql: str) -> CheckedSql:
        import sqlglot
        from sqlglot import exp
        try:
            trees = [t for t in sqlglot.parse(sql, read="postgres") if t is not None]
        except (sqlglot.errors.ParseError, sqlglot.errors.TokenError) as e:
            raise SqlProblem("not valid SQL: " + str(e).splitlines()[0][:200]) from None
        if len(trees) != 1:
            raise SqlProblem(f"write exactly one statement (found {len(trees)})")
        tree = trees[0]
        if not isinstance(tree, exp.Query):
            raise SqlProblem("only a SELECT statement (or WITH ... SELECT) may be run")
        forbidden = [getattr(exp, n) for n in ("Insert", "Update", "Delete", "Create", "Drop", "Alter", "Command", "Copy",
                                               "Merge", "Set", "Into", "Lock", "Placeholder", "Parameter", "Grant")
                     if hasattr(exp, n)]
        for node in tree.walk():
            node = node[0] if isinstance(node, tuple) else node      # older sqlglot walks (node, parent, key)
            if isinstance(node, tuple(forbidden)):
                raise SqlProblem(f"not allowed in a query: {type(node).__name__.upper()} "
                                 "(only reading, with no parameters, locks or INTO)")
        ctes = {c.alias_or_name for c in tree.find_all(exp.CTE)}
        used = set()
        for t in tree.find_all(exp.Table):
            name, db, catalog = t.name, t.args.get("db"), t.args.get("catalog")
            if not name:
                raise SqlProblem(f"read only the policy tables, not {t.sql(dialect='postgres')!r}")
            db_name = db.name if db is not None else ""
            if not db_name and not catalog and name in ctes:
                continue
            if catalog is not None or db_name not in ("", self.schema) or name not in self.allowed:
                raise SqlProblem(f"unknown table {t.sql(dialect='postgres')!r}: use only "
                                 + ", ".join(sorted(self.allowed)))
            if not db_name:
                t.set("db", exp.to_identifier(self.schema))
            used.add(name)
        if not used:
            raise SqlProblem("the query reads none of the policy tables")
        for f in tree.find_all(exp.Func):
            if isinstance(f, (exp.Binary, exp.Connector)):
                continue                                              # operators: AND, ~, ||, ...
            name = (f.name if isinstance(f, exp.Anonymous) else f.sql_name()).lower()
            if name not in FUNCTIONS:
                raise SqlProblem(f"function {name}() is not allowed; allowed: " + ", ".join(sorted(FUNCTIONS)))
        for cmp in tree.find_all(exp.Between, exp.GT, exp.GTE, exp.LT, exp.LTE):
            text = sorted({c.name for c in cmp.find_all(exp.Column) if c.name in self.text_columns})
            if text:
                raise SqlProblem(f"{', '.join(text)} is text: compare it with = or IN to values written exactly as "
                                 "listed, not with BETWEEN, <, >, <= or >=")
        for sel in _output_selects(tree):
            for proj in sel.expressions:
                if proj.find(exp.Column) is None and proj.find(exp.Star) is None:
                    raise SqlProblem(f"every selected value must come from a table column (found the constant "
                                     f"{proj.sql(dialect='postgres')!r})")
        limit = tree.args.get("limit")
        fetch = None
        if limit is None:
            fetch = self.max_rows + 1
        else:
            value = limit.args.get("expression")
            if not isinstance(value, exp.Literal) or value.is_string or not str(value.this).isdigit():
                raise SqlProblem("LIMIT must be a whole number")
            if int(value.this) > self.max_rows:
                fetch = self.max_rows + 1
        named: Dict[str, set] = {}                          # aliases anywhere in the query (subqueries too)
        for a in tree.find_all(exp.Alias):
            if isinstance(a.this, exp.Column):
                named.setdefault(a.alias, set()).add(a.this.name)
        aliases = {alias: next(iter(cols)) for alias, cols in named.items() if len(cols) == 1}
        keys = sorted({c.name for cmp in tree.find_all(exp.EQ, exp.In) for c in [cmp.this]
                       if isinstance(c, exp.Column) and (isinstance(cmp, exp.In) or isinstance(cmp.expression, exp.Literal))})
        subtractions = {}
        for sel in _output_selects(tree)[:1]:
            for proj in sel.expressions:
                expr = proj.this if isinstance(proj, exp.Alias) else proj
                while isinstance(expr, exp.Paren):
                    expr = expr.this
                if isinstance(expr, exp.Sub) and proj.alias_or_name:
                    subtractions[proj.alias_or_name] = (_operand(expr.this), _operand(expr.expression))
        read = sorted({c.name for c in tree.find_all(exp.Column)})
        filters: Dict[str, List[Tuple[str, object]]] = {}
        for cmp in tree.find_all(exp.EQ, exp.In, exp.Like, exp.ILike):
            col, other = cmp.this, cmp.expression
            if not isinstance(col, exp.Column) and isinstance(other, exp.Column):
                col, other = other, col
            if not isinstance(col, exp.Column):
                continue
            if isinstance(cmp, exp.In):
                values = [v for v in cmp.expressions if isinstance(v, exp.Literal)]
            else:
                values = [other] if isinstance(other, exp.Literal) else []
            op = "like" if isinstance(cmp, (exp.Like, exp.ILike)) else "="
            for v in values:
                value = v.this if v.is_string else (int(v.this) if str(v.this).lstrip("-").isdigit() else v.this)
                filters.setdefault(col.name, []).append((op, value))
        if fetch is not None:
            tree = tree.limit(fetch)
        return CheckedSql(tree.sql(dialect="postgres"), sorted(used), fetch, aliases, keys, subtractions, read, filters)


def _operand(node) -> Tuple[Optional[str], str]:
    """One side of a subtraction the query does: (the column it reads, the WHERE condition that picks its row) for a
    column or a one-value subquery; (None, its SQL text) for anything else."""
    from sqlglot import exp
    while isinstance(node, (exp.Paren, exp.Subquery)):
        node = node.this
    if isinstance(node, exp.Column):
        return node.name, ""
    if isinstance(node, exp.Select) and len(node.expressions) == 1:
        value = node.expressions[0]
        value = value.this if isinstance(value, exp.Alias) else value
        where = node.args.get("where")
        if isinstance(value, exp.Column):
            return value.name, where.this.sql(dialect="postgres") if where is not None else ""
    return None, node.sql(dialect="postgres")


def _output_selects(node) -> list:
    from sqlglot import exp
    if isinstance(node, exp.Select):
        return [node]
    if isinstance(node, (exp.Union, exp.Intersect, exp.Except)):
        return _output_selects(node.left) + _output_selects(node.right)
    if isinstance(node, exp.Subquery):
        return _output_selects(node.this)
    return []


# ---------------------------------------------------------------------------------------------------- 6b. arithmetic
_MONEY_LABEL = re.compile(r"\b(rs\.?|inr|premium|amount|limit)\b", re.I)
DERIVE_MAX_ROWS = 8               # no arithmetic over longer results (listings, not calculations)


def column_kind(label: str) -> Optional[str]:
    """'percent' when the printed heading has a %, 'amount' when it names money (Rs., INR, premium, amount, limit)."""
    if "%" in label:
        return "percent"
    return "amount" if _MONEY_LABEL.search(label) else None


def money(d: Decimal) -> str:
    """'Rs 2,025' / 'Rs 17,212.5' (Indian grouping; the fraction kept as computed)."""
    d = d.normalize()
    whole = int(d.to_integral_value(rounding="ROUND_DOWN")) if d >= 0 else -int((-d).to_integral_value(rounding="ROUND_DOWN"))
    frac = format(abs(d - whole), "f")[1:] if d != whole else ""
    return f"Rs {group_inr(whole)}{frac.rstrip('0').rstrip('.') if frac else ''}"


def derive(columns: Sequence[str], labels: Sequence[str], sources: Sequence[Optional[str]], rows: Sequence[Sequence],
           keys: Sequence[str] = ()) -> List[dict]:
    """Calculations done by code, in Decimal, from the rows a query returned, so the model never has to do the
    arithmetic (and a difference never depends on which way round the model subtracted):

    - two rows: for each amount column, the difference (higher minus lower, naming both rows) and the total;
      more rows (up to 8): the total;
    - one row with exactly two amount columns of the same table: their difference;
    - one row with an amount and a percentage from different tables (a premium and a discount): the amount after
      the percentage is taken off, and the amount taken off.

    Amount and percent columns are recognised by their printed headings (``column_kind``); columns that chose the
    rows (``keys``: = or IN in the query) and computed columns (no source table) are never calculated with."""
    def num(v):
        return Decimal(str(v)) if isinstance(v, (int, float, Decimal)) and not isinstance(v, bool) else None
    kinds = [column_kind(l) if src and c not in keys else None for c, l, src in zip(columns, labels, sources)]
    def ident(r):
        """What tells a row apart: its text values and the values that chose it (not page, not computed numbers)."""
        parts = [f"{columns[i]} {rows[r][i]}" for i, k in enumerate(kinds)
                 if k is None and columns[i] != "page" and rows[r][i] is not None
                 and (num(rows[r][i]) is None or columns[i] in keys)]
        return ", ".join(dict.fromkeys(parts)) or f"row {r + 1}"
    out = []
    if not rows or len(rows) > DERIVE_MAX_ROWS:
        return out
    amounts = [i for i, k in enumerate(kinds) if k == "amount"]
    percents = [i for i, k in enumerate(kinds) if k == "percent"]
    if len(rows) >= 2:
        for i in amounts:
            vals = [num(r[i]) for r in rows]
            if any(v is None for v in vals):
                continue
            if len(rows) == 2:
                hi, lo = (0, 1) if vals[0] >= vals[1] else (1, 0)
                out.append({"kind": "difference", "value": vals[hi] - vals[lo],
                            "text": f"Difference in {labels[i]}: {ident(hi)} ({money(vals[hi])}) minus {ident(lo)} "
                                    f"({money(vals[lo])}) = {money(vals[hi] - vals[lo])}"})
            out.append({"kind": "total", "value": sum(vals),
                        "text": f"Total of {labels[i]} over the {len(rows)} rows = {money(sum(vals))}"})
    for r, row in enumerate(rows):
        by_table: Dict[str, List[int]] = {}
        for i in amounts:
            if num(row[i]) is not None:
                by_table.setdefault(sources[i], []).append(i)
        for cols in by_table.values():
            if len(cols) == 2:
                a, b = sorted(cols, key=lambda i: num(row[i]), reverse=True)
                d = num(row[a]) - num(row[b])
                out.append({"kind": "difference", "value": d,
                            "text": f"Difference in {ident(r)}: {labels[a]} ({money(num(row[a]))}) minus {labels[b]} "
                                    f"({money(num(row[b]))}) = {money(d)}"})
        for p in percents:
            pct = num(row[p])
            if pct is None:
                continue
            for a in amounts:
                if sources[a] == sources[p] or num(row[a]) is None:
                    continue
                amount = num(row[a])
                off = amount * pct / 100
                out.append({"kind": "after_percent", "value": amount - off,
                            "text": f"{labels[a]} {money(amount)} less {_plain(pct)}% ({labels[p]}): "
                                    f"{money(amount)} x (1 - {_plain(pct)}/100) = {money(amount - off)} "
                                    f"(taken off: {money(off)})" + (f" [{ident(r)}]" if len(rows) > 1 else "")})
    return out


def explain_differences(columns: Sequence[str], rows: Sequence[Sequence], subtractions: Dict[str, tuple]) -> List[dict]:
    """A subtraction the query did itself (``a - b AS difference``) gives a signed result whose sign depends only on
    the order the model wrote: -1300 can mean "1300 cheaper". For each such column, code states the size and which
    side is larger, so the answer never has to read the sign. ``subtractions``: column -> (first, second, amounts),
    the two sides described in words and whether both are amounts (Rs)."""
    out = []
    if not rows or len(rows) > DERIVE_MAX_ROWS:
        return out
    for i, c in enumerate(columns):
        if c not in subtractions:
            continue
        first, second, amounts = subtractions[c]
        show = money if amounts else _plain
        for r, row in enumerate(rows):
            v = row[i]
            if not isinstance(v, (int, float, Decimal)) or isinstance(v, bool):
                continue
            v = Decimal(str(v))
            size = abs(v)
            larger = ("the first is larger" if v > 0 else "the second is larger") + f", by {show(size)}" if v else \
                "the two are equal"
            row_no = f" (row {r + 1})" if len(rows) > 1 else ""
            out.append({"kind": "difference_by_query", "value": size,
                        "text": f"{c} = {show(v)}{row_no} is ({first}) minus ({second}): {larger}"})
    return out


# ---------------------------------------------------------------------------------------------------- 5. runner
class DatabaseUnavailable(RuntimeError):
    pass


class SqlRunner:
    """EXPLAIN, then the query, on a read-only ``Database`` session with its statement timeout. A statement error
    (unknown column, permission, timeout, ...) becomes a ``SqlProblem`` for the model; a lost connection becomes
    ``DatabaseUnavailable``."""

    def __init__(self, db):
        self.db = db

    def run(self, sql: str) -> Tuple[List[str], List[tuple]]:
        try:
            self.db.query("EXPLAIN " + sql, None)
            return self.db.query_columns(sql, None)
        except Exception as e:
            state = getattr(e, "sqlstate", None)
            if isinstance(state, str) and state and not state.startswith("08"):
                first = (str(e).strip().splitlines() or [type(e).__name__])[0]
                if state == "57014":
                    first = "the query took too long and was cancelled: " + first
                raise SqlProblem(f"the database refused the query: {first[:300]}") from None
            raise DatabaseUnavailable(f"{type(e).__name__}: {str(e).strip()[:200]}") from e


# ---------------------------------------------------------------------------------------------------- 6-7. outcome
@dataclass
class SqlOutcome:
    question: str
    kinds: List[str] = field(default_factory=list)
    similarity: Optional[float] = None                        # closeness to the tables' reference text (None: no embedder)
    status: str = "skipped"
    sql: str = ""
    tables: List[str] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)          # printed heading (or name) of each column
    sources: List[Optional[str]] = field(default_factory=list)  # the table each column comes from (None: computed)
    derived: List[dict] = field(default_factory=list)        # calculations done by code from the rows (``derive``)
    rows: List[tuple] = field(default_factory=list)
    printed: List[dict] = field(default_factory=list)        # the printed tables behind the rows
    linked_text: List[dict] = field(default_factory=list)    # text chunks to read with them {relation, label, text}
    clarification: str = ""                                   # status "ask": the question for the user
    reason: str = ""                                          # the model's reason (not_in_tables, text_only), or why it failed
    attempts: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)            # assumptions code could not check exactly (sql_checks.py)

    @property
    def answered(self) -> bool:
        return self.status == "answered"

    def value_text(self, label: str, value) -> str:
        if value is None:
            return "(empty)"
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, int):
            return group_inr(value) if abs(value) >= 1000 else str(value)
        if isinstance(value, Decimal):
            text = _plain(value)
            return text + "%" if "%" in label else text
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    def block(self) -> str:
        """The rows for the answer model's context (empty unless answered)."""
        if not self.answered:
            return ""
        lines = ["Policy tables: rows read with SQL from the policy's printed tables (exact values as printed)."]
        for p in self.printed:
            lines.append(printed_line(p))
        for line in dict.fromkeys(p["introduced_by"] for p in self.printed if p["introduced_by"]):
            lines.append(f"Introduced by: {line}")
        lines.append(f"Query: {self.sql}")
        lines.append(f"Rows ({len(self.rows)}):")
        lines += self.row_lines()
        if self.derived:
            lines.append("Calculated by code from these rows (exact):")
            lines += [f"- {d['text']}" for d in self.derived]
        for note in dict.fromkeys(n for p in self.printed for n in p["footnotes"] if n):
            lines.append(f"Footnote: {note}")
        lines += [f"Note: {n}" for n in self.notes]                # written by code (sql_checks.py)
        return "\n".join(lines)

    def row_lines(self) -> List[str]:
        return ["- " + "; ".join(f"{label}: {self.value_text(label, v)}" for label, v in zip(self.labels, row))
                for row in self.rows]

    def plain_text(self) -> str:
        """The rows written out plainly (no answer model): what replaces an answer whose figures are not in the
        context (``sql.on_unsupported_figures: rows``)."""
        if not self.answered:
            return ""
        lines = ["From the policy's tables (exact values as printed):"] + self.row_lines()
        if self.derived:
            lines += ["Calculated from these values:"] + [f"- {d['text']}" for d in self.derived]
        lines += [f"Note: {n}" for n in dict.fromkeys(n for p in self.printed for n in p["footnotes"] if n)]
        lines += [f"Note: {n}" for n in self.notes]
        where_ = "; ".join(dict.fromkeys(printed_line(p).replace("Printed table: ", "") for p in self.printed))
        return "\n".join(lines + ([f"Source: {where_}"] if where_ else []))

    def sources_line(self) -> str:
        """'Annexure C, pages 19-20; ...': where the rows are printed."""
        return "; ".join(dict.fromkeys(", ".join(x for x in (where(p["section_id"]) if p.get("section_id") else "",
                                                               pages_text(p.get("pages"))) if x)
                                       for p in self.printed))

    def verify(self, answer: str, context: str = "") -> List[str]:
        """Figures in ``answer`` found in neither the rows (with ``context``, e.g. the text passages) nor the question."""
        from .evaluation.checks import unsupported_figures
        return unsupported_figures(answer, self.block() + "\n" + (context or ""), self.question)

    def to_dict(self) -> dict:
        def plain(v):                                    # numbers stay numbers in JSON (exact for numeric(5,2))
            if isinstance(v, Decimal):
                return int(v) if v == v.to_integral_value() else float(v)
            return v
        return {"question": self.question, "kinds": self.kinds, "similarity": self.similarity, "status": self.status,
                "sql": self.sql,
                "tables": self.tables, "columns": self.columns, "labels": self.labels,
                "rows": [[plain(v) for v in r] for r in self.rows], "sources": self.sources,
                "derived": [{"kind": d["kind"], "value": plain(d["value"]), "text": d["text"]} for d in self.derived],
                "clarification": self.clarification,
                "reason": self.reason, "linked_text": [l.get("label") for l in self.linked_text],
                "attempts": self.attempts, "notes": self.notes, "block": self.block()}


# ---------------------------------------------------------------------------------------------------- the step
class SqlStep:
    """Gate, prompt, reply, checks, query, retries. ``answerer``: the pipeline's ``generation.Answerer`` (the answer
    model with its configured settings and cache). ``db``: a read-only ``Database`` as a login in group rag_sql."""

    def __init__(self, cfg, answerer, db=None, card: Optional[SchemaCard] = None, embedder=None):
        if db is None:
            from .database import Database
            db = Database(cfg)
        self.cfg, self.answerer, self.db = cfg, answerer, db
        self.card = card if card is not None else SchemaCard.load(db, cfg.examples)
        self.card_text, self.names = self.card.text, self.card.names()
        self.gate = SqlGate(self.names, embedder, table_profiles(self.card) if embedder is not None else (),
                            getattr(cfg, "gate_similarity", None))
        self.validator = SqlValidator(list(self.card.tables), cfg.max_rows, text_columns=self.card.text_columns())
        self.runner = SqlRunner(db)
        self.question_check = None
        if getattr(cfg, "check_question", True):
            from .sql_checks import QuestionCheck
            self.question_check = QuestionCheck(self.card, db)

    def run(self, question: str) -> SqlOutcome:
        kinds, sim = self.gate.check(question)
        out = SqlOutcome(question=question, kinds=kinds, similarity=sim)
        if not out.kinds:
            return out
        failed: List[Tuple[str, str]] = []
        last: Optional[SqlProblem] = None
        for n in range(1 + self.cfg.max_retries):
            prompt = build_sql_prompt(question, self.card_text, self.cfg.max_rows, self.cfg.template, failed)
            rec = self.answerer.answer(prompt, label="sql", max_new_tokens=self.cfg.max_new_tokens)
            reply = rec.get("answer") or ""
            att = {"attempt": n + 1, "key": rec.get("key"), "reply": reply, "from_cache": rec.get("from_cache"),
                   "think_tokens": rec.get("think_tokens"), "answer_tokens": rec.get("answer_tokens"),
                   "seconds": rec.get("seconds")}
            out.attempts.append(att)
            try:
                action = parse_action(reply)
                att["action"] = action["action"]
                if action["action"] != "query":
                    out.status = action["action"]
                    out.clarification, out.reason = action["question"], action["reason"]
                    return out
                checked = self.validator.check(action["sql"])
                att["sql"] = checked.sql
                columns, rows = self.runner.run(checked.sql)
                att["rows"] = len(rows)
                if len(rows) > self.cfg.max_rows:
                    raise SqlProblem(f"the query returns more than {self.cfg.max_rows} rows: choose the rows the "
                                     "question asks about, or ask for the missing detail")
                if not rows:
                    raise SqlProblem("the query ran but returned no rows. If a value in the query is not written "
                                     "exactly as listed, correct it; if the tables hold no such row, reply "
                                     "not_in_tables", no_rows=True)
                notes = self.question_check.check(question, checked) if self.question_check is not None else []
            except NotInTables as e:
                att["not_in_tables"] = str(e)
                out.status, out.reason = "not_in_tables", str(e)
                return out
            except SqlProblem as e:
                att["problem"] = str(e)
                failed.append((reply, str(e)))
                last = e
                continue
            except DatabaseUnavailable as e:
                att["problem"] = str(e)
                out.status, out.reason = "unavailable", str(e)
                return out
            out.status, out.sql, out.tables = "answered", checked.sql, checked.tables
            out.columns, out.rows = list(columns), [tuple(r) for r in rows]
            out.labels = [self._label(checked, c) for c in columns]
            out.sources = [self.card.table_of(checked.aliases.get(c, c), checked.tables) for c in columns]
            out.derived = derive(out.columns, out.labels, out.sources, out.rows, checked.keys)
            out.derived += explain_differences(out.columns, out.rows, self._subtractions(checked))
            out.printed = self.card.printed(checked.tables)
            out.linked_text = self.card.linked_text(checked.tables)
            out.notes = notes
            return out
        out.status = "no_rows" if last is not None and last.no_rows else "failed"
        out.reason = str(last) if last is not None else ""
        return out

    def _subtractions(self, checked: CheckedSql) -> Dict[str, tuple]:
        """The query's own subtractions, each side in words: its printed heading and the condition that picks its
        row ('Sum Insured 5,000,000 (premium in Rs.) where plan = ... AND tier = 2'), or its SQL text."""
        out = {}
        for name, sides in checked.subtractions.items():
            words, kinds = [], []
            for column, where_ in sides:
                if column is None:
                    words.append(where_)
                    kinds.append(None)
                    continue
                label = self._label(checked, column)
                words.append(label + (f" where {where_}" if where_ else ""))
                kinds.append(column_kind(label))
            out[name] = (words[0], words[1], kinds == ["amount", "amount"])
        return out

    def _label(self, checked: CheckedSql, name: str) -> str:
        """A result column's printed heading; a renamed column keeps its new name too ('plan_b (Sum Insured ...)')."""
        source = checked.aliases.get(name, name)
        heading = self.card.label(checked.tables, source)
        if name == source or heading == source:
            return heading if name == source else name
        return f"{name} ({heading})"
