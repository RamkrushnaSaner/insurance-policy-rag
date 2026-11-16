"""The SQL step (v3.5): questions about the policy's numeric tables are answered from SQL that the answer model writes.

The numeric tables (premiums, preventive health check-up limits, aggregate deductible discounts) are stored in their
printed layout (migration 003) and shown, for the active version only, as the views of schema ``policy_sql``. Every
other printed table is text chunks, searched like the rest of the policy.

1. ``gate``: a quick pattern check. Only questions about a premium or cost, a check-up or a deductible get the SQL
   step (one more call of the answer model); every other question goes straight to the text search.
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
7. ``SqlOutcome.block()``: the rows for the answer model's context, each value under its printed heading, with the
   query and the tables' printed titles, pages, introducing sentences and footnotes; ``linked_text``: the text
   chunks to read with the tables; ``verify(answer)``: figures in an answer that are in neither the rows nor the
   question.

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
# money words that count when the question also names a plan the tables hold, or an age ("What would X cost ...")
_MONEY = re.compile(r"\b(costs?|costing|prices?|priced|pricing|charges?|rates?|spend|shell(?:ing)? out|fork(?:ing)? out)\b",
                    re.I)
_CHECKUP = re.compile(r"\bcheck[\s-]?ups?\b|\bhealth[\s-]+checks?\b|\bscreenings?\b", re.I)
# an aggregate deductible described in words: "the first Rs 3 lakh of claims"
_DEDUCTIBLE = re.compile(r"\bdeductibles?\b|\bfirst\b[^?]{0,40}?\bof\s+(?:the\s+|each\s+|every\s+|my\s+|our\s+)?claims?\b",
                         re.I)


def gate(question: str, names: Sequence[str] = ()) -> List[str]:
    """What the question is about, among premium, checkup and deductible ([] = no SQL step). ``names``: names the
    tables hold (plans, ...); a cost question counts when it names one, a policy word or an age, and a money word
    ("cost", "price", "shell out") when it names one or an age."""
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
        aliases = {}
        for sel in _output_selects(tree):
            for proj in sel.expressions:
                if proj.find(exp.Column) is None and proj.find(exp.Star) is None:
                    raise SqlProblem(f"every selected value must come from a table column (found the constant "
                                     f"{proj.sql(dialect='postgres')!r})")
                if isinstance(proj, exp.Alias) and isinstance(proj.this, exp.Column):
                    aliases[proj.alias] = proj.this.name
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
        if fetch is not None:
            tree = tree.limit(fetch)
        return CheckedSql(tree.sql(dialect="postgres"), sorted(used), fetch, aliases)


def _output_selects(node) -> list:
    from sqlglot import exp
    if isinstance(node, exp.Select):
        return [node]
    if isinstance(node, (exp.Union, exp.Intersect, exp.Except)):
        return _output_selects(node.left) + _output_selects(node.right)
    if isinstance(node, exp.Subquery):
        return _output_selects(node.this)
    return []


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
    status: str = "skipped"
    sql: str = ""
    tables: List[str] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)          # printed heading (or name) of each column
    rows: List[tuple] = field(default_factory=list)
    printed: List[dict] = field(default_factory=list)        # the printed tables behind the rows
    linked_text: List[dict] = field(default_factory=list)    # text chunks to read with them {relation, label, text}
    clarification: str = ""                                   # status "ask": the question for the user
    reason: str = ""                                          # the model's reason (not_in_tables, text_only), or why it failed
    attempts: List[dict] = field(default_factory=list)

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
        for row in self.rows:
            lines.append("- " + "; ".join(f"{label}: {self.value_text(label, v)}" for label, v in zip(self.labels, row)))
        for note in dict.fromkeys(n for p in self.printed for n in p["footnotes"] if n):
            lines.append(f"Footnote: {note}")
        return "\n".join(lines)

    def verify(self, answer: str, context: str = "") -> List[str]:
        """Figures in ``answer`` found in neither the rows (with ``context``, e.g. the text passages) nor the question."""
        from .evaluation.checks import unsupported_figures
        return unsupported_figures(answer, self.block() + "\n" + (context or ""), self.question)

    def to_dict(self) -> dict:
        def plain(v):                                    # numbers stay numbers in JSON (exact for numeric(5,2))
            if isinstance(v, Decimal):
                return int(v) if v == v.to_integral_value() else float(v)
            return v
        return {"question": self.question, "kinds": self.kinds, "status": self.status, "sql": self.sql,
                "tables": self.tables, "columns": self.columns, "labels": self.labels,
                "rows": [[plain(v) for v in r] for r in self.rows], "clarification": self.clarification,
                "reason": self.reason, "linked_text": [l.get("label") for l in self.linked_text],
                "attempts": self.attempts, "block": self.block()}


# ---------------------------------------------------------------------------------------------------- the step
class SqlStep:
    """Gate, prompt, reply, checks, query, retries. ``answerer``: the pipeline's ``generation.Answerer`` (the answer
    model with its configured settings and cache). ``db``: a read-only ``Database`` as a login in group rag_sql."""

    def __init__(self, cfg, answerer, db=None, card: Optional[SchemaCard] = None):
        if db is None:
            from .database import Database
            db = Database(cfg)
        self.cfg, self.answerer, self.db = cfg, answerer, db
        self.card = card if card is not None else SchemaCard.load(db, cfg.examples)
        self.card_text, self.names = self.card.text, self.card.names()
        self.validator = SqlValidator(list(self.card.tables), cfg.max_rows, text_columns=self.card.text_columns())
        self.runner = SqlRunner(db)

    def run(self, question: str) -> SqlOutcome:
        out = SqlOutcome(question=question, kinds=gate(question, self.names))
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
            out.printed = self.card.printed(checked.tables)
            out.linked_text = self.card.linked_text(checked.tables)
            return out
        out.status = "no_rows" if last is not None and last.no_rows else "failed"
        out.reason = str(last) if last is not None else ""
        return out

    def _label(self, checked: CheckedSql, name: str) -> str:
        """A result column's printed heading; a renamed column keeps its new name too ('plan_b (Sum Insured ...)')."""
        source = checked.aliases.get(name, name)
        heading = self.card.label(checked.tables, source)
        if name == source or heading == source:
            return heading if name == source else name
        return f"{name} ({heading})"
