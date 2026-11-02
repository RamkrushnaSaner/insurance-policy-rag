"""Answers from the policy's tables in the database: premiums, claim limits, instalment grace periods, check-up
limits, deductible discounts and city tiers.

Deterministic, so every figure can be traced to a printed table cell:

1. ``route``: which kind of table question this is (patterns; a fixed list of kinds);
2. fields read from the question: plans, ages, Sums Insured, city or tier, years, frequencies, ...;
3. checks: every field against what the tables hold (a plan without a premium table, a Sum Insured that is not
   offered, ...). A failed check is explained, never guessed around;
4. fixed, parameterised SQL over the active version (``TableData``; no generated SQL). Every row keeps its table,
   row, column, page and the cell text as printed;
5. calculations in Decimal (how much more one plan or tier costs), shown with their inputs;
6. ``TableAnswer.text``: a plain answer written from the rows; ``facts_block()``: the same facts for the answer
   model's context;
7. ``TableAnswer.verify(answer)``: figures in an answer that are neither in the facts nor in the question, and
   ``TableAnswer.trace()``: rows whose typed value differs from the printed cell.

Plan names, places, Sums Insured, ages and frequencies come from the database, so nothing here is specific to one
policy except the question patterns (English) and the item codes of the schema.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Sequence

INTENTS = ("premium", "deductible_discount", "instalment", "claim_limit", "checkup_limit", "city_tier")

# ---------------------------------------------------------------------------------------------------- patterns
_PREMIUM = re.compile(
    r"\b(annual|yearly) premium|\bpremium (for|of|amount|rates?|payable|would|will)\b"
    r"|\bwhat (is|are|would be|will be) (the|my) premium|\bhow much\b[^?]*\bpremium", re.I)
_COST = re.compile(r"\bhow much\b[^?]*\bcosts?\b|\bhow much (more |less |extra )?(do|does|would|will|should)\s+"
                   r"(i|we|you|a|an|my|someone|one|\d+)\b[^?]*\bpay\b|\b(cost|price)s? (of|for)\b|\bcosts? (more|less)\b"
                   r"|\b(cheaper|costlier|expensive)\b", re.I)
_INSURANCE = re.compile(r"\b(policy|plan|cover|insurance|premium)\b", re.I)
_PREMIUM_NOT = re.compile(r"instal+ment|grace period|refund|deduction|income tax|\bgst\b|\btax\b|discount|loading"
                          r"|cash ?back|\bmonthly\b", re.I)
_DEDUCTIBLE = re.compile(r"\bdeductible\b[^?]*\bdiscount|\bdiscount\b[^?]*\bdeductible", re.I)
_INSTALMENT = re.compile(r"\binstal+ments?\b|\b(monthly|quarterly|half[\s-]?yearly|every (month|quarter|six months)"
                         r"|each month|per month)\b", re.I)
_PAYING = re.compile(r"\b(pay|paid|paying|premium|instal+ments?|grace)\b", re.I)
_CLAIM = re.compile(r"\bsingle[\s-]+(hospitali[sz]ation[\s-]+)?claim|\ball\s+(the\s+)?(hospitali[sz]ation\s+)?claims\b"
                    r"|\bmaximum\b[^?]*\b(payable|hospitali[sz]ation claim)|\b(how much|amount)\b[^?]*\b(secure|restore) benefit",
                    re.I)
_COMPARE = re.compile(r"\b(difference|differ|different|compare|comparison|versus|vs\.?|better)\b", re.I)
_CHECKUP = re.compile(r"\bcheck[\s-]?ups?\b", re.I)
_TIER_Q = re.compile(r"\b(which|what)\b[^?]*\btier\b|\btier\b[^?]*\b(cit(y|ies)|place|location)s?\b"
                     r"|\bis\s+\w+(\s+\w+)?\s+(a\s+|in\s+)?tier\b", re.I)
_YES_NO = re.compile(r"^\s*(can|could|may|is|are|do|does|will|would|shall)\b", re.I)

_AGE = [(re.compile(r"\b(\d{1,3})\s*-?\s*(?:years?|yrs?)\s*-?\s*old\b", re.I), 1),
        (re.compile(r"\baged?\s*(?:of|is|=|:)?\s*(\d{1,3})\b(?!\s*(?:lakh|lac|crore|%))", re.I), 1),
        (re.compile(r"\b(\d{1,3})\s*(?:years?|yrs?)\s+of\s+age\b", re.I), 1),
        (re.compile(r"\b(\d{1,2})\s*-?\s*months?\s*-?\s*old\b", re.I), 0)]
_LAKH = re.compile(r"(?:₹|\brs\.?|\binr)?\s*(\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|cr)\b", re.I)
_RUPEES = re.compile(r"(?:₹|\brs\.?|\binr)\s*(\d{1,3}(?:,\d{2,3})+|\d{4,})(?![\d,])", re.I)
_TIER = re.compile(r"\btier[\s-]*(1|2|i{1,2}|one|two)\b", re.I)
_TIER_WORD = {"1": 1, "i": 1, "one": 1, "2": 2, "ii": 2, "two": 2}
_LIVING = re.compile(r"\b(?:living|lives|live|reside|resides|residing|resident of|based|staying|stays|stay|located"
                     r"|settled|from)\s+(?:in\s+|at\s+)?([A-Za-z][A-Za-z.]+(?:\s+[A-Za-z][A-Za-z.]+)?)", re.I)
_IN_CITY = re.compile(r"\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")
_IS_PLACE = re.compile(r"\b(?:[Ii]s|[Aa]re)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:a\s+|in\s+(?:a\s+)?)?[Tt]ier\b")
_NOT_A_PLACE = {"a", "an", "the", "my", "our", "india", "tier", "year", "years", "annexure", "section", "this", "that",
                "which", "city", "metro", "town", "village", "rural", "urban", "policy", "plan", "age", "rs", "rs.",
                "rupees", "next", "last", "day", "month", "date", "renewal", "hospital", "home", "abroad", "network"}
_PLACE_STOP = {"with", "and", "for", "who", "city", "having", "taking", "buying", "opting", "under", "on", "in", "of"}
# Other names for places (general knowledge, applied only when the target place is in the policy's tier list)
PLACE_ALIASES = {"bombay": "mumbai", "new delhi": "delhi", "ncr": "national capital region (ncr)",
                 "delhi ncr": "national capital region (ncr)", "gurgaon": "national capital region (ncr)",
                 "gurugram": "national capital region (ncr)", "noida": "national capital region (ncr)",
                 "greater noida": "national capital region (ncr)", "ghaziabad": "national capital region (ncr)",
                 "faridabad": "national capital region (ncr)"}
_YEAR = re.compile(r"\byear\s*-?\s*([123])\b|\b(first|second|third|1st|2nd|3rd)\s+(?:policy\s+)?year\b", re.I)
_YEAR_RANGE = re.compile(r"\byear\s*([123])\s*(?:to|through|till|until|-|–)\s*(?:year\s*)?([123])\b", re.I)
_YEAR_WORD = {"1": 1, "first": 1, "1st": 1, "2": 2, "second": 2, "2nd": 2, "3": 3, "third": 3, "3rd": 3}
ITEMS = {  # rag.claim_limit_illustration.item -> what a question says
    "max_single_claim": r"single[\s-]+(?:hospitali[sz]ation[\s-]+)?claim|any one claim|per claim|one claim",
    "max_all_claims": r"all\s+(?:the\s+)?(?:hospitali[sz]ation\s+)?claims|total (?:of )?(?:all )?claims|aggregate",
    "secure_benefit": r"secure benefit", "restore": r"\brestore", "bonus_or_plus": r"plus benefit|cumulative bonus|\bbonus\b",
    "base_si": r"base sum insured"}
ITEM_ORDER = ["base_si", "secure_benefit", "bonus_or_plus", "restore", "max_single_claim", "max_all_claims"]
FREQUENCIES = {  # canonical key -> question wording
    "monthly": r"\bmonthly\b|\bevery month\b|\beach month\b|\bper month\b|\bmonth by month\b",
    "quarterly": r"\bquarterly\b|\bevery quarter\b|\bevery three months\b",
    "half yearly": r"\bhalf[\s-]?yearly\b|\bsix[\s-]?monthly\b|\bevery six months\b|\bsemi[\s-]?annual(ly)?\b",
    "yearly": r"(?<!half )(?<!half-)\byearly\b|\bannual(ly)?\b|\bevery year\b|\bonce a year\b"}
_FLOATER = re.compile(r"\bfloater\b|\bfamily\b", re.I)
_INDIVIDUAL = re.compile(r"\bindividual\b", re.I)


# ---------------------------------------------------------------------------------------------------- formatting
def group_inr(n) -> str:
    """Indian digit grouping: 1000000 -> '10,00,000'."""
    n = int(n)
    s, sign = str(abs(n)), "-" if n < 0 else ""
    if len(s) <= 3:
        return sign + s
    head, parts = s[:-3], []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    return sign + ",".join(([head] if head else []) + parts) + "," + s[-3:]


def _plain(d: Decimal) -> str:
    s = format(d.normalize(), "f")
    return s.rstrip("0").rstrip(".") if "." in s else s


def inr(n, words: bool = False) -> str:
    """'Rs. 13,500'; with ``words``: 'Rs. 10 lakh', 'Rs. 2 crore' for round amounts."""
    d = Decimal(n)
    if words and d >= 10_000_000 and d % 100_000 == 0:
        return f"Rs. {_plain(d / 10_000_000)} crore"
    if words and d >= 100_000 and d % 1_000 == 0:
        return f"Rs. {_plain(d / 100_000)} lakh"
    return f"Rs. {group_inr(d)}"


def printed_number(text: str) -> Optional[Decimal]:
    """The number a printed cell shows ('13,500' -> 13500, 'NIL' -> 0, '25%' -> 0.25, '15 Days' -> 15)."""
    t = (text or "").strip()
    if re.fullmatch(r"(?i)nil|-|–", t):
        return Decimal(0)
    m = re.search(r"\d[\d,]*(?:\.\d+)?", t)
    if not m:
        return None
    v = Decimal(m.group(0).replace(",", ""))
    return v / 100 if "%" in t[m.end():m.end() + 2] else v


def where(section_id: str) -> str:
    """'Annexure-C' -> 'Annexure C'; '12.13' -> 'Section 12.13'."""
    s = str(section_id or "")
    return s.replace("-", " ") if s.lower().startswith(("annexure", "schedule", "appendix")) else f"Section {s}"


def _excl_tax(title: str) -> str:
    """', excluding GST' when the table's title says the amounts exclude a tax."""
    m = re.search(r"\bexcl(?:\.|uding|usive of)?\s*(?:of\s+)?([A-Z]{2,5})\b", title or "", re.I)
    return f", excluding {m.group(1).upper()}" if m else ""


def _and(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _note(n: str) -> str:
    """'^Note: Maximum ...' -> '^ Maximum ...'; 'Note: Secure ...' -> 'Secure ...'."""
    return re.sub(r"^(\([^)]*\)\s*)?([*^#†]*)\s*note\s*:\s*",
                  lambda m: (m.group(1) or "") + ((m.group(2) + " ") if m.group(2) else ""), n, flags=re.I).strip()


# ---------------------------------------------------------------------------------------------------- data
_ACTIVE = "JOIN rag.active_versions v ON v.version_id = x.version_id AND v.doc_id = %(doc)s"
_CELL = ("JOIN rag.table_cells c ON c.version_id = x.version_id AND c.table_id = x.table_id AND c.row = x.row "
         "AND c.col = x.col JOIN rag.tables t ON t.version_id = x.version_id AND t.table_id = x.table_id")
_CITE = "x.table_id, x.row, x.col, c.text AS printed, t.pages[1] AS page, t.section_id, coalesce(t.title, t.caption) AS table_title"


class TableData:
    """Fixed, parameterised queries over the typed tables of a document's active version (read-only ``Database``)."""

    def __init__(self, db, doc_id: str):
        self.db, self.doc = db, doc_id

    def _rows(self, sql: str, params: dict) -> List[dict]:
        return self.db.query_dicts(sql, {"doc": self.doc, **params})

    def catalog(self) -> dict:
        head = self._rows("SELECT d.title, v.version_id, v.version_label FROM rag.documents d "
                          "JOIN rag.active_versions v ON v.doc_id = d.doc_id WHERE d.doc_id = %(doc)s", {})
        if not head:
            raise LookupError(f"no active version of document {self.doc!r} in the database")
        plans = self._rows(f"""
            SELECT x.plan FROM rag.plan_sum_insured x {_ACTIVE}
            UNION SELECT x.plan FROM rag.claim_limit_illustration x {_ACTIVE}
            UNION SELECT x.plan FROM rag.premium_rates x {_ACTIVE}
            UNION SELECT x.plan FROM rag.deductible_discounts x {_ACTIVE}""", {})
        premium = self._rows(f"""
            SELECT x.plan, array_agg(DISTINCT x.sum_insured ORDER BY x.sum_insured) AS sums_insured,
                   min(x.age_min) AS age_min, max(coalesce(x.age_max, x.age_min)) AS age_max,
                   bool_or(x.age_max IS NULL) AS open_ended, array_agg(DISTINCT x.tier ORDER BY x.tier) AS tiers,
                   array_agg(DISTINCT x.table_id ORDER BY x.table_id) AS tables, min(t.section_id) AS section
              FROM rag.premium_rates x {_ACTIVE}
              JOIN rag.tables t ON t.version_id = x.version_id AND t.table_id = x.table_id
             GROUP BY x.plan""", {})
        places = self._rows(f"""
            SELECT x.place, x.place_key, x.tier, x.source_block_id, b.page, b.record->>'section' AS section,
                   b.text AS printed
              FROM rag.city_tiers x {_ACTIVE}
              JOIN rag.text_blocks b ON b.version_id = x.version_id AND b.block_id = x.source_block_id
             ORDER BY x.tier, x.place""", {})
        illustration = self._rows(f"""
            SELECT array_agg(DISTINCT x.plan) AS plans, array_agg(DISTINCT x.year ORDER BY x.year) AS years
              FROM rag.claim_limit_illustration x {_ACTIVE}""", {})
        return {"doc_id": self.doc, **head[0], "plans": sorted(r["plan"] for r in plans),
                "premium": {r["plan"]: r for r in premium}, "places": places,
                "illustration": {"plans": illustration[0]["plans"] or [], "years": illustration[0]["years"] or []}}

    def premiums(self, plans: Sequence[str], tiers: Sequence[int], sums_insured: Sequence[int], age: int) -> List[dict]:
        return self._rows(f"""
            SELECT x.plan, x.tier, x.age_min, x.age_max, x.renewal_only, x.sum_insured, x.premium_inr AS value,
                   a.text AS row_label, {_CITE}
              FROM rag.premium_rates x {_ACTIVE} {_CELL}
              JOIN rag.table_cells a ON a.version_id = x.version_id AND a.table_id = x.table_id AND a.row = x.row
                                    AND a.col = 0
             WHERE x.plan = ANY(%(plans)s) AND x.tier = ANY(%(tiers)s::int[])
               AND x.sum_insured = ANY(%(sis)s::bigint[])
               AND %(age)s BETWEEN x.age_min AND coalesce(x.age_max, 1000)
             ORDER BY x.plan, x.tier, x.sum_insured""",
                          {"plans": list(plans), "tiers": list(tiers), "sis": list(sums_insured), "age": int(age)})

    def claim_limits(self, plans: Sequence[str], years: Sequence[int], items: Sequence[str]) -> List[dict]:
        return self._rows(f"""
            SELECT x.plan, x.year, x.item, x.item_printed, x.amount_inr AS value, {_CITE}
              FROM rag.claim_limit_illustration x {_ACTIVE} {_CELL}
             WHERE x.plan = ANY(%(plans)s) AND x.year = ANY(%(years)s::int[]) AND x.item = ANY(%(items)s)
             ORDER BY x.plan, x.item, x.year""", {"plans": list(plans), "years": list(years), "items": list(items)})

    def instalments(self) -> List[dict]:
        return self._rows(f"""
            SELECT x.option, x.frequency, x.grace_period_days AS value, {_CITE}
              FROM rag.instalment_options x {_ACTIVE} {_CELL} ORDER BY x.option""", {})

    def checkups(self, policy_types: Sequence[str], sums_insured: Optional[Sequence[int]]) -> List[dict]:
        return self._rows(f"""
            SELECT x.policy_type, x.applies, x.sum_insured, x.si_printed, x.limit_inr AS value, {_CITE}
              FROM rag.checkup_limits x {_ACTIVE} {_CELL}
             WHERE x.policy_type = ANY(%(types)s) AND (%(any_si)s OR x.sum_insured = ANY(%(sis)s::bigint[]))
             ORDER BY x.policy_type DESC, x.sum_insured""",
                          {"types": list(policy_types), "any_si": not sums_insured, "sis": list(sums_insured or [])})

    def deductibles(self, plans: Sequence[str], deductibles: Optional[Sequence[int]], sum_insured: Optional[int]) -> List[dict]:
        return self._rows(f"""
            SELECT x.plan, x.deductible, x.si_gt, x.si_le, x.band_printed, x.discount AS value, {_CITE}
              FROM rag.deductible_discounts x {_ACTIVE} {_CELL}
             WHERE x.plan = ANY(%(plans)s) AND (%(any_d)s OR x.deductible = ANY(%(deds)s::int[]))
               AND (%(si)s::bigint IS NULL OR ((x.si_gt IS NULL OR %(si)s::bigint > x.si_gt)
                                           AND (x.si_le IS NULL OR %(si)s::bigint <= x.si_le)))
             ORDER BY x.plan, x.deductible, x.si_gt NULLS FIRST""",
                          {"plans": list(plans), "any_d": not deductibles, "deds": list(deductibles or []),
                           "si": sum_insured})

    def notes(self, table_ids: Sequence[str]) -> List[dict]:
        """Notes and footnotes of the tables these parts belong to."""
        return self._rows("""
            SELECT r.table_id, r.parts, r.record->'notes' AS notes, r.record->'footnotes' AS footnotes
              FROM rag.table_refs r JOIN rag.active_versions v ON v.version_id = r.version_id AND v.doc_id = %(doc)s
             WHERE r.parts && %(ids)s::text[] ORDER BY r.table_id""", {"ids": list(table_ids)})


# ---------------------------------------------------------------------------------------------------- the answer
@dataclass
class TableAnswer:
    question: str
    intent: Optional[str] = None
    status: str = "not_routed"          # answered | needs_input | not_in_policy | not_routed
    fields: Dict = field(default_factory=dict)
    lead: str = ""                       # first sentence of the plain answer
    body: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    calculations: List[dict] = field(default_factory=list)
    rows: List[dict] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)

    @property
    def answered(self) -> bool:
        return self.status == "answered"

    @property
    def text(self) -> str:
        """The plain answer, written only from the rows (no answer model)."""
        parts = [self.lead] + self.body + [f"Note: {_note(n)}" for n in self.notes]
        if self.sources:
            parts.append("Source: " + "; ".join(self.sources))
        return "\n".join(p for p in parts if p)

    def facts_block(self) -> str:
        """The facts for the answer model's context (exact values from the database, with their sources)."""
        if not self.answered:
            return ""
        head = "Policy tables (exact values read from the database"
        return "\n".join([f"{head}; {self.lead}" if self.lead else head + ")"] + self.body
                         + [f"Note: {_note(n)}" for n in self.notes] + ["Source: " + "; ".join(self.sources)])

    def verify(self, answer: str) -> List[str]:
        """Figures in ``answer`` that are not in these facts or in the question."""
        from .evaluation.checks import unsupported_figures
        derived = [("inr", int(c["result"])) for c in self.calculations if c.get("unit") == "inr"]
        return unsupported_figures(answer, self.text, self.question, derived)

    def trace(self) -> List[str]:
        """Rows whose value differs from the number printed in their cell (should be none)."""
        out = []
        for r in self.rows:
            shown = printed_number(r.get("printed"))
            if shown is None or Decimal(str(r["value"])) != shown:
                out.append(f"{r['table_id']} row {r['row']} col {r['col']}: value {r['value']} vs printed {r.get('printed')!r}")
        return out

    def to_dict(self) -> dict:
        return {"question": self.question, "intent": self.intent, "status": self.status, "fields": self.fields,
                "text": self.text, "calculations": [{**c, "result": str(c["result"])} for c in self.calculations],
                "rows": [{k: (str(v) if isinstance(v, Decimal) else v) for k, v in r.items()} for r in self.rows],
                "sources": self.sources}


# ---------------------------------------------------------------------------------------------------- the route
class TableRoute:
    def __init__(self, data: TableData):
        self.data = data
        self.cat = data.catalog()
        self.plans: List[str] = self.cat["plans"]
        self._plan_res = self._plan_patterns(self.plans)
        title = self.cat.get("title") or ""
        self._product_re = re.compile(re.sub(r"\\ ", r"\\s*", re.escape(title)).replace(":", r"\s*:"), re.I) \
            if title else None
        self._plan_words = {w.lower() for p in self.plans for w in p.split()}
        first = {p.split()[0].lower() for p in self.plans if p.split()}
        self._brand = next(iter(first)) if len(first) == 1 and all(len(p.split()) > 1 for p in self.plans) else None
        self.places = {r["place_key"]: r for r in self.cat["places"]}
        self._rest = next((r for r in self.cat["places"] if "rest of" in r["place_key"]), None)
        keys = [k for k in self.places if k != (self._rest or {}).get("place_key")]
        names = {k: k for k in keys}
        names.update({a: t for a, t in PLACE_ALIASES.items() if t in self.places})
        self._place_res = [(names[n], n != names[n], re.compile(r"(?<![a-z])" + r"\s+".join(map(re.escape, n.split()))
                                                               + r"(?![a-z])", re.I))
                           for n in sorted(names, key=len, reverse=True)]

    @staticmethod
    def _plan_patterns(plans):
        """Full plan names; also without the shared first word ("Super Secure") unless that leaves one word that
        is part of another plan's name ("Secure")."""
        first = {p.split()[0].lower() for p in plans}
        brand = len(first) == 1
        out = []
        for p in sorted(plans, key=len, reverse=True):
            words = p.split()
            alts = [r"\s+".join(map(re.escape, words))]
            rest = words[1:]
            if brand and rest and (len(rest) > 1 or not any(rest[0].lower() in [w.lower() for w in q.split()[1:]]
                                                            for q in plans if q != p)):
                alts.append(r"\s+".join(map(re.escape, rest)))
            out.append((p, re.compile(r"(?<![a-z])(?:" + "|".join(alts) + r")(?![a-z])(?!\s+benefit)", re.I)))
        return out

    # ------------------------------------------------------------------ routing and fields
    def route(self, q: str) -> Optional[str]:
        cost = _COST.search(q) and (_INSURANCE.search(q) or self.find_plans(q)[0] or self.find_ages(q))
        if (_PREMIUM.search(q) or cost) and not _PREMIUM_NOT.search(q):
            return "premium"
        if _DEDUCTIBLE.search(q):
            return "deductible_discount"
        if _INSTALMENT.search(q) and _PAYING.search(q):
            return "instalment"
        if _CLAIM.search(q) or (_COMPARE.search(q) and len(self.find_plans(q)[0]) >= 2):
            return "claim_limit"
        if _CHECKUP.search(q):
            return "checkup_limit"
        if _TIER_Q.search(q):
            return "city_tier"
        return None

    def find_plans(self, q: str):
        """(plans named, unknown names that look like plans, the product named)."""
        work, product = q, False
        if self._product_re is not None and self._product_re.search(work):
            product = True
            work = self._product_re.sub(lambda m: " " * len(m.group(0)), work)
        found = []
        for name, pat in self._plan_res:
            if pat.search(work):
                found.append(name)
                work = pat.sub(lambda m: " " * len(m.group(0)), work)
        unknown = []
        if self._brand:
            for m in re.finditer(rf"(?<![a-z]){re.escape(self._brand)}\s+([a-z]+)", work, re.I):
                if m.group(1).lower() not in ("plan", "plans", "policy", "for", "and", "with"):
                    unknown.append(f"{self._brand.title()} {m.group(1).title()}")
        return [p for p in self.plans if p in found], unknown, product

    @staticmethod
    def find_ages(q: str) -> List[int]:
        out = []
        for pat, scale in _AGE:
            for m in pat.finditer(q):
                age = int(m.group(1)) if scale else 0
                if age not in out:
                    out.append(age)
        return out

    @staticmethod
    def find_amounts(q: str) -> List[tuple]:
        """(rupees, start position) of every amount: '10 lakh', 'Rs. 25,000', '1 crore'."""
        out, taken = [], []
        for m in _LAKH.finditer(q):
            unit = m.group(2).lower()
            out.append((int(Decimal(m.group(1)) * (10_000_000 if unit.startswith("c") else 100_000)), m.start()))
            taken.append((m.start(), m.end()))
        for m in _RUPEES.finditer(q):
            if not any(a <= m.start() < b for a, b in taken):
                out.append((int(m.group(1).replace(",", "")), m.start()))
        return sorted(out, key=lambda t: t[1])

    def find_places(self, q: str) -> List[tuple]:
        """Every place named, in order: (place row or None, the words used, an alias was used, a name that is not
        in the tier list)."""
        out, work = [], q
        for key, alias, pat in self._place_res:
            for m in pat.finditer(work):
                out.append((m.start(), self.places[key], m.group(0), alias, None))
            work = pat.sub(lambda m: " " * len(m.group(0)), work)
        for pat in (_LIVING, _IN_CITY, _IS_PLACE):
            for m in pat.finditer(work):
                words = m.group(1).split()
                while words and words[-1].lower() in _PLACE_STOP:
                    words.pop()
                if words and words[0].lower() not in _NOT_A_PLACE | self._plan_words \
                        and not any(w.lower() in _PLACE_STOP for w in words):
                    name = " ".join(w.strip(".") for w in words).title()
                    if all(o[4] != name for o in out):
                        out.append((m.start(), None, None, False, name))
        return [o[1:] for o in sorted(out, key=lambda o: o[0])]

    @staticmethod
    def find_tiers(q: str) -> List[int]:
        return sorted({_TIER_WORD[m.group(1).lower()] for m in _TIER.finditer(q)})

    @staticmethod
    def find_years(q: str, available) -> List[int]:
        years = set()
        for m in _YEAR_RANGE.finditer(q):
            a, b = sorted((int(m.group(1)), int(m.group(2))))
            years.update(range(a, b + 1))
        for m in _YEAR.finditer(q):
            years.add(_YEAR_WORD[(m.group(1) or m.group(2)).lower()])
        return sorted(y for y in years if y in available) or list(available)

    @staticmethod
    def find_frequencies(q: str) -> List[str]:
        return [k for k, pat in FREQUENCIES.items() if re.search(pat, q, re.I)]

    # ------------------------------------------------------------------ answering
    def answer(self, question: str) -> TableAnswer:
        intent = self.route(question)
        res = TableAnswer(question=question, intent=intent)
        if intent is None:
            return res
        getattr(self, f"_{intent}")(question, res)
        if res.answered:
            self._add_notes(res)
        return res

    def _tier_of(self, q: str, res: TableAnswer, asked_tier_is_the_question: bool = False) -> List[int]:
        """Tiers from the places named (each explained in a note), else from "Tier 1/2" in the question."""
        named = self.find_tiers(q)
        found = {}
        for place, words, alias, unknown in self.find_places(q):
            if place is not None:
                how = f"{words.strip().title()} counts as {place['place']}, which" if alias else place["place"]
                res.notes.append(f"{how} is listed as Tier {place['tier']} in {where(place['section'])} "
                                 f"(page {place['page']}).")
                found[words.strip().title() if alias else place["place"]] = place["tier"]
            elif unknown and self._rest is not None:
                res.notes.append(f"{unknown} is not named in the city list of {where(self._rest['section'])} (page "
                                 f"{self._rest['page']}), so Tier {self._rest['tier']} ({self._rest['place']}) applies.")
                found[unknown] = self._rest["tier"]
        if not found:
            return named
        res.fields["places"] = found
        if named and set(named) - set(found.values()) and not asked_tier_is_the_question:
            res.notes.append(f"The question also names Tier {_and([str(t) for t in named])}; the places decide.")
        return sorted(set(found.values()))

    def _premium(self, q: str, res: TableAnswer):
        cat = self.cat["premium"]
        named, unknown, product = self.find_plans(q)
        priced = [p for p in self.plans if p in cat]
        res.fields.update(plans_named=named, unknown_plans=unknown, product_named=product)
        if unknown:
            res.status = "not_in_policy"
            res.lead = (f"The policy does not include a plan called {', '.join(unknown)}: its plans are "
                        f"{_and(self.plans)}.")
            return
        no_table = [p for p in named if p not in cat]
        plans = [p for p in named if p in cat] or ([] if named else priced)
        if not plans:
            res.status = "not_in_policy"
            res.lead = (f"The policy does not include a premium table for {_and(no_table)}: "
                        f"{where(cat[priced[0]]['section'])} gives premiums only for {_and(priced)}.")
            return
        if no_table:
            res.notes.append(f"The policy does not include a premium table for {', '.join(no_table)}.")
        ages = self.find_ages(q)
        res.fields["ages"] = ages
        if not ages:
            res.status = "needs_input"
            res.lead = "The premium depends on the age of the insured person; the question does not give one."
            return
        lo, hi = min(cat[p]["age_min"] for p in plans), max(cat[p]["age_max"] for p in plans)
        bad = [a for a in ages if a < lo or (a > hi and not all(cat[p]["open_ended"] for p in plans)) or a > 120]
        if bad:
            res.status = "not_in_policy"
            res.lead = f"The premium table does not include age {bad[0]}: it covers ages {lo} to {hi}."
            return
        offered = sorted(set.intersection(*(set(cat[p]["sums_insured"]) for p in plans)))
        asked = [a for a, _ in self.find_amounts(q)]
        sis = [a for a in asked if a in offered]
        res.fields["sums_insured"] = asked
        if asked and not sis:
            res.status = "not_in_policy"
            res.lead = (f"The premium table does not include a Sum Insured of {inr(asked[0], words=True)}: it lists "
                        f"{', '.join(inr(s, words=True) for s in offered)}.")
            return
        if len(sis) < len(asked):
            res.notes.append("Not in the premium table: " + ", ".join(inr(a, words=True) for a in asked if a not in offered))
        tiers = self._tier_of(q, res) or sorted({t for p in plans for t in cat[p]["tiers"]})
        res.fields.update(plans=plans, tiers=tiers, sums_insured_used=sis or offered)
        if not res.fields.get("places") and not self.find_tiers(q):
            res.notes.append("The premium also depends on the city tier (Section 14); both tiers are shown.")
        rows = [r for age in ages for r in self.data.premiums(plans, tiers, sis or offered, age)]
        if not rows:
            res.status = "not_in_policy"
            res.lead = "The premium table has no row for these details."
            return
        res.rows, res.status = rows, "answered"
        for r in rows:
            r["age"] = next(a for a in ages if r["age_min"] <= a <= (r["age_max"] if r["age_max"] is not None else 1000))
        self._premium_text(res, rows, ages, plans, tiers, sis or offered)

    def _premium_text(self, res, rows, ages, plans, tiers, sis):
        src = where(rows[0]["section_id"]) + _excl_tax(rows[0].get("table_title"))
        def label(r, skip=()):
            bits = [r["plan"] if "plan" not in skip else None, f"Tier {r['tier']}" if "tier" not in skip else None,
                    f"age {r['age']}" if "age" not in skip else None,
                    f"Sum Insured {inr(r['sum_insured'], words=True)}" if "si" not in skip else None]
            return ", ".join(b for b in bits if b)
        if len(rows) == 1:
            r = rows[0]
            res.lead = f"{inr(r['value'])} a year ({label(r)}; {src})."
        else:
            res.lead = f"Annual premiums ({src}):"
            if len(rows) <= 6:
                res.body += [f"- {label(r)}: {inr(r['value'])}" for r in rows]
            else:
                for age in ages:
                    for tier in tiers:
                        block = [r for r in rows if r["age"] == age and r["tier"] == tier]
                        if not block:
                            continue
                        cols = sorted({r["sum_insured"] for r in block})
                        res.body.append(f"Age {age}, Tier {tier} (Sum Insured across):")
                        res.body.append("| Plan | " + " | ".join(inr(s, words=True) for s in cols) + " |")
                        res.body.append("|---" * (len(cols) + 1) + "|")
                        for p in plans:
                            vals = {r["sum_insured"]: r["value"] for r in block if r["plan"] == p}
                            if vals:
                                res.body.append(f"| {p} | " + " | ".join(inr(vals[s]) if s in vals else "–" for s in cols) + " |")
        # how much more: two plans (or two tiers) with everything else equal
        for key, other in (("plan", ("tier", "age", "sum_insured")), ("tier", ("plan", "age", "sum_insured"))):
            groups = {}
            for r in rows:
                groups.setdefault(tuple(r[k] for k in other), []).append(r)
            for g in groups.values():
                if len(g) == 2 and len(rows) == 2:
                    lo_r, hi_r = sorted(g, key=lambda r: r["value"])
                    diff = Decimal(hi_r["value"]) - Decimal(lo_r["value"])
                    a, b = (hi_r[key], lo_r[key]) if key == "plan" else (self._tier_label(res, hi_r["tier"]),
                                                                          self._tier_label(res, lo_r["tier"]))
                    res.calculations.append({"what": f"{a} minus {b}", "inputs": [int(hi_r["value"]), int(lo_r["value"])],
                                             "result": diff, "unit": "inr"})
                    res.lead = (f"{a} costs {inr(diff)} more a year than {b}: {inr(hi_r['value'])} against "
                                f"{inr(lo_r['value'])} ({label(hi_r, skip=(key,))}; {src}).")
                    res.body = [f"- {label(r)}: {inr(r['value'])}" for r in g]
        for r in rows:
            if r["renewal_only"] and f"Age {r['age']} is printed as '{r['row_label']}'." not in res.notes:
                res.notes.append(f"Age {r['age']} is printed as '{r['row_label']}'.")

    @staticmethod
    def _tier_label(res, tier) -> str:
        names = [n for n, t in (res.fields.get("places") or {}).items() if t == tier]
        return f"Tier {tier}" + (f" ({_and(names)})" if names else "")

    def _claim_limit(self, q: str, res: TableAnswer):
        avail = self.cat["illustration"]
        named, unknown, product = self.find_plans(q)
        res.fields.update(plans_named=named, unknown_plans=unknown)
        if unknown:
            res.status = "not_in_policy"
            res.lead = (f"The policy does not include a plan called {', '.join(unknown)}: its plans are "
                        f"{_and(self.plans)}.")
            return
        plans = [p for p in named if p in avail["plans"]] or sorted(avail["plans"], key=self.plans.index)
        years = self.find_years(q, avail["years"])
        items = [k for k in ITEM_ORDER if re.search(ITEMS[k], q, re.I)]
        compare = len(named) >= 2 and _COMPARE.search(q)
        if compare or not items:
            items = ITEM_ORDER if compare else ["max_single_claim", "max_all_claims"]
        res.fields.update(plans=plans, years=years, items=items)
        rows = self.data.claim_limits(plans, years, items)
        if not rows:
            res.status = "not_in_policy"
            res.lead = "The claim illustration has no row for these details."
            return
        res.rows, res.status = rows, "answered"
        res.lead = (f"From the illustration in {where(rows[0]['section_id'])} (maximum amounts payable for "
                    "hospitalization claims, per plan and policy year):")
        res.body.append("| Plan – item | " + " | ".join(f"Year {y}" for y in years) + " |")
        res.body.append("|---" * (len(years) + 1) + "|")
        for p in plans:
            for it in items:
                vals = {r["year"]: r for r in rows if r["plan"] == p and r["item"] == it}
                if vals:
                    name = next(iter(vals.values()))["item_printed"]
                    cells = [("NIL" if Decimal(vals[y]["value"]) == 0 else inr(vals[y]["value"], words=True))
                             if y in vals else "–" for y in years]
                    res.body.append(f"| {p} – {name} | " + " | ".join(cells) + " |")

    def _instalment(self, q: str, res: TableAnswer):
        rows = self.data.instalments()
        if not rows:
            res.status = "not_in_policy"
            res.lead = "The policy does not include an instalment table."
            return
        key = lambda f: "half yearly" if "half" in f.lower() else next((k for k in ("monthly", "quarterly", "yearly")
                                                                        if k in f.lower()), f.lower())
        asked = self.find_frequencies(q)
        res.fields["frequencies"] = asked
        offered = {key(r["frequency"]) for r in rows}
        picked = [r for r in rows if key(r["frequency"]) in asked] or rows
        res.rows, res.status = picked, "answered"
        section = where(picked[0]["section_id"])
        if _YES_NO.search(q) and asked:
            missing = [a for a in asked if a not in offered]
            res.lead = (("Yes. " if not missing else "No. ") +
                        (f"Premium can be paid in {', '.join(r['frequency'] for r in picked if key(r['frequency']) in asked)} "
                         f"instalments ({section})." if not missing else
                         f"The policy offers only {_and([r['frequency'] for r in rows])} instalments ({section})."))
        else:
            res.lead = f"Premium instalment options and their grace periods ({section}):"
        res.body += [f"- {r['frequency']} ({r['option']}): grace period {r['value']} days" for r in picked]

    def _checkup_limit(self, q: str, res: TableAnswer):
        types = (["family_floater"] if _FLOATER.search(q) else []) + (["individual"] if _INDIVIDUAL.search(q) else [])
        sis = [a for a, _ in self.find_amounts(q)]
        rows = self.data.checkups(types or ["individual", "family_floater"], sis)
        res.fields.update(policy_types=types, sums_insured=sis)
        if not rows:
            res.status = "not_in_policy"
            res.lead = ("The check-up table does not include that Sum Insured." if sis else
                        "The policy does not include a check-up limit table.")
            return
        res.rows, res.status = rows, "answered"
        res.lead = f"Preventive Health Check-up limits per policy year ({where(rows[0]['section_id'])}):"
        for t in dict.fromkeys(r["policy_type"] for r in rows):
            block = [r for r in rows if r["policy_type"] == t]
            res.body.append(block[0]["applies"].strip())
            res.body += [f"- {'Family floater' if t == 'family_floater' else 'Individual'} policy, Base Sum Insured "
                         f"{inr(r['sum_insured'], words=True)} (printed: {r['si_printed']}): up to {inr(r['value'])}"
                         for r in block]

    def _deductible_discount(self, q: str, res: TableAnswer):
        named, unknown, _ = self.find_plans(q)
        if unknown:
            res.status = "not_in_policy"
            res.lead = f"The policy does not include a plan called {', '.join(unknown)}."
            return
        amounts = [v for v, _ in self.find_amounts(q)]
        deds = [v for v in amounts if v < 500_000]                 # deductibles are below the smallest Sum Insured
        si = next((v for v in amounts if v >= 500_000), None)
        plans = named or self.plans
        rows = self.data.deductibles(plans, deds, si)
        res.fields.update(plans=plans, deductibles=deds, sum_insured=si)
        if not rows:
            res.status = "not_in_policy"
            res.lead = "The deductible discount table has no row for these details."
            return
        res.rows, res.status = rows, "answered"
        res.lead = f"Aggregate Deductible discounts ({where(rows[0]['section_id'])}):"
        for r in rows:
            pct = Decimal(r["value"]) * 100
            band = f", {r['band_printed']}" if r["band_printed"] else ""
            res.body.append(f"- {r['plan']}, deductible {inr(r['deductible'])}{band}: {_plain(pct)}% discount")
            res.calculations.append({"what": f"{r['plan']} deductible {r['deductible']} as a percentage",
                                     "inputs": [str(r["value"])], "result": pct, "unit": "pct"})

    def _city_tier(self, q: str, res: TableAnswer):
        if not self.places:
            res.status = "not_in_policy"
            res.lead = "The policy does not include a city tier list."
            return
        first = self.cat["places"][0]
        res.status = "answered"
        if self.find_places(q):
            self._tier_of(q, res, asked_tier_is_the_question=True)
            places, named = res.fields["places"], self.find_tiers(q)
            answer = "; ".join(f"{n}: Tier {t}" for n, t in places.items()) + "."
            if _YES_NO.search(q) and len(named) == 1:
                answer = ("Yes. " if all(t == named[0] for t in places.values()) else "No. ") + answer
            res.lead = answer
            return
        by_tier = {}
        for r in self.cat["places"]:
            by_tier.setdefault(r["tier"], []).append(r["place"])
        res.lead = f"City tiers for the premium ({where(first['section'])}, page {first['page']}):"
        res.body += [f"- Tier {t}: {', '.join(ps)}" for t, ps in sorted(by_tier.items())]
        res.sources.append(f"{where(first['section'])}, page {first['page']}")

    def _add_notes(self, res: TableAnswer):
        """Table notes, and the footnotes whose mark appears in a cell used; one source line per table."""
        ids = sorted({r["table_id"] for r in res.rows if r.get("table_id")})
        if not ids:
            return
        marks = " ".join(str(r.get(k) or "") for r in res.rows for k in ("printed", "row_label", "item_printed"))
        seen = {}
        for r in res.rows:
            seen.setdefault(r["table_id"], r)
        refs = self.data.notes(ids)
        for n in refs:
            part = next((seen[t] for t in n["parts"] if t in seen), None)
            prefix = f"({part['table_title']}) " if len(refs) > 1 and part and part.get("table_title") \
                and len(part["table_title"]) < 40 else ""
            for text in n["notes"] or []:
                if text not in res.notes:
                    res.notes.append(text)
            for f in n["footnotes"] or []:
                if f.get("marker") and f["marker"] in marks and prefix + f["text"] not in res.notes:
                    res.notes.append(prefix + f["text"])
        for tid, r in seen.items():
            title = f" – {r['table_title']}" if r.get("table_title") and len(seen) <= 4 else ""
            res.sources.append(f"{where(r['section_id'])}, page {r['page']} (table {tid}){title}")
