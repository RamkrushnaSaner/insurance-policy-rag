"""Answers from the policy's tables in the database: premiums, claim limits, instalment grace periods, check-up
limits, deductible discounts and city tiers; and (migration 002) the printed lists (waiting-period conditions,
non-payable items, Major Medical Illnesses, Global Health Cover benefits, relationships), claim deadlines, grievance
contacts and the Sum Insured utilisation examples.

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
   ``TableAnswer.trace()``: rows whose typed value (or item text) differs from the printed cell;
8. ``TableAnswer.evidence_text``: the printed cells behind the facts (with their headings), and
   ``TableAnswer.linked_chunks``: the text chunks linked to the tables used (the clause that introduces or refers to
   them), so the text around a table can reach the answer model too.

List questions are answered by keyword search on the lists' full-text indexes, with a precision rule: an item is
kept when every distinctive word of it is in the question, when two or more are, or when it starts with one of the
question's distinctive words and holds at least half of them. Words common to coverage questions ("charges",
"covered") and words of the list's own name ("Major Medical Illness") do not count, and a match only on words of a
text section's title is left to that section ("Does the policy pay for an ambulance?").

Plan names, places, Sums Insured, ages and frequencies come from the database, so nothing here is specific to one
policy except the question patterns (English) and the item codes of the schema.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Sequence

INTENTS = ("premium", "deductible_discount", "instalment", "claim_limit", "checkup_limit", "city_tier",
           "claim_deadline", "grievance_contact", "si_utilisation", "list_lookup")
COMPLETE_INTENTS = INTENTS[:6]     # the facts are the whole answer (a looked-up or calculated value)

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

# migration 002 tables
_DEADLINE = re.compile(
    r"\bhow\b[^?]*\b(file|make|submit|lodge|raise|claim)\w*\b[^?]*\breimburs\w*|\breimburs\w*\s+claims?\b[^?]*\b(procedure|"
    r"process|steps?|stepwise|deadline|time limit|within|how (many|long|soon))\b|\b(deadline|time limit|how many days|how long"
    r"|how soon|within)\b[^?]*\b(submit|send|file|lodge)\w*\b[^?]*\b(claims?|documents?|bills?)\b|\b(submit|send|file|lodge)"
    r"\w*\b[^?]*\b(claims?|documents?|bills?)\b[^?]*\b(deadline|time limit|how many days|how long|how soon|within)\b", re.I)
_PAID_WHEN = re.compile(r"\b(get|be|being|am|is|are)\s+(paid|reimbursed|settled)\b|\bsettle\w*|\bpayment of (the )?claim", re.I)
_GRIEVANCE = re.compile(r"\bgrievances?\b|\bcomplain(t|ts|ing)?\b|\bescalat\w*", re.I)
_SI_USE = re.compile(r"\butili[sz]ation of (the )?sum insured\b|\b(illustration|example)s?\b[^?]*\bsum insured\b|\bsum insured\b"
                     r"[^?]*\b(utili[sz]ed|used|consumed)\b[^?]*\b(claims?|example|illustration)\b|\bhow\b[^?]*\bsum insured\b"
                     r"[^?]*\b(used|utili[sz]ed|consumed)\b", re.I)
_ABOUT_ITEM = re.compile(r"\b(cover\w*|pa(y|id|yable)|reimburs\w*|claim\w*|exclu\w*|waiting|includ\w*|allowed|admissible|"
                         r"eligible|available|entitled|count(s|ed)? as|considered|insur\w*|add(ed)?|list\w*|major medical|"
                         r"non[\s-]?medical|annexure|relationships?)\b", re.I)
_PERIOD = re.compile(r"\b(\d+)[\s-]*(day|month|year)s?\b", re.I)
_LIST_ASK = r"\b(which|what|list|all|name)\b(?:\s+[\w'-]+){0,4}?\s+"
LIST_TABLES = {   # table -> its item column, order and extra columns, and "the whole list is asked for" patterns
    "waiting_period_conditions": {
        "item": "condition", "no": "x.list_no", "order": "x.category COLLATE \"C\", x.list_no",
        "extra": "x.exclusion_code, x.category, x.waiting_months, x.waiting_block, b.page AS waiting_page, "
                 "b.record->>'section' AS waiting_section, b.text AS waiting_text",
        "join": "JOIN rag.text_blocks b ON b.version_id = x.version_id AND b.block_id = x.waiting_block",
        "whole": re.compile(_LIST_ASK + r"(conditions|diseases|illnesses|ailments|procedures|surgeries|treatments)\b[^?]*"
                            r"\bwaiting\b|\blist of\b[^?]*\bwaiting\b|\bwaiting\b[^?]*\b(list|which)\b", re.I)},
    "non_payable_items": {
        "item": "item", "no": "x.item_no", "order": "x.list COLLATE \"C\", x.item_no", "extra": "x.list, x.list_title",
        "context": "x.list_title",
        "whole": re.compile(r"(?=[^?]*\b(which|what|list)\b)[^?]*\b(non[\s-]?(medical|payable)|annexure[\s-]*b)\b|"
                            + _LIST_ASK + r"(items|consumables|things)\b[^?]*\b(not|non)[\s-]+(covered|payable|paid|"
                            r"reimbursable|reimbursed)\b", re.I)},
    "major_medical_illnesses": {
        "item": "illness", "no": "x.item_no", "order": "x.item_no", "extra": "", "context": "coalesce(t.caption, t.title)",
        "whole": re.compile(r"\b(which|what|list|all)\b[^?]*\bmajor medical illness(es)?\b|\bmajor medical illnesses\b[^?]*"
                            r"\b(which|what|list|include)\b", re.I)},
    "global_cover_benefits": {
        "item": "benefit", "no": "NULL::int", "order": "string_to_array(x.cover_section, '.')::int[], string_to_array(x.base_section, '.')::int[]",
        "extra": "x.cover_section, x.cover, x.benefit_code, x.base_section", "context": "x.cover",
        "whole": re.compile(_LIST_ASK + r"(benefits|covers|treatments|expenses)\b[^?]*\bglobal (health )?cover\b|\bglobal "
                            r"(health )?cover\b[^?]*\b(which|what|list)\b[^?]*\b(benefits|covers|include\w*|available)\b|"
                            r"\bwhat does (the )?global (health )?cover (include|cover)", re.I)},
    "eligible_relationships": {
        "item": "relationship", "no": "x.item_no", "order": "x.item_no", "extra": "x.item_printed",
        "tsv": "to_tsvector('english'::regconfig, x.relationship)",
        "whole": re.compile(r"\b(which|what)\b[^?]*\b(relationships?|relatives|family members)\b|\bwho (all )?can (i |we )?"
                            r"(be )?(cover|covered|insure|insured|include|included|add|added)\b|\bwho is eligible\b", re.I)},
}
# Words common to questions about cover: they do not tell one list item from another ("attendant charges").
GENERIC_WORDS = ("cover covered covers coverage covering policy policies plan plans insurance insured insurer company "
                 "claim claims claimed pay paid payable payment reimburse reimbursed reimbursement reimbursable expense "
                 "expenses cost costs charge charges amount amounts benefit benefits treatment treatments treated hospital "
                 "hospitals hospitalisation hospitalization room admission admitted patient medical health list listed "
                 "include included includes including item items condition conditions illness illnesses disease diseases "
                 "procedure procedures surgery surgeries surgical waiting period periods month months year years day days "
                 "exact specific specified complete details detail full give tell know need want get getting count counts "
                 "considered consider part type types kind kinds category categories annexure section clause page "
                 "exclusion exclusions excluded exclude available allowed eligible admissible apply applies applicable "
                 "also still able possible required require use used many much take taken")
SYNONYMS = [(re.compile(r"\b(tv|telly)\b", re.I), "television")]


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
                   array_agg(DISTINCT x.table_id COLLATE "C" ORDER BY x.table_id COLLATE "C") AS tables, min(t.section_id) AS section
              FROM rag.premium_rates x {_ACTIVE}
              JOIN rag.tables t ON t.version_id = x.version_id AND t.table_id = x.table_id
             GROUP BY x.plan""", {})
        places = self._rows(f"""
            SELECT x.place, x.place_key, x.tier, x.source_block_id, b.page, b.record->>'section' AS section,
                   b.text AS printed
              FROM rag.city_tiers x {_ACTIVE}
              JOIN rag.text_blocks b ON b.version_id = x.version_id AND b.block_id = x.source_block_id
             ORDER BY x.tier, x.place COLLATE "C" """, {})
        illustration = self._rows(f"""
            SELECT array_agg(DISTINCT x.plan) AS plans, array_agg(DISTINCT x.year ORDER BY x.year) AS years
              FROM rag.claim_limit_illustration x {_ACTIVE}""", {})
        has_lists = self._rows("SELECT to_regclass('rag.non_payable_items') IS NOT NULL AS ok", {})[0]["ok"]
        return {"doc_id": self.doc, **head[0], "plans": sorted(r["plan"] for r in plans), "has_lists": bool(has_lists),
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
             ORDER BY x.plan COLLATE "C", x.tier, x.sum_insured""",
                          {"plans": list(plans), "tiers": list(tiers), "sis": list(sums_insured), "age": int(age)})

    def claim_limits(self, plans: Sequence[str], years: Sequence[int], items: Sequence[str]) -> List[dict]:
        return self._rows(f"""
            SELECT x.plan, x.year, x.item, x.item_printed, x.amount_inr AS value, {_CITE}
              FROM rag.claim_limit_illustration x {_ACTIVE} {_CELL}
             WHERE x.plan = ANY(%(plans)s) AND x.year = ANY(%(years)s::int[]) AND x.item = ANY(%(items)s)
             ORDER BY x.plan COLLATE "C", x.item COLLATE "C", x.year""", {"plans": list(plans), "years": list(years), "items": list(items)})

    def instalments(self) -> List[dict]:
        return self._rows(f"""
            SELECT x.option, x.frequency, x.grace_period_days AS value, {_CITE}
              FROM rag.instalment_options x {_ACTIVE} {_CELL} ORDER BY x.option COLLATE "C" """, {})

    def checkups(self, policy_types: Sequence[str], sums_insured: Optional[Sequence[int]]) -> List[dict]:
        return self._rows(f"""
            SELECT x.policy_type, x.applies, x.sum_insured, x.si_printed, x.limit_inr AS value, {_CITE}
              FROM rag.checkup_limits x {_ACTIVE} {_CELL}
             WHERE x.policy_type = ANY(%(types)s) AND (%(any_si)s OR x.sum_insured = ANY(%(sis)s::bigint[]))
             ORDER BY x.policy_type COLLATE "C" DESC, x.sum_insured""",
                          {"types": list(policy_types), "any_si": not sums_insured, "sis": list(sums_insured or [])})

    def deductibles(self, plans: Sequence[str], deductibles: Optional[Sequence[int]], sum_insured: Optional[int]) -> List[dict]:
        return self._rows(f"""
            SELECT x.plan, x.deductible, x.si_gt, x.si_le, x.band_printed, x.discount AS value, {_CITE}
              FROM rag.deductible_discounts x {_ACTIVE} {_CELL}
             WHERE x.plan = ANY(%(plans)s) AND (%(any_d)s OR x.deductible = ANY(%(deds)s::int[]))
               AND (%(si)s::bigint IS NULL OR ((x.si_gt IS NULL OR %(si)s::bigint > x.si_gt)
                                           AND (x.si_le IS NULL OR %(si)s::bigint <= x.si_le)))
             ORDER BY x.plan COLLATE "C", x.deductible, x.si_gt NULLS FIRST""",
                          {"plans": list(plans), "any_d": not deductibles, "deds": list(deductibles or []),
                           "si": sum_insured})

    # ---- migration 002 tables
    def stems(self, *texts: str):
        """The words of a text as the full-text indexes store them (English stemming, stop words dropped); one list
        per text when several are given (one query)."""
        rows = self.db.query_dicts("""
            SELECT coalesce(tsvector_to_array(to_tsvector('english'::regconfig, t)), '{}') AS s
              FROM unnest(%(t)s::text[]) WITH ORDINALITY AS u(t, n) ORDER BY n""", {"t": list(texts)})
        out = [list(r["s"] or []) for r in rows]
        return out[0] if len(texts) == 1 else out

    def list_lexicon(self, table: str) -> dict:
        """The indexed words of a list ({"items": number of items, "df": {word: items with it}}) and the words of
        its name ("context": "Major Medical Illness", the cover a benefit belongs to, ...)."""
        spec = LIST_TABLES[table]
        tsv = spec.get("tsv", "x.tsv")
        rows = self._rows(f"""
            SELECT w, count(*) AS n FROM rag.{table} x {_ACTIVE}, unnest(tsvector_to_array({tsv})) w GROUP BY w""", {})
        items = self._rows(f"SELECT count(*) AS n FROM rag.{table} x {_ACTIVE}", {})[0]["n"]
        context = self._rows(f"""
            SELECT DISTINCT unnest(tsvector_to_array(to_tsvector('english'::regconfig, coalesce({spec['context']}, '')))) AS w
              FROM rag.{table} x {_ACTIVE} JOIN rag.tables t ON t.version_id = x.version_id AND t.table_id = x.table_id""",
                             {}) if spec.get("context") else []
        return {"items": items, "df": {r["w"]: r["n"] for r in rows}, "context": {r["w"] for r in context}}

    def section_titles(self) -> List[dict]:
        """The titles of the text sections, with their indexed words."""
        return self._rows("""
            SELECT DISTINCT x.section_title AS title,
                   tsvector_to_array(to_tsvector('english'::regconfig, x.section_title)) AS stems
              FROM rag.chunks x JOIN rag.active_versions v ON v.version_id = x.version_id AND v.doc_id = %(doc)s
             WHERE coalesce(x.section_title, '') <> ''""", {})

    def list_rows(self, table: str, stems: Optional[Sequence[str]] = None) -> List[dict]:
        """A whole list (``stems`` None), or its items whose index has any of ``stems`` (in list order)."""
        spec = LIST_TABLES[table]
        tsv = spec.get("tsv", "x.tsv")
        extra = spec["extra"] + ", " if spec["extra"] else ""
        return self._rows(f"""
            SELECT {spec['no']} AS item_no, x.{spec['item']} AS item, {extra}{_CITE},
                   (SELECT array_agg(u.lexeme ORDER BY u.positions[1]) FROM unnest({tsv}) u) AS stems
              FROM rag.{table} x {_ACTIVE} {_CELL} {spec.get('join', '')}
             WHERE %(all)s OR tsvector_to_array({tsv}) && %(stems)s::text[]
             ORDER BY {spec['order']}""", {"all": stems is None, "stems": list(stems or [])})

    def claim_deadlines(self) -> List[dict]:
        return self._rows(f"""
            SELECT x.claim_type, x.deadline_days AS value, x.counted_from, {_CITE}
              FROM rag.claim_deadlines x {_ACTIVE} {_CELL} ORDER BY x.row""", {})

    def grievance_contacts(self) -> List[dict]:
        return self._rows(f"""
            SELECT x.contact_point, x.level_no, x.level, x.details AS item, x.phones, x.emails, {_CITE}
              FROM rag.grievance_contacts x {_ACTIVE} {_CELL} ORDER BY x.row, x.level_no""", {})

    def si_examples(self) -> List[dict]:
        return self._rows(f"""
            SELECT x.example, x.claim_no AS value, x.claim_inr, x.base_available_inr, x.plus_available_inr,
                   x.secure_available_inr, x.restore_available_inr, x.admissible_inr, x.utilisation,
                   r.reference_text, {_CITE}
              FROM rag.si_utilisation_examples x {_ACTIVE} {_CELL}
              LEFT JOIN rag.table_refs r ON r.version_id = x.version_id AND x.table_id = ANY(r.parts)
             ORDER BY x.example, x.claim_no""", {})

    def evidence(self, cells: Sequence[tuple]) -> List[dict]:
        """Printed cells, each with its table's caption and its headings: ``cells`` = (table_id, row, col), or
        (table_id, row, None) for every cell of a row."""
        if not cells:
            return []
        return self._rows("""
            SELECT c.table_id, c.row, c.col, c.text, c.header_path, c.row_path, coalesce(t.caption, t.title) AS caption
              FROM rag.table_cells c JOIN rag.active_versions v ON v.version_id = c.version_id AND v.doc_id = %(doc)s
              JOIN rag.tables t ON t.version_id = c.version_id AND t.table_id = c.table_id
              JOIN unnest(%(tids)s::text[], %(rows)s::int[], %(cols)s::int[]) AS k(table_id, row, col)
                ON k.table_id = c.table_id AND k.row = c.row AND (k.col IS NULL OR k.col = c.col)
             WHERE NOT coalesce(c.is_header, false)
             ORDER BY c.table_id COLLATE "C", c.row, c.col""", {"tids": [c[0] for c in cells], "rows": [c[1] for c in cells],
                                                     "cols": [c[2] for c in cells]})

    def linked_chunks(self, table_ids: Sequence[str]) -> List[dict]:
        """Text chunks linked to the tables these parts belong to (introduced by, same clause, notes, refers to)."""
        if not table_ids:
            return []
        return self._rows("""
            SELECT ct.chunk_uid, ct.relation, ct.table_id
              FROM rag.chunk_tables ct JOIN rag.active_versions v ON v.version_id = ct.version_id AND v.doc_id = %(doc)s
              JOIN rag.table_refs r ON r.version_id = ct.version_id AND r.table_id = ct.table_id
             WHERE r.parts && %(ids)s::text[]
             ORDER BY array_position(ARRAY['introduced_by', 'same_clause', 'notes_in', 'refers_to'], ct.relation),
                      ct.table_id COLLATE "C", ct.chunk_uid COLLATE "C" """, {"ids": list(table_ids)})

    def all_rows_text(self) -> str:
        """Every printed cell of the active version, row by row (for the evaluation: what the tables hold)."""
        rows = self._rows("""
            SELECT coalesce(t.caption, t.title, '') || ' ' || string_agg(coalesce(c.text, ''), ' ' ORDER BY c.col) AS line
              FROM rag.table_cells c JOIN rag.active_versions v ON v.version_id = c.version_id AND v.doc_id = %(doc)s
              JOIN rag.tables t ON t.version_id = c.version_id AND t.table_id = c.table_id
             GROUP BY c.table_id, c.row, t.caption, t.title ORDER BY c.table_id COLLATE "C", c.row""", {})
        return "\n".join(r["line"] for r in rows)

    def notes(self, table_ids: Sequence[str]) -> List[dict]:
        """Notes and footnotes of the tables these parts belong to."""
        return self._rows("""
            SELECT r.table_id, r.parts, r.record->'notes' AS notes, r.record->'footnotes' AS footnotes
              FROM rag.table_refs r JOIN rag.active_versions v ON v.version_id = r.version_id AND v.doc_id = %(doc)s
             WHERE r.parts && %(ids)s::text[] ORDER BY r.table_id COLLATE "C" """, {"ids": list(table_ids)})


# ---------------------------------------------------------------------------------------------------- the answer
@dataclass
class TableAnswer:
    question: str
    intent: Optional[str] = None
    status: str = "not_routed"          # answered | needs_input | not_in_policy | no_match | not_routed
    fields: Dict = field(default_factory=dict)
    lead: str = ""                       # first sentence of the plain answer
    body: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    calculations: List[dict] = field(default_factory=list)
    rows: List[dict] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    cells: List[dict] = field(default_factory=list)          # the printed cells behind the facts
    evidence_text: str = ""
    linked_chunks: List[str] = field(default_factory=list)   # chunk_uid of text chunks linked to the tables used

    @property
    def answered(self) -> bool:
        return self.status == "answered"

    @property
    def complete(self) -> bool:
        """The facts are the whole answer (a looked-up or calculated value), not part of it."""
        return self.answered and self.intent in COMPLETE_INTENTS

    @property
    def table_ids(self) -> List[str]:
        return sorted({r["table_id"] for r in self.rows if r.get("table_id")})

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
        head = "Policy tables (exact values read from the database)."
        return "\n".join([f"{head} {self.lead}" if self.lead else head] + self.body
                         + [f"Note: {_note(n)}" for n in self.notes] + ["Source: " + "; ".join(self.sources)])

    def verify(self, answer: str) -> List[str]:
        """Figures in ``answer`` that are not in these facts or in the question."""
        from .evaluation.checks import unsupported_figures
        derived = [("inr", int(c["result"])) for c in self.calculations if c.get("unit") == "inr"]
        return unsupported_figures(answer, self.text, self.question, derived)

    def trace(self) -> List[str]:
        """Rows whose value differs from the number printed in their cell, list items that differ from their cell,
        and amounts of a row that are not printed in that row (should be none)."""
        out = []
        for r in self.rows:
            where_ = f"{r['table_id']} row {r['row']} col {r['col']}"
            if "value" in r:
                shown = printed_number(r.get("printed"))
                if shown is None or Decimal(str(r["value"])) != shown:
                    out.append(f"{where_}: value {r['value']} vs printed {r.get('printed')!r}")
            elif " ".join(str(r.get("item") or "").split()) != " ".join(str(r.get("printed") or "").split()):
                out.append(f"{where_}: item {r.get('item')!r} vs printed {r.get('printed')!r}")
            amounts = [r[k] for k in r if k.endswith("_inr") and r[k] is not None]
            if amounts:
                printed = {printed_number(c["text"]) for c in self.cells
                           if c["table_id"] == r["table_id"] and c["row"] == r["row"]}
                out += [f"{where_}: amount {a} not printed in the row" for a in amounts if Decimal(a) not in printed]
        return out

    def to_dict(self) -> dict:
        return {"question": self.question, "intent": self.intent, "status": self.status, "fields": self.fields,
                "text": self.text, "calculations": [{**c, "result": str(c["result"])} for c in self.calculations],
                "rows": [{k: (str(v) if isinstance(v, Decimal) else v) for k, v in r.items()} for r in self.rows],
                "sources": self.sources, "table_ids": self.table_ids, "linked_chunks": self.linked_chunks}


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
        self.has_lists = bool(self.cat.get("has_lists"))
        self._generic = set(data.stems(GENERIC_WORDS)) if self.has_lists else set()
        self._titles = [set(r["stems"] or []) - self._generic for r in data.section_titles()] if self.has_lists else []
        self._titles = [t for t in self._titles if t]
        self._lexicons: Dict[str, dict] = {}

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
        if self.has_lists:
            if _DEADLINE.search(q) and not _PAID_WHEN.search(q):
                return "claim_deadline"
            if _GRIEVANCE.search(q):
                return "grievance_contact"
            if _SI_USE.search(q):
                return "si_utilisation"
            if _ABOUT_ITEM.search(q) or any(spec["whole"].search(q) for spec in LIST_TABLES.values()):
                return "list_lookup"
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
            self._evidence(res)
            self._add_notes(res)
            res.linked_chunks = list(dict.fromkeys(r["chunk_uid"] for r in self.data.linked_chunks(res.table_ids)))
        return res

    _WHOLE_ROWS = ("claim_deadline", "si_utilisation")      # the facts show every cell of these rows

    def _evidence(self, res: TableAnswer):
        """The printed cells behind the facts: each with its table caption and headings, whole rows where the facts
        show the whole row, and list items with their printed number (what reached the answer model)."""
        whole = res.intent in self._WHOLE_ROWS
        keys = sorted({(r["table_id"], r["row"], None if whole else r["col"]) for r in res.rows if r.get("table_id")},
                      key=lambda k: (k[0], k[1], -1 if k[2] is None else k[2]))
        res.cells = self.data.evidence(keys)
        by_row, lines = {}, []
        for c in res.cells:
            by_row.setdefault((c["table_id"], c["row"]), []).append(c)
        for cs in by_row.values():
            cap = cs[0]["caption"] or ""
            if whole:
                lines.append(" ".join([cap] + [c["text"] or "" for c in cs]))
            lines += [" ".join([cap] + list(c["header_path"] or []) + list(c["row_path"] or []) + [c["text"] or ""]) for c in cs]
        lines += [f"{r['item_no']} {r['item']}" for r in res.rows if r.get("item_no") is not None and r.get("item")]
        res.evidence_text = "\n".join(lines)

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

    # ---- migration 002 tables
    def _question_stems(self, q: str):
        """(the question's distinctive words, the same with words written together: "kidney stones" also gives
        "kidneystones", as some lists print it)."""
        for pat, word in SYNONYMS:
            q = pat.sub(word, q)
        tokens = re.findall(r"[A-Za-z]+", q)
        base, joined = self.data.stems(q, " ".join(a + b for a, b in zip(tokens, tokens[1:])))
        base = set(base) - self._generic
        return base, base | (set(joined) - self._generic)

    def _item_kept(self, item_stems, words, base, context) -> bool:
        """The precision rule for one list item (see the module docstring); words of the list's own name do not
        count, and a match only on words of a text section's title is a question about that section ("Does the
        policy pay for an ambulance?" is about the ambulance cover's own section, not a list item "Ambulance")."""
        order = [w for w in item_stems if w not in self._generic and w not in context]
        item, words, base = set(order), words - context, base - context
        hit = item & words
        if not hit or any(hit <= t for t in self._titles):
            return False
        return item <= words or len(hit) >= 2 or (order[0] in hit and 2 * len(hit) >= len(base))

    def _lexicon(self, table: str) -> dict:
        if table not in self._lexicons:
            self._lexicons[table] = self.data.list_lexicon(table)
        return self._lexicons[table]

    def _list_lookup(self, q: str, res: TableAnswer):
        """A whole list ("Which illnesses count as Major Medical Illnesses?") or the items a question names ("Are
        attendant charges paid?"), kept by the precision rule of ``_item_kept``."""
        whole = [t for t, spec in LIST_TABLES.items() if spec["whole"].search(q)]
        found = {}
        if whole:
            res.fields["mode"] = "whole list"
            for t in whole:
                rows = self.data.list_rows(t)
                if t == "waiting_period_conditions" and rows and not self._same_period(q, rows[0]["waiting_months"]):
                    res.fields.setdefault("skipped", []).append(f"{t}: the question names another waiting period")
                    continue
                found[t] = rows
            if "global_cover_benefits" in found:          # only the cover the question names, if it names one
                rows = found["global_cover_benefits"]
                named = [c for c in dict.fromkeys(r["cover"] for r in rows) if self._names_cover(q, c)]
                if named:
                    found["global_cover_benefits"] = [r for r in rows if r["cover"] in named]
        else:
            base, words = self._question_stems(q)
            res.fields.update(mode="items", words=sorted(base))
            for t in LIST_TABLES:
                lex = self._lexicon(t)
                cand = sorted(w for w in words - lex["context"] if w in lex["df"])
                keep = [r for r in (self.data.list_rows(t, cand) if cand else [])
                        if self._item_kept(r["stems"] or [], words, base, lex["context"])]
                if keep:
                    found[t] = keep
        if not found:
            res.status = "no_match"
            return
        res.status = "answered"
        res.lead = "From the policy's lists:" if res.fields["mode"] == "items" else "The policy's list:"
        for t, rows in found.items():
            res.rows += rows
            res.body += self._render_list(t, rows, whole=res.fields["mode"] == "whole list")

    @staticmethod
    def _names_cover(q: str, cover: str) -> bool:
        """The question names this cover by the words in its brackets ("(Emergency & Planned Treatments)")."""
        norm = lambda s: " ".join(re.findall(r"[a-z0-9]+", s.lower().replace("&", " and ")))
        m = re.search(r"\(([^)]+)\)", cover)
        return bool(m) and f" {norm(m.group(1))} " in f" {norm(q)} "

    @staticmethod
    def _same_period(q: str, months: int) -> bool:
        """No period named in the question, or one equal to ``months``."""
        named = [int(m.group(1)) * {"day": Decimal(1) / 30, "month": 1, "year": 12}[m.group(2).lower()]
                 for m in _PERIOD.finditer(q)]
        return not named or any(n == months for n in named)

    @staticmethod
    def _render_list(table: str, rows: List[dict], whole: bool) -> List[str]:
        r0 = rows[0]
        src = f"{where(r0['section_id'])}, page {r0['page']}"
        if table == "waiting_period_conditions":
            text = " ".join((r0.get("waiting_text") or "").split())
            out = [f"{where(r0['waiting_section'])}, {r0['exclusion_code']} – {r0['waiting_months']}-month waiting period. "
                   f"The policy text (page {r0['waiting_page']}) says: \"{text}\""]
            for cat in dict.fromkeys(r["category"] for r in rows):
                block = [r for r in rows if r["category"] == cat]
                name = block[0].get("table_title") or cat.replace("_", " ")
                out.append(f"- {name} ({where(block[0]['section_id'])}, page {block[0]['page']}"
                           f"{f', {len(block)} listed' if whole else ''}): " + "; ".join(r["item"] for r in block))
            return out
        if table == "global_cover_benefits":
            return [f"- {c} ({where(rows_c[0]['cover_section'])}, page {rows_c[0]['page']}) includes: "
                    + "; ".join(f"{r['benefit_code']} {r['item']}" for r in rows_c)
                    for c, rows_c in ((c, [r for r in rows if r["cover"] == c]) for c in dict.fromkeys(r["cover"] for r in rows))]
        title = {"non_payable_items": r0.get("list_title"), "major_medical_illnesses": r0.get("table_title"),
                 "eligible_relationships": "Relationships that can be covered"}.get(table) or table
        num = (lambda r: r["item_printed"]) if table == "eligible_relationships" else (lambda r: f"{r['item_no']}.")
        head = f"{title} ({src}{f', {len(rows)} listed' if whole else ''}):"
        return [head, "; ".join(f"{num(r)} {r['item']}" for r in rows)] if whole else \
            [head] + [f"- {num(r)} {r['item']}" for r in rows]

    def _claim_deadline(self, q: str, res: TableAnswer):
        rows = self.data.claim_deadlines()
        if not rows:
            res.status = "no_match"
            return
        res.rows, res.status = rows, "answered"
        res.lead = f"Time limits for submitting reimbursement claim documents ({where(rows[0]['section_id'])}):"
        res.body += [f"- {r['claim_type']}: {r['printed']}" for r in rows]

    def _grievance_contact(self, q: str, res: TableAnswer):
        rows = self.data.grievance_contacts()
        if not rows:
            res.status = "no_match"
            return
        res.rows, res.status = rows, "answered"
        res.lead = f"Grievance contact points, by escalation level ({where(rows[0]['section_id'])}):"
        res.body += [f"- {r['contact_point']} – {r['level']}: {r['item']}" for r in rows]

    def _si_utilisation(self, q: str, res: TableAnswer):
        rows = self.data.si_examples()
        if not rows:
            res.status = "no_match"
            return
        res.rows, res.status = rows, "answered"
        res.lead = f"Worked examples of how the Sum Insured is used, claim by claim ({where(rows[0]['section_id'])}):"
        amt = lambda v: "-" if v is None else inr(v, words=True)
        for ex in dict.fromkeys(r["example"] for r in rows):
            block = [r for r in rows if r["example"] == ex]
            res.body.append(f"Example {ex}" + (f" ({block[0]['reference_text']})" if block[0].get("reference_text") else "") + ":")
            res.body.append("| Claim | Claim amount | Base Sum Insured available | Plus Benefit available | Secure Benefit "
                            "available | Restore available | Admissible amount | Sum Insured used |")
            res.body.append("|---" * 8 + "|")
            res.body += [f"| {r['value']} | {amt(r['claim_inr'])} | {amt(r['base_available_inr'])} | "
                         f"{amt(r['plus_available_inr'])} | {amt(r['secure_available_inr'])} | {amt(r['restore_available_inr'])} | "
                         f"{amt(r['admissible_inr'])} | {r['utilisation']} |" for r in block]

    def _add_notes(self, res: TableAnswer):
        """Table notes, and the footnotes whose mark appears in a cell used; one source line per table."""
        ids = sorted({r["table_id"] for r in res.rows if r.get("table_id")})
        if not ids:
            return
        marks = " ".join([str(r.get(k) or "") for r in res.rows for k in ("printed", "row_label", "item_printed")]
                         + ([c["text"] or "" for c in res.cells] if res.intent in self._WHOLE_ROWS else []))
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
