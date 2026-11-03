"""An invented policy's tables ("my: Zenith Care"; every number and list item made up): the same rows feed the
in-memory stand-in used by the unit tests and a real Postgres in the opt-in integration test, so both must give the
same answers. ``dataset(lists=True)`` adds the tables of migration 002 (lists, claim deadlines, grievance contacts,
a Sum Insured example) and text chunks linked to the tables."""
import hashlib
import re
from decimal import Decimal

from policy_rag.tables import LIST_TABLES, group_inr

DOC = "pytest-tables"
TITLE = "my: Zenith Care"
PRICED = ["Zenith Care", "Zenith Care Plus"]
PLANS = PRICED + ["Zenith Lite"]
AGES = [0, 1, 2, 30, 31, 45, 90]                    # 90 = "90 and above", renewal only
SUMS_INSURED = [500_000, 1_000_000]
ITEM_ROWS = [("base_si", "Base Sum Insured"), ("secure_benefit", "Secure Benefit"),
             ("bonus_or_plus", "Cumulative Bonus{star} / Plus Benefit"), ("restore", "Automatic Restore Benefit"),
             ("max_single_claim", "Maximum permissible amount for a single Hospitalization claim in a Policy Year"),
             ("max_all_claims", "Maximum permissible amount for all Hospitalization claims in a Policy Year")]


def premium_value(plan, tier, age, si) -> int:
    return 1000 + 10 * age + 100 * tier + 5 * si // 100_000 + (37 if plan.endswith("Plus") else 0)


def claim_value(plan, item, year) -> int:
    base = 500_000 if plan == "Zenith Lite" else 1_000_000
    extra = {"Zenith Care": 0, "Zenith Care Plus": 500_000, "Zenith Lite": 0}[plan]
    return {"base_si": base, "secure_benefit": 0 if plan == "Zenith Lite" else base + extra,
            "bonus_or_plus": (year - 1) * base // 2, "restore": base,
            "max_single_claim": 2 * base + extra + (year - 1) * base // 2,
            "max_all_claims": 3 * base + extra + (year - 1) * base // 2}[item]


LISTS = {
    "waiting_period_conditions": [("illness", "Kidneystone and bladder stone disorders"), ("illness", "Cataract"),
                                  ("illness", "Varicose veins"), ("surgical_procedure", "Hernia repair"),
                                  ("surgical_procedure", "Joint replacement surgeries"),
                                  ("surgical_procedure", "Tonsillectomy")],
    "non_payable_items": ["Baby Food", "Attendant Charges", "Television Charges", "Gloves", "Ambulance Collar",
                          "Ambulance", "Kit with no details [Delivery Kit, Recovery Kit]"],
    "major_medical_illnesses": ["Cancer of specified severity", "Kidney failure requiring regular dialysis",
                                "Stroke resulting in permanent symptoms", "Major Organ Transplant"],
    "global_cover_benefits": [("4.9", "Global Health Cover (Emergency Treatments Only)", "B 3.1", "Hospitalization Expenses"),
                              ("4.9", "Global Health Cover (Emergency Treatments Only)", "B 4.1", "Emergency Air Ambulance"),
                              ("4.10", "Global Health Cover (Emergency & Planned Treatments)", "B 3.1", "Hospitalization Expenses"),
                              ("4.10", "Global Health Cover (Emergency & Planned Treatments)", "B 3.4", "AYUSH Treatment"),
                              ("4.10", "Global Health Cover (Emergency & Planned Treatments)", "B 4.1", "Emergency Air Ambulance")],
    "eligible_relationships": [("i.", "Spouse"), ("ii.", "Son"), ("iii.", "Mother-in-law")],
}
SECTION_TITLES = {"3.2": "Emergency Air Ambulance", "3.4": "AYUSH Treatment", "4.3": "Protect Benefit",
                  "10.1": "Standard Exclusions", "11.6": "Procedure for Reimbursement Claims",
                  "12.18": "Redressal of Grievance", "13.2": "Utilization of Sum Insured"}


def chunk_uid(text: str) -> str:
    return hashlib.sha256(f"{DOC}\0{text}".encode()).hexdigest()[:32]


def dataset(lists: bool = False) -> dict:
    tables, typed, refs = {}, {k: [] for k in ("premium_rates", "plan_sum_insured", "claim_limit_illustration",
                                                "instalment_options", "checkup_limits", "deductible_discounts")}, []

    def table(tid, section, page, caption=None, title=None):
        tables[tid] = {"section_id": section, "page": page, "caption": caption, "title": title, "cells": {}}

    def cell(tid, row, col, text):
        tables[tid]["cells"][(row, col)] = text

    def ref(tid, notes=(), footnotes=(), reference=None):
        refs.append({"table_id": tid, "parts": [tid], "notes": list(notes), "footnotes": list(footnotes),
                     "reference": reference})

    # premiums: one table per plan and tier
    for n, (plan, tier) in enumerate([(p, t) for p in PRICED for t in (1, 2)]):
        tid = f"p{20 + n}-t1"
        table(tid, "Annexure-Z", 20 + n, title=f"{TITLE} - {plan} Plan Gross Premium (Excl. GST) - Tier {tier}")
        ref(tid, footnotes=[{"marker": "*", "text": "*Only for Renewal Purposes"}])
        for r, age in enumerate(AGES, start=3):
            cell(tid, r, 0, f">={age}*" if age == 90 else str(age))
            for c, si in enumerate(SUMS_INSURED, start=1):
                v = premium_value(plan, tier, age, si)
                cell(tid, r, c, group_inr(v))
                typed["premium_rates"].append(dict(plan=plan, tier=tier, age_min=age, age_max=None if age == 90 else age,
                                                   renewal_only=age == 90, sum_insured=si, premium_inr=v,
                                                   table_id=tid, row=r, col=c))
    # plans and Sums Insured
    table("p1-t5", "2", 1, caption="2. Plans & Sum Insured")
    for c, plan in enumerate(PLANS, start=1):
        for r, si in enumerate(SUMS_INSURED if plan != "Zenith Lite" else SUMS_INSURED[:1], start=1):
            cell("p1-t5", r, c, group_inr(si))
            typed["plan_sum_insured"].append(dict(plan=plan, sum_insured=si, table_id="p1-t5", row=r, col=c))
    # claim limit illustration: one table per year
    for year in (1, 2, 3):
        tid = f"p1-t{year}"
        table(tid, "2.1", 1, caption=f"Year {year}")
        ref(tid, notes=["Note: Restore applies to claims in India only"],
            footnotes=[{"marker": "*", "text": f"*Assuming no claim in year {year - 1}"}] if year > 1 else [])
        for r, (item, printed) in enumerate(ITEM_ROWS, start=1):
            name = printed.format(star="*" if year > 1 else "")
            cell(tid, r, 0, name)
            for c, plan in enumerate(PLANS, start=1):
                v = claim_value(plan, item, year)
                cell(tid, r, c, "NIL" if v == 0 else group_inr(v))
                typed["claim_limit_illustration"].append(dict(year=year, plan=plan, item=item, item_printed=name,
                                                              amount_inr=v, printed=cell_text(tables, tid, r, c),
                                                              table_id=tid, row=r, col=c))
    # instalments
    table("p12-t1", "9.1", 12, caption="9.1. Premium Payment in Instalments")
    ref("p12-t1")
    for r, (opt, freq, days, printed) in enumerate([("Option 1", "Yearly", 30, "30 days"),
                                                   ("Option 2", "Monthly", 15, "15 Days")], start=1):
        cell("p12-t1", r, 0, opt)
        cell("p12-t1", r, 1, freq)
        cell("p12-t1", r, 2, printed)
        typed["instalment_options"].append(dict(option=opt, frequency=freq, grace_period_days=days, grace_printed=printed,
                                                table_id="p12-t1", row=r, col=2))
    # check-up limits
    for tid, kind, applies, limits in [("p7-t1", "individual", "For Individual Policies, limits apply per person.", (1500, 2000)),
                                       ("p7-t2", "family_floater", "For Family Floater Policies, limits apply to all.", (2500, 5000))]:
        table(tid, "5", 7, caption="5. Preventive Health Check-up" if kind == "individual" else None)
        ref(tid)
        for c, (si, lim) in enumerate(zip(SUMS_INSURED, limits), start=1):
            cell(tid, 0, c, f"{si // 100_000} Lakhs")
            cell(tid, 1, c, f"Rs. {group_inr(lim)}")
            typed["checkup_limits"].append(dict(policy_type=kind, applies=applies, sum_insured=si,
                                                si_printed=f"{si // 100_000} Lakhs", limit_inr=lim,
                                                table_id=tid, row=1, col=c))
    # deductible discounts
    table("p7-t3", "8", 7)
    ref("p7-t3")
    r = 0
    for plan in PRICED:
        for ded, low, high in [(25_000, "0.25", "0.15"), (50_000, "0.40", "0.30")]:
            for gt, le, band, d in [(None, 2_000_000, "Base SI <= 20 lakhs", low), (2_000_000, None, "Base SI > 20 lakhs", high)]:
                r += 1
                cell("p7-t3", r, 1, f"{int(Decimal(d) * 100)}%")
                typed["deductible_discounts"].append(dict(plan=plan, deductible=ded, si_gt=gt, si_le=le, band_printed=band,
                                                          discount=Decimal(d), table_id="p7-t3", row=r, col=1))
    blocks = [{"block_id": "p15-b1", "page": 15, "section": "14", "text": "a. Tier 1: Mumbai, National Capital Region (NCR)."},
              {"block_id": "p15-b2", "page": 15, "section": "14", "text": "b. Tier 2: Rest of India."}]
    places = [dict(place="Mumbai", tier=1, source_block_id="p15-b1"),
              dict(place="National Capital Region (NCR)", tier=1, source_block_id="p15-b1"),
              dict(place="Rest of India", tier=2, source_block_id="p15-b2")]
    ds = {"doc_id": DOC, "title": TITLE, "tables": tables, "typed": typed, "refs": refs, "blocks": blocks,
          "places": places, "chunks": [], "chunk_tables": []}
    if lists:
        _add_lists(ds, table, cell, ref)
    return ds


def _add_lists(ds, table, cell, ref):
    typed, blocks = ds["typed"], ds["blocks"]
    blocks.append({"block_id": "p8-b5", "page": 8, "section": "10.1",
                   "text": "i. The listed conditions are excluded until 24 months of continuous coverage have passed. "
                           "Not applicable to claims due to an Accident."})
    rows = typed["waiting_period_conditions"] = []
    for tid, cat, caption in (("p8-t1", "illness", "Illnesses"), ("p8-t2", "surgical_procedure", "Surgical Procedures")):
        table(tid, "10.1", 8, caption=caption)
        ref(tid, reference="vi. List of specific diseases/procedures is provided below:")
        items = [c for k, c in LISTS["waiting_period_conditions"] if k == cat]
        for n, item in enumerate(items, start=1):
            cell(tid, (n - 1) // 2, (n - 1) % 2, item)
            rows.append(dict(exclusion_code="Excl02", category=cat, list_no=n, condition=item, waiting_months=24,
                             waiting_block="p8-b5", table_id=tid, row=(n - 1) // 2, col=(n - 1) % 2))
    table("p18-t1", "Annexure-B", 18, caption="Annexure B – List I - Items not covered (Non-Medical Expenses)")
    ref("p18-t1")
    typed["non_payable_items"] = []
    for n, item in enumerate(LISTS["non_payable_items"], start=1):
        cell("p18-t1", n, 0, str(n))
        cell("p18-t1", n, 1, item)
        typed["non_payable_items"].append(dict(list="List I", list_title="Annexure B – List I - Items not covered "
                                               "(Non-Medical Expenses)", item_no=n, item=item, table_id="p18-t1", row=n, col=1))
    table("p5-t1", "4.8", 5, caption="Major Medical Illness")
    ref("p5-t1")
    typed["major_medical_illnesses"] = []
    for n, item in enumerate(LISTS["major_medical_illnesses"], start=1):
        cell("p5-t1", n, 0, str(n))
        cell("p5-t1", n, 1, item)
        typed["major_medical_illnesses"].append(dict(item_no=n, illness=item, table_id="p5-t1", row=n, col=1))
    typed["global_cover_benefits"] = []
    for sec, tid in (("4.9", "p5-t2"), ("4.10", "p6-t1")):
        rows = [r for r in LISTS["global_cover_benefits"] if r[0] == sec]
        table(tid, sec, 5 if sec == "4.9" else 6, caption=f"{sec}. {rows[0][1]}")
        ref(tid)
        for n, (_, cover, code, benefit) in enumerate(rows, start=1):
            cell(tid, n, 0, code)
            cell(tid, n, 1, benefit)
            typed["global_cover_benefits"].append(dict(cover_section=sec, cover=cover, benefit_code=code, benefit=benefit,
                                                       table_id=tid, row=n, col=1))
    table("p1-t4", "1", 1)
    ref("p1-t4")
    typed["eligible_relationships"] = []
    for n, (printed, rel) in enumerate(LISTS["eligible_relationships"], start=1):
        cell("p1-t4", n, 0, printed)
        cell("p1-t4", n, 1, rel)
        typed["eligible_relationships"].append(dict(item_no=n, item_printed=printed, relationship=rel,
                                                    table_id="p1-t4", row=n, col=1))
    table("p10-t1", "11.6", 10, caption="11.6. Procedure for Reimbursement Claims")
    ref("p10-t1")
    typed["claim_deadlines"] = []
    for r, (kind, days, frm) in enumerate([("Reimbursement of Hospitalization Expenses", 20, "date of discharge"),
                                           ("Reimbursement of Post-Hospitalization Expenses", 10, "end of treatment")], start=1):
        cell("p10-t1", r, 0, kind)
        cell("p10-t1", r, 1, f"Within {days} days of {frm}.")
        typed["claim_deadlines"].append(dict(claim_type=kind, deadline_days=days, counted_from=frm,
                                             printed=f"Within {days} days of {frm}.", table_id="p10-t1", row=r, col=1))
    table("p13-t1", "12.18", 13)
    ref("p13-t1")
    typed["grievance_contacts"] = []
    for r, (point, details) in enumerate([("Call us", ("1800 000 111", "1800 000 222")),
                                          ("Write to us", ("care@zenith.example", "escalate@zenith.example"))], start=1):
        cell("p13-t1", r, 0, point)
        for level_no, (level, d) in enumerate(zip(("First Contact Point", "Escalation level 1"), details), start=1):
            cell("p13-t1", r, level_no, d)
            typed["grievance_contacts"].append(dict(contact_point=point, level_no=level_no, level=level, details=d,
                                                    phones=[d] if "@" not in d else [], emails=[d] if "@" in d else [],
                                                    table_id="p13-t1", row=r, col=level_no))
    table("p14-t1", "13.2", 14, caption="Illustration 1")
    ref("p14-t1", footnotes=[{"marker": "*", "text": "*A single claim cannot exceed the Sum Insured available."}],
        reference="Illustration for Utilization of Sum Insured: Base Sum Insured 5,00,000")
    typed["si_utilisation_examples"] = []
    for r, (claim, base, restore, adm, use) in enumerate([(700_000, 500_000, 0, 500_000, "Base"),
                                                         (300_000, None, 500_000, 300_000, "Automatic Restore")], start=1):
        values = [f"{r}st claim" if r == 1 else f"{r}nd claim", group_inr(claim), "-" if base is None else group_inr(base),
                  group_inr(restore), group_inr(adm) + ("*" if r == 1 else ""), use]
        for c, v in enumerate(values):
            cell("p14-t1", r, c, v)
        typed["si_utilisation_examples"].append(dict(example=1, claim_no=r, claim_inr=claim, base_available_inr=base,
                                                     plus_available_inr=None, secure_available_inr=None,
                                                     restore_available_inr=restore, admissible_inr=adm, utilisation=use,
                                                     table_id="p14-t1", row=r, col=0))
    for n, (sid, title) in enumerate(SECTION_TITLES.items()):
        text = f"{title}: invented text of Section {sid}."
        ds["chunks"].append({"idx": n, "chunk_uid": chunk_uid(text), "section_id": sid, "section_title": title,
                             "label": f"Section {sid} - {title}", "text": text, "embed_text": text})
    uid = {c["section_id"]: c["chunk_uid"] for c in ds["chunks"]}
    ds["chunk_tables"] = [("p8-t1", uid["10.1"], "introduced_by"), ("p8-t2", uid["10.1"], "introduced_by"),
                          ("p18-t1", uid["4.3"], "refers_to"), ("p10-t1", uid["11.6"], "same_clause"),
                          ("p13-t1", uid["12.18"], "same_clause"), ("p14-t1", uid["13.2"], "introduced_by")]


def cell_text(tables, tid, row, col):
    return tables[tid]["cells"][(row, col)]


_STOP = set("a an the and or of in on at to for with by from is are was were be been being do does did i my me we our "
            "you your it its this that these those what which who whom how when where why will would can could should "
            "may might shall there here not no if as than then so too very all any each some such only own same other "
            "about into over under again once both more most few nor just now am has have having he she they them his "
            "her their him up down out off before after above below between through during while until because".split())


def crude_stem(w: str) -> str:
    """A small stand-in for Postgres' English stemmer (the same word always gives the same stem)."""
    for suf, rep in (("sses", "ss"), ("ies", "i"), ("ss", "ss"), ("s", "")):
        if w.endswith(suf):
            w = w[:len(w) - len(suf)] + rep
            break
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 3 and re.search("[aeiou]", w[:-len(suf)]):
            w = w[:-len(suf)]
            break
    return w[:-1] if w.endswith("e") and len(w) > 3 else w


def crude_stems(text: str, ordered: bool = False):
    out = []
    for tok in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", (text or "").lower()):
        parts = tok.split("-")
        for w in ([tok] if len(parts) > 1 else []) + parts:
            if w not in _STOP and w:
                out.append(crude_stem(w) if "-" not in w else "-".join(crude_stem(x) for x in w.split("-")))
    out = list(dict.fromkeys(out))
    return out if ordered else sorted(out)


class MemoryTableData:
    """The ``TableData`` queries answered from ``dataset()`` in Python (same columns, same order)."""

    ITEM = {"waiting_period_conditions": "condition", "non_payable_items": "item", "major_medical_illnesses": "illness",
            "global_cover_benefits": "benefit", "eligible_relationships": "relationship"}

    def __init__(self, ds=None):
        self.ds = ds or dataset()
        self.has_lists = "non_payable_items" in self.ds["typed"]
        self.doc = self.ds["doc_id"]

    def _cite(self, r):
        t = self.ds["tables"][r["table_id"]]
        return {"table_id": r["table_id"], "row": r["row"], "col": r["col"], "printed": t["cells"][(r["row"], r["col"])],
                "page": t["page"], "section_id": t["section_id"], "table_title": t["title"] or t["caption"]}

    def catalog(self):
        typed, ds = self.ds["typed"], self.ds
        premium = {}
        for plan in sorted({r["plan"] for r in typed["premium_rates"]}):
            rows = [r for r in typed["premium_rates"] if r["plan"] == plan]
            premium[plan] = {"plan": plan, "sums_insured": sorted({r["sum_insured"] for r in rows}),
                             "age_min": min(r["age_min"] for r in rows),
                             "age_max": max(r["age_max"] if r["age_max"] is not None else r["age_min"] for r in rows),
                             "open_ended": any(r["age_max"] is None for r in rows),
                             "tiers": sorted({r["tier"] for r in rows}), "tables": sorted({r["table_id"] for r in rows}),
                             "section": min(self.ds["tables"][r["table_id"]]["section_id"] for r in rows)}
        blocks = {b["block_id"]: b for b in ds["blocks"]}
        places = [{"place": p["place"], "place_key": " ".join(p["place"].lower().split()), "tier": p["tier"],
                   "source_block_id": p["source_block_id"], "page": blocks[p["source_block_id"]]["page"],
                   "section": blocks[p["source_block_id"]]["section"], "printed": blocks[p["source_block_id"]]["text"]}
                  for p in sorted(ds["places"], key=lambda p: (p["tier"], p["place"]))]
        plans = sorted({r["plan"] for k in ("plan_sum_insured", "claim_limit_illustration", "premium_rates",
                                            "deductible_discounts") for r in typed[k]})
        ill = typed["claim_limit_illustration"]
        return {"doc_id": ds["doc_id"], "title": ds["title"], "version_id": 1, "version_label": "v1", "plans": plans,
                "premium": premium, "places": places, "has_lists": self.has_lists,
                "illustration": {"plans": sorted({r["plan"] for r in ill}), "years": sorted({r["year"] for r in ill})}}

    def premiums(self, plans, tiers, sums_insured, age):
        out = []
        for r in self.ds["typed"]["premium_rates"]:
            hi = r["age_max"] if r["age_max"] is not None else 1000
            if r["plan"] in plans and r["tier"] in tiers and r["sum_insured"] in sums_insured and r["age_min"] <= age <= hi:
                label = self.ds["tables"][r["table_id"]]["cells"][(r["row"], 0)]
                out.append({"plan": r["plan"], "tier": r["tier"], "age_min": r["age_min"], "age_max": r["age_max"],
                            "renewal_only": r["renewal_only"], "sum_insured": r["sum_insured"], "value": r["premium_inr"],
                            "row_label": label, **self._cite(r)})
        return sorted(out, key=lambda r: (r["plan"], r["tier"], r["sum_insured"]))

    def claim_limits(self, plans, years, items):
        out = [{"plan": r["plan"], "year": r["year"], "item": r["item"], "item_printed": r["item_printed"],
                "value": r["amount_inr"], **self._cite(r)}
               for r in self.ds["typed"]["claim_limit_illustration"]
               if r["plan"] in plans and r["year"] in years and r["item"] in items]
        return sorted(out, key=lambda r: (r["plan"], r["item"], r["year"]))

    def instalments(self):
        return sorted(({"option": r["option"], "frequency": r["frequency"], "value": r["grace_period_days"], **self._cite(r)}
                       for r in self.ds["typed"]["instalment_options"]), key=lambda r: r["option"])

    def checkups(self, policy_types, sums_insured):
        out = [{"policy_type": r["policy_type"], "applies": r["applies"], "sum_insured": r["sum_insured"],
                "si_printed": r["si_printed"], "value": r["limit_inr"], **self._cite(r)}
               for r in self.ds["typed"]["checkup_limits"]
               if r["policy_type"] in policy_types and (not sums_insured or r["sum_insured"] in sums_insured)]
        return sorted(sorted(out, key=lambda r: r["sum_insured"]), key=lambda r: r["policy_type"], reverse=True)

    def deductibles(self, plans, deductibles, sum_insured):
        def fits(r):
            return sum_insured is None or ((r["si_gt"] is None or sum_insured > r["si_gt"])
                                           and (r["si_le"] is None or sum_insured <= r["si_le"]))
        out = [{"plan": r["plan"], "deductible": r["deductible"], "si_gt": r["si_gt"], "si_le": r["si_le"],
                "band_printed": r["band_printed"], "value": r["discount"], **self._cite(r)}
               for r in self.ds["typed"]["deductible_discounts"]
               if r["plan"] in plans and (not deductibles or r["deductible"] in deductibles) and fits(r)]
        return sorted(out, key=lambda r: (r["plan"], r["deductible"], -1 if r["si_gt"] is None else r["si_gt"]))

    def notes(self, table_ids):
        return sorted(({"table_id": f["table_id"], "parts": f["parts"], "notes": f["notes"], "footnotes": f["footnotes"]}
                       for f in self.ds["refs"] if set(f["parts"]) & set(table_ids)), key=lambda r: r["table_id"])

    # ---- migration 002 tables
    def stems(self, *texts):
        out = [crude_stems(t) for t in texts]
        return out[0] if len(texts) == 1 else out

    def section_titles(self):
        return [{"title": t, "stems": crude_stems(t)} for t in dict.fromkeys(c["section_title"] for c in self.ds["chunks"])]

    def _list(self, table):
        typed, tables, blocks = self.ds["typed"], self.ds["tables"], {b["block_id"]: b for b in self.ds["blocks"]}
        rows = []
        for r in typed[table]:
            item = r[self.ITEM[table]]
            row = {"item_no": r.get("list_no", r.get("item_no")), "item": item, **self._cite(r),
                   "stems": crude_stems(item, ordered=True)}
            if table == "waiting_period_conditions":
                b = blocks[r["waiting_block"]]
                row.update(exclusion_code=r["exclusion_code"], category=r["category"], waiting_months=r["waiting_months"],
                           waiting_block=r["waiting_block"], waiting_page=b["page"], waiting_section=b["section"],
                           waiting_text=b["text"])
            elif table == "non_payable_items":
                row.update(list=r["list"], list_title=r["list_title"])
            elif table == "global_cover_benefits":
                row.update(item_no=None, cover_section=r["cover_section"], cover=r["cover"], benefit_code=r["benefit_code"],
                           base_section=re.sub(r"^[A-Z]+ ?", "", r["benefit_code"]))
            elif table == "eligible_relationships":
                row.update(item_printed=r["item_printed"])
            rows.append(row)
        key = {"waiting_period_conditions": lambda r: (r["category"], r["item_no"]),
               "non_payable_items": lambda r: (r["list"], r["item_no"]),
               "global_cover_benefits": lambda r: ([int(x) for x in r["cover_section"].split(".")],
                                                   [int(x) for x in r["base_section"].split(".")])
               }.get(table, lambda r: r["item_no"])
        return sorted(rows, key=key)

    def list_lexicon(self, table):
        df = {}
        for r in self._list(table):
            for w in set(r["stems"]):
                df[w] = df.get(w, 0) + 1
        context_of = {"non_payable_items": lambda r: r["list_title"], "global_cover_benefits": lambda r: r["cover"],
                      "major_medical_illnesses": lambda r: r["table_title"]}.get(table)
        context = {w for r in self._list(table) for w in crude_stems(context_of(r))} if context_of else set()
        return {"items": len(self._list(table)), "df": df, "context": context}

    def list_rows(self, table, stems=None):
        return [r for r in self._list(table) if stems is None or set(r["stems"]) & set(stems)]

    def claim_deadlines(self):
        return [{"claim_type": r["claim_type"], "value": r["deadline_days"], "counted_from": r["counted_from"], **self._cite(r)}
                for r in sorted(self.ds["typed"]["claim_deadlines"], key=lambda r: r["row"])]

    def grievance_contacts(self):
        return [{"contact_point": r["contact_point"], "level_no": r["level_no"], "level": r["level"], "item": r["details"],
                 "phones": r["phones"], "emails": r["emails"], **self._cite(r)}
                for r in sorted(self.ds["typed"]["grievance_contacts"], key=lambda r: (r["row"], r["level_no"]))]

    def si_examples(self):
        tables = self.ds["tables"]
        refs = {t: f.get("reference") or tables[f["table_id"]]["caption"] or tables[f["table_id"]]["title"] or ""
                for f in self.ds["refs"] for t in f["parts"]}
        keys = ("claim_inr", "base_available_inr", "plus_available_inr", "secure_available_inr", "restore_available_inr",
                "admissible_inr", "utilisation")
        return [{"example": r["example"], "value": r["claim_no"], **{k: r[k] for k in keys},
                 "reference_text": refs.get(r["table_id"]), **self._cite(r)}
                for r in sorted(self.ds["typed"]["si_utilisation_examples"], key=lambda r: (r["example"], r["claim_no"]))]

    def evidence(self, cells):
        out = []
        for tid, row, col in cells:
            t = self.ds["tables"][tid]
            for (r, c), text in sorted(t["cells"].items()):
                if r == row and (col is None or c == col):
                    out.append({"table_id": tid, "row": r, "col": c, "text": text, "header_path": [], "row_path": [],
                                "caption": t["caption"] or t["title"]})
        return sorted({(o["table_id"], o["row"], o["col"]): o for o in out}.values(),
                      key=lambda o: (o["table_id"], o["row"], o["col"]))

    def linked_chunks(self, table_ids):
        order = ["introduced_by", "same_clause", "notes_in", "refers_to"]
        parts = {f["table_id"]: f["parts"] for f in self.ds["refs"]}
        rows = [{"chunk_uid": u, "relation": rel, "table_id": t} for t, u, rel in self.ds["chunk_tables"]
                if set(parts.get(t, [])) & set(table_ids)]
        return sorted(rows, key=lambda r: (order.index(r["relation"]), r["table_id"], r["chunk_uid"]))

    def all_rows_text(self):
        lines = []
        for tid in sorted(self.ds["tables"]):
            t = self.ds["tables"][tid]
            for row in sorted({r for r, _ in t["cells"]}):
                cells = [t["cells"][(r, c)] for r, c in sorted(t["cells"]) if r == row]
                lines.append(" ".join([t["caption"] or t["title"] or ""] + cells))
        return "\n".join(lines)
