"""An invented policy's tables ("my: Zenith Care"; every number made up): the same rows feed the in-memory stand-in
used by the unit tests and a real Postgres in the opt-in integration test, so both must give the same answers."""
from decimal import Decimal

from policy_rag.tables import group_inr

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


def dataset() -> dict:
    tables, typed, refs = {}, {k: [] for k in ("premium_rates", "plan_sum_insured", "claim_limit_illustration",
                                                "instalment_options", "checkup_limits", "deductible_discounts")}, []

    def table(tid, section, page, caption=None, title=None):
        tables[tid] = {"section_id": section, "page": page, "caption": caption, "title": title, "cells": {}}

    def cell(tid, row, col, text):
        tables[tid]["cells"][(row, col)] = text

    def ref(tid, notes=(), footnotes=()):
        refs.append({"table_id": tid, "parts": [tid], "notes": list(notes), "footnotes": list(footnotes)})

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
    return {"doc_id": DOC, "title": TITLE, "tables": tables, "typed": typed, "refs": refs, "blocks": blocks,
            "places": places}


def cell_text(tables, tid, row, col):
    return tables[tid]["cells"][(row, col)]


class MemoryTableData:
    """The ``TableData`` queries answered from ``dataset()`` in Python (same columns, same order)."""

    def __init__(self, ds=None):
        self.ds = ds or dataset()

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
                "premium": premium, "places": places,
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
