from policy_rag.evaluation.checks import (answer_polarity, cited_sections, declines_at_start, figures, opens_with_hedge,
                                         score_answer, stated_ages, unsupported_figures)
from policy_rag.evaluation.stats import cohen_kappa, holm, mcnemar_exact, mde, wilson
from policy_rag.evaluation.table_checks import get_table_checker
from policy_rag.evaluation.testset import anchor_in, fact_found, norm_text, words


def test_figures_unify_amount_formats():
    assert ("inr", 1_500_000) in figures("Rs. 15 Lakhs")
    assert ("inr", 1_500_000) in figures("₹15,00,000")
    assert ("months", 24) in figures("twenty four months") or ("months", 24) in figures("24 months")
    assert ("pct", 50.0) in figures("up to 50%")


def test_figures_read_amount_abbreviations():
    assert figures("Covered up to 5 L") == {("inr", 500_000)} == figures("Rs. 5 lakh")
    assert figures("25k/50k/ 100K") == {("inr", 25_000), ("inr", 50_000), ("inr", 100_000)}
    assert figures("Rs. 2 Cr") == {("inr", 20_000_000)}
    assert figures("5 litres of lather") == set() and figures("Plan L") == set()
    assert unsupported_figures("Paid up to Rs. 5 lakh.", "[Annexure]\nAir travel: up to 5 L") == []


def test_unsupported_figures():
    ctx = "Expenses in the 30 days before admission; max upto 6000 per year."
    assert unsupported_figures("Paid for 30 days.", ctx) == []
    assert unsupported_figures("Paid for 45 days.", ctx) == ["45 days"]
    assert unsupported_figures("Up to Rs 6,000.", ctx) == []              # bare number in the context counts


def test_ages_given_without_their_unit():
    q = "I'm 35 and my cover is 15 lakh. What do I pay?"
    assert stated_ages("", q) == {35}
    assert unsupported_figures("For a 35-year-old: Rs. 9,000.", "Rs. 9,000", q) == []
    assert unsupported_figures("Valid for 15 years.", "Rs. 9,000", q) == ["15 years"]       # 15 is an amount there
    table = "Premiums\n70: Age of the member (in Years): 70; Premium (Rs.): 41,000\nCover: 30 days"
    assert stated_ages(table) == {70}
    assert unsupported_figures("The 70 year old member pays Rs. 41,000.", table) == []
    assert unsupported_figures("Wait 41 years.", table) == ["41 years"]                   # 41,000 is an amount
    assert unsupported_figures("Wait 70 days.", table) == ["70 days"]                     # only ages, in years


def test_fact_matching_word_forms():
    text = norm_text("It needs pre-authorization, and it is not available for reimbursement. Admissions excluded.")
    assert fact_found(["pre authorized"], text)
    assert fact_found(["not available on reimbursement"], text)
    assert fact_found(["admission"], text) and fact_found(["excludes"], text)
    assert not fact_found(["available on reimbursement basis"], text)                    # every content word needed
    assert not fact_found(["not excluded"], norm_text("It is excluded."))                    # "not" is never dropped
    assert not fact_found(["covered after 24 months"], norm_text("covers it after 24 months"))  # with a number: exact
    assert not fact_found(["65 years"], norm_text("for 6 years"))


def test_fact_matching_rules():
    assert fact_found(["5 years"], norm_text("after 5 years")) and not fact_found(["5 years"], norm_text("65 years"))
    assert fact_found(["30 days"], norm_text("a 30-day period"))
    assert fact_found(["pre-hospitalization"], norm_text("pre- and post-hospitalisation expenses"))
    assert anchor_in(["covered only after 24 months"], words("It is covered only after 24 months of cover."))


def test_refusal_and_polarity():
    assert declines_at_start("Not specified in the provided policy text.")
    assert declines_at_start("The policy does not mention pet cover. It covers people.")
    assert not declines_at_start("Pets are covered. Dogs too. The policy does not mention fish.")
    assert answer_polarity("No, it is not covered.") == "no" and answer_polarity("Covered.") is None
    assert cited_sections("see Sections 3.1 and 4.2") == {"3.1", "4.2"}


def test_hedge_at_the_start():
    assert opens_with_hedge("Not specified in the provided policy text. It is paid for later claims.")
    assert opens_with_hedge("The policy text does not explicitly list this illness. It may be covered.")
    assert opens_with_hedge("It is not possible to confirm whether this illness is covered.")
    assert not opens_with_hedge("The policy does not cover cosmetic surgery (Section 10).")       # an answer
    assert not opens_with_hedge("The policy does not directly pay the hospital for this; claims are reimbursed.")
    assert not opens_with_hedge("No, it is not covered. The text does not explicitly say why.")
    assert not opens_with_hedge("Cosmetic surgery is excluded. The policy does not mention tattoos.")


ITEM = {"id": "S01", "answerable": True, "question": "What is the waiting period for kidney stones?",
        "facts": [["24 months"]], "type": "fact", "must_not": [r"\b12 months\b"]}


def test_score_answer_verdicts():
    ctx = "[Section 10.1 - Waiting Periods]\nKidney stones are covered after 24 months."
    ok = score_answer(ITEM, "Kidney stones are covered after 24 months (Section 10.1).", ctx, norm_text(ctx))
    assert ok["fully_correct"] and ok["first_stage"] is None
    missing = score_answer(ITEM, "There is a waiting period.", ctx, norm_text(ctx))
    assert not missing["fully_correct"] and missing["first_stage"] == "generation"
    false = score_answer(ITEM, "After 24 months, or 12 months for Section 10.1.", ctx, norm_text(ctx))
    assert "known_false" in false["critical"]
    gap = score_answer(ITEM, "Not specified in the provided policy text.", "", norm_text(ctx), gate_route="no_answer")
    assert gap["first_stage"] == "context" and not gap["fully_correct"]
    hedge = score_answer(ITEM, "The policy does not explicitly say. It may be 24 months.", ctx, norm_text(ctx))
    assert not hedge["fully_correct"] and "hedge" in {i["kind"] for i in hedge["issues"]} and not hedge["critical"]
    partial = dict(ITEM, reference="The policy does not mention it; 24 months for stones.", expect_admit=True)
    assert "hedge" not in {i["kind"] for i in score_answer(partial, "The policy does not mention it. Stones: 24 months.",
                                                           ctx, norm_text(ctx))["issues"]}
    neg = {"id": "N", "answerable": False, "question": "Pets?", "facts": [], "type": "out_of_scope"}
    assert score_answer(neg, "Not specified in the provided policy text.", "", "")["fully_correct"]
    assert "not_declined" in score_answer(neg, "Yes, pets are covered.", "", "")["critical"]


def test_optima_secure_table_checker():
    check = get_table_checker("optima_secure")
    res = check("For Optima Secure, a single claim in Year 2 can be paid up to Rs. 25 lakh.", [["secure", "single", 2]], "")
    assert res["correct"] == 1 and not res["wrong"]
    res = check("For Optima Secure, a single claim in Year 2 can be paid up to Rs. 30 lakh.", [["secure", "single", 2]], "")
    assert res["wrong"]
    cells = [["secure", "single", y] for y in (1, 2, 3)]
    bold = ("Single claim limit:\n- Optima Secure: **Rs. 20 lakh** (Year 1), rising to **Rs. 25 lakh** (Year 2) and "
            "**Rs. 30 lakh** (Year 3).")
    res = check(bold, cells, "")
    assert res["correct"] == 3 and not res["wrong"]
    res = check(bold.replace("25 lakh", "35 lakh"), cells, "")
    assert res["wrong"] == ["Optima Secure, single claim, Year 2: answer 35 lakh, policy 25 lakh"]


def test_statistics():
    lo, hi = wilson(79, 88)
    assert round(lo, 2) == 0.82 and round(hi, 2) == 0.95
    assert mcnemar_exact(0, 6) == 0.03125 and mcnemar_exact(3, 3) == 1.0
    assert holm({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert 0.1 < mde(88) < 0.2
