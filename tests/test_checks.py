from policy_rag.evaluation.checks import (answer_polarity, cited_sections, declines_at_start, figures, score_answer,
                                         unsupported_figures)
from policy_rag.evaluation.stats import cohen_kappa, holm, mcnemar_exact, mde, wilson
from policy_rag.evaluation.table_checks import get_table_checker
from policy_rag.evaluation.testset import anchor_in, fact_found, norm_text, words


def test_figures_unify_amount_formats():
    assert ("inr", 1_500_000) in figures("Rs. 15 Lakhs")
    assert ("inr", 1_500_000) in figures("₹15,00,000")
    assert ("months", 24) in figures("twenty four months") or ("months", 24) in figures("24 months")
    assert ("pct", 50.0) in figures("up to 50%")


def test_unsupported_figures():
    ctx = "Expenses in the 30 days before admission; max upto 6000 per year."
    assert unsupported_figures("Paid for 30 days.", ctx) == []
    assert unsupported_figures("Paid for 45 days.", ctx) == ["45 days"]
    assert unsupported_figures("Up to Rs 6,000.", ctx) == []              # bare number in the context counts


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
    neg = {"id": "N", "answerable": False, "question": "Pets?", "facts": [], "type": "out_of_scope"}
    assert score_answer(neg, "Not specified in the provided policy text.", "", "")["fully_correct"]
    assert "not_declined" in score_answer(neg, "Yes, pets are covered.", "", "")["critical"]


def test_optima_secure_table_checker():
    check = get_table_checker("optima_secure")
    res = check("For Optima Secure, a single claim in Year 2 can be paid up to Rs. 25 lakh.", [["secure", "single", 2]], "")
    assert res["correct"] == 1 and not res["wrong"]
    res = check("For Optima Secure, a single claim in Year 2 can be paid up to Rs. 30 lakh.", [["secure", "single", 2]], "")
    assert res["wrong"]


def test_statistics():
    lo, hi = wilson(79, 88)
    assert round(lo, 2) == 0.82 and round(hi, 2) == 0.95
    assert mcnemar_exact(0, 6) == 0.03125 and mcnemar_exact(3, 3) == 1.0
    assert holm({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert 0.1 < mde(88) < 0.2
