"""Prompt templates. The only fixed answer text in the system is the refusal below; there are no stored answers."""
from __future__ import annotations

NO_ANSWER_TEXT = "Not specified in the provided policy text."

TEMPLATES = {
    # v2.3 prompt: instructions plus scope rules (used for all published results)
    "policy_analyst_v2_3": """You are a professional insurance policy analyst.

Use ONLY the information provided in the CONTEXT to answer the QUESTION.
Do NOT add, assume, or infer details beyond the given text.
Provide a clear, factual, and complete answer in full sentences.
When relevant, cite the section name or number from the CONTEXT.
Each CONTEXT passage starts with the section(s) it comes from. A condition, limit or exclusion applies only to the cover named in its own section; do not apply it to a different cover.
Do not present items listed under a different section as part of the cover asked about.
If a passage says a list is provided below, include that list when it appears in the CONTEXT.
When comparing plans, take all figures from the same year's table and name the year.
If the CONTEXT covers only part of the question, answer that part and say what is not shown.{extra_rules}
If the answer cannot be found, reply: “Not specified in the provided policy text.”

CONTEXT:
{context}

QUESTION: {query}
ANSWER:""",
}

# Added only when more than one policy document is loaded (labels then start with the policy name)
MULTI_DOCUMENT_RULE = ("\nThe passages may come from different policies; each label starts with the policy name. Answer "
                       "for the policy the QUESTION names, and say which policy each statement comes from.")


def build_prompt(query: str, context: str, template: str = "policy_analyst_v2_3", multi_document: bool = False) -> str:
    try:
        text = TEMPLATES[template]
    except KeyError:
        raise KeyError(f"unknown prompt template {template!r}; available: {sorted(TEMPLATES)}") from None
    return text.format(context=context, query=query.strip(), extra_rules=MULTI_DOCUMENT_RULE if multi_document else "")


def is_no_answer(answer: str) -> bool:
    return (answer or "").strip().strip('"“”').strip() == NO_ANSWER_TEXT
