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


# ---------------------------------------------------------------------------------------------------- follow-ups
# Conversation memory: a follow-up is rewritten into a standalone question before it is searched and answered
# (policy_rag/memory.py). The rewrite runs on the answer model with its configured generation settings.
CONDENSE_TEMPLATES = {
    "condense_v1": """Rewrite the user's last question so that it can be understood without the conversation.
Keep the user's meaning and wording. Add only what the question leaves out (the plan, cover, benefit, condition, person
or amount it refers to), taken from the conversation. Do not answer the question and do not add facts.
If the last question already makes sense on its own, or it starts a new topic, return it unchanged.
Reply with the rewritten question only.

CONVERSATION:
{history}

LAST QUESTION: {question}
STANDALONE QUESTION:""",
    # v2 (after the first dialogue run, where v1 rewrites copied figures and sentences from earlier answers)
    "condense_v2": """Rewrite the user's last question so that it can be understood without the conversation.
Replace words such as "it", "that", "those", "there" or "the first one" with the name they stand for: the plan, cover,
benefit, condition or person. If the question leaves out what is asked (as in "And for X?"), take that from the user's
earlier question, and keep what the user asks about (for example the premium, the waiting period, the limit or the
documents).
Figures may come only from the user's own questions. Take nothing else from the assistant's answers: no amounts,
periods, percentages, section numbers, conditions or explanations.
Do not answer the question. Keep the user's wording for everything else.
If the last question already makes sense on its own, or it starts a new topic, return it unchanged.
Reply with the rewritten question only, on one line.

CONVERSATION:
{history}

LAST QUESTION: {question}
STANDALONE QUESTION:""",
    # v3 (after the second dialogue run, where v2 rewrites often used "the policy" or "the plan" instead of the name,
    # and reworded complete questions). The examples are invented, so no policy wording is suggested.
    "condense_v3": """Rewrite the user's last question so that it can be understood without the conversation.
Replace each word that points back to the conversation ("it", "its", "that", "those", "they", "there", "the first
one", "the same") with the exact name of what it points to, written as in the conversation: the specific cover,
benefit, condition, plan or person, never a general word such as "the policy", "the plan", "the cover" or "the
benefit" (unless the user means the policy itself).
If the question leaves out what is asked (as in "And for X?"), take that from the user's earlier question.
Keep every other word and detail of the last question: ages, amounts, periods, places and names.
Figures may come only from the user's questions. Take nothing else from the assistant's answers: no amounts,
periods, percentages, section numbers, conditions or explanations.
Do not answer the question. Reply with the rewritten question only, on one line.

Examples (invented):
User: What does the Sunrise Recovery Benefit pay?
Assistant: It pays a daily allowance after discharge ...
Last question: Is it paid for ICU days too?
Standalone question: Is the Sunrise Recovery Benefit paid for ICU days too?

User: How long is the waiting period for knee surgery under the Gold plan?
Assistant: It is 24 months ...
Last question: And for hernia?
Standalone question: How long is the waiting period for hernia under the Gold plan?

CONVERSATION:
{history}

LAST QUESTION: {question}
STANDALONE QUESTION:""",
}


def build_condense_prompt(history, question: str, template: str = "condense_v1") -> str:
    """The rewrite prompt. ``history``: [{"question", "answer"}, ...], oldest first (already shortened)."""
    try:
        text = CONDENSE_TEMPLATES[template]
    except KeyError:
        raise KeyError(f"unknown condense template {template!r}; available: {sorted(CONDENSE_TEMPLATES)}") from None
    turns = "\n".join(f"User: {h['question']}\nAssistant: {h['answer']}" for h in history)
    return text.format(history=turns, question=question.strip())
