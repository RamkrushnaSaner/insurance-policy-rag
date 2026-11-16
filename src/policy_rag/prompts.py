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


# ---------------------------------------------------------------------------------------------------- SQL step
# The SQL step (policy_rag/sql_step.py): the answer model reads the schema card of the policy's numeric tables and
# replies with one JSON object. It runs with the answer model's configured generation settings (thinking included).
SQL_TEMPLATES = {
    "sql_v1": """You read an insurance policy's numeric tables with SQL (PostgreSQL). Each table is listed below with
its printed titles, footnotes and columns: every column's type and printed heading, and the values its text columns
hold, exactly as printed in the policy.

TABLES
{card}

Decide how the QUESTION can be answered from these tables, and reply with ONE JSON object and nothing else:
{{"action": "query", "sql": "<one SELECT statement>"}}
    when the tables hold the values the question asks about.
{{"action": "ask", "question": "<one short question to the user>"}}
    when a detail needed to choose the rows is missing (for example the age, the plan, the Sum Insured or the
    place) and the rows for every possible choice would not fit in 8 rows. Ask only for what is missing.
{{"action": "not_in_tables", "reason": "<what the tables do not hold>"}}
    when the question asks for a value these tables do not hold (for example an age, a Sum Insured or a plan
    that is not listed).
{{"action": "text_only", "reason": "<why the tables are not needed>"}}
    when the question is not about a value in these tables (for example what a benefit covers or how a rule
    works).

Rules for the SQL:
- Use only the tables and columns listed. Write text values exactly as listed (a value printed with a mark keeps
  the mark).
- Select the columns that hold the answer and the columns that identify each row (such as the plan, tier or age),
  so every value can be traced to its printed row. Do not select constants.
- When a missing detail has only a few possible choices (such as two plans), return the rows for all of them.
- When the question names a place instead of a tier, use the tier whose listed places include it, or the tier
  listed for all other places.
- Never guess a value, and never round or change one. Return at most {max_rows} rows.

QUESTION: {question}
{attempts}JSON:""",
    # v2 (after the first dev run: results not computed, lakh read as the wrong column, ages compared as a text range,
    # a general city classification used instead of the policy's tiers, an amount outside a column's printed amounts).
    # General rules with invented example values, so no rule is written for one question.
    "sql_v2": """You read an insurance policy's numeric tables with SQL (PostgreSQL). Each table is listed below with
its printed titles, footnotes and columns: every column's type and printed heading, and the values the columns hold
when they hold few, exactly as printed in the policy.

TABLES
{card}

Decide how the QUESTION can be answered from these tables, and reply with ONE JSON object and nothing else:
{{"action": "query", "sql": "<one SELECT statement>"}}
    when the tables hold the values the question asks about.
{{"action": "ask", "question": "<one short question to the user>"}}
    when a detail needed to choose the rows is missing (for example the age, the plan, the Sum Insured or the
    place) and the rows for every possible choice would not fit in 8 rows. Ask only for what is missing.
{{"action": "text_only", "reason": "<why the tables are not needed>"}}
    when the question is about how a benefit, rule or condition works (for example when it applies, whether it
    carries forward, who can get it). The policy's text answers these, not the tables.
{{"action": "not_in_tables", "reason": "<what the tables do not hold>"}}
    only when the question asks for an amount, a limit or a percentage that these tables do not hold (for example
    for an age, a Sum Insured, a deductible or a plan that is not listed).

Rules for the SQL:
- Amounts: 1 lakh = 100,000 and 1 crore = 100 lakh = 10,000,000. Convert every amount in the question to rupees
  before choosing a column or a value, and check it against the column's printed amount (shown with its lakh or
  crore value).
- A column holds values only for the amounts its heading names. A heading that lists amounts (such as
  "12, 18 & 40 Lakhs") covers exactly those amounts; a heading with a limit (such as "> 60 lakhs") covers only the
  amounts on that side. If no column covers the asked amount, reply not_in_tables.
- Text values: compare with = (or IN for several) to values written exactly as listed, any mark included (a value
  printed as '5*' is written '5*'). Never compare text with BETWEEN, <, >, <= or >=.
- Categories: use the categories exactly as the tables list them (plans, tiers and their places, policy types).
  Something belongs to a category only when the category's listed values name it or something that contains it
  (a town inside a listed region); never place it by a classification from outside the policy. A category listed
  for all others covers everything not named elsewhere.
- Calculations: when the question asks for a difference, a total, or an amount after a discount, compute the result
  in the SQL as its own column (for example (SELECT ...) - (SELECT ...) AS difference, or
  premium * (1 - discount / 100.0) AS after_discount), and select the values it is computed from too.
- Select the columns that hold the answer and the columns that identify each row (such as the plan, tier or age),
  so every value can be traced to its printed row. Do not select constants.
- When a missing detail has only a few possible choices (such as two plans), return the rows for all of them.
- Never guess a value, and never round or change one. Return at most {max_rows} rows.

QUESTION: {question}
{attempts}JSON:""",
}


def build_sql_prompt(question: str, card: str, max_rows: int = 50, template: str = "sql_v2", attempts=()) -> str:
    """The SQL step's prompt. ``attempts``: [(earlier reply, why it could not be used), ...], oldest first."""
    try:
        text = SQL_TEMPLATES[template]
    except KeyError:
        raise KeyError(f"unknown SQL template {template!r}; available: {sorted(SQL_TEMPLATES)}") from None
    earlier = ""
    if attempts:
        earlier = "\nEARLIER REPLIES (they could not be used):\n" + "".join(
            f"Reply {n}: {reply.strip()}\nProblem: {problem}\n" for n, (reply, problem) in enumerate(attempts, 1)
        ) + "Write a corrected reply.\n\n"
    return text.format(card=card.strip(), question=question.strip(), max_rows=max_rows, attempts=earlier)
