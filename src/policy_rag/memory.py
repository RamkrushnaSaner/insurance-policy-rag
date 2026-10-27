"""Conversation memory: follow-up questions are rewritten into standalone questions.

How it works:

1. A ``Conversation`` keeps the turns of one chat (in memory; it can be saved to JSON). The caller owns it and
   passes it to ``RAGSystem.ask``; the pipeline itself stays stateless.
2. When a question arrives with earlier turns, the ``QuestionRewriter`` shows the answer model the last
   ``memory.max_turns`` turns (each turn's standalone question and the start of its answer) and asks for the new
   question in standalone form ("And after discharge?" -> "How many days of post-hospitalization expenses are
   covered after discharge?"). It uses the answer model with its configured settings, thinking included, and the
   answer cache, so a repeated rewrite costs nothing.
3. Guards reject a rewrite that is empty, too long, looks like an answer, contains a figure the user did not type,
   or copies a run of words from an earlier answer (answers may only supply a name, "it" -> "the Plus Benefit");
   the original question is then used.
4. Only the standalone question is searched and answered. The answer prompt never contains the conversation, so
   answers still come from the policy text alone, and a question without earlier turns is handled exactly as before.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from .prompts import NO_ANSWER_TEXT, build_condense_prompt


@dataclass
class Turn:
    question: str                 # what the user typed
    standalone: str               # the question that was searched and answered
    answer: str
    sources: List[str] = field(default_factory=list)   # labels of the passages the answer model read
    route: str = ""               # "answer" or "no_answer"


@dataclass
class Conversation:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    turns: List[Turn] = field(default_factory=list)
    documents: Optional[List[str]] = None   # document filter kept for every turn (None = all documents)

    def add(self, question: str, standalone: str, answer: str, sources=None, route: str = ""):
        self.turns.append(Turn(question, standalone, answer, list(sources or []), route))

    def recent(self, max_turns: int, answer_chars: int) -> List[dict]:
        """The last ``max_turns`` turns as the rewriter sees them: standalone questions, shortened answers."""
        out = []
        for t in self.turns[-max_turns:] if max_turns > 0 else []:
            a = t.answer if len(t.answer) <= answer_chars else t.answer[:answer_chars].rsplit(" ", 1)[0] + " ..."
            out.append({"question": t.standalone or t.question, "answer": a})
        return out

    def clear(self):
        self.turns.clear()

    def to_dict(self) -> dict:
        return {"id": self.id, "documents": self.documents, "turns": [asdict(t) for t in self.turns]}

    @classmethod
    def from_dict(cls, d: dict) -> "Conversation":
        return cls(d.get("id") or uuid.uuid4().hex[:12], [Turn(**t) for t in d.get("turns", [])], d.get("documents"))

    def save(self, path) -> Path:
        """Write the conversation as JSON. It contains policy text: keep it out of public repositories."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path) -> "Conversation":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------------------------------- guards
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_LABEL = re.compile(r"^(standalone question|rewritten question|last question|question)\s*:\s*", re.I)
_ANSWER_START = re.compile(r"^(yes|no)\b[\s,.:;!-]", re.I)


_WORD = re.compile(r"[a-z0-9]+")


def _numbers(text: str) -> set:
    return {n.replace(",", "") for n in _NUMBER.findall(text or "")}


def _ngrams(text: str, n: int) -> list:
    w = _WORD.findall((text or "").lower())
    return [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]


def check_rewrite(raw: str, question: str, history: List[dict], max_chars: int = 400,
                  copy_words: int = 6) -> Tuple[str, Optional[str]]:
    """(question to use, reason for falling back or None). The rewrite is the first line of the model's reply,
    without a label or quotes. What the user typed (this question and the earlier questions) is the only source a
    rewrite may take figures or longer phrases from; the assistant's answers may only supply a name ("it" -> "the
    Plus Benefit"). ``copy_words``: a run of this many words that the rewrite shares with an earlier answer, and that
    the user never typed, rejects the rewrite (0 turns the check off)."""
    original = question.strip()
    lines = [l.strip() for l in (raw or "").splitlines() if l.strip()]
    q = _LABEL.sub("", lines[0]).strip().strip('"“”\'').strip() if lines else ""
    if len(q) < 3:
        return original, "empty rewrite"
    if len(q) > max_chars:
        return original, "rewrite too long"
    if q == NO_ANSWER_TEXT or _ANSWER_START.match(q):
        return original, "rewrite looks like an answer"
    typed = [question] + [h["question"] for h in history]
    new = _numbers(q) - {n for t in typed for n in _numbers(t)}
    if new:
        return original, f"rewrite adds figures that are not in the user's questions: {', '.join(sorted(new))}"
    if copy_words > 0:
        said = {g for t in typed for g in _ngrams(t, copy_words)}
        answers = {g for h in history for g in _ngrams(h["answer"], copy_words)}
        copied = [g for g in _ngrams(q, copy_words) if g in answers and g not in said]
        if copied:
            return original, f"rewrite copies an earlier answer: '{' '.join(copied[0])} ...'"
    return q, None


class QuestionRewriter:
    """Follow-up -> standalone question, generated by the answer model through the ``Answerer`` (same settings,
    same cache, seed from the prompt)."""

    def __init__(self, answerer, cfg):
        self.answerer, self.cfg = answerer, cfg

    def rewrite(self, question: str, history: List[dict]) -> dict:
        """{"question": the question to search and answer, "rewritten": changed?, "fallback": reason or None, ...}.
        Without history the question is returned as it is, with no model call."""
        if not history:
            return {"question": question.strip(), "rewritten": False, "fallback": None, "turns_used": 0}
        prompt = build_condense_prompt(history, question, self.cfg.template)
        gen = self.answerer.answer(prompt, label="rewrite")
        standalone, reason = check_rewrite(gen["answer"], question, history, self.cfg.max_question_chars,
                                           self.cfg.copy_words)
        return {"question": standalone, "rewritten": standalone != question.strip(), "fallback": reason,
                "turns_used": len(history), "raw": (gen["answer"] or "")[:500], "key": gen.get("key"),
                "seconds": gen.get("seconds"), "think_tokens": gen.get("think_tokens"),
                "thinking_forced": gen.get("thinking_forced"), "from_cache": gen.get("from_cache")}
