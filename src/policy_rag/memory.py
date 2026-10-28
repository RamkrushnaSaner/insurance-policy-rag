"""Conversation memory: follow-up questions are rewritten into standalone questions.

How it works:

1. A ``Conversation`` keeps the turns of one chat (in memory; it can be saved to JSON). The caller owns it and
   passes it to ``RAGSystem.ask``; the pipeline itself stays stateless.
2. When a question arrives with earlier turns, the ``QuestionRewriter`` shows the answer model the last
   ``memory.max_turns`` turns (each turn's standalone question and the start of its answer) and asks for the new
   question in standalone form ("And after discharge?" -> "How many days of post-hospitalization expenses are
   covered after discharge?"). It uses the answer model with its configured settings, thinking included, and the
   answer cache, so a repeated rewrite costs nothing.
   A question that does not point back to the conversation (no "it", "that", "And ...") is not rewritten at all
   (``memory.skip_complete``), and the rewrite's answer is short (``memory.max_new_tokens``; thinking unchanged).
3. Guards reject a rewrite that is empty, too long, looks like an answer, contains a figure the user did not type,
   drops one the user typed, names nothing specific ("the policy"), or copies a run of words from an earlier answer
   (answers may only supply a name, "it" -> "the Plus Benefit"); the original question is then used.
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

# Words that point back to the conversation. A question without any of them is complete and is not rewritten.
# A pronoun after "and"/"or" + a new clause usually points inside the question ("What is X and when is it covered?").
_REFERS = re.compile(r"""(?ix)
    \b(it|its|they|them|their|theirs|those|these|such|same|former|latter)\b
  | \b(?:does|do|did|will|would|can|could|is|are|was|were|has|have|should|must|may|might)\s+
       (?:that\b|this\b(?!\s+(?:policy|plan|insurance|product|document)\b))
  | \bthat\b(?!\s+(?:is|are|was|were|has|have|had|can|could|will|would|may|might|must|should|applies|apply|covers|pays
                     |includes|include|requires|require)\b)
  | \bthis\b(?!\s+(?:policy|plan|insurance|product|document)\b)
  | (?<!\bis\s)(?<!\bare\s)(?<!\bwas\s)(?<!\bwere\s)(?<!\bbe\s)\bthere\b
  | \b(?:first|second|third|last|other|which|another|same|special|new|old|cheaper|each)\s+ones?\b
  | (?<!\bday\s)(?<!\byear\s)(?<!\bweek\s)(?<!\bmonth\s)\bones?\b(?=\s*(?:[?.,!]|$))
  | (?<!\bthe\s)\bones?\s+(?:of|for|in|with|from)\b
""")
_CONTINUES = re.compile(r"^\s*(?:and|also|then|so|but|what\s+about|how\s+about|what\s+if|in\s+that\s+case)\b", re.I)
_NEW_CLAUSE = re.compile(r",?\s(?:and|or)\s+(?=(?:when|which|what|how|where|who|why|whether|is|are|was|were|does|do|did"
                         r"|can|could|will|would|should|if)\b)", re.I)
# Replacing "it" with only these words names nothing ("What does the policy not pay for?").
GENERIC = {"policy", "plan", "cover", "coverage", "benefit", "benefits", "insurance", "scheme", "product", "option",
           "feature", "approval", "thing", "one"}
_FILLER = {"the", "a", "an", "this", "that", "these", "those", "of", "for", "to", "in", "on", "under", "with", "by",
           "and", "or", "is", "are", "be", "been", "being", "does", "do", "did", "i", "my", "me", "it", "its", "s"}


def needs_rewrite(question: str) -> bool:
    """True when the question points back to the conversation ("it", "that one", "And for X?"); False for a question
    that is complete as it stands. Questions that point back only through "the" ("the papers") are not recognised."""
    if _CONTINUES.search(question or ""):
        return True
    head = _NEW_CLAUSE.split((question or "").strip(), maxsplit=1)[0]
    return bool(_REFERS.search(head))


def _numbers(text: str) -> set:
    return {n.replace(",", "") for n in _NUMBER.findall(text or "")}


def _ngrams(text: str, n: int) -> list:
    w = _WORD.findall((text or "").lower())
    return [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]


def check_rewrite(raw: str, question: str, history: List[dict], max_chars: int = 400,
                  copy_words: int = 8) -> Tuple[str, Optional[str]]:
    """(question to use, reason for falling back or None). The rewrite is the first line of the model's reply,
    without a label or quotes. What the user typed (this question and the earlier questions) is the only source a
    rewrite may take figures or longer phrases from; the assistant's answers may only supply a name ("it" -> "the
    Plus Benefit"). A rewrite must keep the figures the user typed and must name something specific: one that only
    adds "the policy" or "the plan" is rejected. ``copy_words``: a run of this many words that the rewrite shares
    with an earlier answer, and that the user never typed, rejects the rewrite (0 turns the check off)."""
    original = question.strip()
    lines = [l.strip() for l in (raw or "").splitlines() if l.strip()]
    q = _LABEL.sub("", lines[0]).strip().strip('"“”\'').strip() if lines else ""
    if len(q) < 3:
        return original, "empty rewrite"
    if len(q) > max_chars:
        return original, "rewrite too long"
    if q == NO_ANSWER_TEXT or _ANSWER_START.match(q):
        return original, "rewrite looks like an answer"
    if q == original:
        return q, None
    typed = [question] + [h["question"] for h in history]
    new = _numbers(q) - {n for t in typed for n in _numbers(t)}
    if new:
        return original, f"rewrite adds figures that are not in the user's questions: {', '.join(sorted(new))}"
    lost = _numbers(question) - _numbers(q)
    if lost:
        return original, f"rewrite drops figures the user typed: {', '.join(sorted(lost))}"
    added = set(_WORD.findall(q.lower())) - set(_WORD.findall(original.lower())) - _FILLER
    if not added - GENERIC:
        return original, ("rewrite names nothing specific" + (f" (adds only: {', '.join(sorted(added))})" if added else ""))
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
        Without history, or (``memory.skip_complete``) for a question that does not point back to the conversation,
        the question is returned as it is, with no model call."""
        if not history:
            return {"question": question.strip(), "rewritten": False, "fallback": None, "turns_used": 0}
        if self.cfg.skip_complete and not needs_rewrite(question):
            return {"question": question.strip(), "rewritten": False, "fallback": None, "turns_used": len(history),
                    "skipped": "complete question"}
        prompt = build_condense_prompt(history, question, self.cfg.template)
        gen = self.answerer.answer(prompt, label="rewrite", max_new_tokens=self.cfg.max_new_tokens or None)
        standalone, reason = check_rewrite(gen["answer"], question, history, self.cfg.max_question_chars,
                                           self.cfg.copy_words)
        if reason is None and gen.get("hit_limit") and len([l for l in (gen["answer"] or "").splitlines() if l.strip()]) < 2:
            standalone, reason = question.strip(), "rewrite cut off at the length limit"
        return {"question": standalone, "rewritten": standalone != question.strip(), "fallback": reason,
                "turns_used": len(history), "raw": (gen["answer"] or "")[:500], "key": gen.get("key"),
                "seconds": gen.get("seconds"), "think_tokens": gen.get("think_tokens"),
                "thinking_forced": gen.get("thinking_forced"), "from_cache": gen.get("from_cache")}
