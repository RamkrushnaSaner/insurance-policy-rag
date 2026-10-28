"""Small stand-in components so the whole system can be tested on any CPU without downloading models.

They are registered as component types (``fake_hash``, ``fake_overlap``, ``fake_extractive``); a config selects them
with ``plugins: [fakes]`` and the ``type`` fields. They are deliberately simple: the tests check the plumbing
(retrieval, routing, context, gate, cache, evaluation), not answer quality.
"""
import hashlib
import re

import numpy as np

from policy_rag.prompts import NO_ANSWER_TEXT
from policy_rag.registry import register

_WORD = re.compile(r"[a-z0-9]+")


def _words(s):
    return _WORD.findall(s.lower())


@register("embedder", "fake_hash")
class HashEmbedder:
    """Bag of hashed words (64 dimensions)."""
    name = "fake-hash"

    def __init__(self, dim=64):
        self.dim = dim

    @classmethod
    def from_config(cls, cfg, **_):
        return cls()

    def _vec(self, text):
        v = np.zeros(self.dim, dtype="float32")
        for w in _words(text):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        return v

    def encode_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def encode_query(self, text):
        return self._vec(text)[None, :]

    def revision(self):
        return None


_STOP = {"what", "which", "does", "with", "that", "this", "from", "under", "have", "policy", "cover", "covered",
         "covers", "paid", "pays", "there", "their", "will", "when", "how", "many", "much"}


def content_words(s):
    return {w for w in _words(s) if len(w) >= 4 and w not in _STOP}


@register("reranker", "fake_overlap")
class OverlapReranker:
    """Score = number of content words of the question found in the passage, minus 6: a passage sharing no such
    word scores -6, below the default gate threshold of -5."""
    name = "fake-overlap"

    @classmethod
    def from_config(cls, cfg, **_):
        return cls()

    def score(self, query, docs):
        q = content_words(query)
        return np.array([len(q & set(_words(d))) - 6.0 for d in docs], dtype="float32")

    def revision(self):
        return None


@register("generator", "fake_extractive")
class ExtractiveGenerator:
    """Answers with the context sentence that shares most words with the question (or refuses). Counts calls, so
    tests can check that the cache works."""
    name = "fake-extractive"
    precision = "none"
    decoding = {"do_sample": False}

    def __init__(self):
        self.calls = 0                # answer calls
        self.rewrite_calls = 0        # follow-up rewrite calls (the condense prompt)
        self.rewrites = {}            # scripted rewrites: follow-up question -> model reply

    @classmethod
    def from_config(cls, cfg, **_):
        return cls()

    def signature(self):
        return {"llm": self.name}

    def revision(self):
        return None

    def count_tokens(self, text):
        return len(text.split())

    def self_check(self):
        return []

    def tokens_per_second(self, prompt):
        return 1000.0

    def _rewrite(self, prompt):
        """A follow-up rewrite: the scripted reply if there is one; otherwise questions with a reference word ("it",
        "that", "and ...") get the previous question added, other questions come back unchanged."""
        self.rewrite_calls += 1
        question = prompt.rsplit("LAST QUESTION:", 1)[1].rsplit("STANDALONE QUESTION:", 1)[0].strip()
        if question in self.rewrites:
            return self.rewrites[question]
        previous = [l[len("User: "):] for l in prompt.splitlines() if l.startswith("User: ")][-1]
        refers = re.search(r"\b(it|its|that|there|this|one|them|those)\b|^(and|so|then|what about|how about)\b", question, re.I)
        return f"{previous.rstrip('?')}: {question}" if refers else question

    def generate(self, prompt, seed, max_new_tokens=None):
        """The context passage that shares most content words with the question, or the refusal."""
        if prompt.rstrip().endswith("STANDALONE QUESTION:"):
            self.rewrite_max_new_tokens = max_new_tokens
            text = self._rewrite(prompt)
            return {"answer": text, "thinking": "", "prompt_tokens": len(prompt.split()), "think_tokens": 0,
                    "answer_tokens": len(text.split()), "thinking_forced": False, "hit_limit": False}
        self.calls += 1
        context = prompt.split("CONTEXT:", 1)[1].rsplit("QUESTION:", 1)[0].strip()
        question = prompt.rsplit("QUESTION:", 1)[1].rsplit("ANSWER:", 1)[0]
        q = content_words(question)
        bodies = [b.partition("\n")[2].strip() for b in re.split(r"\n\n(?=\[)", context)]
        best = max(bodies, key=lambda b: len(q & set(_words(b))), default="")
        answer = best if best and q & set(_words(best)) else NO_ANSWER_TEXT
        return {"answer": answer, "thinking": "", "prompt_tokens": len(prompt.split()), "think_tokens": 0,
                "answer_tokens": len(answer.split()), "thinking_forced": False, "hit_limit": False}
