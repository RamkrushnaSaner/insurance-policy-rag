"""The test set and how its answer keys are matched against text.

A test-set file is JSON: ``{"name", "document", "table_checker", "evidence_removed_candidates", "items": [...]}``.
Fields per item:

- ``id``, ``split`` ("dev" for debugging, "test" held out for adopt/reject decisions), ``type``, ``answerable``,
  ``paraphrase`` (everyday wording), ``question``, ``reference`` (a reference answer for human reading);
- ``sections``: section ids that hold the answer;
- ``evidence``: evidence passages, each a list of alternative anchor phrases. A chunk is relevant when it contains
  one, so retrieval is scored for any chunk file of the same document;
- ``facts``: key facts, each a list of alternative phrasings (``re:`` = a regular expression on normalised text);
- optional checks: ``table_cells`` (plan-table values), ``polarity`` (yes/no), ``ordered``, ``off_scope`` (sections
  of another cover), ``must_not`` (known false statements), ``derived_figures`` (figures the answer may compute),
  ``expect_admit`` (must say what is not shown), ``absent_terms`` (not-answerable: terms absent from the policy).

The version is a hash of the items: changing anything makes a new version, and baselines must be re-run.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..corpus import Corpus


@dataclass
class TestSet:
    __test__ = False                          # not a pytest test class
    name: str
    items: List[dict]
    document: Optional[str] = None             # the document id the questions are about (multi-document configs)
    table_checker: Optional[str] = None
    evidence_removed_candidates: List[str] = field(default_factory=list)
    path: Optional[str] = None

    @property
    def version(self) -> str:
        return hashlib.sha1(json.dumps(self.items, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]

    @property
    def by_id(self) -> Dict[str, dict]:
        return {t["id"]: t for t in self.items}

    def select(self, splits) -> List[dict]:
        return [t for t in self.items if t["split"] in splits]


def load_testset(path) -> TestSet:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"name": Path(path).stem, "items": data}
    return TestSet(name=data.get("name", Path(path).stem), items=data["items"], document=data.get("document"),
                   table_checker=data.get("table_checker"),
                   evidence_removed_candidates=data.get("evidence_removed_candidates", []), path=str(path))


# ---------------------------------------------------------------------------------------------------- matching
def norm_text(s: str) -> str:
    """Lower-case; drop currency marks; dashes and slashes to spaces; -isation/-ised to -ization/-ized;
    "pre- and post-hospitalization" spelled out."""
    s = (s or "").lower().replace("₹", " ")
    s = re.sub(r"\bpre\s*[-‐‑–]?\s*(?:and|or|&|/)\s*post\s*[-‐‑–]?\s*(hospitali[sz](?:ation|ed))",
               r"pre-\1 and post-\1", s)
    s = re.sub(r"\brs\.?\s*", " ", s)
    s = re.sub(r"[-‐‑‒–—/]", " ", s)
    s = re.sub(r"isation", "ization", s)
    s = re.sub(r"ised\b", "ized", s)
    return re.sub(r"\s+", " ", s).strip()


def _alt_re(alt: str) -> str:
    """A phrase matches at a word start ("5 years" is not found inside "65 years"); a phrase ending in a digit must
    also end there ("10,000" is not found inside "10,000,000"); "N days/months/years" also matches the singular."""
    if alt.startswith("re:"):
        return alt[3:]
    a = norm_text(alt)
    if re.search(r"\d (day|week|month|year|hour)s$", a):
        return r"(?<![0-9a-z.,])" + re.escape(a[:-1]) + r"s?(?![a-z])"
    return r"(?<![0-9a-z.,])" + re.escape(a) + (r"(?![0-9]|[.,][0-9])" if a[-1:].isdigit() else "")


def fact_pos(alternatives, text_norm: str):
    hits = [m.start() for m in (re.search(_alt_re(a), text_norm) for a in alternatives) if m]
    return min(hits) if hits else None


def fact_found(alternatives, text_norm: str) -> bool:
    return fact_pos(alternatives, text_norm) is not None


def fact_label(f) -> str:
    return re.sub(r"\(\?[<!=][^)]*\)|\\[bd]|re:|[?(){}\[\]\\^$*+|]", "", f[0])[:40]


def words(s: str) -> str:
    return " " + " ".join(re.findall(r"[a-z0-9]+", norm_text(s))) + " "


def anchor_in(alternatives, words_text: str) -> bool:
    """An evidence passage is in a text (as returned by ``words``) if one anchor's words occur in order; anchors of
    6+ words also match on 60% of their words in a row (a chunk boundary may cut them)."""
    for a in alternatives:
        w = words(a).split()
        if " " + " ".join(w) + " " in words_text:
            return True
        n = max(4, math.ceil(0.6 * len(w)))
        if len(w) >= 6 and any(" " + " ".join(w[i:i + n]) + " " in words_text for i in range(len(w) - n + 1)):
            return True
    return False


class CorpusText:
    """Normalised texts of a corpus, computed once (used by the checks and retrieval metrics)."""

    def __init__(self, corpus: Corpus, doc_id: Optional[str] = None):
        self.corpus = corpus
        self.chunk_words = [words(c.text) for c in corpus.chunks]
        own = [c for c in corpus.chunks if doc_id is None or c.doc_id == doc_id]
        self.norm = norm_text(" ".join(c.label + " " + c.text for c in own))
        self.all_words = " ".join(self.chunk_words[c.idx] for c in own)

    def words_of(self, ids) -> str:
        return " ".join(self.chunk_words[i] for i in sorted(set(ids)))


def check_testset(items, corpus: Corpus, text: CorpusText) -> List[dict]:
    """Facts and evidence against the loaded chunk file(s); absent terms really absent."""
    rows = []
    for t in items:
        if t["answerable"]:
            miss_f = [fact_label(f) for f in t["facts"] if not fact_found(f, text.norm)
                      and not any(fact_found(f, str(v)) for _, v in t.get("derived_figures", []))]
            miss_e = sum(not anchor_in(e, text.all_words) for e in t["evidence"])
            if miss_f or miss_e:
                rows.append({"id": t["id"], "problem": "evidence not in this chunk file",
                             "detail": f"{miss_e}/{len(t['evidence'])} passages; facts: {', '.join(miss_f) or '-'}"})
        for term in t.get("absent_terms", []):
            if not t["answerable"] and re.search(r"\b" + re.escape(norm_text(term)) + r"\b", text.norm):
                rows.append({"id": t["id"], "problem": "out-of-scope term occurs in the chunk file", "detail": term})
    return rows
