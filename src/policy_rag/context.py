"""From reranked candidates to the context the answer model reads.

1. **Title pinning**: sections whose heading words all appear in the question go first (with their sub-sections).
2. **Seeds**: reranked chunks scoring at least ``seed_score_floor``, best first.
3. **Neighbour expansion**: each seed brings the text it depends on: the next chunk of the same group or a
   sub-section (once more after a "... provided below" pointer), the previous chunk of the same group or the parent
   section, the neighbouring year of a plan table. Chunks of routed indexes (premium tables) are self-contained.
4. **Budget**: groups are added until ``token_budget`` (answer-model tokens, labels included) is used up.

Each group reaches the model as one passage under a label line such as ``[Section 4.4 - Plus Benefit]``.
Neighbours never cross from one document into another.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .corpus import Chunk, Corpus

_TITLE_STOP = {"of", "for", "the", "a", "an", "in", "to", "and", "on", "with", "under", "by", "procedure", "list",
               "section", "notes", "note", "rs", "my"}
_POINTER = re.compile(r"(below|following|as follows|hereunder)\W*$|:\s*$", re.I)


def split_sentences(t: str) -> List[str]:
    return [p.strip() for p in re.split(r"(?<=[.;])\s+(?=[A-Z0-9·₹(])", t) if p.strip()]


def _words_of(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def title_core(title: str) -> frozenset:
    """Heading words that must all appear in a question for the section to be pinned."""
    title = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", title)
    return frozenset(w for w in _words_of(title) if w not in _TITLE_STOP)


def _parent(sid: str) -> Optional[str]:
    return sid.rsplit(".", 1)[0] if "." in sid else None


def _is_child(a: str, b: str) -> bool:
    return a.startswith(b + ".")


class ContextBuilder:
    def __init__(self, corpus: Corpus, cfg, count_tokens):
        """``count_tokens(text) -> int`` should use the answer model's tokenizer (the budget is in its tokens)."""
        self.corpus, self.cfg, self.count_tokens = corpus, cfg, count_tokens
        self.chunks: List[Chunk] = corpus.chunks
        self._block: Dict[int, int] = {}
        self.section_cores = {key: title_core(s.title) for key, s in corpus.sections.items() if s.index == "policy"}

    # ------------------------------------------------------------------ sizes
    def block_tokens(self, i: int) -> int:
        """Tokens a chunk adds to the prompt: label line + text + separator."""
        if i not in self._block:
            c = self.chunks[i]
            self._block[i] = self.count_tokens(f"[{c.label}]") + self.count_tokens(c.text) + 2
        return self._block[i]

    # ------------------------------------------------------------------ title pinning
    def title_matches(self, query: str, docs: Optional[Set[str]] = None) -> List[Tuple[str, str]]:
        """Sections whose heading words all appear in the question; a match inside a longer match is dropped."""
        q = set(_words_of(query))
        cores = self.section_cores if docs is None else {k: v for k, v in self.section_cores.items() if k[0] in docs}
        hits = [k for k, core in cores.items() if core and core <= q]
        hits = [a for a in hits if not any(cores[a] < cores[b] for b in hits)]
        hits.sort(key=lambda k: -len(cores[k]))
        return hits[:self.cfg.max_pinned_sections]

    def pinned_group(self, key: Tuple[str, str]) -> List[int]:
        """All chunks of a title-matched section, then its sub-sections, in document order."""
        doc_id, sid = key
        idxs = list(self.corpus.sections[key].chunk_ids)
        j = idxs[-1] + 1
        while j < len(self.chunks) and self.chunks[j].doc_id == doc_id and _is_child(self.chunks[j].section_id, sid):
            idxs.append(j)
            j += 1
        return idxs

    # ------------------------------------------------------------------ neighbours
    def _same_table_family(self, a: Chunk, b: Chunk) -> bool:
        return a.is_table and b.is_table and _parent(a.section_id) is not None \
            and _parent(a.section_id) == _parent(b.section_id)

    def group_for(self, i: int) -> List[int]:
        ch, seed, out = self.chunks, self.chunks[i], [i]
        if not seed.is_main:
            return out
        if i + 1 < len(ch) and ch[i + 1].doc_id == seed.doc_id:
            nxt = ch[i + 1]
            if nxt.is_main and (nxt.unit == seed.unit or _is_child(nxt.section_id, seed.section_id)
                                or self._same_table_family(seed, nxt)):
                out.append(i + 1)
                if _POINTER.search(nxt.text.strip()) and i + 2 < len(ch) and ch[i + 2].doc_id == seed.doc_id:
                    nn = ch[i + 2]
                    if nn.unit == nxt.unit or _is_child(nn.section_id, nxt.section_id) \
                            or _is_child(nn.section_id, seed.section_id):
                        out.append(i + 2)
        if i > 0 and ch[i - 1].doc_id == seed.doc_id:
            prv = ch[i - 1]
            if prv.is_main and (prv.unit == seed.unit or prv.section_id == _parent(seed.section_id)
                                or self._same_table_family(seed, prv)):
                out.append(i - 1)
        return out

    # ------------------------------------------------------------------ the context
    def build(self, query: str, candidate_ids: Sequence[int], scores, docs: Optional[Set[str]] = None) -> List[List[int]]:
        """Groups of chunk ids (document order inside a group, most relevant group first), within the budget."""
        budget, floor = self.cfg.token_budget, self.cfg.seed_score_floor
        score_of = {int(i): float(s) for i, s in zip(candidate_ids, scores)}
        ranked = sorted(score_of, key=lambda i: (-score_of[i], i))
        plan = [self.pinned_group(k) for k in self.title_matches(query, docs)] if self.cfg.title_pinning else []
        plan += [self.group_for(i) if self.cfg.neighbour_expansion else [i] for i in ranked if score_of[i] >= floor]
        used, taken, groups = 0, set(), []
        for g in plan:
            group = []
            for i in g:
                if i in taken:
                    continue
                cost = self.block_tokens(i)
                if used + cost > budget:
                    if group or groups:
                        break
                    continue
                group.append(i)
                taken.add(i)
                used += cost
            if group:
                groups.append(sorted(group))
            if used >= budget - 40:
                break
        return groups

    def shorten(self, groups: List[List[int]], limit: int) -> List[List[int]]:
        """The first ``limit`` tokens of a context (used for the out-of-memory retry)."""
        keep, used = [], 0
        for g in groups:
            part = []
            for i in g:
                t = self.block_tokens(i)
                if used + t > limit and (keep or part):
                    break
                part.append(i)
                used += t
            if part:
                keep.append(part)
            if used >= limit:
                break
        return keep

    def tokens(self, groups: List[List[int]]) -> int:
        return sum(self.block_tokens(i) for g in groups for i in g)

    # ------------------------------------------------------------------ formatting
    def merge_group_text(self, group: List[int]) -> str:
        """Consecutive chunks of one group as one passage; a sentence repeated as overlap is dropped once."""
        ch, parts = self.chunks, []
        for k, i in enumerate(group):
            t = ch[i].text
            if k and group[k - 1] == i - 1 and ch[i - 1].section_key == ch[i].section_key:
                sents, prev = split_sentences(t), split_sentences(ch[i - 1].text)
                if sents and prev and sents[0] == prev[-1]:
                    t = " ".join(sents[1:])
            parts.append(t)
        return " ".join(p for p in parts if p)

    def group_label(self, group: List[int]) -> str:
        labels = list(dict.fromkeys(self.chunks[i].label for i in group))
        return "[" + " → ".join(labels) + "]"

    def format(self, groups: List[List[int]]) -> str:
        return "\n\n".join(f"{self.group_label(g)}\n{self.merge_group_text(g)}" for g in groups)
