"""Deterministic answer checks (no LLM judge) and retrieval metrics.

Per question, **fully correct** (the primary metric) = every key fact, no critical error, nothing missing or cut
off; for a not-answerable question: says so at the start, with no critical error.

Critical errors: wrong plan-table value, known false statement, another cover's rule, wrong yes/no, wrong order,
a figure that is not in the context, answering a not-answerable question.

Every problem is tagged with the pipeline stage where it started (reliability > corpus > retrieval > context >
gate > generation), so a failure points at the component to change.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

import numpy as np

from ..prompts import is_no_answer
from .testset import CorpusText, anchor_in, fact_found, fact_label, fact_pos, norm_text

# ---------------------------------------------------------------------------------------------------- figures
_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty five": 45, "sixty": 60, "ninety": 90}
_UNIT = {"day": "days", "month": "months", "year": "years", "hour": "hours"}


def figures(text: str) -> set:
    """Amounts (lakh/crore, rupee and comma-grouped forms unified), percentages and durations."""
    t = (text or "").lower().replace("₹", " ₹ ").replace("-", " ")
    for w, n in sorted(_WORD_NUM.items(), key=lambda x: -len(x[0])):
        t = re.sub(rf"\b{w}\b", str(n), t)
    out = set()
    for m in re.finditer(r"(?:₹|\brs\.?|\binr)?\s*(\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|cr)\b", t):
        out.add(("inr", int(round(float(m.group(1)) * (1e5 if m.group(2).startswith("la") else 1e7)))))
    for m in re.finditer(r"(?:₹|\brs\.?|\binr)\s*(\d[\d,]*(?:\.\d+)?)(?!\d|[.,]\d)(?!\s*(?:lakhs?|lacs?|crores?|cr)\b)", t):
        out.add(("inr", int(round(float(m.group(1).replace(",", ""))))))
    for m in re.finditer(r"(?<![\d.,])(\d{1,3}(?:,\d{2,3})+)(?![\d,])(?!\s*(?:lakhs?|lacs?|crores?|cr)\b)", t):
        out.add(("inr", int(m.group(1).replace(",", ""))))           # premium tables: "Age 40: 10,300 / 13,300"
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", t):
        out.add(("pct", float(m.group(1))))
    for m in re.finditer(r"(\d+)\s*(?:(?:consecutive|continuous|calendar|policy|completed|full)\s+){0,2}(day|month|year|hour)s?\b", t):
        out.add((_UNIT[m.group(2)], int(m.group(1))))
    return out


def _plain_numbers(text: str) -> set:
    return {float(x.replace(",", "")) for x in re.findall(r"(?<![\d.])\d[\d,]*(?:\.\d+)?", text or "")}


def unsupported_figures(answer: str, context: str, question: str = "", derived=()) -> list:
    """Figures in the answer that are not in the context or the question. An amount also counts as supported when
    the context has the same number without a currency mark (tables write "max upto 6000")."""
    extra, plain = {tuple(d) for d in derived}, _plain_numbers(context) | _plain_numbers(question)
    left = {(k, v) for k, v in figures(answer) - figures(context) - figures(question) - extra
            if not (k == "inr" and float(v) in plain)}
    return sorted(f"Rs {v:,.0f}" if k == "inr" else f"{v:g}%" if k == "pct" else f"{v:g} {k}" for k, v in left)


def cited_sections(text: str) -> set:
    ids = set()
    for m in re.finditer(r"sections?\s+((?:\d+(?:\.\d+)*(?:\.[a-z])?(?:\s*(?:,|and|&)\s*)?)+)", (text or "").lower()):
        ids.update(s.rstrip(".") for s in re.findall(r"\d+(?:\.\d+)*", m.group(1)))
    return ids


def unsupported_citations(answer: str, context: str) -> list:
    ctx_ids = set(re.findall(r"\d+(?:\.\d+)+|\b\d+\b", context or ""))
    return sorted(s for s in cited_sections(answer) if s not in ctx_ids)


# ---------------------------------------------------------------------------------------------------- refusals, yes/no
_DECLINE = re.compile(
    r"not (specified|mentioned|stated|provided|included|covered|listed|available|given|shown|found|offered|described|"
    r"defined|addressed|part of|applicable|contain)|(does|do|did|is|are)( not|n[’']t) (specify|mention|cover|state|provide|"
    r"include|contain|list|offer|have|say|describe|address|apply|exist|cover|appear|explicitly)|"
    r"\bno (information|mention|details|reference|data|such|specific|indication|provision)\b|\bthere (is|are) no\b|"
    r"cannot (be )?(found|determined|answered|confirmed|provide)|unable to (find|determine|answer|confirm|provide)|"
    r"\bnot (possible|able) to (determine|answer|say)|outside the scope|beyond the (scope|provided)", re.I)


def sentences_of(text: str):
    return [s.strip() for s in re.split(r"\n+|(?<=[.!?])\s+", text or "") if s.strip()]


def declines_at_start(answer: str) -> bool:
    """A refusal or "the policy does not say" within the first two sentences (an answer that invents first and
    hedges at the end does not count)."""
    if is_no_answer(answer):
        return True
    head = " ".join(re.sub(r"^[\W_]+", "", s) for s in sentences_of(answer)[:2])
    return bool(_DECLINE.search(head))


def admits_gap(answer: str) -> bool:
    return bool(_DECLINE.search(answer or "")) or bool(re.search(
        r"\bno\b[^.;:\n]{0,60}\b(mention|mentioned|information|specified|stated|provided|details|listed)\b", answer or "", re.I))


def answer_polarity(answer: str):
    """'yes' / 'no' when the answer opens with one (e.g. "No, return transportation is not covered")."""
    m = re.match(r"[\W_]*(?:(?:based on|according to|as per)[^,\n]{0,80},\s*)?(yes|no)\b\s*[,.!:;–—-]", (answer or "").lower())
    return m.group(1) if m else None


# ---------------------------------------------------------------------------------------------------- scope
def context_blocks(context: str):
    """(section ids, text) for each passage of a formatted context."""
    out = []
    for b in re.split(r"\n\n(?=\[)", context or ""):
        head, _, body = b.partition("\n")
        out.append((re.findall(r"Section ([\w.\-]+?)(?: -|\]| →|$)", head), body))
    return out


def _bigrams(text: str) -> set:
    w = re.findall(r"[a-z0-9]+", norm_text(text))
    return set(zip(w, w[1:]))


def _in_sections(sid: str, prefixes) -> bool:
    return any(sid == p or sid.startswith(p + ".") for p in prefixes)


def off_scope_hits(answer: str, context: str, prefixes):
    """Sections of another cover or procedure that the answer cites, or copies a sentence from."""
    if not prefixes or not answer or is_no_answer(answer):
        return []
    hits = [f"cites Section {s}" for s in sorted(cited_sections(answer)) if _in_sections(s, prefixes)]
    blocks = [(ids, _bigrams(body)) for ids, body in context_blocks(context)]
    for s in sentences_of(answer):
        sb = _bigrams(s)
        if len(sb) < 5 or not blocks:
            continue
        ids, bg = max(blocks, key=lambda b: len(sb & b[1]))
        in_scope = max((len(sb & b) for i2, b in blocks if not any(_in_sections(i, prefixes) for i in i2)), default=0)
        if len(sb & bg) / len(sb) >= 0.5 and any(_in_sections(i, prefixes) for i in ids) \
                and not any(not _in_sections(i, prefixes) for i in ids) and in_scope / len(sb) < 0.5:
            hits.append(f"copies Section {ids[0]} (\"{s[:50]}…\")")    # not when the right section has it too
    return hits


def cut_off(answer: str, gen):
    if gen and gen.get("hit_limit") is not None:
        return bool(gen["hit_limit"])
    a = (answer or "").rstrip()
    return bool(a) and not is_no_answer(a) and not re.search(r"[.!?:)\]\"”'*|]$", a)


# ---------------------------------------------------------------------------------------------------- verdict
STAGES = ["reliability", "corpus", "retrieval", "context", "gate", "generation"]
CRITICAL = {"table_wrong", "known_false", "off_scope", "polarity", "wrong_order", "unsupported_figures", "not_declined"}
INCOMPLETE = {"missing_fact", "table_missing", "refused", "no_admit", "cut_off"}


def score_answer(item: dict, answer: str, context: str, corpus_norm: str, gen=None, gate_route="answer", error=None,
                 cand_norm: Optional[str] = None, table_checker: Optional[Callable] = None) -> dict:
    """Verdict for one answer. ``corpus_norm`` = normalised text of the document (to tell a corpus gap from a
    retrieval miss); ``cand_norm`` = normalised text of all search candidates (live runs only), to tell a search
    miss from a context miss."""
    issues = []
    add = lambda stage, kind, text: issues.append({"stage": stage, "kind": kind, "issue": text})
    a, a_norm, c_norm = answer or "", norm_text(answer), norm_text(context)
    v = {"refused": is_no_answer(a), "fact_recall": None, "facts_missing": [], "table": None}
    if error:
        add("reliability", "error", f"pipeline error: {str(error)[:80]}")
    if item["answerable"]:
        found = [fact_found(f, a_norm) for f in item["facts"]]
        v["fact_recall"] = sum(found) / len(found) if found else None
        v["facts_missing"] = [fact_label(f) for f, ok in zip(item["facts"], found) if not ok]
        v["false_refusal"] = v["refused"] or (declines_at_start(a) and not any(found))
        for f, ok in zip(item["facts"], found):
            if ok:
                continue
            derived = any(fact_found(f, str(x[1])) for x in item.get("derived_figures", []))
            if not derived and not fact_found(f, corpus_norm):
                add("corpus", "missing_fact", f"not in the chunk file: {fact_label(f)}")
            elif not derived and cand_norm is not None and not fact_found(f, cand_norm):
                add("retrieval", "missing_fact", f"search did not find: {fact_label(f)}")
            elif not derived and not fact_found(f, c_norm):
                add("context", "missing_fact", f"not in the context: {fact_label(f)}")
            elif not v["refused"] and gate_route == "answer":
                add("generation", "missing_fact", f"in the context, not in the answer: {fact_label(f)}")
        if gate_route == "no_answer":
            add("gate", "refused", "refused by the relevance gate")
        elif v["false_refusal"]:
            add("generation", "refused", "model refused")
        if not v["refused"] and a:
            if item.get("table_cells") and table_checker is not None:
                v["table"] = tc = table_checker(a, item["table_cells"], item["question"])
                if tc["wrong"]:
                    add("generation", "table_wrong", f"wrong table values ({len(tc['wrong'])}): " + "; ".join(tc["wrong"][:3]))
                if tc["missing"]:
                    add("generation", "table_missing", f"table values missing: {len(tc['missing'])} of {tc['expected']}")
            if item.get("ordered"):
                pos = [fact_pos(f, a_norm) for f in item["facts"]]
                pos = [p for p in pos if p is not None]
                if pos != sorted(pos):
                    add("generation", "wrong_order", "key facts in the wrong order")
            if item.get("polarity") and answer_polarity(a) and answer_polarity(a) != item["polarity"]:
                add("generation", "polarity", f"answers '{answer_polarity(a)}', the policy says '{item['polarity']}'")
            if item.get("expect_admit") and not admits_gap(a):
                add("generation", "no_admit", "does not say which part the policy does not answer")
            if cut_off(a, gen):
                add("generation", "cut_off", "stopped at the answer length limit")
            missing_said = _DECLINE.search(a) and [fact_label(f) for f, ok in zip(item["facts"], found)
                                                    if not ok and fact_found(f, c_norm)]
            if missing_said:
                add("generation", "says_missing", "says something is not shown, but the context has: " + ", ".join(missing_said))
    elif not declines_at_start(a):
        add("generation", "not_declined", "answered instead of saying the policy does not cover it")
    if a and not is_no_answer(a):
        for p in item.get("must_not", []):
            m = re.search(p, a_norm)
            if m:
                add("generation", "known_false", f"known false or wrong-row statement: \"{m.group(0)[:60]}\"")
        off = off_scope_hits(a, context, item.get("off_scope"))
        if off:
            add("generation", "off_scope", "uses another cover's or procedure's section: " + "; ".join(off[:2]))
        uf = unsupported_figures(a, context, item["question"], item.get("derived_figures", []))
        if uf:
            add("generation", "unsupported_figures", "figures not in the context: " + ", ".join(uf))
        uc = unsupported_citations(a, context)
        if uc:
            add("generation", "unsupported_citations", "cites section numbers not in the context: " + ", ".join(uc))
    kinds = {i["kind"] for i in issues}
    v["critical"] = sorted(kinds & CRITICAL)
    if item["answerable"]:
        v["fully_correct"] = not error and not (kinds & (CRITICAL | INCOMPLETE))
    else:
        v["abstained"] = declines_at_start(a)
        v["fully_correct"] = not error and v["abstained"] and not (kinds & CRITICAL)
    v["issues"] = issues
    v["first_stage"] = None if v["fully_correct"] else next((s for s in STAGES if any(i["stage"] == s for i in issues)), None)
    return v


# ---------------------------------------------------------------------------------------------------- retrieval metrics
def retrieval_metrics(item: dict, cand_ids, ranked_ids, ctx_ids, text: CorpusText, block_tokens) -> dict:
    """From the evidence passages: search recall (evidence among the candidates), context recall (evidence in the
    context), MRR@10 (first relevant chunk among the top 10 reranked), context precision (share of context tokens
    in relevant chunks)."""
    ev = item["evidence"]
    if not item["answerable"] or not ev:
        return {}
    chunks = text.corpus.chunks
    relevant = lambda i: any(anchor_in(e, text.chunk_words[i]) for e in ev)
    cand_w, ctx_w = text.words_of(cand_ids), text.words_of(ctx_ids)
    rr = next((1.0 / r for r, i in enumerate(ranked_ids[:10], start=1) if relevant(i)), 0.0)
    tok = sum(block_tokens(i) for i in ctx_ids)
    rel_tok = sum(block_tokens(i) for i in ctx_ids if relevant(i))
    ctx_norm = norm_text(" ".join(chunks[i].label + " " + chunks[i].text for i in ctx_ids))
    return {"evidence_in_chunk_file": float(np.mean([anchor_in(e, text.all_words) for e in ev])),
            "search_recall": float(np.mean([anchor_in(e, cand_w) for e in ev])),
            "context_recall": float(np.mean([anchor_in(e, ctx_w) for e in ev])),
            "mrr10": rr, "context_precision": rel_tok / tok if tok else 0.0,
            "facts_in_context": float(np.mean([fact_found(f, ctx_norm) for f in item["facts"]])) if item["facts"] else None}
