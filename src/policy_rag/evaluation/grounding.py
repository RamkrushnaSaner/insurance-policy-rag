"""Groundedness: is each answer sentence supported by the context the model was given? (HHEM-2.1-Open)

Runs after all answers exist, in fp32 (T5-based; fp16 overflows), so it costs no answer-model calls. Sentences that
talk about the context ("the policy text does not mention ...") are not claims and are skipped. Known weakness: HHEM
flags correct paraphrases and summaries, so the hallucination rate it reports is pessimistic; read the flagged
sentences (``weakest`` in results.csv) before acting on it.
"""
from __future__ import annotations

import re
import time

import numpy as np

from ..device import dtype_kw, free_gpu, is_oom, model_revision, resolve_device
from ..prompts import is_no_answer

_META = re.compile(r"\b(context|policy text|provided text|provided policy|the text|information provided|document|excerpt"
                   r"|provided sections?|given sections?)s?\b[^.]{0,80}\b(not|no)\b"
                   r"|\b(?:context|provided sections?|given sections?|policy sections)\b[^.]{0,40}\b(?:outlines?|includes?"
                   r"|discuss(?:es)?|address(?:es)?|focus(?:es)?|describes?)\b"
                   r"|\bnot (specified|mentioned|stated|provided|shown|given|listed|included) in\b|^not specified in the provided"
                   r" policy text|\bthere is no (?:mention|reference|provision|information)\b|\bno (?:mention|reference) of\b"
                   r"|\b(?:is|are) not (?:explicitly )?(?:mentioned|specified|stated|defined)\b|\bcannot be determined\b"
                   r"|\bdoes not (?:explicitly )?(?:mention|specify|state|address|describe|include)\b", re.I)
_ABBREV = re.compile(r"\b(Rs|No|Nos|Sr|vs|approx|e\.g|i\.e|etc|Dr|Ltd|Co|Excl|Sec)\.\s+", re.I)   # not sentence ends


def claim_units(answer: str):
    """The answer's claim sentences (list markers removed; "Rs. 15,000" stays one sentence)."""
    units = []
    for line in (answer or "").splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)]|[a-zA-Z][.)]|#+)\s*", "", line).replace("**", "").strip()
        line = _ABBREV.sub(lambda m: m.group(1) + "․ ", line)
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(₹\d])", line):
            s = s.replace("․", ".").strip()
            if len(re.findall(r"[A-Za-z]+", s)) < 4 or s.endswith(":") or _META.search(s) or "|" in s[:1]:
                continue
            units.append(s)
    return units


class HHEMGrounder:
    def __init__(self, model: str = "vectara/hallucination_evaluation_model", threshold: float = 0.5, device=None,
                 batch_size: int = 4):
        self.name, self.threshold, self.batch_size = model, threshold, batch_size
        self.device = resolve_device(device)
        self.model = None

    def load(self):
        if self.model is None:
            import torch
            from transformers import AutoModelForSequenceClassification
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.name, trust_remote_code=True, **dtype_kw(torch.float32)).to(self.device).eval()
        return self

    def revision(self):
        return model_revision(self.model, self.name)

    def scores(self, pairs):
        import torch
        self.load()
        out = []
        for i in range(0, len(pairs), self.batch_size):
            part = pairs[i:i + self.batch_size]
            try:
                with torch.no_grad():
                    out += [float(x) for x in self.model.predict(part)]
            except Exception as e:                     # long context: one sentence at a time
                if not is_oom(e):
                    raise
                free_gpu()
                for pr in part:
                    with torch.no_grad():
                        out.append(float(self.model.predict([pr])[0]))
        return out

    def ground_record(self, rec: dict) -> dict:
        """Adds rec["grounding"]: share of supported claim sentences, and the weakest sentence."""
        a = rec.get("answer")
        if not a or is_no_answer(a) or not rec.get("context"):
            rec["grounding"] = None
            return rec
        units = claim_units(a)
        if not units:
            rec["grounding"] = {"units": 0, "supported": None, "groundedness": None, "hallucinated": False}
            return rec
        sc = self.scores([(rec["context"], u) for u in units])
        thr, worst = self.threshold, int(np.argmin(sc))
        rec["grounding"] = {"units": len(units), "supported": sum(s >= thr for s in sc),
                            "groundedness": sum(s >= thr for s in sc) / len(units), "hallucinated": any(s < thr for s in sc),
                            "weakest": units[worst][:200], "weakest_score": round(sc[worst], 3),
                            "unsupported": [u[:160] for u, s in zip(units, sc) if s < thr][:5]}
        return rec

    def ground_run(self, run: dict, log=print, label: str = "") -> int:
        t0, n = time.perf_counter(), 0
        for rec in run["records"]:
            a = rec.get("answer")
            if a and not is_no_answer(a) and rec.get("context") and rec.get("grounding") is None:
                self.ground_record(rec)
                n += 1
        log(f"Groundedness{label}: {n} answers checked in {time.perf_counter() - t0:.0f}s")
        return n
