"""Answer generation: the answer model, its tokenizer (for the context budget) and the answer cache.

Interface of a generator (what the pipeline calls):

- ``signature() -> dict``: everything that changes the output for a given prompt (model, precision, decoding,
  budgets). It is part of the cache key, so changing a setting never reuses an old answer.
- ``generate(prompt, seed) -> dict`` with ``answer``, ``thinking``, ``prompt_tokens``, ``think_tokens``,
  ``answer_tokens``, ``thinking_forced``, ``hit_limit``.
- ``count_tokens(text) -> int`` (the context budget is counted in the answer model's tokens).

The model weights load on the first ``generate`` call, so retrieval-only runs and fully cached runs never load them.
"""
from __future__ import annotations

import datetime
import hashlib
import io
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Dict, Optional

from .device import dtype_kw, free_gpu, model_revision, resolve_device
from .registry import register

# Qwen3's own wording for stopping the thinking at a budget (Qwen3 technical report, "thinking budget")
FORCE_ANSWER = ("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking "
                "directly now.\n</think>\n\n")


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>|</?think>", "", text, flags=re.S).strip()


@register("generator", "hf_causal_lm")
class HFCausalLMGenerator:
    """A Hugging Face causal LM with a chat template (tested with Qwen3-8B).

    With thinking on, generation runs in two phases: (1) the model thinks until ``</think>`` or ``think_budget``
    tokens; if the budget runs out it is told to stop thinking and answer (Qwen3's budget forcing); (2) the answer,
    with its own ``max_new_tokens``. The thinking is returned separately and never mixed into the answer.

    Decoding: Qwen advises against greedy decoding in thinking mode (it loops). Its recommended settings are
    T=0.6, top_p=0.95, top_k=20 (thinking) and T=0.7, top_p=0.8, top_k=20 (no thinking). The seed comes from the
    prompt, so the same prompt gives the same answer on the same hardware and software.
    """

    def __init__(self, model: str, load_in_4bit: bool = True, enable_thinking: bool = True, think_budget: int = 1024,
                 max_new_tokens: int = 1000, sampling: bool = True, device=None):
        from transformers import AutoTokenizer
        self.name, self.load_in_4bit, self.enable_thinking = model, load_in_4bit, enable_thinking
        self.think_budget, self.max_new_tokens, self.sampling = think_budget, max_new_tokens, sampling
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = None
        self.offloaded_modules = []
        if not sampling:
            self.decoding = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
        elif enable_thinking:
            self.decoding = {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
        else:
            self.decoding = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
        self.precision = ("4-bit NF4" if load_in_4bit and self.device == "cuda" else
                          "fp16" if self.device == "cuda" else "fp32 (CPU)")
        vocab = self.tokenizer.get_vocab()
        self.think_end = self.tokenizer.convert_tokens_to_ids("</think>") if "</think>" in vocab else None

    @classmethod
    def from_config(cls, cfg, **_):
        return cls(cfg.model, cfg.load_in_4bit, cfg.enable_thinking, cfg.think_budget, cfg.max_new_tokens,
                   cfg.sampling, cfg.device)

    # ------------------------------------------------------------------ identity
    def signature(self) -> dict:
        """Cache-key settings. The model revision is recorded with results but kept out of the key, so that keys do
        not depend on whether the Hub answered a revision lookup (and answers cached by the notebook stay valid)."""
        sig = {"llm": self.name, "revision": None, "precision": self.precision, "thinking": self.enable_thinking,
               "max_new_tokens": self.max_new_tokens,
               "decoding": {k: v for k, v in self.decoding.items() if v is not None}}
        if self.enable_thinking:
            sig.update(think_budget=self.think_budget, phases="think-then-answer v1")
        return sig

    def revision(self):
        return model_revision(self.tokenizer, self.name)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    # ------------------------------------------------------------------ loading
    def load(self):
        if self.model is not None:
            return self
        import torch
        from transformers import AutoModelForCausalLM
        kw = {"device_map": "auto"} if self.device == "cuda" else {}
        if self.load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_use_double_quant=True,
                                                           bnb_4bit_compute_dtype=torch.float16)   # T4: no bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name, **dtype_kw(torch.float16 if self.device == "cuda" else torch.float32), **kw).eval()
        self.offloaded_modules = [k for k, v in (getattr(self.model, "hf_device_map", None) or {}).items()
                                  if v in ("cpu", "disk")]
        eos = self.model.generation_config.eos_token_id
        self._eos = [eos] if isinstance(eos, int) else list(eos or [self.tokenizer.eos_token_id])
        return self

    def unload(self):
        self.model = None
        free_gpu()

    # ------------------------------------------------------------------ generation
    def chat_text(self, prompt: str, thinking: Optional[bool] = None) -> str:
        thinking = self.enable_thinking if thinking is None else thinking
        return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=thinking)

    def _inputs(self, prompt: str, thinking=None):
        text = self.chat_text(prompt, thinking)
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        bos = self.tokenizer.bos_token_id
        if bos is not None and self.tokenizer.bos_token and not text.startswith(self.tokenizer.bos_token):
            enc = self.tokenizer(text, return_tensors="pt")          # template without BOS: add exactly one
        return enc.to(self.model.device)

    def _generate(self, ids, max_new_tokens: int, eos, seed: int, min_new_tokens: int = 0):
        import torch
        torch.manual_seed(seed)
        with torch.no_grad():
            out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                      min_new_tokens=min_new_tokens or None, eos_token_id=eos,
                                      pad_token_id=self.tokenizer.pad_token_id, **self.decoding)
        return out[0, ids.shape[1]:]

    def generate_plain(self, prompt: str, max_new_tokens: int, min_new_tokens: int = 0, thinking=None, seed: int = 0):
        """One generation without the thinking phases (smoke and speed tests): (text, prompt tokens, new tokens)."""
        self.load()
        enc = self._inputs(prompt, thinking)
        new = self._generate(enc["input_ids"], max_new_tokens, self._eos, seed, min_new_tokens)
        return strip_thinking(self.tokenizer.decode(new, skip_special_tokens=True)), int(enc["input_ids"].shape[1]), int(len(new))

    def generate(self, prompt: str, seed: int) -> dict:
        import torch
        self.load()
        tok = self.tokenizer
        enc = self._inputs(prompt)
        ids, n_in = enc["input_ids"], int(enc["input_ids"].shape[1])
        thinking, n_think, forced = "", 0, False
        if self.enable_thinking and self.think_end is not None:
            new1 = self._generate(ids, self.think_budget, self._eos + [self.think_end], seed)
            ends = (new1 == self.think_end).nonzero()
            if len(ends):                                         # thinking finished by itself
                cut = int(ends[0]) + 1
                prefix = torch.cat([ids[0], new1[:cut]])
            elif len(new1) and int(new1[-1]) in self._eos:        # answered without closing the thinking block
                text = tok.decode(new1, skip_special_tokens=True)
                return {"answer": strip_thinking(text), "thinking": "", "prompt_tokens": n_in, "think_tokens": 0,
                        "answer_tokens": len(new1), "thinking_forced": False, "hit_limit": False}
            else:                                                 # budget used up: stop thinking, answer now
                cut, forced = len(new1), True
                force = tok(FORCE_ANSWER, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(ids.device)
                prefix = torch.cat([ids[0], new1, force])
            thinking = strip_thinking(tok.decode(new1[:cut], skip_special_tokens=True))
            n_think, ids = cut, prefix.unsqueeze(0)
        new2 = self._generate(ids, self.max_new_tokens, self._eos, seed + 1)
        return {"answer": strip_thinking(tok.decode(new2, skip_special_tokens=True)), "thinking": thinking,
                "prompt_tokens": n_in, "think_tokens": n_think, "answer_tokens": int(len(new2)),
                "thinking_forced": forced, "hit_limit": int(len(new2)) >= self.max_new_tokens}

    def self_check(self) -> list:
        """Load the model and run two quick checks; returns problems (empty list = fine)."""
        self.load()
        problems = []
        if self.offloaded_modules:
            problems.append(f"{len(self.offloaded_modules)} answer-model modules are on the CPU/disk (GPU full): "
                            "answers will be very slow")
        if self.enable_thinking and self.think_end is None:
            problems.append("this model's tokenizer has no </think> token: the thinking budget is not applied")
        if "qwen3" in self.name.lower() and (("<think>\n\n</think>" not in self.chat_text("x")) != self.enable_thinking):
            problems.append("the chat template does not switch thinking as configured")
        text, _, _ = self.generate_plain("Reply with one word: the capital of France.", 8, thinking=False)
        if not re.search(r"[A-Za-z]{3,}", text) or re.fullmatch(r"(.)\1+", text.replace(" ", "")):
            problems.append(f"the answer model's test output looks broken ({text!r}); with fp16 this can be a numeric overflow")
        return problems

    def tokens_per_second(self, prompt: str) -> float:
        t0 = time.perf_counter()
        _, _, n = self.generate_plain(prompt, 32, min_new_tokens=32, thinking=False)
        return round(n / max(time.perf_counter() - t0, 1e-6), 2)


# ---------------------------------------------------------------------------------------------------- answer cache
class AnswerCache:
    """Answers keyed by the exact prompt and the generator signature, one JSON line each (append-only)."""

    def __init__(self, path: Optional[str], enabled: bool = True):
        self.path = Path(path) if path else None
        self.enabled = enabled and self.path is not None
        self.records: Dict[str, dict] = {}
        if self.enabled and self.path.exists():
            self._read_lines(self.path.read_text(encoding="utf-8").splitlines())

    def _read_lines(self, lines) -> int:
        n = 0
        for line in lines:
            try:
                r = json.loads(line)
                self.records[r["key"]] = r
                n += 1
            except Exception:
                pass
        return n

    def add_from(self, source: str) -> int:
        """Merge answers from another cache file (.jsonl) or a results zip; returns how many were read."""
        p = Path(source)
        raw = p.read_bytes()
        if p.suffix.lower() == ".zip":
            z = zipfile.ZipFile(io.BytesIO(raw))
            names = [n for n in z.namelist() if n.endswith("answer_cache.jsonl")]
            raw = z.read(names[0]) if names else b""
        lines = [l for l in raw.decode("utf-8").splitlines() if l.strip()]
        n = self._read_lines(lines)
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write("".join(l + "\n" for l in lines))
        return n

    def get(self, key: str) -> Optional[dict]:
        return self.records.get(key) if self.enabled else None

    def put(self, rec: dict):
        if not self.enabled:
            return
        self.records[rec["key"]] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def __len__(self):
        return len(self.records)


class Answerer:
    """Generator + cache. The seed of each answer comes from its cache key, i.e. from the prompt and settings."""

    def __init__(self, generator, cache: Optional[AnswerCache] = None, verbose: bool = True):
        self.generator, self.verbose = generator, verbose
        self.cache = cache if cache is not None else AnswerCache(None, enabled=False)   # (an empty cache is falsy)
        self._sig = json.dumps(generator.signature(), sort_keys=True)

    def key(self, prompt: str) -> str:
        return hashlib.sha256((self._sig + "\n" + prompt).encode()).hexdigest()[:24]

    def is_cached(self, prompt: str) -> bool:
        return self.cache.get(self.key(prompt)) is not None

    def answer(self, prompt: str, use_cache: bool = True) -> dict:
        """The generation record: key, answer, thinking, token counts, seconds, from_cache."""
        key = self.key(prompt)
        hit = self.cache.get(key) if use_cache else None
        if hit is not None:
            return dict(hit, from_cache=True)
        t0 = time.perf_counter()
        out = self.generator.generate(prompt, seed=int(key[:8], 16))
        rec = {"key": key, **out, "new_tokens": (out.get("think_tokens") or 0) + (out.get("answer_tokens") or 0),
               "seconds": round(time.perf_counter() - t0, 2),
               "created": datetime.datetime.now().isoformat(timespec="seconds")}
        if use_cache:
            self.cache.put(rec)
        if self.verbose:
            print(f"    answer: {rec['answer_tokens']} tokens" + (
                f" after {rec['think_tokens']} thinking tokens"
                + (" (thinking budget reached: told to answer)" if rec["thinking_forced"] else "") if rec["think_tokens"] else "")
                + f" in {rec['seconds']:.0f}s" + (" (answer stopped at the length limit)" if rec["hit_limit"] else ""))
        return dict(rec, from_cache=False)
