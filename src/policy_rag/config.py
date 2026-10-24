"""Configuration: one YAML file per system variant, read into typed dataclasses.

A config file may start with ``extends: other.yaml`` and then list only what differs. Values can also be
overridden from the command line with dotted keys (``--set generator.think_budget=1536``). Unknown keys are
errors, so a misspelled setting cannot be silently ignored in an experiment.
"""
from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_args, get_origin, get_type_hints

import yaml


@dataclass
class DocumentConfig:
    id: str                              # short, unique, stable: it is stored with every chunk and every result
    chunks: str                          # path to the chunk file (.json or .csv)
    name: str = ""                       # shown to the answer model when more than one document is loaded


@dataclass
class EmbedderConfig:
    type: str = "sentence_transformers"
    model: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    batch_size: int = 32
    device: Optional[str] = None         # None = CUDA if available, else CPU


@dataclass
class RerankerConfig:
    type: str = "cross_encoder"
    model: str = "BAAI/bge-reranker-v2-m3"
    max_length: int = 1024
    batch_size: int = 8
    use_label: bool = True               # score "label: text" (labels carry section and clause names)
    device: Optional[str] = None


@dataclass
class RoutedIndexConfig:
    """Chunks whose ``index`` field has this name form their own search index, searched only when the question
    matches ``query_pattern`` (and not ``exclude_pattern``). Used for the premium tables: hundreds of near-identical
    rows would otherwise crowd the policy text out of the candidate lists."""
    k: int = 12                          # candidates from each search leg of this index
    query_pattern: str = ""
    exclude_pattern: str = ""


_PREMIUM_QUERY = (r"\bhow much\b[^?]*\b(cost|premium|pay for (the|this) (policy|plan|cover))|\b(annual|yearly) premium"
                  r"|\bpremium (for|of|amount|rates?|payable for)\b|\bwhat (is|are|would be) the premium"
                  r"|\b(cost|price)s? (of|for)\b|\bcost (more|less)\b|\b(cheaper|expensive)\b")
_PREMIUM_EXCLUDE = r"instal+ment|grace period|refund|deduction|income tax|discount|loading"


@dataclass
class RetrievalConfig:
    k_dense: int = 32
    k_bm25: int = 32
    use_bm25: bool = True
    rrf_k: int = 60                      # reciprocal rank fusion only orders the union; nothing is cut
    routing: bool = True                 # False = routed-index chunks are searched like all others
    routed_indexes: Dict[str, RoutedIndexConfig] = field(default_factory=lambda: {
        "premium": RoutedIndexConfig(k=12, query_pattern=_PREMIUM_QUERY, exclude_pattern=_PREMIUM_EXCLUDE)})


@dataclass
class ContextConfig:
    token_budget: int = 1500             # hard cap, counted with the answer model's tokenizer, labels included
    title_pinning: bool = True           # sections whose heading words all appear in the question go first
    max_pinned_sections: int = 2
    neighbour_expansion: bool = True     # add the chunk a selected chunk continues into or depends on
    seed_score_floor: float = -5.0       # reranked chunks below this do not fill the context


@dataclass
class GateConfig:
    min_score: float = -5.0              # refuse when the best reranker score is below this


@dataclass
class PromptConfig:
    template: str = "policy_analyst_v2_3"


@dataclass
class GeneratorConfig:
    type: str = "hf_causal_lm"
    model: str = "Qwen/Qwen3-8B"
    load_in_4bit: bool = True            # NF4 with fp16 compute: about 6.5 GB on a T4
    enable_thinking: bool = True
    think_budget: int = 1024             # thinking tokens; then the model is told to stop thinking and answer
    max_new_tokens: int = 1000           # answer tokens, after the thinking
    sampling: bool = True                # Qwen's recommended sampling, seeded from the prompt; False = greedy
    oom_retry_context_tokens: int = 1000  # on CUDA out-of-memory, retry once with a context of this size
    device: Optional[str] = None


@dataclass
class CacheConfig:
    enabled: bool = True
    path: str = "runs/answer_cache.jsonl"  # shared by all runs: keys include the prompt and generation settings


@dataclass
class EvaluationConfig:
    testset: str = "testsets/optima_secure_v3.json"
    splits: List[str] = field(default_factory=lambda: ["dev", "test"])
    output_dir: str = "runs"
    groundedness: bool = True
    grounding_model: str = "vectara/hallucination_evaluation_model"
    hhem_threshold: float = 0.5          # a sentence is supported when HHEM scores it at or above this
    evidence_removed_test: bool = True   # up to 4 extra answers with the evidence taken out of the context
    repeat_check: bool = True            # 1 answer regenerated without the cache (determinism)


@dataclass
class DecisionConfig:
    """Fixed before results are seen: when does a candidate replace the baseline?"""
    alpha: float = 0.05                  # McNemar exact test on "fully correct", held-out test split
    n_bootstrap: int = 10_000
    max_median_seconds: float = 150.0    # per generated answer
    max_peak_gpu_gb: float = 14.0
    kappa_target: float = 0.7            # agreement of the automatic verdict with human labels


@dataclass
class Config:
    documents: List[DocumentConfig] = field(default_factory=list)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    plugins: List[str] = field(default_factory=list)   # modules imported first (they can register new component types)
    source: Optional[str] = None         # the file this config was read from (set by load_config)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        import hashlib
        d = self.to_dict()
        d.pop("source", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------------------------------- loading
class ConfigError(ValueError):
    pass


def _build(cls, data: Any, where: str):
    """Dataclass ``cls`` from a dict, recursively; unknown keys are errors."""
    if not dataclasses.is_dataclass(cls):
        return data
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{where}: unknown setting(s) {sorted(unknown)}; allowed: {sorted(names)}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        kwargs[f.name] = _convert(hints[f.name], data[f.name], f"{where}.{f.name}")
    return cls(**kwargs)


def _convert(tp, value, where):
    origin, args = get_origin(tp), get_args(tp)
    if origin is Union:                                   # Optional[X]
        if value is None:
            return None
        tp = next(a for a in args if a is not type(None))
        origin, args = get_origin(tp), get_args(tp)
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, where)
    if origin in (list, List):
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list")
        return [_convert(args[0], v, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin in (dict, Dict):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping")
        return {k: _convert(args[1], v, f"{where}.{k}") for k, v in value.items()}
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if tp in (int, float, str, bool) and not isinstance(value, tp):
        raise ConfigError(f"{where}: expected {tp.__name__}, got {value!r}")
    return value


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "routed_indexes":
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path, seen=()) -> dict:
    path = Path(path)
    if path.resolve() in seen:
        raise ConfigError(f"{path}: circular 'extends'")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    if parent:
        base = _read_yaml((path.parent / parent), seen + (path.resolve(),))
        data = _merge(base, data)
    return data


def parse_override(text: str):
    """'generator.think_budget=1536' -> (['generator', 'think_budget'], 1536). The value is read as YAML."""
    if "=" not in text:
        raise ConfigError(f"override {text!r}: expected key=value")
    key, value = text.split("=", 1)
    return key.strip().split("."), yaml.safe_load(value)


def apply_overrides(data: dict, overrides) -> dict:
    data = copy.deepcopy(data)
    for item in overrides or []:
        keys, value = parse_override(item) if isinstance(item, str) else item
        cur = data
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = value
    return data


def load_config(path: Union[str, Path, None] = None, overrides=None, data: Optional[dict] = None) -> Config:
    """Read a YAML config (following ``extends``), apply ``key=value`` overrides and validate everything."""
    raw = dataclasses.asdict(Config())                    # defaults first, so overrides can reach every setting
    raw = _merge(raw, _read_yaml(Path(path)) if path else {})
    raw = _merge(raw, data or {})
    raw = apply_overrides(raw, overrides)
    cfg = _build(Config, raw, "config")
    cfg.source = str(path) if path else None
    import importlib
    for module in cfg.plugins:
        importlib.import_module(module)
    ids = [d.id for d in cfg.documents]
    if len(ids) != len(set(ids)):
        raise ConfigError(f"documents: ids must be unique, got {ids}")
    return cfg


def resolve_input(path: Union[str, Path], cfg: Optional[Config] = None) -> Path:
    """An input path from a config: as given (relative to the working directory) if it exists, else relative to the
    config file's folder or one of its parents (so ``configs/experiments/x.yaml`` finds ``data/...`` at the root)."""
    p = Path(path)
    if p.is_absolute() or p.exists() or cfg is None or not cfg.source:
        return p
    here = Path(cfg.source).resolve().parent
    for folder in [here, *here.parents][:4]:
        if (folder / p).exists():
            return folder / p
    return p
