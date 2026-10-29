import pytest

from policy_rag.config import ConfigError, load_config

from conftest import ROOT


def test_default_config_is_the_baseline():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert cfg.generator.model == "Qwen/Qwen3-8B" and cfg.generator.enable_thinking
    assert cfg.reranker.model == "BAAI/bge-reranker-v2-m3"
    assert (cfg.context.token_budget, cfg.gate.min_score, cfg.context.seed_score_floor) == (1500, -5.0, -5.0)
    assert set(cfg.retrieval.routed_indexes) == {"premium"} and cfg.retrieval.routed_indexes["premium"].k == 12
    assert [d.id for d in cfg.documents] == ["optima-secure"]
    assert cfg.memory.enabled is False                        # not adopted (docs/evaluation.md)


def test_yaml_values_equal_the_code_defaults():
    """configs/default.yaml writes every setting out; it must say the same as the dataclass defaults."""
    from policy_rag.config import Config
    cfg, defaults = load_config(ROOT / "configs" / "default.yaml").to_dict(), Config().to_dict()
    for section in ("embedder", "reranker", "retrieval", "context", "gate", "prompt", "generator", "memory", "cache",
                    "evaluation", "decision"):
        assert cfg[section] == defaults[section], section


def test_extends_and_overrides():
    cfg = load_config(ROOT / "configs" / "experiments" / "no_thinking.yaml", ["generator.think_budget=2048"])
    assert cfg.generator.enable_thinking is False and cfg.generator.think_budget == 2048
    assert cfg.reranker.model == "BAAI/bge-reranker-v2-m3"            # inherited


def test_override_inside_a_routed_index_keeps_its_pattern():
    cfg = load_config(ROOT / "configs" / "default.yaml", ["retrieval.routed_indexes.premium.k=20"])
    assert cfg.retrieval.routed_indexes["premium"].k == 20
    assert "premium" in cfg.retrieval.routed_indexes["premium"].query_pattern


def test_unknown_or_mistyped_settings_are_errors():
    with pytest.raises(ConfigError, match="unknown setting"):
        load_config(ROOT / "configs" / "default.yaml", ["generator.think_budjet=10"])
    with pytest.raises(ConfigError, match="expected int"):
        load_config(ROOT / "configs" / "default.yaml", ["context.token_budget=lots"])


def test_duplicate_document_ids_are_rejected():
    with pytest.raises(ConfigError, match="unique"):
        load_config(data={"documents": [{"id": "a", "chunks": "x.json"}, {"id": "a", "chunks": "y.json"}]})
