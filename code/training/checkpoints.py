"""
This module provides utilities for loading and saving model checkpoints.
"""

from dataclasses import asdict, is_dataclass
from typing import Any

from training.grassman import (
    GrassmannConfig,
    GrassmannLM,
    TransformerConfig,
    TransformerLM,
)


def serialize_config(cfg: Any) -> dict[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(vars(cfg))
    raise TypeError(f"Unsupported config type: {type(cfg)!r}")


def infer_model_type(checkpoint: dict[str, Any]) -> str:
    model_type = checkpoint.get("model_type")
    if model_type is None:
        return "grassmann" if any("mix." in key for key in checkpoint["model"]) else "transformer"
    if model_type not in {"grassmann", "transformer"}:
        raise ValueError(f"Unsupported model_type {model_type!r}")
    return model_type


def build_config(model_type: str, raw_config: Any | None):
    config_cls = GrassmannConfig if model_type == "grassmann" else TransformerConfig
    if raw_config is None:
        return config_cls()
    if isinstance(raw_config, config_cls):
        return raw_config
    config_dict = serialize_config(raw_config)
    return config_cls(**config_dict)


def build_model(model_type: str, cfg):
    if model_type == "grassmann":
        return GrassmannLM(cfg)
    if model_type == "transformer":
        return TransformerLM(cfg)
    raise ValueError(f"Unsupported model_type {model_type!r}")


def load_model_from_checkpoint(checkpoint: dict[str, Any]) -> tuple[Any, str]:
    model_type = infer_model_type(checkpoint)
    cfg = build_config(model_type, checkpoint.get("config"))
    model = build_model(model_type, cfg)
    model.load_state_dict(checkpoint["model"])
    return model, model_type
