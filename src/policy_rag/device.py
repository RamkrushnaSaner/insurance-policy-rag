"""Small helpers around PyTorch devices and memory. PyTorch is imported only when a helper is called, so the
retrieval, context and evaluation logic can be imported (and tested) without it."""
from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Optional


def torch_or_none():
    try:
        import torch
        return torch
    except ImportError:
        return None


def resolve_device(preferred: Optional[str] = None) -> str:
    if preferred:
        return preferred
    torch = torch_or_none()
    return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


def dtype_kw(dtype) -> dict:
    """``from_pretrained`` takes ``dtype=`` from transformers 4.56 on (``torch_dtype=`` before)."""
    import transformers
    major_minor = tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2])
    return {"dtype": dtype} if major_minor >= (4, 56) else {"torch_dtype": dtype}


def free_gpu():
    gc.collect()
    torch = torch_or_none()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def gpu_gb() -> Optional[float]:
    """GPU memory in use (all processes), GB."""
    torch = torch_or_none()
    if torch is None or not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return round((total - free) / 1e9, 2)


def reset_peak_memory():
    torch = torch_or_none()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_gpu_gb() -> Optional[float]:
    torch = torch_or_none()
    if torch is None or not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1e9, 2)


def gpu_name() -> Optional[str]:
    torch = torch_or_none()
    return torch.cuda.get_device_name(0) if torch is not None and torch.cuda.is_available() else None


def is_oom(exc: BaseException) -> bool:
    torch = torch_or_none()
    return torch is not None and isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ()))


def model_revision(model=None, name: Optional[str] = None) -> Optional[str]:
    """The model's commit on the Hugging Face Hub, so a result can be traced to exact weights (None if unknown)."""
    for cfg in (getattr(model, "config", None), getattr(getattr(model, "auto_model", None), "config", None)):
        h = getattr(cfg, "_commit_hash", None)
        if h:
            return h
    if name and not Path(name).exists():
        try:
            from huggingface_hub import HfApi
            return HfApi().model_info(name).sha
        except Exception:
            return None
    return None
