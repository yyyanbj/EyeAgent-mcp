"""Lightweight logging setup for core framework.

Users can override log level with EYETOOLS_LOG_LEVEL env var.

Also includes helpers to safely summarize potentially large or complex
objects (model outputs, arrays, tensors, images) for logging without
dumping full payloads to the logs.
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Any, Dict

def _is_numpy(obj: Any) -> bool:
    try:  # pragma: no cover - optional dependency
        import numpy as np  # type: ignore
        return isinstance(obj, np.ndarray)
    except Exception:
        return False

def _is_torch_tensor(obj: Any) -> bool:
    try:  # pragma: no cover - optional dependency
        import torch  # type: ignore
        return isinstance(obj, torch.Tensor)
    except Exception:
        return False

def _summarize_sequence(seq: Any, max_items: int, level: int, max_level: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "type": type(seq).__name__,
        "len": len(seq) if hasattr(seq, "__len__") else None,
    }
    if level >= max_level:
        return out
    try:
        items = list(seq)[:max_items]
    except Exception:
        items = []
    out["preview_types"] = [type(x).__name__ for x in items]
    # shallow preview values (stringified, truncated)
    prev_vals = []
    for x in items:
        try:
            s = str(x)
            if len(s) > 120:
                s = s[:117] + "..."
            prev_vals.append(s)
        except Exception:
            prev_vals.append(f"<{type(x).__name__}>")
    out["preview"] = prev_vals
    return out

def summarize_for_log(obj: Any, *, max_items: int = 8, max_level: int = 2) -> Any:
    """Return a compact, JSON-serializable summary suitable for logging.

    - Dict: show size, keys (truncated), and value types (not full values)
    - List/Tuple/Set: show length and a short preview of types/values
    - NumPy ndarray: shape, dtype
    - torch.Tensor: shape, dtype, device, requires_grad
    - str: length and truncated preview
    - bytes/bytearray: length
    - Other scalars: returned directly when small; otherwise type name
    """
    try:
        # Primitives
        if obj is None or isinstance(obj, (bool, int, float)):
            return obj
        if isinstance(obj, str):
            return {"type": "str", "len": len(obj), "preview": (obj if len(obj) <= 200 else obj[:197] + "...")}
        if isinstance(obj, (bytes, bytearray)):
            return {"type": type(obj).__name__, "len": len(obj)}

        # NumPy
        if _is_numpy(obj):  # type: ignore
            try:  # pragma: no cover
                import numpy as np  # type: ignore
                arr = obj  # type: ignore
                return {
                    "type": "ndarray",
                    "shape": tuple(getattr(arr, "shape", ())),
                    "dtype": str(getattr(arr, "dtype", "unknown")),
                }
            except Exception:
                return {"type": "ndarray"}

        # Torch
        if _is_torch_tensor(obj):  # type: ignore
            t = obj  # type: ignore
            def _safe_get(attr, default=None):
                try:
                    return getattr(t, attr)
                except Exception:
                    return default
            return {
                "type": "torch.Tensor",
                "shape": tuple(_safe_get("shape", ()) or ()),
                "dtype": str(_safe_get("dtype", "unknown")),
                "device": str(_safe_get("device", "cpu")),
                "requires_grad": bool(_safe_get("requires_grad", False)),
            }

        # Mappings
        if isinstance(obj, dict):
            out: Dict[str, Any] = {"type": "dict", "len": len(obj)}
            keys = list(obj.keys())[:max_items]
            out["keys"] = [str(k) for k in keys]
            if max_level > 0:
                out["value_types"] = {str(k): type(obj[k]).__name__ for k in keys}
            return out

        # Sequences
        if isinstance(obj, (list, tuple, set)):
            return _summarize_sequence(obj, max_items=max_items, level=0, max_level=max_level)

        # PIL Image
        try:  # pragma: no cover - optional
            from PIL import Image  # type: ignore
            if isinstance(obj, Image.Image):
                return {"type": "PIL.Image", "size": getattr(obj, "size", None), "mode": getattr(obj, "mode", None)}
        except Exception:
            pass

        # Fallback: just return type name
        return {"type": type(obj).__name__}
    except Exception:
        return {"type": "unprintable"}

LOG_LEVEL = os.getenv("EYETOOLS_LOG_LEVEL", "DEBUG").upper()


def get_logger(name: str = "eyetools") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(stream_handler)
        # Optional file handler if EYETOOLS_LOG_DIR is set
        log_dir = os.getenv("EYETOOLS_LOG_DIR")
        if log_dir:
            try:
                p = Path(log_dir)
                p.mkdir(parents=True, exist_ok=True)
                file_path = p / "eyetools.log"
                fh = logging.FileHandler(file_path, encoding="utf-8")
                fh.setFormatter(logging.Formatter(fmt))
                logger.addHandler(fh)
            except Exception:  # noqa: BLE001
                pass
        logger.setLevel(LOG_LEVEL)
        logger.propagate = False
    return logger

core_logger = get_logger("eyetools.core")

__all__ = ["get_logger", "core_logger", "summarize_for_log"]
