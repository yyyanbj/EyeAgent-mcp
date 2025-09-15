"""Lightweight logging setup for core framework.

Users can override log level with EYETOOLS_LOG_LEVEL env var.
"""
from __future__ import annotations
import logging
import os
from pathlib import Path

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

__all__ = ["get_logger", "core_logger"]
