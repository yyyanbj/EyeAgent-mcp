"""Lightweight logging setup for core framework.

Users can override log level with EYETOOLS_LOG_LEVEL env var.
"""
from __future__ import annotations
import logging
import os

LOG_LEVEL = os.getenv("EYETOOLS_LOG_LEVEL", "INFO").upper()


def get_logger(name: str = "eyetools") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.setLevel(LOG_LEVEL)
        logger.propagate = False
    return logger

core_logger = get_logger("eyetools.core")

__all__ = ["get_logger", "core_logger"]
