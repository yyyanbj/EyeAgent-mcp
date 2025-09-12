"""Misc helper utilities (hashing, paths, logging setup placeholder)."""
from __future__ import annotations
import hashlib
from typing import Iterable


def stable_hash(parts: Iterable[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
    return h.hexdigest()[:16]

__all__ = ["stable_hash"]
