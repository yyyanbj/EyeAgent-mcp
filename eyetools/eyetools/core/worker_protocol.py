"""Worker protocol skeleton.

A real worker would:
1. Start, read INIT message -> load module/class, instantiate tool.
2. Optionally LOAD_MODEL.
3. Handle PREDICT messages.
4. Respond with JSON lines.

MVP here only documents protocol; actual executable worker entrypoint can be added later.
"""
from __future__ import annotations

INIT = "INIT"
LOAD_MODEL = "LOAD_MODEL"
PREDICT = "PREDICT"
SHUTDOWN = "SHUTDOWN"
STATS = "STATS"

__all__ = [
    "INIT",
    "LOAD_MODEL",
    "PREDICT",
    "SHUTDOWN",
    "STATS",
]
