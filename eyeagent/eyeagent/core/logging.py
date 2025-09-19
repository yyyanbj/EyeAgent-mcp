from __future__ import annotations
import os
from loguru import logger

_CONFIGURED = False


def setup_logging() -> None:
    """Configure loguru logger once based on environment variables.

    Env vars:
    - EYEAGENT_LOG_LEVEL: log level (DEBUG/INFO/WARNING/ERROR), default INFO
    - EYEAGENT_LOG_FILE: optional path to write logs in addition to stderr
    - EYEAGENT_LOG_FORMAT: optional log format string for loguru
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = os.getenv("EYEAGENT_LOG_LEVEL", "DEBUG")
    fmt = os.getenv(
        "EYEAGENT_LOG_FORMAT",
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    # Remove default handler then add our sink(s)
    try:
        logger.remove()
    except Exception:
        pass
    logger.add(lambda msg: print(msg, end=""), level=level, format=fmt, enqueue=True)

    log_file = os.getenv("EYEAGENT_LOG_FILE")
    if log_file:
        # Rotation daily by default
        logger.add(log_file, level=level, format=fmt, rotation="00:00", retention="7 days", enqueue=True)

    _CONFIGURED = True
