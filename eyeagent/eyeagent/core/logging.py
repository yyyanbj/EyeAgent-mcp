from __future__ import annotations
from loguru import logger
from .settings import get_logging_config

_CONFIGURED = False


def setup_logging() -> None:
    """Configure loguru logger once based on centralized settings (no env).

    Values are resolved from the loaded config under `logging` and can be
    overridden per-process via settings.set_overrides().
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    cfg = get_logging_config()
    level = cfg.get("level", "DEBUG")
    fmt = cfg.get("format") or "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    # Remove default handler then add our sink(s)
    try:
        logger.remove()
    except Exception:
        pass
    logger.add(lambda msg: print(msg, end=""), level=level, format=fmt, enqueue=True)

    log_file = cfg.get("file")
    if log_file:
        # Rotation daily by default
        logger.add(log_file, level=level, format=fmt, rotation="00:00", retention="7 days", enqueue=True)

    _CONFIGURED = True
