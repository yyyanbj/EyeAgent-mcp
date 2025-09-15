import os
from pathlib import Path

def test_file_logging_creation(monkeypatch, tmp_path):
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("EYETOOLS_LOG_DIR", str(log_dir))
    monkeypatch.setenv("EYETOOLS_LOG_LEVEL", "DEBUG")
    from eyetools.core.logging import get_logger
    logger = get_logger("eyetools.core.test")
    logger.debug("test debug line")
    logger.info("info line")
    file_path = log_dir / "eyetools.log"
    assert file_path.exists()
    content = file_path.read_text(encoding="utf-8")
    assert "test debug line" in content
    assert "info line" in content