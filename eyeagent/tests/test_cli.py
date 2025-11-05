from __future__ import annotations
import os
from pathlib import Path

from eyeagent.cli import main as eyeagent_main


def test_list_agents_with_config():
    repo_root = Path(__file__).resolve().parents[1]
    cfg = repo_root / "eyeagent" / "config" / "eyeagent.imaging.yml"
    assert cfg.is_file()
    rc = eyeagent_main(["list-agents", "--config", str(cfg)])
    assert rc == 0


essential_keys = ["llm", "workflow", "agents"]


def test_print_config():
    repo_root = Path(__file__).resolve().parents[1]
    cfg = repo_root / "eyeagent" / "config" / "eyeagent.triage.yml"
    assert cfg.is_file()
    rc = eyeagent_main(["print-config", "--config", str(cfg)])
    assert rc == 0
