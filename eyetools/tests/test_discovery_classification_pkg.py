import os
from pathlib import Path
from eyetools.core.registry import ToolRegistry
from eyetools.core.loader import discover_tools


def test_discover_classification_tool(monkeypatch):
    root_tools = Path(__file__).resolve().parent.parent / "tools"
    assert (root_tools / "classification" / "config.yaml").exists()
    registry = ToolRegistry()
    errors = discover_tools([root_tools], registry)
    assert not errors
    ids = {m.id for m in registry.list()}
    assert any(i.startswith("classification:") for i in ids), ids