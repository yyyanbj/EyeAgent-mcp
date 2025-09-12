from pathlib import Path
import pytest
from eyetools.core.registry import ToolRegistry
from eyetools.core.loader import discover_tools
from eyetools.core.tool_manager import ToolManager
from eyetools.core.errors import ToolNotFoundError


def test_metrics_increment_and_aggregate():
    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    registry = ToolRegistry()
    discover_tools([examples_dir], registry)
    tm = ToolManager(registry)
    # choose first tool
    metas = registry.list()
    assert metas
    tool_id = metas[0].id
    for _ in range(3):
        tm.predict(tool_id, {"inputs": {"input": "x"}})
    metrics = tm.get_metrics(tool_id)
    assert metrics.get("predict_count") == 3
    assert metrics.get("avg_latency_ms") >= 0
    all_metrics = tm.get_metrics()
    assert all_metrics["__aggregate__"]["predict_total"] >= 3


def test_tool_not_found_error():
    registry = ToolRegistry()
    tm = ToolManager(registry)
    with pytest.raises(ToolNotFoundError):
        tm.predict("non.existent.tool", {"inputs": {"input": "x"}})
