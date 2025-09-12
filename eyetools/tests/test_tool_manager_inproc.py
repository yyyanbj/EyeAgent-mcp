from pathlib import Path
from eyetools.core.registry import ToolRegistry
from eyetools.core.loader import discover_tools
from eyetools.core.tool_manager import ToolManager


def test_inproc_predict_template_variant():
    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    registry = ToolRegistry()
    discover_tools([examples_dir], registry)
    tm = ToolManager(registry)
    # pick one variant
    tool_id = None
    for meta in registry.list():
        if meta.id.startswith("demo_template:"):
            tool_id = meta.id
            break
    assert tool_id, "No demo_template variant discovered"
    result = tm.predict(tool_id, {"inputs": {"input": "dummy"}})
    assert result["status"] == "ok"
    assert "outputs" in result
