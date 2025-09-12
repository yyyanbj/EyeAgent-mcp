from pathlib import Path
from eyetools.core.registry import ToolRegistry
from eyetools.core.loader import discover_tools
from eyetools.core.tool_manager import ToolManager


def test_subprocess_variant():
    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    registry = ToolRegistry()
    discover_tools([examples_dir], registry)
    # force one variant to subprocess by adjusting runtime
    target = None
    for m in registry.list():
        if m.id.startswith("demo_template:large"):
            m.runtime["load_mode"] = "subprocess"
            target = m.id
            break
    if not target:  # skip if not present
        return
    tm = ToolManager(registry, workspace_root=examples_dir.parent)
    result = tm.predict(target, {"inputs": {"input": "dummy"}})
    assert result["status"] == "ok"
