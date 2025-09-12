from pathlib import Path
from eyetools.core.registry import ToolRegistry
from eyetools.core.loader import discover_tools


def test_discover_examples():
    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    registry = ToolRegistry()
    errors = discover_tools([examples_dir], registry)
    # At least the template variants should be discovered
    ids = {m.id for m in registry.list()}
    assert any(i.startswith("demo_template:") for i in ids), f"Expected demo_template variants in {ids}"  # noqa
    # No config parse errors
    assert not errors, f"Discovery errors: {errors}"  # noqa
