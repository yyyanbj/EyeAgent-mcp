import sys, importlib
from pathlib import Path

def test_classification_tool_impl_import_out_of_package():
    cls_dir = Path('tools/classification').resolve()
    if str(cls_dir) not in sys.path:
        sys.path.insert(0, str(cls_dir))
    mod = importlib.import_module('tool_impl')
    assert hasattr(mod, 'ClassificationTool')
