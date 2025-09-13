# Segmentation Tool Package

This package provides multiple medical image segmentation tasks using pre-trained nnUNetv2 models located under `weights/segmentation`.

## Variants
See `config.yaml` for full list of supported task variants (cfp_* , oct_*, ffa_*).

## Usage (Programmatic)
```python
from eyetools.core.registry import ToolRegistry
from eyetools.core.loader import discover_tools
from eyetools.core.tool_manager import ToolManager
from pathlib import Path

reg = ToolRegistry()
discover_tools([Path('tools')], reg, [])
manager = ToolManager(registry=reg, workspace_root=Path('.'))

# pick a variant id (after discovery) e.g. vision.segmentation:cfp_DR
for meta in reg.list():
    if meta.id.endswith('segmentation:cfp_DR'):
        result = manager.predict(meta.id, {"inputs": {"image_path": "sample.jpg"}})
        print(result)
```

## Environment
A dedicated uv environment `envs/seg` is defined to pin segmentation heavy dependencies. The tool config references it via `environment_ref: seg`.

## Notes
Model weights are expected at `weights/segmentation/DatasetXXX_*` folders (nnUNet structure). The tool will raise a clear error if a required dataset folder is missing.
