# Segmentation Tool (nnUNetv2)

The `segmentation` tool package provides multiple medical image segmentation tasks (CFP, OCT, FFA) backed by pretrained nnUNetv2 models.

## Environment
- Environment reference: `py312-seg`
- Defined at: `envs/py312-seg/pyproject.toml`
- Heavy dependencies (torch, nnunetv2, SimpleITK, batchgenerators, etc.) are isolated from the base environment to keep core startup fast.

## Variants
See `tools/segmentation/config.yaml` for the full list (`cfp_*`, `oct_*`, `ffa_lesion`). Each variant maps to an internal dataset/model id via inlined configuration.

## Usage
Programmatic example:
```python
from pathlib import Path
from eyetools.core.registry import ToolRegistry
from eyetools.core.loader import discover_tools
from eyetools.core.tool_manager import ToolManager

reg = ToolRegistry()
discover_tools([Path('tools')], reg, [])
manager = ToolManager(registry=reg, workspace_root=Path('.'))

seg_tool_id = next(m.id for m in reg.list() if m.id.endswith('segmentation:cfp_DR'))
result = manager.predict(seg_tool_id, {"inputs": {"image_path": "sample.jpg"}})
print(result)
```

## Weights
Expected under `weights/segmentation/` with nnUNet folder naming: `DatasetXYZ_*`. The code searches for the prefix matching the resolved model id and loads `checkpoint_final.pth`.

## Outputs
The tool returns JSON with:
- `counts`: object with lesion -> instance count
- `areas`: object with lesion -> list of contour areas
- `output_paths`: merged, colorized, overlay image paths
- `inference_time`: seconds

## Testing Strategy
Tests can mock the predictor to avoid downloading weights: patch `SegmentationTool.load_model` to inject a dummy predictor producing a synthetic mask.

### CLI Example
You can launch the MCP server and access segmentation tools:
```bash
uv run --python=python3.12 eyetools-mcp serve --tools-dir tools --host 0.0.0.0 --port 8000
```
Then pick the discovered tool id (e.g. `vision.segmentation:cfp_artifact`) from registry listing.

### Dependency Decoupling
The implementation is self-contained; all lesion color and model id mapping logic lives inside `tools/segmentation/tool_impl.py` with no dependency on `langchain_tool_src`.

### Notes
- Heavy nnUNet inference is only triggered on first prediction (lazy load).
- Mock test `test_segmentation_mock_predict` ensures CI stays lightweight.

