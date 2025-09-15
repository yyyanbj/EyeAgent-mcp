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

## Demo Scripts
Two helper scripts illustrate real vs. fallback inference and a simple MCP client interaction.

### `scripts/demo_segmentation.py`
Run a segmentation variant on a sample image. Supports two modes:

- `real` (default): attempts to load nnUNet model weights under `weights/segmentation/`.
- `fallback`: skips model loading and produces a synthetic mask so you can observe the output structure instantly.

Example (fallback for quick preview):
```bash
python scripts/demo_segmentation.py \
	--variant cfp_artifact \
	--image examples/test_images/Artifact.jpg \
	--mode fallback
```

Attempt real inference (requires proper nnUNet weights present):
```bash
python scripts/demo_segmentation.py \
	--variant cfp_artifact \
	--image examples/test_images/Artifact.jpg
```

Outputs (merged mask, colorized label map, overlay) are written next to the input image with suffixes, and a JSON summary is printed.

### `scripts/mcp_client_demo.py`
Illustrates how an MCP client could call the segmentation tool through the running server. Adapt this script for integration into your own agent or UI layer.

---
If you invoke these scripts from within a subdirectory, they dynamically insert the project root into `sys.path` so imports remain robust.

