# EyeAgent-mcp

## Tools Overview

Included tool packages (auto-discovered under `tools/`):
- Classification (`tools/classification`) – image classification variants
- Segmentation (`tools/segmentation`) – medical image segmentation (nnUNetv2) using dedicated environment `py312-seg`

Helper demo scripts:
- `scripts/demo_segmentation.py` – run a segmentation variant on a sample image (real or fallback synthetic inference)
- `scripts/mcp_client_demo.py` – example of invoking the tool via an MCP client workflow

See `docs/modules` for architecture and module details.

## Installation


Start with `uv` installation:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
cd eyetools
uv venv --python 3.12
source .venv/bin/activate
uv sync  # install base deps
```

Run the MCP server (auto-discovers tools). If you omit an explicit path and a local `./tools` directory exists, it will be used automatically.

```bash
uv run eyetools-mcp serve --host 0.0.0.0 --port 8000
```
Specify one or more tool roots explicitly (repeat flag) using the new `--tools-dir` alias (or legacy `--tool-path`):
```bash
uv run eyetools-mcp serve --host 0.0.0.0 --port 8000 --tools-dir tools --tools-dir extra_tools
```

Preload (instantiate + load models) to reduce first-request latency (may consume GPU memory immediately):
```bash
uv run eyetools-mcp serve --preload --tools-dir tools
```
Also preload subprocess tools (if any configured):
```bash
uv run eyetools-mcp serve --preload --preload-subprocess
```

### Segmentation Environment (Optional Heavy)
Segmentation variants use a dedicated environment `py312-seg` (see `envs/py312-seg/pyproject.toml`). You can warm it up:

```bash
uv run --with nnunetv2 --python=python3.12 python -c "import nnunetv2; print('nnUNet OK')"
```

Or just invoke a segmentation tool; dependencies will resolve on first use via `EnvManager`.

#### Quick Demo (Fallback Mode)
Run a fast synthetic segmentation (no model load) to see outputs structure:
```bash
python scripts/demo_segmentation.py --variant cfp_artifact --image examples/test_images/Artifact.jpg --mode fallback
```

#### Real Inference
Place nnUNet weights under `weights/segmentation/Dataset000_artifact` (etc.) then:
```bash
python scripts/demo_segmentation.py --variant cfp_artifact --image examples/test_images/Artifact.jpg
```

### Quick Programmatic Example
```python
from pathlib import Path
from eyetools.core.registry import ToolRegistry
from eyetools.core.loader import discover_tools
from eyetools.core.tool_manager import ToolManager

reg = ToolRegistry(); discover_tools([Path('tools')], reg, [])
manager = ToolManager(registry=reg, workspace_root=Path('.'))
seg_meta = next(m for m in reg.list() if m.package=='segmentation' and m.variant=='cfp_artifact')
print(manager.predict(seg_meta.id, {"inputs": {"image_path": "sample.png"}}))
```