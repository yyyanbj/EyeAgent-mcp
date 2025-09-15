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

### Lifecycle Modes (NEW)
You can now choose a lifecycle strategy controlling when models are instantiated / loaded and when they are unloaded:

| Mode | Description |
|------|-------------|
| `eager` (default) | Discover & preload all tools at startup (always includes subprocess tools). Fastest first inference; highest initial memory usage. |
| `lazy` | Discover tools only; instantiate/load on first request. Lower startup memory; first-call latency higher. |
| `dynamic` | Same as `lazy`, plus an automatic background maintenance loop that marks tools idle / unloads them after configurable inactivity. |

Flags:
```bash
--lifecycle-mode {eager|lazy|dynamic}
--dynamic-mark-idle-s <seconds>   # inactivity to mark IDLE (dynamic mode)
--dynamic-unload-s <seconds>      # inactivity to unload after being IDLE
--dynamic-interval-s <seconds>    # background sweep interval
```

Examples:
```bash
# 1. Eager (all models into RAM/VRAM) – default
uv run eyetools-mcp serve --lifecycle-mode eager

# 2. Lazy (on-demand loading)
uv run eyetools-mcp serve --lifecycle-mode lazy

# 3. Dynamic (auto idle/unload after 10m idle; unload after 30m)
uv run eyetools-mcp serve --lifecycle-mode dynamic \
	--dynamic-mark-idle-s 600 --dynamic-unload-s 1800 --dynamic-interval-s 60
```

Legacy preload flags (still supported):
```bash
--preload                # preload all inproc tools
--preload-subprocess     # also preload subprocess workers
```
Precedence: if `--lifecycle-mode` is specified it takes priority; legacy `--preload` flags are ignored under `eager` because eager always preloads everything (including subprocess). If you omit `--lifecycle-mode`, using `--preload` reproduces the previous startup behavior.

Eager & subprocess notes:
* Subprocess tools spawn worker processes during eager preload.
* The worker will attempt an explicit `LOAD_MODEL` command (if the tool supports `load_model`). If it fails (e.g., weights missing) the tool remains lazily loaded; the first prediction will then trigger loading.
* Check `/admin/config` to see active lifecycle configuration.

### Logging
Default log level is now `DEBUG` (can be noisy). Override with environment variable:
```bash
export EYETOOLS_LOG_LEVEL=INFO   # or WARNING / ERROR
```
All logs share the prefix `[eyetools.core]` or similar logger names.
To also persist logs to a file, use the CLI flag:
```bash
uv run eyetools-mcp serve --log-dir logs
```
This creates (or appends) `logs/eyetools.log`. Equivalent environment-based setup:
```bash
export EYETOOLS_LOG_DIR=logs
uv run eyetools-mcp serve
```

### Management & Introspection Endpoints
The server exposes a lightweight operational surface (all JSON):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Basic status (count of discovered tools) |
| `/tools` | GET | List tools (optionally filtered by role) |
| `/predict` | POST | Run inference `{tool_id, request, role?}` |
| `/metrics` | GET | Aggregated metrics; add `?tool_id=<id>` for single tool |
| `/lifecycle` | GET | Lifecycle states; add `?detailed=true` for `last_used` + loaded flag |
| `/admin/reload` | POST | Rediscover tools; `?preload_models=true` to eagerly (re)load |
| `/admin/preload/{tool_id}` | POST | Preload a single tool: inproc => instantiate (+lazy model); subprocess => spawn worker |
| `/admin/evict?tool_id=<id>` | POST | Unload an inproc tool or terminate subprocess worker |
| `/admin/resources` | GET | Resource snapshot: counts, memory RSS, optional GPU, metrics aggregate |
| `/admin/maintenance` | POST | Run maintenance (query: mark_idle_s, unload_idle_s) |

Example single-tool preload:
```bash
curl -X POST http://localhost:8000/admin/preload/classification:modality
```

Evict a tool to reclaim RAM/VRAM:
```bash
curl -X POST 'http://localhost:8000/admin/evict?tool_id=classification:modality'
```

Fetch metrics for just one tool:
```bash
curl 'http://localhost:8000/metrics?tool_id=classification:modality'
```

Resource snapshot (memory & counts):
```bash
curl http://localhost:8000/admin/resources
```

Mark tools idle (>5s since last use) then unload those idle >10s (two-step or combined calls):
```bash
curl -X POST 'http://localhost:8000/admin/maintenance?mark_idle_s=5'
curl -X POST 'http://localhost:8000/admin/maintenance?unload_idle_s=10'
```

Combined (only unload threshold):
```bash
curl -X POST 'http://localhost:8000/admin/maintenance?unload_idle_s=600'
```

Detailed lifecycle snapshot (includes `last_used` timestamps):
```bash
curl 'http://localhost:8000/lifecycle?detailed=true'
```

Notes:
* Subprocess (segmentation) tools are now supported by `/admin/preload/{tool_id}` (spawns worker; model remains lazy if implemented that way).
* `evict` now supports terminating subprocess workers as well.
* Idle/unload automation hooks (`mark_idle`, `unload_idle`) exist internally and can be wired to a background task later.

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