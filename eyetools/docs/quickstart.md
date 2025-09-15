# Quick Start

## Environment
The project uses `uv` for dependency and multi-Python management. Install `uv`:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

See the official docs for alternative installation methods.

## Install Dependencies
```
uv sync
```

## Run Tests
```
uv run pytest -q
```

## Launch MCP Server
Basic run (will auto-load local `./tools` directory if it exists):
```
uv run eyetools-mcp serve --host 0.0.0.0 --port 8000
```
Add examples and an explicit tools directory (repeat `--tools-dir` to add multiple roots):
```
uv run eyetools-mcp serve --host 0.0.0.0 --port 8000 --include-examples --tools-dir tools
```
`--tools-dir` is an alias of `--tool-path`; both are accepted.

Eagerly load all tool models (GPU memory allocation) at startup:
```
uv run eyetools-mcp serve --preload
```
Include subprocess-mode tools too (if defined):
```
uv run eyetools-mcp serve --preload --preload-subprocess
```

Open http://localhost:8000/docs to view the FastAPI interactive docs.

## Tool Discovery
Use one (or many) `--tools-dir` / `--tool-path` flags to point at directories containing tool packages. You can also set environment variables:

* `EYETOOLS_TOOL_PATHS` (colon separated)
* `EYETOOLS_EXTRA_TOOL_PATHS` (colon separated)

If neither flags nor env vars are provided and a local `./tools` directory exists, it will be used automatically.

See `examples/tool_package_template` for a reference structure.

