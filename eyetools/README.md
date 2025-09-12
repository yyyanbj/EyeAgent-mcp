# EyeAgent-mcp

## Installation


Start with `uv` installation:

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
cd eyetools
uv venv --python 3.12.0
source .venv/bin/activate
uv sync # --extra dev
uv run python main_server.py 
```