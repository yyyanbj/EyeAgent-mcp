# Quick Start

## Install
We recommend using uv for dependency management:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then in the repository root:
```
uv sync
```

## Run UI
Launch a simple Gradio UI that streams tool/agent steps:

```
uv run eyeagent-ui --port 7860
```

By default, workflow mode is unified unless overridden in `eyeagent/config/eyeagent.yml`.
We have set it to `profile` in this repo so that the multi-agent pipeline runs via `pipelines.yml`.
To explicitly select the default profile at runtime:

```
export EYEAGENT_PIPELINE_PROFILE=default
uv run eyeagent-ui --mcp-url "http://localhost:8000/mcp" --port 7860
```

## Benchmark (batch mode)

Use the repository script `bin/run_benchmark.py` for headless runs over many cases. Example:

```
python bin/run_benchmark.py \
	--config-file eyeagent/eyeagent/config/eyeagent.imaging.yml \
	--profile imaging --backend profile --routing llm \
	--images-dir /data/fundus_eval
```

## Knowledge step (RAG / PubMed)

If your MCP server exposes the tools `rag:query` and `web_search:pubmed`, EyeAgent will run a knowledge step between specialist and follow-up when using the `default` profile.

The UI and benchmark flows will include knowledge between specialist and follow-up when the selected profile enables it.

Tool contracts (normalized):
- rag:query args: `{ "query": string, "top_k"?: number }` → `{ items: [{ title?, text?, source?, score? }], source: "rag" }`
- web_search:pubmed args: `{ "query": string, "top_k"?: number }` → `{ items: [{ id?, title?, abstract?, url?, year? }], source: "pubmed" }`

You can remap tool IDs via `eyeagent/config/tools.yml`.

## Test
```
uv run pytest -q
```
