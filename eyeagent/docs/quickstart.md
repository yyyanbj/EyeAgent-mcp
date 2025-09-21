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

## Run CLI (diagnosis)
The package exposes a CLI entrypoint that runs the workflow once:

```
uv run eyeagent-diagnose \
	--patient '{"patient_id":"P001","age":63}' \
	--images '[{"image_id":"IMG001","path":"/path/to/cfp.jpg"}]'
```

To avoid calling external services while exploring, enable dry-run:

```
export EYEAGENT_DRY_RUN=1
uv run eyeagent-diagnose --patient '{"patient_id":"P001"}' --images '[{"image_id":"I1","path":"/tmp/od.jpg"}]'
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

## Test
## Knowledge step (RAG / PubMed)

If your MCP server exposes the tools `rag:query` and `web_search:pubmed`, EyeAgent will run a knowledge step between specialist and follow-up when using the `default` profile.

- Or use the demo runner:

```
uv run python -m eyeagent.run_ophthalmology_demo
```

Tool contracts (normalized):
- rag:query args: `{ "query": string, "top_k"?: number }` → `{ items: [{ title?, text?, source?, score? }], source: "rag" }`
- web_search:pubmed args: `{ "query": string, "top_k"?: number }` → `{ items: [{ id?, title?, abstract?, url?, year? }], source: "pubmed" }`

You can remap tool IDs via `eyeagent/config/tools.yml`.
```
uv run pytest -q
```
