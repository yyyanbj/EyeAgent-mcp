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

## Test
```
uv run pytest -q
```
