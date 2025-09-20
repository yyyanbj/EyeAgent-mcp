# EyeAgent: Ophthalmology Diagnostic Workflow (MCP-powered)

EyeAgent is a diagnostic workflow that orchestrates multiple agents (Orchestrator → Image Analysis → Specialist → Follow-up → Report) and integrates with MCP tools for image analysis and disease grading. It includes a CLI and an optional Gradio UI, structured tracing, and a final JSON report.

## Quick start

1) Install deps (use your preferred tool; examples here use uv):
```bash
cd /home/bingjie/workspace/EyeAgent-mcp/eyeagent
uv sync
```

2) Configure API access and MCP server:
- Set OpenAI-compatible credentials (DeepSeek or OpenAI):
  - in .env: OPENAI_API_KEY=...
  - or export DEEPSEEK_API_KEY=...
- Ensure an MCP server exposing the required tools is running and reachable via MCP_SERVER_URL (default http://localhost:8000/mcp/)

3) Run the CLI once:
```bash
uv run eyeagent-diagnose --patient '{"patient_id":"P001","age":63}' --images '[{"image_id":"IMG001","path":"/data/cfp1.jpg"}]'
```

4) Or launch the UI:
```bash
uv run eyeagent-ui

To point the UI at a specific MCP server without setting a global env var, pass the flag:

```bash
uv run eyeagent-ui --mcp-url http://localhost:8000/mcp/ --port 7860
```
```

Outputs are written to cases/<case_id> by default (trace.json, final_report.json). Override with EYEAGENT_CASES_DIR or EYEAGENT_DATA_DIR.

## Architecture (agents)
- Orchestrator: infers modality/laterality and plans the pipeline
- ImageAnalysis: quality and segmentation per modality, optional multidisease screening
- Specialist: disease-specific grading based on candidates
- FollowUp: management plan based on grades and patient data
- Report: consolidates everything into a clinician-friendly report

All agents share a diagnostic base that handles MCP tool calls with trace logging and optional LLM reasoning. See `eyeagent/agents/` for details.

## Configuration
- Prompts: `eyeagent/config/prompts.yml` (override system prompts, UI presets)
- Tools overlay: `eyeagent/config/tools.yml` (add/override tool metadata)
- Pipeline profiles: `eyeagent/config/pipelines.yml` (conditional step lists)
- Global settings (LLM): `eyeagent/config/eyeagent.yml`

Environment knobs:
- EYEAGENT_LOG_LEVEL, EYEAGENT_LOG_FILE, EYEAGENT_LOG_FORMAT
- EYEAGENT_DRY_RUN=1 to bypass real MCP calls and LLM planning/reasoning
- EYEAGENT_USE_LANGGRAPH=1 to prefer LangGraph; 0 uses a simple fallback runner (deprecated; prefer config)
- EYEAGENT_PIPELINE_PROFILE selects a pipeline from pipelines.yml (optional)
- EYEAGENT_MCP_ADAPTER_BIND=1 to use langchain-mcp-adapters tool binding

Workflow mode precedence:
- Preferred: set `workflow.mode` in `eyeagent/config/eyeagent.yml` (values: unified | graph | interaction | profile)
- Env overrides (deprecated but kept for compatibility):
  - `EYEAGENT_UNIFIED=1` forces unified mode
  - `EYEAGENT_USE_LANGGRAPH=1` prefers LangGraph when available
The config value takes precedence when present; env toggles will emit deprecation warnings.

## Development notes
- CLI entrypoints:
  - eyeagent-diagnose → `eyeagent/run_diagnosis.py:main`

## Multi-agent topologies (star & ring)

For quick experimentation beyond the unified agent flow, we provide lightweight executors for star and ring topologies in `eyeagent/core/topologies.py`.

- Star: a hub/planner selects agents to fan-out to; results are aggregated back.
- Ring: a fixed order of agents run in cycles with an optional early stop.

Minimal example:

```python
import asyncio
from eyeagent.core.topologies import run_star, run_ring


async def planner(state):
  # choose agents based on context/messages
  return ["image_analysis", "specialist"]


async def image_analysis(state):
  # ... perform work, append messages
  return {"messages": ["IA: found lesions"]}


async def specialist(state):
  # ... perform work, read context set by IA
  return {"messages": ["SP: graded DR R2"]}


async def aggregate(state, results):
  # combine results into final context/messages
  return {"context": {"summary": list(results.keys())}}


async def demo_star():
  state = {"messages": ["start"], "context": {}}
  agents = {"image_analysis": image_analysis, "specialist": specialist}
  await run_star(state, planner, agents, aggregate)
  return state


async def demo_ring():
  state = {"messages": ["start"], "context": {}}
  agents = [("image_analysis", image_analysis), ("specialist", specialist)]
  await run_ring(state, agents, rounds=2, stop_condition=lambda s: "stop" in s.get("context", {}))
  return state


if __name__ == "__main__":
  print(asyncio.run(demo_star()))
  print(asyncio.run(demo_ring()))
```

You can adapt these callables to wrap your existing agents (`eyeagent/agents/*.py`) by writing thin async adapters that accept and return partial state updates.
  - eyeagent-ui → `eyeagent/ui/app.py:main`
- Tests: `eyeagent/tests/test_smoke_dry_run.py` (uses dry run)
- Tracing: `eyeagent/tracing/trace_logger.py`
- MCP registry: `eyeagent/tools/tool_registry.py` (+ overlay)

  ### Examples (non-core)
  - Generic multi-agent + MCP demo lives in the repository top-level `examples/` only.
  - In-package re-exports/shims have been removed to avoid duplication.
  - Launch the demo UI (optional):
    - Run the top-level script: `python examples/run_multiagent.py`
      - Note: In-package shims `eyeagent/run_multiagent.py` and `eyeagent/multiagent_framework.py` were removed. Use the examples at repo root.

## MCP tool expectations (examples)
- classification:modality (CFP/OCT/FFA), classification:laterality, classification:cfp_quality
- segmentation:cfp_* and segmentation:oct_* tools for lesion detection
- disease-specific grading tools for Specialist (e.g., DR, AMD, etc.)

If your server uses different names, use `eyeagent/config/tools.yml` to map and augment metadata.

## License
Apache-2.0 (see repository root).
