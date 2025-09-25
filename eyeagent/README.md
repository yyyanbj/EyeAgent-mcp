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
```

To point the UI at a specific MCP server without setting a global env var, pass the flag:

```bash
uv run eyeagent-ui --mcp-url http://localhost:8000/mcp/ --port 7860
```

UI workflow backend selection:
- Per-run: use the “Workflow Backend” dropdown in the Run Diagnosis tab (choices: langgraph | profile | interaction).
- Session default: pass a startup flag so the whole UI uses a backend by default (still overridable per run by dropdown):

```bash
uv run eyeagent-ui --workflow-backend profile
```

Outputs are written to cases/<case_id> by default (trace.json, final_report.json). Override with EYEAGENT_CASES_DIR or EYEAGENT_DATA_DIR.

5) Optional: run the ophthalmology demo (includes knowledge step):

```bash
uv run python -m eyeagent.run_ophthalmology_demo
```

## Architecture (agents)
- Orchestrator: infers modality/laterality and plans the pipeline
- ImageAnalysis: quality and segmentation per modality, optional multidisease screening
- Specialist: disease-specific grading based on candidates
- FollowUp: management plan based on grades and patient data
- Knowledge: queries internal RAG and optionally PubMed for evidence and summaries
- Report: consolidates everything into a clinician-friendly report

All agents share a diagnostic base that handles MCP tool calls with trace logging and optional LLM reasoning. See `eyeagent/agents/` for details.

## Configuration
- Prompts: `eyeagent/config/prompts.yml` (override system prompts, UI presets)
- Tools overlay: `eyeagent/config/tools.yml` (add/override tool metadata)
- Pipeline profiles: `eyeagent/config/pipelines.yml` (conditional step lists); default profile includes `knowledge` between specialist and follow_up
- Global settings (LLM & workflow mode): `eyeagent/config/eyeagent.yml`

Recommended multi-agent setup:

1) Ensure `eyeagent/config/eyeagent.yml` contains:

```
workflow:
  backend: profile
```

2) Run the UI with the default profile:

```bash
export EYEAGENT_PIPELINE_PROFILE=default
uv run eyeagent-ui --mcp-url "http://localhost:8000/mcp" --port 7860
```

Environment knobs:
- EYEAGENT_LOG_LEVEL, EYEAGENT_LOG_FILE, EYEAGENT_LOG_FORMAT
- EYEAGENT_DRY_RUN=1 to bypass real MCP calls and LLM planning/reasoning
- EYEAGENT_USE_LANGGRAPH=1 to prefer LangGraph; 0 uses a simple fallback runner (deprecated; prefer config)
- EYEAGENT_PIPELINE_PROFILE selects a pipeline from pipelines.yml (optional)
- EYEAGENT_MCP_ADAPTER_BIND=1 to use langchain-mcp-adapters tool binding
 - MCP_SERVER_URL to point agents to your MCP server (the UI flag --mcp-url sets this for the process)

Workflow mode precedence:
- Preferred: set `workflow.mode` in `eyeagent/config/eyeagent.yml` (values: unified | graph | interaction | profile)
- Env overrides (deprecated but kept for compatibility):
  - `EYEAGENT_UNIFIED=1` forces unified mode
  - `EYEAGENT_USE_LANGGRAPH=1` prefers LangGraph when available
The config value takes precedence when present; env toggles will emit deprecation warnings.

## Workflow backends (default: LangGraph)

We provide multiple orchestration backends under `eyeagent/workflows/`:

- LangGraph (default): `eyeagent/workflows/langgraph.py`
  - Public API: `run_diagnosis_async`, `run_diagnosis`
  - Default entry re-export: `eyeagent/diagnostic_workflow.py`
- Profile-driven: `eyeagent/workflows/profile.py`
  - Uses `eyeagent/config/pipelines.yml` to define a step list with optional conditions
  - Public API: `run_diagnosis_async`, `run_diagnosis`
  - Select profile with `EYEAGENT_PIPELINE_PROFILE` (default: `default`)
- Spec/interaction-driven: `eyeagent/workflows/interaction.py`
  - Accepts a custom spec (nodes/edges) or falls back to a simple orchestrator-led sequence
  - Public API: `run_diagnosis_async`, `run_diagnosis`

Example usage:

```python
# Default (LangGraph)
from eyeagent.diagnostic_workflow import run_diagnosis_async

# Profile
from eyeagent.workflows.profile import run_diagnosis_async as run_profile

# Interaction / custom spec
from eyeagent.workflows.interaction import run_diagnosis_async as run_interaction

final = await run_diagnosis_async(patient, images)
final_profile = await run_profile(patient, images)
final_interaction = await run_interaction(patient, images, spec={...})
```

To select backend globally without changing imports, set in `eyeagent/config/eyeagent.yml`:

```yaml
workflow:
  backend: langgraph   # or: profile | interaction
```

Or via environment variable for the current process:

```bash
export EYEAGENT_WORKFLOW_BACKEND=profile
```

CLI one-off override (no config change):

```bash
uv run eyeagent-diagnose \
  --workflow-backend profile \
  --patient '{"patient_id":"P001","age":63}' \
  --images '[{"image_id":"IMG001","path":"/data/cfp1.jpg"}]'
```

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

Knowledge step tools (add to your MCP server):
- rag:query → returns top-k passages from your internal ophthalmology corpus
- web_search:pubmed → returns recent/relevant PubMed references

You can map/override tool IDs and descriptions in `eyeagent/config/tools.yml`.

If your server uses different names, use `eyeagent/config/tools.yml` to map and augment metadata.

## License
Apache-2.0 (see repository root).
