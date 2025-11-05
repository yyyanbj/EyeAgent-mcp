# EyeAgent: Ophthalmology Diagnostic Workflow (MCP-powered)

EyeAgent is a diagnostic workflow that orchestrates multiple agents (Orchestrator → Image Analysis → Specialist → Follow-up → Report) and integrates with MCP tools for image analysis and disease grading. It includes a CLI and an optional Gradio UI, structured trace, and a final JSON report.

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

3) Launch the UI:
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
uv run eyeagent-ui --backend profile
```

Outputs are written to cases/<case_id> by default (trace.json, final_report.json). Override with EYEAGENT_CASES_DIR or EYEAGENT_DATA_DIR.

4) Unified CLI (headless, config-driven)

Run once with JSON input and a chosen backend/profile:

```bash
uv run eyeagent run \
  --config eyeagent/eyeagent/config/eyeagent.triage.yml \
  --backend profile \
  --profile triage \
  --patient '{"patient_id":"P001","age":63,"gender":"F"}' \
  --images '[{"image_id":"IMG001","path":"/path/to/fundus.jpg"}]'
```

Overrides via startup flags (no need to edit config file):
- `--backend` to switch backend (langgraph | profile | interaction | single)
- `--profile` to pick a pipeline profile when using the profile backend
- `--spec` to pass a custom interaction spec file when using the interaction backend (YAML/JSON)
- `--enable-agent role` / `--disable-agent role` to toggle agents for this run (can be repeated)

Utilities:

```bash
# List available agents after loading config
uv run eyeagent list-agents --config eyeagent/eyeagent/config/eyeagent.imaging.yml

# Print the fully-resolved config EyeAgent will use
uv run eyeagent print-config --config eyeagent/eyeagent/config/eyeagent.triage.yml
```

4) Headless benchmark (batch mode):

Use the repository script `bin/run_benchmark.py` to process a JSONL file or scan an images directory. Examples:

```bash
# Triage profile, profile backend, LLM routing; input as JSONL
python bin/run_benchmark.py \
  --config-file eyeagent/eyeagent/config/eyeagent.triage.yml \
  --profile triage --backend profile --routing llm \
  --input-jsonl ./cases_list.jsonl \
  --output-dir ./cases

# Imaging-only profile; scan a directory of images
python bin/run_benchmark.py \
  --config-file eyeagent/eyeagent/config/eyeagent.imaging.yml \
  --profile imaging --backend profile --routing llm \
  --images-dir ./datasets/fundus --glob "*.jpg" \
  --output-dir ./benchmark_out
```

### CLI incremental runs (append-only)

You can continue a previous case from the CLI and process only newly added images. This preserves the same case ID and merges results with the prior run.

Key flags:
- `--case-id <ID>`: the existing case to continue (found under `cases/<ID>/`)
- `--incremental`: enable incremental mode; only new images are processed

Example:

```bash
uv run eyeagent run \
  --config eyeagent/eyeagent/config/eyeagent.imaging.yml \
  --case-id 01234567-89ab-cdef-0123-456789abcdef \
  --incremental \
  --images '[
    {"image_id":"NEW_OD","path":"/data/cases/P002/new_OD.jpg"},
    {"image_id":"NEW_OCT","path":"/data/cases/P002/new_OCT.png"}
  ]'
```

Behavior and notes:
- The CLI loads `cases/<case_id>/trace.json` and merges its `images` with the newly provided `--images` list (no duplicates by `image_id` fallback `path`).
- If you don’t specify a backend, the CLI prefers `profile` in incremental mode to ensure `prior` is supported.
- The workflow receives `prior={"images": images_prev}`; agents detect `context.incremental` and process only new images.
- Patient object: if not provided for this run, it’s loaded from the previous case.
- Image identity: matched by `image_id`, falling back to `path` when missing.

## Architecture (agents)
- Orchestrator: infers modality/laterality and plans the pipeline
- ImageAnalysis: quality and segmentation per modality, optional multidisease screening
- Specialist: disease-specific grading based on candidates
- FollowUp: management plan based on grades and patient data
- Knowledge: queries internal RAG and optionally PubMed for evidence and summaries
- Report: consolidates everything into a clinician-friendly report

All agents share a diagnostic base that handles MCP tool calls with trace logging and optional LLM reasoning. See `eyeagent/agents/` for details.

## Multi-image input and per-image routing
- Inputs: pass a list of images as `[{"image_id": "IMG001", "path": "/path/to/xxx.jpg"}, ...]`.
- Single-image tools: every MCP tool accepts exactly one image. EyeAgent automatically fans out per image, choosing tools per image modality and running them in parallel.
- Modality routing: `PreliminaryAgent` runs `classification:modality` and `classification:laterality` for each image; `ImageAnalysisAgent` then dispatches only CFP tools to CFP images, OCT tools to OCT images, and FFA tools to FFA images.
- Aggregation: `ImageAnalysisAgent` returns `per_image` blocks (quality, lesions, diseases) and a merged summary; `SpecialistAgent` returns per-image grading; `ReportAgent` consolidates to the final fragment while preserving `per_image` details.

Behavior with mixed modalities:
- Example input containing both CFP and OCT images → IA will run `classification:cfp_quality` and `segmentation:cfp_*` on CFP images, `segmentation:oct_*` on OCT images; no cross-modality calls are made to wrong images.
- If modality cannot be determined, CFP is used as a safe default for that specific image.

Performance and concurrency:
- Per-image analysis runs concurrently to minimize overall latency (async fan-out per image and tool).
- You can globally disable heavy tools or restrict tools per agent via `eyeagent/config/tools.yml` or runtime filters to fit your environment.

Examples

- CLI with multiple images:

```bash
uv run eyeagent run \
  --config eyeagent/eyeagent/config/eyeagent.imaging.yml \
  --backend profile --profile imaging \
  --patient '{"patient_id":"P002","age":58}' \
  --images '[
    {"image_id":"OD-1","path":"/data/cases/P002/OD.jpg"},
    {"image_id":"OCT-1","path":"/data/cases/P002/OCT.png"}
  ]'
```

- UI: upload multiple files in the Run Diagnosis tab; EyeAgent will auto-route each image by modality and show per-image tool calls in the chat panel.

### Incremental runs (append images after a case is finished)

UI supports appending new images to an existing case and processing only the new images:

- Open the "Continue / Incremental" tab in the UI.
- Enter the existing Case ID (from the first run; see cases/<case_id>/).
- Upload additional images and optionally provide extra instruction.
- Click "Append and Incremental Run".

Under the hood:

- The workflow is called with `prior` state (previous images and agent outputs).
- Agents read `context.incremental` and `context.new_image_ids` and only process the new images; outputs are merged with prior results.
- The chat panel streams only the newly added events; final report gets updated.

Programmatic usage (optional)

```python
import json
from pathlib import Path
from eyeagent.diagnostic_workflow import run_diagnosis_async

# Load prior state from an existing case
CASES = Path('cases')
case_id = 'YOUR_CASE_ID'
doc = json.loads((CASES / case_id / 'trace.json').read_text(encoding='utf-8'))

patient = doc.get('patient', {})
images_prev = doc.get('images', [])
# Append new images
images_merged = list(images_prev) + [
  {"image_id": "NEW_OD", "path": "/path/to/new_OD.jpg"},
  {"image_id": "NEW_OCT", "path": "/path/to/new_OCT.png"},
]

prior = {
  "images": images_prev,
  "orchestrator_outputs": doc.get("orchestrator_outputs"),
  "preliminary": doc.get("preliminary"),
  "image_analysis": doc.get("image_analysis"),
  "specialist": doc.get("specialist"),
  "knowledge": doc.get("knowledge"),
  "follow_up": doc.get("follow_up"),
}

# Reuse same case_id to keep a single trace
final = await run_diagnosis_async(patient, images_merged, case_id=case_id, prior=prior)
```

Notes

- Image identity is matched by `image_id` (fallback to `path` when missing). UI derives `image_id` from file name stem.
- If a new upload shares the same `image_id` as an existing image, it is treated as already processed and skipped during incremental processing.
- Incremental mode currently optimizes Preliminary/ImageAnalysis/Specialist steps; Report/Follow-up will re-synthesize a fresh final fragment with merged per-image details.

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
- Profile-driven (pipelines unified入口): `eyeagent/workflows/profile.py`
  - Uses `eyeagent/config/pipelines.yml` to define a step list with optional conditions
  - Public API: `run_diagnosis_async`, `run_diagnosis`
  - Select profile with `EYEAGENT_PIPELINE_PROFILE` (default: `default`)
- Spec/interaction-driven: `eyeagent/workflows/interaction.py`
  - Accepts a custom spec (nodes/edges) or falls back to a simple orchestrator-led sequence
  - Public API: `run_diagnosis_async`, `run_diagnosis`
  
- Topology-driven: `eyeagent/workflows/topology.py`
  - Lightweight执行器，支持 star/ring 两类拓扑，不依赖 LangGraph
  - 通过环境变量或 settings.workflow.topology 配置：
    - EYEAGENT_TOPOLOGY: star|ring（默认 ring）
    - EYEAGENT_TOPOLOGY_AGENTS: 逗号分隔的 agent 角色列表（环形顺序 / 星型集合）
    - EYEAGENT_TOPOLOGY_ROUNDS: 环形轮数（默认 1）

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
  backend: langgraph   # or: profile | interaction | topology | single
```

Or via environment variable for the current process:

```bash
export EYEAGENT_WORKFLOW_BACKEND=profile
```

CLI one-off override for the UI (set for the process only): pass `--backend` when launching `eyeagent-ui`.

## Development notes
Supported entry modes:
  - eyeagent-ui → `eyeagent/ui/app.py:main`
  - Unified CLI → `eyeagent/cli.py:main` (console script `eyeagent`)
  - Headless benchmark → `bin/run_benchmark.py`

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
  
- Tracing: `eyeagent/trace/trace_logger.py`
- MCP registry: `eyeagent/tools/tool_registry.py` (+ overlay)

  ### Examples (non-core)
  - Generic multi-agent + MCP demo lives in the repository top-level `examples/` only.
  - In-package re-exports/shims and demo runners were removed to avoid duplication.

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
