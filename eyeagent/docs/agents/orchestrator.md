# Orchestrator Agent

The Orchestrator decides exactly one `next_agent` per step and never calls tools. It operates in a strict cadence with the interaction engine:

1) route (decide `next_agent`) → 2) execute that agent → 3) collect outputs → 4) re-route … → finish at `report`.

## Outputs

- `next_agent`: one of
  - `symptom_triage | preliminary | image_analysis | specialist | decision | follow_up | report`
- `routing_reasons` (optional): short list of reasons for debugging/UI.

Planned pipelines are deprecated in Orchestrator outputs. UIs and engines should consume `next_agent` only.

## Guardrails & Heuristics

- Prefer `symptom_triage` early (if enabled and not completed).
- If images exist, ensure `preliminary` and `image_analysis` precede `specialist`/`decision` when enabled.
- Otherwise pick the first enabled unfinished step; default to `report`.

## Routing strategies

- `llm` (default): The LLM proposes `next_agent` with minimal JSON; guardrails enforce safe ordering.
- `config`: Deterministic policy (e.g., derived from profile/steps) if configured.

Select with:

- Env: `EYEAGENT_ROUTING_STRATEGY=llm|config`
- CLI (UI): `--routing-strategy`
- CLI (headless): `python -m eyeagent.cli run --routing-strategy llm`

## Notes

- The Orchestrator’s system prompt and capabilities are aligned with the next-agent-only contract.
- The engine enforces sequential agent execution; tools are also called sequentially.
