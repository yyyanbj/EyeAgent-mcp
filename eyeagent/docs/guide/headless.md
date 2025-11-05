# Headless (No UI) Guide

This guide shows how to run EyeAgent end-to-end from the command line without the Gradio UI. The CLI prints the final JSON report to stdout for easy batch processing.

## Basic run

```bash
python -m eyeagent.cli run \
  --mcp-url http://localhost:8000/mcp/ \
  --backend interaction \
  --routing-strategy llm \
  --patient '{"patient_id":"P001"}' \
  --images '[{"image_id":"I1","path":"/data/fundus.jpg"}]'
```

- `--backend interaction` enables step-by-step routing (route → execute → route), strictly sequential.
- `--routing-strategy llm` lets the Orchestrator pick `next_agent` using an LLM (with guardrails).

## Incremental runs

Append new images to an existing case:

```bash
python -m eyeagent.cli run \
  --case-id <existing_case_uuid> \
  --incremental \
  --images '[{"image_id":"I2","path":"/data/new.jpg"}]'
```

The engine re-runs image-related steps for new images and merges outputs.

## Configuration

- `--config /path/to/eyeagent.yml` will be respected via the Settings loader.
- You can also use `--enable-agent role` and `--disable-agent role` multiple times to toggle agents on the fly.

## Flags ↔ Env mapping

All runtime environment variables have CLI counterparts so you can avoid ad-hoc env exports:

- `--mcp-url` → MCP_SERVER_URL
- `--routing-strategy {llm,config}` → EYEAGENT_ROUTING_STRATEGY
- `--mcp-adapter-bind` → EYEAGENT_MCP_ADAPTER_BIND=1
- `--max-classes <int>` → EYEAGENT_MAX_CLASSES
- `--cases-dir` → EYEAGENT_CASES_DIR
- `--data-dir` → EYEAGENT_DATA_DIR

## Output

The CLI prints a JSON object that includes the final report and trace references. You can redirect to a file or post-process with `jq`.

```bash
python -m eyeagent.cli run ... > report.json
```
