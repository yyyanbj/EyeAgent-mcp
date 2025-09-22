# EyeTools Documentation

Welcome to the EyeTools framework docs. This folder provides an overview, architecture, per-module reference, testing practices and contribution guide.

## Contents
- [Quick Start](quickstart.md)
- [Architecture Overview](architecture.md)
- Module docs (modules/*.md)
- [Testing & Quality](testing.md)
- [Contributing](contributing.md)

## Unified Output Contract

All tools should return plain JSON with minimal shaping. Common fields where applicable:

- prediction or label: the primary predicted class or value
- probabilities: map of label -> probability for classification-like tools
- inference_time: seconds spent inside the tool (optional)
- counts/areas/output_paths: for segmentation-style tools

The MCP server returns a pass-through envelope with meta:

{ "output": <tool_output>, "__meta__": { "tool_id": "...", "inputs": {...}, "ts": <epoch_seconds> } }

Framework layers unwrap the envelope and attach `mcp_meta` to traces so the UI can show tool name and time.

