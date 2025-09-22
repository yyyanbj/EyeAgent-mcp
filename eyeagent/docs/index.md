# EyeAgent Documentation

Welcome to the EyeAgent documentation. This project provides multi-agent orchestration capabilities and tool integration, working with EyeTools for automatic tool discovery and invocation.

## Contents
- [Quick Start](quickstart.md)
- [Architecture](architecture.md)
- (More sections will be added)

## Tool Outputs & UI

EyeAgent expects unified tool outputs across types. The agents standardize common fields (label, probabilities, probability/predicted for disease-specific, counts/areas for segmentation) while preserving original keys.

The MCP server wraps raw tool output with a `__meta__` block (tool_id, inputs, ts). Agents unwrap this and attach `mcp_meta` to trace events. The UI shows core fields and a small gray header with the tool id and timestamp for each tool bubble.
