#!/usr/bin/env python
"""Minimal MCP client demo for invoking a segmentation tool.
Requires an MCP-compatible client interface. This script illustrates a *conceptual* flow
(if you already have an MCP client library you can adapt the call section).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import sys

# Ensure repository root (parent of scripts dir) is in sys.path for imports when executed from other CWDs
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Pseudocode placeholder (replace with actual MCP client library usage):
# from mcp_client_lib import MCPClient

EXAMPLE_REQUEST = {
    "tool_id": "vision.segmentation:cfp_artifact",  # final discovered id may include category prefix
    "inputs": {"image_path": "examples/test_images/Artifact.jpg"}
}

def main():
    print("This is a conceptual MCP client demo. Integrate with your actual MCP client.")
    print("Example request body:")
    print(json.dumps(EXAMPLE_REQUEST, indent=2))
    print("Steps:")
    print(" 1. Start server: uv run eyetools-mcp serve --tools-dir tools --host 127.0.0.1 --port 8000")
    print(" 2. Use your MCP/HTTP client to list tools -> locate segmentation variant id")
    print(" 3. POST prediction: { 'id': <tool_id>, 'inputs': { 'image_path': '...'}}")
    print(" 4. Receive JSON with counts/areas/output_paths")

if __name__ == "__main__":
    main()
