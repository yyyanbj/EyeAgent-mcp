#!/usr/bin/env python3
"""
Test RAG via MCP server endpoints and optionally reload/warmup the tool.

Usage examples:
  # 1) Start the server (in another terminal):
  #    uv run eyetools-mcp serve --tools-dir eyetools/tools --host 127.0.0.1 --port 8000

  # 2) Local quick test (no qdrant ingestion):
  #    python eyetools/scripts/test_mcp_rag.py --query "retinal detachment" --base-url http://127.0.0.1:8000

  # 3) With reload + warmup (qdrant mode, if dependencies available in rag env):
  #    python eyetools/scripts/test_mcp_rag.py --reload --warmup --query "retinal detachment" --base-url http://127.0.0.1:8000

Notes:
 - The server integrates FastMCP; this script also checks /mcp/tools to confirm MCP tool registration.
 - To point RAG at your own corpus, run the server with env var:
     export EYETOOLS_RAG_DIRS="/abs/path/docs:/abs/path/more"
"""
from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict

import requests


def _get(url: str):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r


def _post(url: str, params: Dict[str, Any] | None = None, json_body: Dict[str, Any] | None = None):
    r = requests.post(url, params=params or {}, json=json_body or {}, timeout=120)
    r.raise_for_status()
    return r


def find_rag_tool_id(base_url: str) -> str | None:
    r = _get(f"{base_url}/tools")
    data = r.json()
    # New shape: {"tools": [{"id": "rag:query", ...}, ...]}
    if isinstance(data, dict) and isinstance(data.get("tools"), list):
        for t in data["tools"]:
            if isinstance(t, dict) and str(t.get("id", "")).startswith("rag:"):
                return str(t["id"])
    # Legacy fallback: plain list of ids
    if isinstance(data, dict) and isinstance(data.get("tools"), list):
        for tid in data["tools"]:
            if isinstance(tid, str) and tid.startswith("rag:"):
                return tid
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", help="MCP server base URL")
    ap.add_argument("--query", required=True, help="Query text for RAG")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--reload", action="store_true", help="Call /admin/reload before querying")
    ap.add_argument("--warmup", action="store_true", help="Call /admin/warmup for rag:query (qdrant ingestion)")
    ap.add_argument("--role", help="Optional role name for /predict role filtering")
    ap.add_argument("--skip-mcp-check", action="store_true", help="Skip /mcp/tools verification")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")

    # 0) Health check
    try:
        hr = _get(f"{base}/health")
        print("[health]", hr.json())
    except Exception as e:
        print("[error] health check failed:", e)
        return 2

    # 1) Optional MCP tool registration check
    if not args.skip_mcp_check:
        try:
            mr = _get(f"{base}/mcp/tools")
            mdata = mr.json()
            # Show brief summary
            names = mdata.get("tool_names") or mdata.get("tools") or []
            print(f"[mcp] tools registered (count={len(names)}):", names[:10])
        except Exception as e:
            print("[warn] MCP tools check failed:", e)

    # 2) Discover tools and locate rag:query id
    try:
        tid = find_rag_tool_id(base)
        print("[tools] rag tool id:", tid)
        if not tid:
            print("[error] rag tool not discovered; ensure server started with --tools-dir eyetools/tools")
            return 3
    except Exception as e:
        print("[error] listing tools failed:", e)
        return 3

    # 3) Optional reload
    if args.reload:
        try:
            rr = _post(f"{base}/admin/reload")
            print("[reload]", rr.json())
        except Exception as e:
            print("[error] reload failed:", e)
            return 4

    # 4) Optional warmup (vector ingestion for qdrant mode)
    if args.warmup:
        try:
            wr = _post(f"{base}/admin/warmup", params={"tool_id": tid})
            print("[warmup]", json.dumps(wr.json(), ensure_ascii=False))
        except Exception as e:
            print("[warn] warmup failed (ok to skip for local mode):", e)

    # 5) Predict via MCP server
    try:
        payload = {"tool_id": tid, "request": {"inputs": {"query": args.query, "top_k": args.top_k}}}
        if args.role:
            payload["role"] = args.role
        pr = _post(f"{base}/predict", json_body=payload)
        out = pr.json()
        print("[predict]", json.dumps(out, ensure_ascii=False))
    except Exception as e:
        print("[error] predict failed:", e)
        return 5

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
