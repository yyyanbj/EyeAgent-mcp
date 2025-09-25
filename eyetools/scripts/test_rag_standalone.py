#!/usr/bin/env python3
"""
Quick standalone test for rag:query tool without running full server.
- Supports both local and qdrant modes.
- Reads corpus from EYETOOLS_RAG_DIRS or config.yaml defaults.
Usage:
  uv run -q python eyetools/scripts/test_rag_standalone.py --mode local --query "retinal detachment"
  uv run -q python eyetools/scripts/test_rag_standalone.py --mode qdrant --query "retinal detachment" --warmup
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

# Ensure imports work when run from repo root
import sys
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from tools.rag.tool_impl import RAGQueryTool  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "qdrant"], default="local")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--warmup", action="store_true", help="For mode=qdrant, perform ingestion before query")
    ap.add_argument("--maxpages", type=int, default=50, help="Cap PDF pages for quick tests")
    args = ap.parse_args()

    tool = RAGQueryTool(
        meta={"root_dir": str((repo_root / "tools" / "rag").resolve()), "model": {"lazy": True}},
        params={
            "mode": args.mode,
            "top_k": args.top_k,
            "maxpages": args.maxpages,
            # Default to weights/rag/books relative to tool root if not overridden by env
            "corpus_dirs": ["../../weights/rag/books"],
        },
    )

    if args.warmup and args.mode == "qdrant":
        w = tool.warmup()
        print("warmup:", json.dumps(w, ensure_ascii=False))

    out = tool.predict({"inputs": {"query": args.query, "top_k": args.top_k}})
    print("predict:", json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
