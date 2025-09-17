#!/usr/bin/env python3
"""Single Agent MCP client test script

Features:
1. Connect to an MCP server (default: http://localhost:8000/mcp/)
2. Use ResearchAgent to select and call tools
3. Persist each conversation to ui/conversations/<case_uuid>.json
4. Support interactive CLI (multi-turn), type exit/quit to leave
"""
import os
import argparse
import asyncio
from agents.research_agent import ResearchAgent

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp/")

def parse_args():
    ap = argparse.ArgumentParser(description="Single Agent MCP Test")
    ap.add_argument("--mcp", default=MCP_SERVER_URL, help="MCP server URL (default: %(default)s)")
    ap.add_argument("--once", action="store_true", help="Run once without interactive loop")
    ap.add_argument("--query", help="Provide a single-shot query (use with --once)")
    return ap.parse_args()

async def a_main():
    args = parse_args()
    agent = ResearchAgent(mcp_url=args.mcp)

    if args.once:
        if not args.query:
            raise SystemExit("--once requires --query")
        result = await agent.a_run(args.query)
        print("Result:", result)
        return

    print(f"🔗 Connected to MCP: {args.mcp}")
    print("Type your question. Enter exit/quit to stop.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExited")
            break
        if q.lower() in {"exit", "quit"}:
            print("Bye 👋")
            break
        if not q:
            continue
        result = await agent.a_run(q)
        print(f"[{result['role']}][case={result['case_id'][:8]}] -> {result['answer']}")

if __name__ == "__main__":
    asyncio.run(a_main())
