#!/usr/bin/env python3
"""
Multi-Agent Framework Runner
Run the multi-agent system with Gradio UI.
"""

import sys
import os
import asyncio

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from multiagent_framework import create_gradio_interface, MCP_SERVER_URL
from fastmcp import Client

async def check_mcp_server():
    """Check if MCP server is running and accessible."""
    try:
        async with Client(MCP_SERVER_URL) as client:
            await client.ping()
            tools = await client.list_tools()
            return len(tools) > 0
    except Exception:
        return False

async def main():
    print("🔍 Checking MCP server connection...")
    mcp_available = await check_mcp_server()

    if not mcp_available:
        print("❌ MCP server not available at", MCP_SERVER_URL)
        print("Please start the MCP server first:")
        print("  cd /home/bingjie/workspace/EyeAgent-mcp/eyetool")
        print("  uv run python main_server.py")
        print("\nOr test the connection with:")
        print("  uv run python test_mcp.py")
        return

    print("✅ MCP server is running and accessible")
    print("🚀 Starting Multi-Agent Framework...")

    interface = create_gradio_interface()
    interface.launch(server_name="0.0.0.0", server_port=5788)

if __name__ == "__main__":
    asyncio.run(main())
