from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain_core.tools import BaseTool

# Optional official adapter (preferred when available)
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore
    _HAS_MCP_ADAPTER = True
except Exception:  # pragma: no cover - optional dependency
    MultiServerMCPClient = None  # type: ignore
    _HAS_MCP_ADAPTER = False

from fastmcp import Client
from .tool_registry import get_tool


class MCPToolArgs:  # lightweight schema placeholder; avoid pydantic dependency here
    def __init__(self, arguments: Optional[Dict[str, Any]] = None):
        self.arguments = arguments or {}


class MCPTool(BaseTool):
    name: str
    description: str
    mcp_name: str
    mcp_url: str

    # LangChain v0.2+ expects _run/_arun with explicit params
    def _run(self, arguments: Dict[str, Any], run_manager: Optional[Any] = None) -> str:  # type: ignore[override]
        import asyncio

        async def _call():
            async with Client(self.mcp_url) as client:
                res = await client.call_tool(name=self.mcp_name, arguments=arguments or {})
                try:
                    return str(getattr(res, "data", res))
                except Exception:
                    return str(res)

        # Use new loop if none running
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return loop.create_task(_call())  # type: ignore[return-value]
            return loop.run_until_complete(_call())
        except RuntimeError:
            return asyncio.run(_call())

    async def _arun(self, arguments: Dict[str, Any], run_manager: Optional[Any] = None) -> str:  # type: ignore[override]
        async with Client(self.mcp_url) as client:
            res = await client.call_tool(name=self.mcp_name, arguments=arguments or {})
            try:
                return str(getattr(res, "data", res))
            except Exception:
                return str(res)


def build_langchain_mcp_tools(tool_ids: List[str], mcp_url: str) -> List[BaseTool]:
    """Backward-compatible builder that exposes MCP tools as LangChain tools.

    - Default: lightweight wrapper around FastMCP (works without official adapter)
    - Note: Names are the EyeAgent tool_ids to keep the LLM contract stable
    """
    tools: List[BaseTool] = []
    for tid in tool_ids or []:
        meta = get_tool(tid)
        if not meta:
            continue
        desc = meta.get("desc") or meta.get("desc_long") or tid
        schema = meta.get("args_schema")
        if schema:
            try:
                import json as _json
                desc += "\nArgs(JSON Schema): " + _json.dumps(schema)
            except Exception:
                pass
        mcp_name = meta.get("mcp_name", tid)
        tools.append(MCPTool(name=tid, description=desc, mcp_name=mcp_name, mcp_url=mcp_url))
    return tools


async def load_tools_from_mcp(url: str, include_ids: Optional[List[str]] = None) -> List[BaseTool]:
    """Preferred async loader using official adapters when installed.

    If the adapter isn't available, this returns wrappers built via build_langchain_mcp_tools
    filtered by include_ids.
    """
    if _HAS_MCP_ADAPTER and MultiServerMCPClient is not None:
        client = MultiServerMCPClient({
            "default": {
                "transport": "streamable_http",
                "url": url,
            }
        })
        tools = await client.get_tools()
        if include_ids:
            # filter using MCP tool names from registry mapping
            allowed_names = set()
            for tid in include_ids:
                meta = get_tool(tid)
                if meta and meta.get("mcp_name"):
                    allowed_names.add(meta["mcp_name"])
            tools = [t for t in tools if getattr(t, "name", None) in allowed_names]
        return tools
    # Fallback: build wrappers
    return build_langchain_mcp_tools(include_ids or [], url)
