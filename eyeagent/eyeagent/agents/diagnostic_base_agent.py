from typing import Any, Dict, List, Optional
import asyncio
import json
from loguru import logger
from fastmcp import Client
import os

from ..tracing.trace_logger import TraceLogger
from ..tools.tool_registry import get_tool
from ..config.prompts import PromptsConfig

class DiagnosticBaseAgent:
    """Base class for diagnostic agents with reasoning and trace hooks.

    Input context: patient, images, and accumulated intermediate results.
    Output contract: {"agent", "role", "outputs": {...}, "tool_calls": [...], "reasoning": str}
    """
    role: str = "generic"
    name: str = "GenericDiagnosticAgent"
    # tool_id list from tool registry
    allowed_tool_ids: List[str] = []
    system_prompt: str = "You are a diagnostic agent."

    def __init__(self, mcp_url: str, trace_logger: TraceLogger, case_id: str):
        self.mcp_url = mcp_url
        self.trace_logger = trace_logger
        self.case_id = case_id
        # Load system prompt override from config if present
        try:
            cfg = PromptsConfig()
            sp = cfg.get_system_prompt(self.__class__.__name__)
            if sp:
                self.system_prompt = sp
        except Exception:
            pass

    # LLM is disabled for diagnostic agents; planning should be implemented per-agent

    async def _call_tool(self, client: Client, tool_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        meta = get_tool(tool_id)
        event_base = {
            "type": "tool_call",
            "agent": self.name,
            "role": self.role,
            "tool_id": tool_id,
            "mcp_tool": meta.get("mcp_name") if meta else None,
            "version": meta.get("version") if meta else None,
            "arguments": arguments,
        }
        try:
            if not meta:
                raise ValueError(f"Unknown tool_id={tool_id}")
            raw = await client.call_tool(name=meta["mcp_name"], arguments=arguments)
            output = self._normalize_tool_result(raw)
            event = {**event_base, "status": "success", "output": output}
            self.trace_logger.append_event(self.case_id, event)
            return {"tool_id": tool_id, "output": output, "status": "success", "version": meta.get("version")}
        except Exception as e:
            event = {**event_base, "status": "failed", "error": str(e)}
            self.trace_logger.append_event(self.case_id, event)
            return {"tool_id": tool_id, "output": None, "status": "failed", "error": str(e), "version": meta.get("version") if meta else None}

    async def a_run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Override in subclasses to implement planning + tool usage + reasoning."""
        raise NotImplementedError

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return asyncio.run(self.a_run(context))

    async def plan_tools(self, task_desc: str, available: List[str]) -> List[Dict[str, Any]]:
        """Planner disabled. Agents should return explicit tool steps; default empty."""
        return []

    # ---- helpers ----------------------------------------------------
    def _normalize_tool_result(self, res: Any) -> Any:
        """Convert FastMCP CallToolResult (or any object) to a JSON-serializable plain value.

        Preference: extract the tool's returned dict from result.content if present.
        Fallback to model_dump()/dict()/str() for unknown types.
        """
        # Direct attributes used by FastMCP CallToolResult
        try:
            data_attr = getattr(res, "data", None)
            if data_attr is not None:
                return data_attr
            struct_attr = getattr(res, "structured_content", None)
            if struct_attr is not None:
                return struct_attr
        except Exception:
            pass
        # Pydantic v2 object
        try:
            if hasattr(res, "model_dump") and callable(getattr(res, "model_dump")):
                data = res.model_dump()
            elif hasattr(res, "dict") and callable(getattr(res, "dict")):
                data = res.dict()  # type: ignore[attr-defined]
            else:
                data = res
        except Exception:
            data = res
        # FastMCP typically uses a content array with blocks
        if isinstance(data, dict) and "content" in data:
            content = data.get("content")
            blocks = content if isinstance(content, list) else [content]
            norm_blocks = []
            for b in blocks:
                if hasattr(b, "model_dump"):
                    b = b.model_dump()
                norm_blocks.append(b)
            # Try to extract first JSON payload
            for b in norm_blocks:
                if isinstance(b, dict):
                    if "json" in b:
                        return b["json"]
                    if b.get("type") in ("json", "application/json") and "value" in b:
                        return b["value"]
                    if b.get("type") == "text" and "text" in b:
                        # If text looks like JSON, try to parse
                        txt = b.get("text")
                        if isinstance(txt, str) and txt.lstrip().startswith("{"):
                            try:
                                return json.loads(txt)
                            except Exception:
                                return txt
                        return txt
            return norm_blocks
        # Ensure JSON-serializable
        try:
            json.dumps(data)
            return data
        except Exception:
            return str(data)
