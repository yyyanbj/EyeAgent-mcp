"""
Unified base agent class for EyeAgent.

This replaces the older split between a generic BaseAgent and DiagnosticBaseAgent.
We expose BaseAgent (preferred name) and keep DiagnosticBaseAgent as an alias for
compatibility. The implementation corresponds to the former DiagnosticBaseAgent.
"""

from typing import Any, Dict, List, Optional
import asyncio
import json
from loguru import logger
from fastmcp import Client
import os

from ..tracing.trace_logger import TraceLogger
from ..metrics.metrics import tool_timer, add_tokens
from ..tools.tool_registry import get_tool
from ..config.prompts import PromptsConfig
from ..config.settings import build_chat_model, get_llm_config
from ..llm.json_client import JsonLLM
from ..llm.models import ToolPlan, ReasoningOut
from ..tools.langchain_mcp_tools import build_langchain_mcp_tools, load_tools_from_mcp
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from contextlib import asynccontextmanager


class BaseAgent:
    # Capabilities declaration (subclasses should override as needed)
    capabilities: dict = {
        "required_context": [],
        "expected_outputs": [],
        "retry_policy": {"max_attempts": 1, "on_fail": "fail"},
        "modalities": [],
        "tools": [],
    }

    """Base class for diagnostic agents with reasoning and trace hooks.

    Input context: patient, images, and accumulated intermediate results.
    Output contract: {"agent", "role", "outputs": {...}, "tool_calls": [...], "reasoning": str}
    """
    role: str = "generic"
    name: str = "GenericDiagnosticAgent"
    # tool_id list from tool registry
    allowed_tool_ids: List[str] = []
    system_prompt: str = "You are a diagnostic agent."
    # Class-level LLM shared per agent class (can be overridden in subclasses)
    llm = None

    def __init__(self, mcp_url: str, trace_logger: TraceLogger, case_id: str):
        self.mcp_url = mcp_url
        self.trace_logger = trace_logger
        self.case_id = case_id
        # Initialize class-level LLM if not set
        if self.__class__.llm is None:
            try:
                self.__class__.llm = build_chat_model(self.__class__.__name__)
            except Exception:
                self.__class__.llm = None
        # Load system prompt override from config if present
        try:
            cfg = PromptsConfig()
            sp = cfg.get_system_prompt(self.__class__.__name__)
            if sp:
                self.system_prompt = sp
        except Exception:
            pass

    def _dry_run(self) -> bool:
        return os.getenv("EYEAGENT_DRY_RUN", "0").lower() in ("1", "true", "yes")

    @asynccontextmanager
    async def _client_ctx(self):
        """Yield a real MCP client unless DRY-RUN is enabled.

        In DRY-RUN mode, yield None and avoid any network connection attempts."""
        if self._dry_run():
            yield None
            return
        async with Client(self.mcp_url) as client:
            yield client

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
        # DRY-RUN: return mock outputs without calling MCP
        if self._dry_run():
            role = (meta or {}).get("role")
            mock: Dict[str, Any] = {"mock": True}
            try:
                if role == "specialist":
                    mock.update({"disease": meta.get("disease") or "unknown", "probability": 0.0, "predicted": False, "all_probabilities": {"negative": 1.0}, "grade": "ungraded", "confidence": 0.0})
                elif isinstance(tool_id, str) and tool_id.startswith("classification:cfp_quality"):
                    mock.update({"prediction": "unknown"})
                elif isinstance(tool_id, str) and tool_id.startswith("segmentation:"):
                    mock.update({"counts": {}})
            except Exception:
                pass
            std = self._standardize_output(tool_id, mock, meta)
            event = {**event_base, "status": "success", "output": std, "dry_run": True}
            self.trace_logger.append_event(self.case_id, event)
            return {"tool_id": tool_id, "output": std, "status": "success", "version": meta.get("version") if meta else None}
        try:
            if not meta:
                raise ValueError(f"Unknown tool_id={tool_id}")
            # Measure latency and error status via context manager
            with tool_timer(tool_id):
                raw = await client.call_tool(name=meta["mcp_name"], arguments=arguments)
            output = self._normalize_tool_result(raw)
            std = self._standardize_output(tool_id, output, meta)
            event = {**event_base, "status": "success", "output": std}
            self.trace_logger.append_event(self.case_id, event)
            return {"tool_id": tool_id, "output": std, "status": "success", "version": meta.get("version")}
        except Exception as e:
            event = {**event_base, "status": "failed", "error": str(e)}
            self.trace_logger.append_event(self.case_id, event)
            return {"tool_id": tool_id, "output": None, "status": "failed", "error": str(e), "version": meta.get("version") if meta else None}

    async def a_run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Override in subclasses to implement planning + tool usage + reasoning."""
        raise NotImplementedError

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return asyncio.run(self.a_run(context))

    # ---- common helpers for per-image tool invocation ----------------------
    async def call_tool_per_image(self, client: Client | None, tool_id: str, images: List[Dict[str, Any]], arguments: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        """Call a tool for each image (or once if images empty) and attach image_id.

        When client is None (dry-run context manager yields None), _call_tool handles dry-run.
        """
        calls: List[Dict[str, Any]] = []
        img_list = images if images else [None]
        for img in img_list:
            args = dict(arguments or {})
            if isinstance(img, dict) and img.get("path"):
                args.setdefault("image_path", img.get("path"))
            tc = await self._call_tool(client, tool_id, args)
            if isinstance(img, dict):
                tc["image_id"] = img.get("image_id") or img.get("path")
            calls.append(tc)
        return calls

    async def plan_tools(self, task_desc: str, available: List[str]) -> List[Dict[str, Any]]:
        """LLM planner: propose ordered tool calls with optional arguments.

        Returns a list of {tool_id, arguments, reasoning}. If the model fails to
        produce valid JSON or proposes unknown tools, return [].
        """
        available = available or []
        if not available:
            return []
        # DRY-RUN: skip LLM planner so agents fall back to their static plans
        if self._dry_run():
            return []
        # Build tool description hint
        desc_lines: List[str] = []
        for tid in available:
            meta = get_tool(tid) or {"desc": "", "modalities": None, "role": self.role}
            mod = meta.get("modalities")
            mod_txt = f" modalities={','.join(mod)}" if mod else ""
            schema = meta.get("args_schema")
            if schema:
                try:
                    import json as _json
                    schema_txt = _json.dumps(schema)
                except Exception:
                    schema_txt = str(schema)
                desc_lines.append(f"- {tid}:{mod_txt} {meta.get('desc','')}\n  Args(JSON Schema): {schema_txt}")
            else:
                desc_lines.append(f"- {tid}:{mod_txt} {meta.get('desc','')}")
        sys = (
            f"You are the {self.name} ({self.role}) planning tool usage. Use only provided tools. "
            f"Return strictly the JSON schema."
        )
        user = (
            f"Task: {task_desc}\n\nAvailable tools:\n" + "\n".join(desc_lines)
        )
        try:
            llm = JsonLLM(agent_name=self.__class__.__name__)
            # Prefer structured schema
            parsed: ToolPlan = llm.invoke_structured(sys, user, ToolPlan)  # type: ignore[assignment]
            logger.debug(f"[{self.name}] plan_tools structured parsed: {parsed}")
            out: List[Dict[str, Any]] = []
            for step in parsed.plan:
                if step.tool_id in available:
                    out.append({
                        "tool_id": step.tool_id,
                        "arguments": step.arguments,
                        "reasoning": step.reasoning,
                    })
            return out
        except Exception:
            # Fallback to plain JSON
            try:
                schema = '{"plan": [{"tool_id": "...", "arguments": {}, "reasoning": "..."}]}'
                data = llm.invoke_json(system_prompt=sys, user_prompt=user, schema_hint=schema)
                logger.debug(f"[{self.name}] plan_tools json parsed: {data}")
                plan = data.get("plan") if isinstance(data, dict) else None
                out: List[Dict[str, Any]] = []
                if isinstance(plan, list):
                    for step in plan:
                        if not isinstance(step, dict):
                            continue
                        tid = step.get("tool_id")
                        if tid not in available:
                            continue
                        out.append({
                            "tool_id": tid,
                            "arguments": step.get("arguments"),
                            "reasoning": step.get("reasoning")
                        })
                return out
            except Exception:
                return []

    def gen_reasoning(self, context_summary: str, schema_hint: Optional[str] = None) -> str:
        """Generate reasoning/narrative via LLM JSON mode with constrained schema.

        Returns the 'reasoning' field from JSON; falls back to input summary on failure.
        """
        # DRY-RUN: return the summary directly
        if self._dry_run():
            return context_summary
        sys = f"You are the {self.name} ({self.role}). Provide concise medical reasoning."
        user = f"Context summary to explain concisely: {context_summary}"
        try:
            llm = JsonLLM(agent_name=self.__class__.__name__)
            parsed: ReasoningOut = llm.invoke_structured(sys, user, ReasoningOut)  # type: ignore[assignment]
            logger.debug(f"[{self.name}] reasoning structured parsed: {parsed}")
            return parsed.reasoning or parsed.narrative or context_summary
        except Exception:
            # Fallback to plain JSON
            try:
                data = llm.invoke_json(system_prompt=sys, user_prompt=user, schema_hint='{"reasoning": "...", "narrative": "..."}')
                logger.debug(f"[{self.name}] reasoning json parsed: {data}")
                if isinstance(data, dict):
                    return str(data.get("reasoning") or data.get("narrative") or context_summary)
            except Exception:
                pass
            return context_summary

    # ---- LLM bound-tools runtime -------------------------------------------
    async def run_with_bound_tools(self,
                                   messages: List[dict] | None,
                                   allowed_tool_ids: List[str],
                                   system_prompt: Optional[str] = None,
                                   max_steps: int = 6) -> Dict[str, Any]:
        """Drive an LLM that is bound to tools; intercept tool calls and execute via MCP (_call_tool) to keep trace.

        messages: optional list of role/content dicts; if None, a default HumanMessage is built.
        Returns a dict with keys: messages (langchain messages), tool_calls (list), final_response (AI content str).
        """
        # Build chat model using class-level LLM or fallback to settings
        llm = self.__class__.llm or build_chat_model(self.__class__.__name__)

        # Build LangChain tool stubs for tool_calls; execution is handled by _call_tool for tracing
        use_adapter = os.getenv("EYEAGENT_MCP_ADAPTER_BIND", "0").lower() in ("1", "true", "yes")
        logger.debug(f"[{self.name}] bind tools use_adapter={use_adapter} allowed={allowed_tool_ids}")
        # Map between our tool_ids and MCP tool names
        tid_to_mcp = {}
        mcp_to_tid = {}
        for tid in allowed_tool_ids:
            meta = get_tool(tid) or {}
            mcpn = meta.get("mcp_name", tid)
            tid_to_mcp[tid] = mcpn
            mcp_to_tid[mcpn] = tid

        if use_adapter:
            try:
                lc_tools = await load_tools_from_mcp(self.mcp_url, include_ids=allowed_tool_ids)
                logger.debug(f"[{self.name}] adapter tools loaded: {[getattr(t,'name',None) for t in lc_tools]}")
            except Exception as e:
                logger.warning(f"MCP adapter bind failed: {e}; falling back to internal wrappers")
                lc_tools = build_langchain_mcp_tools(allowed_tool_ids, self.mcp_url)
                use_adapter = False
        else:
            lc_tools = build_langchain_mcp_tools(allowed_tool_ids, self.mcp_url)
        logger.debug(f"[{self.name}] binding LLM with {len(lc_tools)} tools")
        model = llm.bind_tools(lc_tools)

        # Convert simple dict messages into LangChain message objects
        lc_messages: List[Any] = []
        if system_prompt:
            lc_messages.append(SystemMessage(content=system_prompt))
        for m in (messages or []):
            role = (m.get("role") or "user").lower()
            content = m.get("content") or ""
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        if not lc_messages:
            lc_messages = [HumanMessage(content="Use the available tools to progress the diagnosis, then summarize.")]

        tool_calls_acc: List[Dict[str, Any]] = []
        for step in range(max_steps):
            logger.debug(f"[{self.name}] step={step} invoking model with {len(lc_messages)} messages")
            resp = model.invoke(lc_messages)
            # Token metrics if available
            try:
                meta = getattr(resp, "response_metadata", {}) or {}
                usage = meta.get("token_usage") or {}
                if usage:
                    if usage.get("prompt_tokens"):
                        add_tokens(self.__class__.__name__, "prompt", int(usage.get("prompt_tokens", 0)))
                    if usage.get("completion_tokens"):
                        add_tokens(self.__class__.__name__, "completion", int(usage.get("completion_tokens", 0)))
                    if usage.get("total_tokens"):
                        add_tokens(self.__class__.__name__, "total", int(usage.get("total_tokens", 0)))
            except Exception:
                pass
            lc_messages.append(resp)
            tcs = getattr(resp, "tool_calls", []) or []
            if not tcs:
                logger.debug(f"[{self.name}] no tool_calls; stopping loop")
                break
            # Execute each tool call via MCP to preserve our trace logging
            for call in tcs:
                name = call.get("name")
                args = call.get("args") or {}
                # Resolve name to our tool_id when using adapter-bound tools
                tool_id = mcp_to_tid.get(name) if use_adapter else name
                # Only execute allowed tools
                if tool_id not in allowed_tool_ids:
                    # Return a tool error message to the model
                    lc_messages.append(ToolMessage(tool_call_id=call.get("id"), name=name or "unknown", args=args, content="{\"error\": \"invalid tool\"}"))
                    continue
                async with self._client_ctx() as client:
                    logger.debug(f"[{self.name}] calling tool tool_id={tool_id} args={args}")
                    tc = await self._call_tool(client, tool_id, args)
                tool_calls_acc.append(tc)
                # Provide the (JSON) result back to the model as ToolMessage
                import json as _json
                content = _json.dumps(tc.get("output")) if tc.get("output") is not None else _json.dumps({"status": tc.get("status")})
                lc_messages.append(ToolMessage(tool_call_id=call.get("id"), name=name, args=args, content=content))

        final_text = ""
        if lc_messages and isinstance(lc_messages[-1], AIMessage):
            final_text = lc_messages[-1].content or ""
        logger.debug(f"[{self.name}] final_response len={len(final_text)} tool_calls={len(tool_calls_acc)}")

        return {"messages": lc_messages, "tool_calls": tool_calls_acc, "final_response": final_text}

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

    def _standardize_output(self, tool_id: str, output: Any, meta: Dict[str, Any] | None = None) -> Any:
        """Standardize tool outputs into a consistent shape while keeping original keys.

        Adds common fields and aligns synonyms:
        - kind: classification | disease_specific | segmentation
        - tool_id: original tool id
        - probabilities: preferred dict for label->prob (copied from all_probabilities when present)
        - label: unified single-label name (copied from prediction when present)
        Keeps original fields to avoid breaking downstream code.
        """
        try:
            if not isinstance(output, dict):
                # Wrap non-dict into a minimal envelope
                return {"kind": None, "tool_id": tool_id, "value": output}
            out = dict(output)  # shallow copy
            kind: Optional[str] = None
            # Infer kind from tool_id prefix or meta
            if isinstance(tool_id, str):
                if tool_id.startswith("classification:"):
                    kind = "classification"
                elif tool_id.startswith("disease_specific_cls:"):
                    kind = "disease_specific"
                elif tool_id.startswith("segmentation:"):
                    kind = "segmentation"
            if not kind and meta and meta.get("role") == "specialist":
                kind = "disease_specific"
            out.setdefault("tool_id", tool_id)
            if kind:
                out.setdefault("kind", kind)

            # Configurable max classes to keep in probabilities (for readability)
            try:
                max_classes = int(os.getenv("EYEAGENT_MAX_CLASSES", "8"))
            except Exception:
                max_classes = 8

            # Harmonize common fields
            # classification: prefer label; accept prediction as alias
            if kind == "classification":
                if "label" not in out and isinstance(out.get("prediction"), str):
                    out["label"] = out.get("prediction")
                # Some classifiers return {probabilities: {...}}; if missing but all_probabilities present, copy
                if "probabilities" not in out and isinstance(out.get("all_probabilities"), dict):
                    out["probabilities"] = dict(out.get("all_probabilities") or {})
                # Compute top-1 prediction if probabilities available
                probs = out.get("probabilities") if isinstance(out.get("probabilities"), dict) else None
                if probs and not out.get("predicted_label"):
                    try:
                        top = max(probs.items(), key=lambda kv: float(kv[1] or 0))
                        out["predicted_label"], out["predicted_prob"] = top[0], float(top[1] or 0)
                    except Exception:
                        pass
                # Trim probabilities to Top-K for compactness (sorted desc)
                if probs and isinstance(probs, dict):
                    try:
                        items = sorted(probs.items(), key=lambda kv: float(kv[1] or 0), reverse=True)[:max_classes]
                        out["probabilities"] = {k: v for k, v in items}
                    except Exception:
                        pass

            # disease specific: copy all_probabilities -> probabilities; ensure probability/predicted
            if kind == "disease_specific":
                if "probabilities" not in out and isinstance(out.get("all_probabilities"), dict):
                    out["probabilities"] = dict(out.get("all_probabilities") or {})
                # Derive probability if only probabilities provided for binary cases
                if "probability" not in out and isinstance(out.get("probabilities"), dict):
                    try:
                        # Heuristic: use max probability as main probability
                        max_prob = max(float(v) for v in out["probabilities"].values() if v is not None)
                        out["probability"] = max_prob
                    except Exception:
                        pass
                if "predicted" not in out and isinstance(out.get("probability"), (int, float)):
                    try:
                        out["predicted"] = bool(float(out.get("probability")) >= 0.5)
                    except Exception:
                        pass
                # Compute predicted_label/predicted_prob from probabilities when present
                probs = out.get("probabilities") if isinstance(out.get("probabilities"), dict) else None
                if probs and not out.get("predicted_label"):
                    try:
                        top = max(probs.items(), key=lambda kv: float(kv[1] or 0))
                        out["predicted_label"], out["predicted_prob"] = top[0], float(top[1] or 0)
                    except Exception:
                        pass
                # Attach disease names (full/abbr) based on tool_id
                try:
                    from ..tools.tool_registry import disease_names_for_tool  # type: ignore
                    names = disease_names_for_tool(tool_id)
                    if names:
                        out.setdefault("disease_names", names)
                        # Ensure top-level disease string is set
                        if not out.get("disease") and names.get("abbr"):
                            out["disease"] = names.get("abbr")
                except Exception:
                    pass
                # Trim probabilities to Top-K
                if probs and isinstance(probs, dict):
                    try:
                        items = sorted(probs.items(), key=lambda kv: float(kv[1] or 0), reverse=True)[:max_classes]
                        out["probabilities"] = {k: v for k, v in items}
                    except Exception:
                        pass

            # segmentation: carry counts/areas/output_paths as-is; attach totals
            if kind == "segmentation":
                counts = out.get("counts") if isinstance(out.get("counts"), dict) else None
                if counts and "total_regions" not in out:
                    try:
                        out["total_regions"] = sum(int(v) for v in counts.values() if v is not None)
                    except Exception:
                        pass

            return out
        except Exception:
            return output

# Backward-compat alias
DiagnosticBaseAgent = BaseAgent
