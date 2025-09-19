from typing import Any, Dict, List
from .diagnostic_base_agent import DiagnosticBaseAgent
from .registry import register_agent
from ..tools.tool_registry import specialist_tools, get_tool, resolve_specialist_tools, role_tool_ids
from ..config.tools_filter import filter_tool_ids
from fastmcp import Client

@register_agent
class SpecialistAgent(DiagnosticBaseAgent):
    role = "specialist"
    name = "SpecialistAgent"
    allowed_tool_ids: List[str] = []  # decided at runtime based on candidate diseases
    system_prompt = (
        "You are the specialist agent. For candidate diseases, call corresponding grading models and return results with reasoning."
    )

    # Capabilities declaration for specialist
    capabilities = {
        "required_context": ["candidate_diseases", "images"],
        "expected_outputs": ["disease_grades", "narrative"],
        "retry_policy": {"max_attempts": 2, "on_fail": "skip"},
        "modalities": ["CFP", "OCT", "FFA"],
        "tools": [],  # dynamic per run
    }

    async def a_run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        candidate_diseases: List[str] = context.get("candidate_diseases", [])
        # Resolve candidate diseases to concrete disease-specific tools (robust mapping)
        tools_meta = resolve_specialist_tools(candidate_diseases)
        # Fallback: dynamic default → allow all specialist tools if none resolved
        if not tools_meta:
            all_specialist_ids = role_tool_ids("specialist")
            # Apply include/exclude filter from config (regex/list)
            filtered_ids = filter_tool_ids(self.__class__.__name__, all_specialist_ids)
            self.allowed_tool_ids = filtered_ids
        else:
            self.allowed_tool_ids = [m.get("tool_id") for m in tools_meta]
        # If still empty, emit a trace note and short-circuit with empty outputs
        if not self.allowed_tool_ids:
            note = {
                "type": "agent_step",
                "agent": self.name,
                "role": self.role,
                "outputs": {"disease_grades": [], "narrative": "No specialist tools matched candidate diseases."},
                "tool_calls": [],
                "reasoning": f"No tools resolved for: {candidate_diseases}"
            }
            self.trace_logger.append_event(self.case_id, note)
            return {"agent": self.name, "role": self.role, "outputs": note["outputs"], "tool_calls": [], "reasoning": note["reasoning"]}

        task_desc = f"For candidate diseases {candidate_diseases}, call grading models one by one."
        plan = await self.plan_tools(task_desc, self.allowed_tool_ids)
        if not plan:
            plan = [
                {"tool_id": tid, "arguments": None, "reasoning": "Run disease-specific grading."}
                for tid in self.allowed_tool_ids
            ]

        images = context.get("images", [])
        img_list = images if images else [None]
        tool_calls: List[Dict[str, Any]] = []
        results_flat: List[Dict[str, Any]] = []
        results_by_image: Dict[str, List[Dict[str, Any]]] = {}

        async with self._client_ctx() as client:
            for step in plan:
                tool_id = step.get("tool_id")
                if tool_id not in self.allowed_tool_ids:
                    continue
                for img in img_list:
                    img_path = (img or {}).get("path") if isinstance(img, dict) else None
                    args = dict(step.get("arguments") or {})
                    if img_path:
                        args["image_path"] = img_path
                    tc = await self._call_tool(client, tool_id, args)
                    tc["reasoning"] = step.get("reasoning")
                    if isinstance(img, dict):
                        tc["image_id"] = img.get("image_id") or img.get("path")
                    tool_calls.append(tc)
                    if tc.get("status") == "success":
                        meta = get_tool(tool_id) or {}
                        output = tc.get("output") or {}
                        entry = {
                            "disease": meta.get("disease") or tool_id.split(":")[-1],
                            "grade": output.get("grade") if isinstance(output, dict) else None,
                            "confidence": output.get("confidence") if isinstance(output, dict) else None,
                            "tool_id": tool_id
                        }
                        if isinstance(img, dict):
                            entry["image_id"] = img.get("image_id") or img.get("path")
                            results_by_image.setdefault(entry["image_id"], []).append(entry)
                        results_flat.append(entry)

        # Narrative summary for specialist grading (per-image aware)
        def _fmt_conf(c):
            try:
                return f"{float(c)*100:.1f}%"
            except Exception:
                return str(c)

        lines: List[str] = []
        if results_by_image:
            for img_id, items in results_by_image.items():
                bits: List[str] = []
                for r in items:
                    d = r.get("disease")
                    g = r.get("grade")
                    if d and g:
                        conf = r.get("confidence")
                        s = f"{d} {g}"
                        if conf is not None:
                            s += f" ({_fmt_conf(conf)})"
                        bits.append(s)
                if bits:
                    lines.append(f"{img_id}: " + ", ".join(bits))
        else:
            # fallback to flat list
            bits: List[str] = []
            for r in results_flat:
                d = r.get("disease")
                g = r.get("grade")
                if d and g:
                    conf = r.get("confidence")
                    s = f"{d} {g}"
                    if conf is not None:
                        s += f" ({_fmt_conf(conf)})"
                    bits.append(s)
            if bits:
                lines.append(", ".join(bits))

        base_summary = ("Specialist grading summary: " + "; ".join(lines)) if lines else "Specialist grading completed."
        reasoning = self.gen_reasoning(base_summary)
        outputs = {
            "disease_grades": results_flat,
            "per_image": {"disease_grades": results_by_image},
            "narrative": reasoning,
        }
        self.trace_logger.append_event(self.case_id, {
            "type": "agent_step",
            "agent": self.name,
            "role": self.role,
            "outputs": outputs,
            "tool_calls": tool_calls,
            "reasoning": reasoning
        })
        return {"agent": self.name, "role": self.role, "outputs": outputs, "tool_calls": tool_calls, "reasoning": reasoning}
