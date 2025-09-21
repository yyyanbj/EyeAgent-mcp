from typing import Any, Dict, List

from .base_agent import BaseAgent as DiagnosticBaseAgent
from .registry import register_agent
from ..config.tools_filter import filter_tool_ids


@register_agent
class KnowledgeAgent(DiagnosticBaseAgent):
    """Knowledge/RAG agent that consults internal documents and optionally web search.

    Inputs: patient context, messages, upstream findings (image_analysis/specialist)
    Tools: rag:query (required), web_search:* (optional)
    Outputs: knowledge_evidence (citations/snippets) and a short narrative.
    """

    role = "knowledge"
    name = "KnowledgeAgent"
    allowed_tool_ids = [
        "rag:query",
        "web_search:pubmed",
        "web_search:tavily",
    ]
    system_prompt = (
        "You are the KnowledgeAgent. Given tentative diagnoses or findings, compose targeted queries, "
        "retrieve relevant evidence from RAG/vector DB, and optionally complement with web search. "
        "Return concise, clinically-relevant evidence summaries and citations."
    )

    capabilities = {
        "required_context": ["patient"],
        "expected_outputs": ["knowledge_evidence", "narrative"],
        "retry_policy": {"max_attempts": 1, "on_fail": "skip"},
        "modalities": ["CFP", "OCT", "FFA"],
        "tools": allowed_tool_ids,
    }

    def _build_query(self, context: Dict[str, Any]) -> str:
        patient = context.get("patient") or {}
        instruction = patient.get("instruction") or ""
        # Prefer diseases from specialist -> image_analysis -> orchestrator screening
        diseases: List[str] = []
        try:
            sp = context.get("specialist") or {}
            for dg in (sp.get("disease_grades") or []):
                d = dg.get("disease")
                if d and d not in diseases:
                    diseases.append(d)
        except Exception:
            pass
        if not diseases:
            ia = context.get("image_analysis") or {}
            probs = ia.get("diseases")
            if isinstance(probs, dict):
                try:
                    diseases = [k for k, _ in sorted(probs.items(), key=lambda kv: float(kv[1] or 0), reverse=True)[:3]]
                except Exception:
                    diseases = list(probs.keys())[:3]
        if not diseases:
            try:
                orch = context.get("orchestrator_outputs") or {}
                scr = (orch.get("screening_results") or [])
                if scr:
                    out = (scr[0] or {}).get("output")
                    if isinstance(out, dict):
                        probs = out.get("probabilities") if isinstance(out.get("probabilities"), dict) else out
                        if isinstance(probs, dict):
                            diseases = list(probs.keys())[:3]
            except Exception:
                pass
        # Construct a focused query
        parts = []
        if diseases:
            parts.append("; ".join(diseases))
        if instruction:
            parts.append(instruction)
        age = patient.get("age")
        if age is not None:
            parts.append(f"age {age}")
        return ", ".join([str(p) for p in parts if p]) or "ophthalmology diagnosis management guidelines"

    async def a_run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        images = context.get("images", [])
        query = self._build_query(context)

        # Prefer RAG; optionally use web search when planner or fallback includes it
        allowed = filter_tool_ids(self.__class__.__name__, list(self.allowed_tool_ids))
        plan = await self.plan_tools(
            f"Query knowledge sources for: {query}",
            allowed,
        )
        if not plan:
            plan = [
                {"tool_id": "rag:query", "arguments": {"query": query, "top_k": 5}, "reasoning": "Retrieve top passages from internal docs."},
                {"tool_id": "web_search:pubmed", "arguments": {"query": query, "top_k": 3}, "reasoning": "Fetch recent PubMed references (optional)."},
            ]

        tool_calls: List[Dict[str, Any]] = []
        evidence_blocks: List[Dict[str, Any]] = []

        async with self._client_ctx() as client:
            for step in plan:
                tid = step.get("tool_id")
                if tid not in allowed:
                    continue
                args = dict(step.get("arguments") or {})
                # Attach image-derived hints only if the tool supports it (we keep args simple to avoid schema mismatch)
                tc = await self._call_tool(client, tid, args)
                tc["reasoning"] = step.get("reasoning")
                tool_calls.append(tc)
                out = tc.get("output")
                if isinstance(out, dict):
                    evidence_blocks.append(out)
                elif isinstance(out, list):
                    evidence_blocks.extend([{"items": out}])

        # Build a concise narrative
        # Prefer titles/snippets from evidence blocks
        bullets: List[str] = []
        for blk in evidence_blocks:
            items = []
            if isinstance(blk, dict):
                items = blk.get("items") or blk.get("documents") or blk.get("results") or []
            if isinstance(items, list):
                for it in items[:3]:
                    if isinstance(it, dict):
                        title = it.get("title") or it.get("id") or it.get("source")
                        if title:
                            bullets.append(str(title))
        base_summary = (
            f"Knowledge evidence for query '{query}': " + "; ".join(bullets[:5])
        ).strip()
        narrative = self.gen_reasoning(base_summary)

        outputs = {
            "query": query,
            "knowledge_evidence": evidence_blocks,
            "narrative": narrative,
        }

        self.trace_logger.append_event(self.case_id, {
            "type": "agent_step",
            "agent": self.name,
            "role": self.role,
            "outputs": outputs,
            "tool_calls": tool_calls,
            "reasoning": narrative,
        })

        return {"agent": self.name, "role": self.role, "outputs": outputs, "tool_calls": tool_calls, "reasoning": narrative}
