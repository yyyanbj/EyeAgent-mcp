"""
Format agent for EyeAgent benchmark module.

This agent standardizes the output of the multiagent diagnostic workflow
to a consistent format suitable for evaluation: "The diagnosis of this image is XXX"
"""

from typing import Any, Dict, List, Optional
import os
import re
import json
from loguru import logger

from eyeagent.agents.base_agent import BaseAgent
from eyeagent.agents.registry import register_agent


@register_agent
class FormatAgent(BaseAgent):
    """Agent that formats diagnostic outputs for benchmark evaluation."""
    
    role = "format"
    name = "FormatAgent"
    llm_config_name = "ReportAgent"
    allowed_tool_ids: List[str] = []  # No tools required
    
    system_prompt = (
        "ROLE: Diagnostic output formatter for benchmark evaluation.\n"
        "GOAL: Extract and standardize the final diagnosis from agent outputs.\n"
        "INPUTS: Complete diagnostic workflow outputs including diagnoses, reasoning, analysis.\n"
        "OUTPUT: Exactly one line: 'The diagnosis of this image is [DIAGNOSIS]'.\n"
        "HARD CONSTRAINTS (strict):\n"
        "1) [DIAGNOSIS] MUST be chosen from the provided VALID DIAGNOSES list (exact token match).\n"
        "2) Output MUST contain no extra text, punctuation, JSON, quotes, or explanations.\n"
        "3) If no class fits, choose 'Normal' (must be in the list).\n"
        "4) Prefer the most confident/primary diagnosis if multiple are present.\n"
        "5) If upstream outputs are negative for a disease family (e.g., 'no diabetic retinopathy'), map to 'Normal' unless a positive class from the VALID DIAGNOSES is clearly indicated.\n"
        "EXAMPLES (format only):\n"
        "The diagnosis of this image is Normal\n"
        "The diagnosis of this image is central serous retinopathy\n"
        "DO NOT: output probabilities, multiple labels, explanations, or non-listed synonyms."
    )
    
    capabilities = {
        "required_context": ["final_fragment"],
        "expected_outputs": ["formatted_diagnosis"],
        "retry_policy": {"max_attempts": 1, "on_fail": "default"},
        "modalities": ["CFP", "OCT", "FFA"],
        "tools": [],
    }
    
    def __init__(self, mcp_url: str, trace_logger, case_id: str, class_names: Optional[List[str]] = None):
        super().__init__(mcp_url, trace_logger, case_id)
        self.class_names = class_names or ["Normal", "DR", "AMD", "Glaucoma"]
        
        # Update system prompt with class names
        class_list = ", ".join(self.class_names)
        self.system_prompt += f"\nVALID DIAGNOSES: {class_list}"
    
    async def a_run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Always use LLM over conversation history to generate single-label conclusion."""

        # Collect final fragment and conversation transcript
        final_fragment = context.get("final_fragment", {})
        convo_text = self._load_conversation_text_safe()
        # Also collect a compact summary of tool calls from trace.json, so LLM can see raw tool outputs
        tools_text, tools_struct = self._load_tools_summary_safe()
        try:
            ff_snippet = json.dumps(final_fragment, ensure_ascii=False)
        except Exception:
            ff_snippet = str(final_fragment)

        # Prepare VALID list and emit debug inputs (system prompt + messages used)
        valid = ", ".join(self.class_names)
        self._emit_debug_inputs(convo_text=convo_text, final_fragment_snippet=ff_snippet, valid_list=valid, tools_summary=tools_text)

        # Dry-run or LLM missing: deterministically return Normal without heuristic extraction
        if os.getenv("EYEAGENT_DRY_RUN", "0").lower() in ("1", "true", "yes") or self.llm is None:
            diagnosis = self._clamp_to_valid("Normal")
            formatted_output = f"The diagnosis of this image is {diagnosis}"
            reasoning = "Dry-run/LLM-unavailable; returned 'Normal' without heuristic extraction (LLM-only policy)."
            outputs = {
                "formatted_diagnosis": formatted_output,
                "extracted_diagnosis": diagnosis,
                "confidence": 0.5,
                "reasoning": reasoning,
            }
            self.trace_logger.append_event(self.case_id, {
                "type": "agent_step",
                "agent": self.name,
                "role": self.role,
                "outputs": outputs,
                "tool_calls": [],
                "reasoning": reasoning,
            })
            return {"agent": self.name, "role": self.role, "outputs": outputs, "tool_calls": [], "reasoning": reasoning}

        prompt = (
            self.system_prompt
            + "\n\nVALID DIAGNOSES: " + valid
            + "\n\nConversation history (most recent last):\n" + (convo_text or "<empty>")
            + "\n\nFinal fragment JSON:\n" + ff_snippet
            + "\n\nTool calls summary (compact):\n" + (tools_text or "<none>")
            + "\n\nAnswer with exactly one line as specified."
        )

        try:
            response = self.llm.invoke(prompt)
            raw = getattr(response, "content", None) if response is not None else None
            line = (raw or "").strip().splitlines()[0] if raw else ""
        except Exception as e:
            logger.warning(f"LLM inference failed in FormatAgent: {e}")
            line = "The diagnosis of this image is Normal"

        # Parse and clamp to valid class list
        extracted = self._parse_output_line(line)
        diagnosis = self._clamp_to_valid(extracted)
        formatted_output = f"The diagnosis of this image is {diagnosis}"
        reasoning = "LLM-based synthesis over conversation and final fragment; output clamped to valid classes."
        
        outputs = {
            "formatted_diagnosis": formatted_output,
            "extracted_diagnosis": diagnosis,
            # Prefer confidence extracted from tool calls if available
            "confidence": self._extract_confidence(final_fragment, tools_struct),
            "reasoning": reasoning
        }
        
        # Log the formatting event
        self.trace_logger.append_event(self.case_id, {
            "type": "agent_step",
            "agent": self.name,
            "role": self.role,
            "outputs": outputs,
            "tool_calls": [],
            "reasoning": reasoning,
        })
        
        return {
            "agent": self.name,
            "role": self.role,
            "outputs": outputs,
            "tool_calls": [],
            "reasoning": reasoning
        }

    def _load_conversation_text_safe(self) -> str:
        """Load conversation.jsonl to a compact transcript string."""
        try:
            from eyeagent.tracing.trace_logger import TraceLogger  # for type reference only
            path = self.trace_logger.get_conversation_path(self.case_id)
            if not path or not os.path.exists(path):
                return ""
            lines: List[str] = []
            with open(path, "r", encoding="utf-8") as f:
                for i, ln in enumerate(f):
                    if i > 4000:
                        break
                    try:
                        rec = json.loads(ln.strip())
                        role = rec.get("role") or "assistant"
                        content = str(rec.get("content") or "").strip()
                        if content:
                            lines.append(f"[{role}] {content}")
                    except Exception:
                        continue
            transcript = "\n".join(lines)
            return transcript[-40000:]
        except Exception:
            return ""

    def _load_tools_summary_safe(self) -> tuple[str, Dict[str, Any]]:
        """Load recent tool_calls from trace.json and build a compact human-readable summary and a structured dict.

        Returns: (summary_text, structured_summary)
        structured_summary keys:
          - modality: Optional[str]
          - laterality: Optional[str]
          - screening: Dict[str, float]  # e.g., classification:multidis probabilities (top few)
          - specialist: List[Dict[str, Any]]  # per call: {tool_id, image_id, predicted_label|grade, predicted_prob|confidence, disease}
          - misc: List[Dict[str, Any]]  # other notable tool outputs
        """
        try:
            from eyeagent.tracing.trace_logger import TraceLogger  # import here to avoid cycles at import time
            doc: Dict[str, Any] = self.trace_logger.load_trace(self.case_id)  # type: ignore[attr-defined]
        except Exception:
            return "", {}

        events = list(doc.get("events") or [])
        tool_calls: List[Dict[str, Any]] = []
        # Prefer the tool_calls array attached to the latest agent_step from UnifiedAgent (or any agent_step)
        for ev in reversed(events):
            if isinstance(ev, dict) and ev.get("type") == "agent_step":
                arr = ev.get("tool_calls")
                if isinstance(arr, list) and arr:
                    tool_calls = arr
                    break
        # Fallback: collect the last ~200 explicit tool_call events
        if not tool_calls:
            for ev in reversed(events):
                if isinstance(ev, dict) and ev.get("type") == "tool_call":
                    # Normalize format to match agent_step.tool_calls entries
                    tc = {
                        "tool_id": ev.get("tool_id"),
                        "arguments": ev.get("arguments"),
                        "output": ev.get("output"),
                        "status": ev.get("status"),
                        "version": ev.get("version"),
                        "mcp_meta": ev.get("mcp_meta"),
                        "image_id": (ev.get("arguments") or {}).get("image_path"),
                    }
                    tool_calls.append(tc)
                    if len(tool_calls) >= 200:
                        break
            tool_calls.reverse()

        # Build structured summary
        summary: Dict[str, Any] = {"modality": None, "laterality": None, "screening": {}, "specialist": [], "misc": []}
        def _flt(x: Any) -> Optional[float]:
            try:
                return float(x)
            except Exception:
                return None
        # Collate
        for tc in tool_calls:
            tid = tc.get("tool_id")
            out = tc.get("output") or {}
            if not isinstance(tid, str):
                continue
            if tid == "classification:modality" and isinstance(out, dict):
                lab = out.get("label") or out.get("prediction")
                if isinstance(lab, str):
                    summary["modality"] = lab
                continue
            if tid == "classification:laterality" and isinstance(out, dict):
                lab = out.get("label") or out.get("prediction")
                if isinstance(lab, str):
                    summary["laterality"] = lab
                continue
            if tid == "classification:multidis" and isinstance(out, dict):
                probs = out.get("probabilities") if isinstance(out.get("probabilities"), dict) else out
                if isinstance(probs, dict):
                    # take top 8
                    try:
                        items = sorted([(k, float(v)) for k, v in probs.items() if v is not None], key=lambda kv: kv[1], reverse=True)[:8]
                        summary["screening"] = {k: v for k, v in items}
                    except Exception:
                        pass
                continue
            if tid.startswith("disease_specific_cls:") and isinstance(out, dict):
                disease_key = tid.split(":", 1)[-1]
                pred_label = out.get("predicted_label") or out.get("label") or out.get("prediction") or out.get("grade")
                prob = _flt(out.get("predicted_prob"))
                if prob is None:
                    # Optionally derive from probabilities map
                    probs_map = out.get("probabilities")
                    if isinstance(probs_map, dict):
                        try:
                            # choose positive class if available
                            pos = None
                            if pred_label and pred_label in probs_map:
                                pos = _flt(probs_map.get(pred_label))
                            if pos is None:
                                pos = max((_flt(v) or 0.0) for v in probs_map.values())
                            prob = float(pos)
                        except Exception:
                            prob = None
                summary["specialist"].append({
                    "tool_id": tid,
                    "image_id": tc.get("image_id"),
                    "disease": disease_key,
                    "predicted_label": pred_label,
                    "predicted_prob": prob,
                })
                continue
            # capture a few other outputs that may be useful
            if tid.startswith("classification:") and isinstance(out, dict):
                summary["misc"].append({"tool": tid, "output": {k: out.get(k) for k in ("prediction", "label", "unit") if k in out}})

        # Turn into a readable text block
        lines: List[str] = []
        if summary.get("modality"):
            lines.append(f"modality: {summary['modality']}")
        if summary.get("laterality"):
            lines.append(f"laterality: {summary['laterality']}")
        if isinstance(summary.get("screening"), dict) and summary["screening"]:
            # e.g. DR:0.92, AMD:0.88
            top_line = ", ".join([f"{k}:{v:.3f}" for k, v in summary["screening"].items()])
            lines.append(f"screening(top): {top_line}")
        # specialist compact lines
        sp: List[Dict[str, Any]] = summary.get("specialist") or []
        # sort by prob desc if available
        try:
            sp = sorted(sp, key=lambda d: (d.get("predicted_prob") is not None, d.get("predicted_prob") or 0.0), reverse=True)
        except Exception:
            pass
        for ent in sp[:32]:  # cap lines
            d = ent.get("disease")
            pl = ent.get("predicted_label")
            pp = ent.get("predicted_prob")
            if pp is not None and pl:
                lines.append(f"{d}: {pl} ({pp:.3f})")
            elif pp is not None:
                lines.append(f"{d}: {pp:.3f}")
            elif pl:
                lines.append(f"{d}: {pl}")
            else:
                lines.append(f"{d}")
        text = "\n".join(lines)
        # Trim overly long text
        if len(text) > 4000:
            text = text[:4000] + "..."
        return text, summary
        

    def _parse_output_line(self, line: str) -> str:
        """Extract label from 'The diagnosis of this image is X' or use line content."""
        if not isinstance(line, str):
            return "Normal"
        m = re.search(r"The diagnosis of this image is\s+(.+)$", line.strip(), re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return line.strip().strip("\"'")

    def _clamp_to_valid(self, diagnosis: str) -> str:
        """Map arbitrary string to nearest valid class name; default Normal."""
        if not isinstance(diagnosis, str) or not self.class_names:
            return "Normal"
        guess = diagnosis.strip()
        # exact
        for c in self.class_names:
            if guess == c:
                return c
        # case-insensitive exact
        for c in self.class_names:
            if guess.lower() == c.lower():
                return c
        # substring match (prefer non-Normal)
        non_norm = [c for c in self.class_names if c.lower() != "normal"]
        for c in non_norm:
            if guess.lower() in c.lower() or c.lower() in guess.lower():
                return c
        return "Normal"
    
    def _extract_diagnosis(self, final_fragment: Dict[str, Any]) -> str:
        """Extract the primary diagnosis from the final diagnostic output."""
        
        # Strategy 1: Look for explicit diagnosis field
        if "diagnoses" in final_fragment:
            diagnoses = final_fragment["diagnoses"]
            if isinstance(diagnoses, list) and diagnoses:
                primary = diagnoses[0]
                if isinstance(primary, dict):
                    return self._normalize_diagnosis(primary.get("disease", primary.get("name", "Normal")))
                else:
                    return self._normalize_diagnosis(str(primary))
            elif isinstance(diagnoses, dict):
                # Get highest probability diagnosis
                return self._get_highest_prob_diagnosis(diagnoses)
        
        # Strategy 2: Look for conclusion or summary
        if "conclusion" in final_fragment:
            conclusion = final_fragment["conclusion"]
            if isinstance(conclusion, str):
                extracted = self._extract_from_text(conclusion)
                if extracted:
                    return extracted
        
        # Strategy 3: Look for specialist outputs
        if "specialist" in final_fragment:
            specialist = final_fragment["specialist"]
            if isinstance(specialist, dict):
                disease = specialist.get("disease")
                if disease:
                    return self._normalize_diagnosis(disease)
        
        # Strategy 4: Look for image analysis outputs
        if "image_analysis" in final_fragment:
            analysis = final_fragment["image_analysis"]
            if isinstance(analysis, dict):
                findings = analysis.get("findings", {})
                if isinstance(findings, dict):
                    # Look for highest probability finding
                    return self._get_highest_prob_diagnosis(findings)
        
        # Strategy 5: Search through narrative text
        narrative = final_fragment.get("narrative", "")
        if isinstance(narrative, str):
            extracted = self._extract_from_text(narrative)
            if extracted:
                return extracted
        
        # Strategy 6: Use LLM to extract diagnosis from complex output
        if final_fragment:
            return self._llm_extract_diagnosis(final_fragment)
        
        # Default fallback
        logger.warning("Could not extract diagnosis from final output, defaulting to Normal")
        return "Normal"
    
    def _normalize_diagnosis(self, diagnosis: str) -> str:
        """Normalize diagnosis to match class names."""
        if not isinstance(diagnosis, str):
            return "Normal"
        
        diagnosis = diagnosis.strip()
        
        # Direct match
        if diagnosis in self.class_names:
            return diagnosis
        
        # Case insensitive match
        for class_name in self.class_names:
            if diagnosis.lower() == class_name.lower():
                return class_name
        
        # Partial match
        for class_name in self.class_names:
            if diagnosis.lower() in class_name.lower() or class_name.lower() in diagnosis.lower():
                return class_name
        
        # Disease name mappings
        mappings = {
            "diabetic retinopathy": "DR",
            "age-related macular degeneration": "AMD", 
            "macular degeneration": "AMD",
            "normal": "Normal",
            "healthy": "Normal",
            "no pathology": "Normal"
        }
        
        diagnosis_lower = diagnosis.lower()
        for key, value in mappings.items():
            if key in diagnosis_lower and value in self.class_names:
                return value
        
        return "Normal"
    
    def _get_highest_prob_diagnosis(self, prob_dict: Dict[str, Any]) -> str:
        """Get diagnosis with highest probability from probability dictionary."""
        if not isinstance(prob_dict, dict):
            return "Normal"
        
        max_prob = 0.0
        best_diagnosis = "Normal"
        
        for disease, prob in prob_dict.items():
            try:
                prob_val = float(prob)
                if prob_val > max_prob:
                    max_prob = prob_val
                    best_diagnosis = self._normalize_diagnosis(disease)
            except (ValueError, TypeError):
                continue
        
        return best_diagnosis
    
    def _extract_from_text(self, text: str) -> Optional[str]:
        """Extract diagnosis from free text using patterns."""
        if not isinstance(text, str):
            return None
        
        # Pattern 1: "diagnosis is X" or "diagnosed with X"
        patterns = [
            r"diagnosis is ([^.,:;]+)",
            r"diagnosed with ([^.,:;]+)",
            r"condition is ([^.,:;]+)", 
            r"shows signs of ([^.,:;]+)",
            r"consistent with ([^.,:;]+)",
            r"suggests ([^.,:;]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                diagnosis = match.group(1).strip()
                normalized = self._normalize_diagnosis(diagnosis)
                if normalized != "Normal":  # Only return if we found a valid disease
                    return normalized
        
        # Pattern 2: Look for class names directly mentioned
        text_lower = text.lower()
        for class_name in self.class_names:
            if class_name.lower() in text_lower and class_name != "Normal":
                return class_name
        
        return None
    
    def _llm_extract_diagnosis(self, final_fragment: Dict[str, Any]) -> str:
        """Use LLM to extract diagnosis from complex output."""
        try:
            if os.getenv("EYEAGENT_DRY_RUN", "0").lower() in ("1", "true", "yes"):
                return "Normal"
            if self.llm is None:
                return "Normal"
            
            prompt = f"""
            Extract the primary diagnosis from this medical diagnostic output.
            
            Available diagnoses: {', '.join(self.class_names)}
            
            Diagnostic output:
            {json.dumps(final_fragment, indent=2)}
            
            Return only the diagnosis name from the available list, or "Normal" if no pathology is detected.
            """
            
            response = self.llm.invoke(prompt)
            if hasattr(response, 'content'):
                extracted = response.content.strip()
                return self._normalize_diagnosis(extracted)
            
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
        
        return "Normal"
    
    def _extract_confidence(self, final_fragment: Dict[str, Any], tools_struct: Optional[Dict[str, Any]] = None) -> float:
        """Extract confidence score if available."""
        # Look for confidence in various places
        confidence_fields = ["confidence", "probability", "score"]
        
        for field in confidence_fields:
            if field in final_fragment:
                try:
                    return float(final_fragment[field])
                except (ValueError, TypeError):
                    continue
        
        # Look in diagnoses
        if "diagnoses" in final_fragment:
            diagnoses = final_fragment["diagnoses"]
            if isinstance(diagnoses, list) and diagnoses:
                primary = diagnoses[0]
                if isinstance(primary, dict):
                    for field in confidence_fields:
                        if field in primary:
                            try:
                                return float(primary[field])
                            except (ValueError, TypeError):
                                continue
        # Otherwise, consider tool_calls summary (e.g., highest predicted_prob among specialist outputs)
        try:
            sp = ((tools_struct or {}).get("specialist") or [])
            vals = [float(x.get("predicted_prob")) for x in sp if x.get("predicted_prob") is not None]
            if vals:
                return max(vals)
        except Exception:
            pass
        return 0.5  # Default confidence

    def _emit_debug_inputs(self, convo_text: str, final_fragment_snippet: str, valid_list: str, tools_summary: Optional[str] = None) -> None:
        """Log the FormatAgent system prompt and the messages it reads for transparency."""
        try:
            # Build full system prompt string
            sys_prompt_full = f"{self.system_prompt}\n\nVALID DIAGNOSES: {valid_list}"
            # Limit sizes to avoid huge logs
            convo_excerpt = (convo_text or "").strip()
            if len(convo_excerpt) > 8000:
                convo_excerpt = "..." + convo_excerpt[-8000:]
            ff_preview = (final_fragment_snippet or "").strip()
            if len(ff_preview) > 4000:
                ff_preview = ff_preview[:4000] + "..."
            tools_preview = (tools_summary or "").strip()
            if len(tools_preview) > 4000:
                tools_preview = tools_preview[:4000] + "..."

            # Resolve conversation file path if available
            conv_path = None
            try:
                conv_path = self.trace_logger.get_conversation_path(self.case_id)
            except Exception:
                conv_path = None

            # Log via logger
            logger.info(
                "[FormatAgent Debug] case_id={}\nSystem Prompt:\n{}\n\nConversation file: {}\nConversation excerpt (tail):\n{}\n\nFinal fragment preview:\n{}\n\nTool calls summary (compact):\n{}",
                self.case_id,
                sys_prompt_full,
                conv_path or "<unknown>",
                convo_excerpt,
                ff_preview,
                tools_preview,
            )

            # Persist into trace events
            self.trace_logger.append_event(self.case_id, {
                "type": "format_agent_debug",
                "agent": self.name,
                "system_prompt": sys_prompt_full,
                "conversation_file": conv_path,
                "conversation_excerpt_tail": convo_excerpt,
                "final_fragment_preview": ff_preview,
                "tools_summary_preview": tools_preview,
            })
        except Exception as e:
            logger.warning(f"Failed to emit FormatAgent debug inputs: {e}")


def format_diagnosis_for_evaluation(diagnostic_output: Dict[str, Any], 
                                   class_names: Optional[List[str]] = None) -> str:
    """
    Standalone function to format diagnosis output for evaluation.
    
    Args:
        diagnostic_output: The complete diagnostic workflow output
        class_names: List of valid class names
        
    Returns:
        Formatted diagnosis string
    """
    from eyeagent.tracing.trace_logger import TraceLogger
    
    # Create a temporary format agent (no conversation available here)
    trace_logger = TraceLogger()
    fmt = FormatAgent("", trace_logger, "temp", class_names)
    # LLM path if available; else Normal
    if os.getenv("EYEAGENT_DRY_RUN", "0").lower() in ("1", "true", "yes") or fmt.llm is None:
        diag = fmt._clamp_to_valid("Normal")
        return f"The diagnosis of this image is {diag}"
    try:
        valid = ", ".join(fmt.class_names)
        ff = json.dumps(diagnostic_output, ensure_ascii=False)
        prompt = (
            fmt.system_prompt
            + "\n\nVALID DIAGNOSES: " + valid
            + "\n\nFinal fragment JSON:\n" + ff
            + "\n\nAnswer with exactly one line as specified."
        )
        resp = fmt.llm.invoke(prompt)
        raw = getattr(resp, "content", None) if resp is not None else None
        line = (raw or "").strip().splitlines()[0] if raw else ""
        extracted = fmt._parse_output_line(line)
        diag = fmt._clamp_to_valid(extracted)
        return f"The diagnosis of this image is {diag}"
    except Exception:
        return "The diagnosis of this image is Normal"