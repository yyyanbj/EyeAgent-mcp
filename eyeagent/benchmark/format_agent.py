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
        "INPUTS: Complete diagnostic workflow outputs including diagnoses, reasoning, and analysis.\n"
        "OUTPUT: Single standardized diagnosis in format 'The diagnosis of this image is [DIAGNOSIS]'.\n"
        "CONSTRAINTS: \n"
        "- Extract the most confident/primary diagnosis\n"
        "- Use exact disease names from the provided class list\n"
        "- If multiple diagnoses exist, choose the one with highest confidence\n"
        "- If no clear diagnosis, output 'The diagnosis of this image is Normal'\n"
        "- Only output the required format, no additional text or explanation"
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
        """Extract and format the primary diagnosis."""
        
        # Get the final diagnostic output
        final_fragment = context.get("final_fragment", {})
        
        # Extract diagnosis using multiple strategies
        diagnosis = self._extract_diagnosis(final_fragment)
        
        # Format the output
        formatted_output = f"The diagnosis of this image is {diagnosis}"
        
        # Create reasoning for transparency
        reasoning = f"Extracted diagnosis '{diagnosis}' from final diagnostic output and formatted for evaluation."
        
        outputs = {
            "formatted_diagnosis": formatted_output,
            "extracted_diagnosis": diagnosis,
            "confidence": self._extract_confidence(final_fragment),
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
    
    def _extract_confidence(self, final_fragment: Dict[str, Any]) -> float:
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
        
        return 0.5  # Default confidence


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
    
    # Create a temporary format agent
    trace_logger = TraceLogger()
    format_agent = FormatAgent("", trace_logger, "temp", class_names)
    
    # Create context with final fragment
    context = {"final_fragment": diagnostic_output}
    
    # Extract diagnosis
    result = format_agent._extract_diagnosis(diagnostic_output)
    
    return f"The diagnosis of this image is {result}"