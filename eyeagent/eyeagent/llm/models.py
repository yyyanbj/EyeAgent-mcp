from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ToolPlanStep(BaseModel):
    """Single tool call planning step."""
    tool_id: str = Field(description="The tool_id to call, must be one of available tool ids")
    arguments: Optional[Dict[str, Any]] = Field(default=None, description="Arguments for the tool call, or null for defaults")
    reasoning: Optional[str] = Field(default=None, description="Short reason for selecting this tool")

class ToolPlan(BaseModel):
    """Planning result: ordered tool calls."""
    plan: List[ToolPlanStep] = Field(description="Ordered list of tool invocations")

class ReasoningOut(BaseModel):
    """Reasoning/narrative output."""
    reasoning: str = Field(description="Concise reasoning text")
    narrative: Optional[str] = Field(default=None, description="Optional narrative duplication or extended text")
