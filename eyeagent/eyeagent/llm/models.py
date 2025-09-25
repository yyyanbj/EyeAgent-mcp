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


class RoutingDecision(BaseModel):
    """LLM routing decision for the Orchestrator.

    Must provide the planned pipeline and the immediate next agent.
    """
    planned_pipeline: List[str] = Field(
        description=(
            "Ordered list of agent roles to execute. Allowed roles only: "
            "['preliminary','image_analysis','specialist','knowledge','follow_up','report']. "
            "Include 'report' exactly once at the end."
        )
    )
    next_agent: str = Field(
        description=(
            "The next agent to run now. Must be one of the allowed roles and ideally the first incomplete step in planned_pipeline."
        )
    )
    routing_reasons: Optional[List[str]] = Field(
        default=None,
        description="Short bullet reasons for key choices (e.g., why skipping specialist or starting with preliminary)."
    )
