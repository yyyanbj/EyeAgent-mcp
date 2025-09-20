import asyncio
import os
from typing import Any, Dict

from .core.topologies import run_ring
from .examples.adapters import (
    orchestrator_plan_adapter,
    image_analysis_adapter,
    specialist_adapter,
    followup_adapter,
    orchestrator_summarize_adapter,
)


async def demo_ring(case: Dict[str, Any]) -> Dict[str, Any]:
    state: Dict[str, Any] = {"messages": [], "context": case}
    agents = [
        ("orchestrator", orchestrator_plan_adapter),
        ("image_analysis", image_analysis_adapter),
        ("specialist", specialist_adapter),
        ("follow_up", followup_adapter),
        ("orchestrator_final", orchestrator_summarize_adapter),
    ]

    # Single cycle is enough; stop after final orchestrator summary
    def stop(s: Dict[str, Any]) -> bool:
        return bool(s.get("context", {}).get("final_summary"))

    await run_ring(state, agents, rounds=1, stop_condition=stop)
    return state


def main():
    os.environ.setdefault("EYEAGENT_DRY_RUN", "1")  # safe default for demo
    # Minimal case
    case = {
        "patient": {"id": "P001", "age": 63, "gender": "M"},
        "images": [
            {"image_id": "IMG001", "path": "./examples/sample_cfp.png"}
        ],
    }
    state = asyncio.run(demo_ring(case))
    print("=== Final Summary ===")
    print(state.get("context", {}).get("final_summary"))
    print("=== Messages ===")
    for m in state.get("messages", []):
        print("- ", m)


if __name__ == "__main__":
    main()
