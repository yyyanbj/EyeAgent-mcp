from __future__ import annotations
import argparse
from typing import Any, Dict

from .core.config_loader import load_config
from .core.trace_logger import TraceLogger
from .core.registry import create_agent


def main():
    parser = argparse.ArgumentParser(description="Run EyeAgent pipeline from config")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    args = parser.parse_args()

    cfg: Dict[str, Any] = load_config(args.config)

    # runtime and cases dir
    base_dir = ((cfg.get("runtime") or {}).get("cases_dir"))
    trace = TraceLogger(base_dir=base_dir)
    case_id = trace.create_case(patient=cfg.get("patient") or {}, images=cfg.get("images") or [])

    context: Dict[str, Any] = {
        "case_id": case_id,
        "input": (cfg.get("input") or ""),
        "config_path": args.config,
    }

    # Orchestrator orchestrates sub agents per config
    orch_cfg = (cfg.get("orchestrator") or {})
    orch = create_agent("orchestrator", config=orch_cfg, trace=trace)
    context = orch.run(context)

    print(f"case_id={case_id}")
    final = context.get("final_report")
    if final:
        print("final_report:")
        print(final)


if __name__ == "__main__":
    main()
