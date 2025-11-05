from __future__ import annotations
import argparse
import os
from typing import Optional
from .settings import set_overrides


def add_common_runtime_flags(
    parser: argparse.ArgumentParser,
    *,
    include_ui: bool = False,
    include_run_io: bool = False,
) -> None:
    """Add a unified set of runtime flags shared by UI and CLI.

    This avoids drift between entrypoints and ensures a single mapping to centralized settings (no env vars).

    Parameters:
      parser: target ArgumentParser to extend
      include_ui: include UI-only flags (port, queue tuning)
      include_run_io: include headless run I/O flags used by CLI (patient/images)
    """

    # Config source
    parser.add_argument("--config", dest="config", type=str, help="Path to eyeagent.yml (override config file)")
    parser.add_argument("--config-dir", dest="config_dir", type=str, help="Directory containing eyeagent.yml (override config directory)")

    # Core runtime toggles (shared)
    parser.add_argument("--backend", dest="backend", choices=["langgraph", "profile", "interaction", "single", "topology"], help="Workflow backend (override)")
    parser.add_argument("--profile", dest="profile", type=str, help="Pipeline profile name (override)")
    parser.add_argument("--routing-strategy", dest="routing_strategy", choices=["llm", "config"], help="Orchestrator routing strategy (override)")
    parser.add_argument("--mcp-adapter-bind", dest="mcp_adapter_bind", action="store_true", help="Bind MCP tools via adapter (override)")
    parser.add_argument("--max-classes", dest="max_classes", type=int, help="Max classes retained in probability outputs (override)")

    # Data and cases roots
    parser.add_argument("--cases-dir", dest="cases_dir", type=str, help="Cases directory (override)")
    parser.add_argument("--data-dir", dest="data_dir", type=str, help="Data directory (override)")

    # Optional MCP URL for adapters
    parser.add_argument("--mcp-url", dest="mcp_url", type=str, help="MCP server base URL (override)")

    # Topology backend tuning
    parser.add_argument("--topology", dest="topology", choices=["star", "ring"], help="Topology when backend=topology (override)")
    parser.add_argument("--topology-agents", dest="topology_agents", type=str, help="Comma-separated agent roles for topology (override)")
    parser.add_argument("--topology-rounds", dest="topology_rounds", type=int, help="Number of rounds for ring topology (override)")

    if include_ui:
        parser.add_argument("--port", dest="port", type=int, default=None, help="UI server port (default 7860)")
        parser.add_argument("--ui-concurrency", dest="ui_concurrency", type=int, help="Gradio queue concurrency (override)")
        parser.add_argument("--ui-queue-size", dest="ui_queue_size", type=int, help="Gradio queue max size (override)")
        parser.add_argument("--ui-status-rate", dest="ui_status_rate", type=float, help="Gradio queue status update rate (override)")

    if include_run_io:
        # These are CLI-run convenience flags; they are not mapped to env directly
        parser.add_argument("--patient", type=str, help="Patient JSON string (CLI only)")
        parser.add_argument("--patient-file", type=str, help="Path to JSON file with patient object (CLI only)")
        parser.add_argument("--images", type=str, help="Images JSON array (CLI only)")
        parser.add_argument("--images-file", type=str, help="Path to JSON file containing images array (CLI only)")


def apply_runtime_env_from_args(args: argparse.Namespace) -> None:
    """Apply unified runtime options without mutating environment variables.

    Stores process-local overrides in settings so other modules can resolve
    configuration deterministically during this run.
    """
    set_overrides(
        config_file=getattr(args, "config", None) or None,
        config_dir=getattr(args, "config_dir", None) or None,
        workflow_backend=getattr(args, "backend", None) or None,
        pipeline_profile=getattr(args, "profile", None) or None,
        mcp_server_url=getattr(args, "mcp_url", None) or None,
        routing_strategy=getattr(args, "routing_strategy", None) or None,
        mcp_adapter_bind=getattr(args, "mcp_adapter_bind", None) or None,
        max_classes=getattr(args, "max_classes", None) or None,
        cases_dir=getattr(args, "cases_dir", None) or None,
        data_dir=getattr(args, "data_dir", None) or None,
    )
