"""CLI entrypoint for eyetools MCP server."""
from __future__ import annotations
import argparse
import uvicorn
from .mcp_server import create_app


def build_parser():
    p = argparse.ArgumentParser(prog="eyetools-mcp", description="Eytools MCP Server")
    sub = p.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="Start MCP server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    # Backward/alternative flags: --tool-path (repeatable) and --tools-dir (single or repeatable)
    serve.add_argument(
        "--tool-path",
        action="append",
        dest="tool_paths",
        default=[],
        help="Additional tool root path (repeatable). Use --tools-dir as an alias.",
    )
    serve.add_argument(
        "--tools-dir",
        action="append",
        dest="tool_paths",
        help="Alias for --tool-path (repeatable)",
    )
    serve.add_argument("--include-examples", action="store_true", help="Include packaged examples")
    serve.add_argument("--role-config", help="Path to role router config (YAML/JSON)")
    serve.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    serve.add_argument("--preload", action="store_true", help="Instantiate and load (lazy) models at startup")
    serve.add_argument("--preload-subprocess", action="store_true", help="Also preload subprocess tools (if any)")
    serve.add_argument(
        "--log-dir",
        help="Directory to write log file (eyetools.log). If not set, only stdout is used.",
    )
    # New lifecycle modes: eager (all models upfront), lazy (on-demand), dynamic (idle eviction)
    serve.add_argument(
        "--lifecycle-mode",
        choices=["eager", "lazy", "dynamic"],
        help="Model/tool lifecycle strategy: eager=preload all (+subprocess optional), lazy=load on first use, dynamic=lazy plus auto idle/unload",
    )
    serve.add_argument(
        "--dynamic-mark-idle-s",
        type=float,
        default=300.0,
        help="(dynamic mode) seconds of inactivity before marking a tool IDLE",
    )
    serve.add_argument(
        "--dynamic-unload-s",
        type=float,
        default=900.0,
        help="(dynamic mode) seconds of inactivity before unloading an IDLE tool",
    )
    serve.add_argument(
        "--dynamic-interval-s",
        type=float,
        default=60.0,
        help="(dynamic mode) maintenance loop interval in seconds",
    )
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "serve":
        parser.print_help()
        return 1
    # Setup log directory env before app creation (logging module reads env once)
    import os, pathlib
    if getattr(args, "log_dir", None):
        log_dir_path = pathlib.Path(args.log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("EYETOOLS_LOG_DIR", str(log_dir_path.resolve()))
    app = create_app(
        include_examples=args.include_examples,
        tool_paths=args.tool_paths,
        role_config_path=args.role_config,
        preload=args.preload,
        preload_subprocess=args.preload_subprocess,
        lifecycle_mode=getattr(args, "lifecycle_mode", None),
        dynamic_mark_idle_s=getattr(args, "dynamic_mark_idle_s", 300.0),
        dynamic_unload_s=getattr(args, "dynamic_unload_s", 900.0),
        dynamic_interval_s=getattr(args, "dynamic_interval_s", 60.0),
    )
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    main()
