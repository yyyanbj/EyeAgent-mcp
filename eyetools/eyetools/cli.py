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
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "serve":
        parser.print_help()
        return 1
    app = create_app(
        include_examples=args.include_examples,
        tool_paths=args.tool_paths,
        role_config_path=args.role_config,
        preload=args.preload,
        preload_subprocess=args.preload_subprocess,
    )
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    main()
