from eyetools.cli import build_parser

def test_cli_preload_flags_parse():
    parser = build_parser()
    ns = parser.parse_args(["serve", "--preload", "--preload-subprocess"])  # should not raise
    assert ns.preload is True
    assert ns.preload_subprocess is True


def test_cli_preload_without_lifecycle_mode(monkeypatch):
    # simulate invoking create_app via parsed args to ensure legacy flags still pass through
    parser = build_parser()
    ns = parser.parse_args(["serve", "--preload", "--preload-subprocess", "--include-examples"])  # no lifecycle flag
    from eyetools.mcp_server import create_app
    app = create_app(include_examples=ns.include_examples, preload=ns.preload, preload_subprocess=ns.preload_subprocess)
    tm = app.state.tool_manager
    # expect at least one inproc tool loaded due to preload
    counts = tm.resource_snapshot().get('counts', {})
    assert counts.get('loaded_inproc', 0) >= 1
