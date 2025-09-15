from eyetools.cli import build_parser

def test_cli_preload_flags_parse():
    parser = build_parser()
    ns = parser.parse_args(["serve", "--preload", "--preload-subprocess"])  # should not raise
    assert ns.preload is True
    assert ns.preload_subprocess is True
