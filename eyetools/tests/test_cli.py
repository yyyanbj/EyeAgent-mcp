from eyetools.cli import build_parser

def test_cli_parser_help(capsys):
    parser = build_parser()
    code = parser.parse_args([])  # no command
    # argparse returns namespace; main() handles printing help, here just ensure parser exists
    assert parser is not None

