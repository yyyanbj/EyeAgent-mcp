from pathlib import Path
import pytest
from eyetools.core.config_loader import parse_config_file, load_configs_in_dir, ConfigError

def test_parse_single(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("entry: mod:Cls\nname: test\ncategory: demo\n")
    items = parse_config_file(cfg)
    assert len(items) == 1
    td = items[0]
    assert td["id"].startswith("demo.")
    assert td["runtime"]["load_mode"] == "auto"


def test_parse_variants(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("""
package: demo_pkg
entry: m:K
variants:
  - variant: small
    runtime: {load_mode: inproc}
  - variant: large
    model: {precision: fp16}
""")
    items = parse_config_file(cfg)
    ids = {i['id'] for i in items}
    assert any(":small" in i for i in ids)
    assert any(":large" in i for i in ids)


def test_invalid_precision(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("entry: a:b\nmodel: {precision: xxx}\n")
    with pytest.raises(ConfigError):
        parse_config_file(cfg)


def test_load_configs_in_dir(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("entry: a:b\nname: demo\n")
    res = load_configs_in_dir(tmp_path)
    assert res and res[0]["entry"] == "a:b"
