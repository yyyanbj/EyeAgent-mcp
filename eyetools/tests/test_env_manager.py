from pathlib import Path
from eyetools.core.env_manager import EnvManager

def test_env_key_stable():
    em = EnvManager(Path.cwd())
    k1 = em.build_env_key("py310", ["a==1", "b==2"])
    k2 = em.build_env_key("py310", ["b==2", "a==1"])  # order independent
    assert k1 == k2


def test_run_in_env_command_composition(monkeypatch):
    em = EnvManager(Path.cwd())
    captured = {}
    def fake_run(cmd, cwd=None, capture_output=True, text=True):  # noqa
        captured['cmd'] = cmd
        class R:  # minimal structure
            returncode = 0
            stdout = "ok"
            stderr = ""
        return R()
    monkeypatch.setattr("subprocess.run", fake_run)
    meta = {"runtime": {"python": "py310", "extra_requires": ["x==1"]}}
    em.run_in_env(meta, ["python", "-V"])
    cmd = captured['cmd']
    assert cmd[0:2] == ["uv", "run"]
    assert "--with" in cmd
    assert any(c.startswith("--python=") for c in cmd)
