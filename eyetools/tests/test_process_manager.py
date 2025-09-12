import json
from pathlib import Path
from eyetools.core.process_manager import ProcessManager
from eyetools.core.registry import ToolMeta
import types

class DummyProc:
    def __init__(self):
        self.stdin = types.SimpleNamespace(write=self._write, flush=lambda: None)
        self._out = []
        self.stdout = types.SimpleNamespace(readline=self._readline)
        self._responses = []
    def _write(self, data):
        payload = json.loads(data)
        if payload.get("cmd") == "INIT":
            self._responses.append(json.dumps({"ok": True}) + "\n")
        elif payload.get("cmd") == "PREDICT":
            self._responses.append(json.dumps({"ok": True, "data": {"status": "ok"}}) + "\n")
        elif payload.get("cmd") == "SHUTDOWN":
            self._responses.append(json.dumps({"ok": True}) + "\n")
    def _readline(self):
        if self._responses:
            return self._responses.pop(0)
        return ""
    def poll(self):
        return None
    def terminate(self):
        pass


def test_process_manager_init_and_predict(monkeypatch, tmp_path: Path):
    pm = ProcessManager(tmp_path)
    def fake_popen(cmd, cwd=None, stdin=None, stdout=None, text=None):  # noqa
        return DummyProc()
    monkeypatch.setattr("subprocess.Popen", fake_popen)
    meta = ToolMeta(id="m1", entry="m:K", version="0.1.0")
    wi = pm.spawn(meta.id, meta=meta)
    assert wi.tool_id == "m1"
    pm.ensure_init(meta)
    resp = pm.request(meta.id, {"cmd": "PREDICT"})
    assert resp["ok"]
    pm.stop(meta.id)
