"""Subprocess worker entrypoint (minimal JSON line protocol).

Protocol messages (one JSON per line stdin->stdout):
    {"cmd": "INIT", "meta": {...}}
    {"cmd": "PREDICT", "request": {...}}
    {"cmd": "LOAD_MODEL"}            (optional: force model load if lazy)
    {"cmd": "SHUTDOWN"}

Responses:
  {"ok": bool, "cmd": <cmd>, "data"|"error": ...}

Model loading is lazy on first PREDICT if meta.model.lazy is true.
"""
from __future__ import annotations

import sys
import json
import traceback
from importlib import import_module
from typing import Any, Dict
from pathlib import Path

tool_instance = None  # type: ignore
tool_meta: Dict[str, Any] | None = None


def _import_entry(entry: str):
    module_name, class_name = entry.split(":", 1)
    try:
        mod = import_module(module_name)
    except ModuleNotFoundError:
        # attempt dynamic path injection based on tool_meta.root_dir
        if tool_meta and tool_meta.get("root_dir"):
            root_dir = Path(tool_meta["root_dir"])  # points to the tool package directory
            if str(root_dir) not in sys.path:
                sys.path.insert(0, str(root_dir))
            try:
                mod = import_module(module_name)
            except ModuleNotFoundError:
                # also try parent directory (so module_name inside the package dir itself)
                parent = root_dir.parent
                if str(parent) not in sys.path:
                    sys.path.insert(0, str(parent))
                mod = import_module(module_name)
        else:
            raise
    return getattr(mod, class_name)


def handle_init(msg: Dict[str, Any]):
    global tool_instance, tool_meta
    tool_meta = msg.get("meta", {})
    entry = tool_meta.get("entry")
    if not entry:
        raise ValueError("meta.entry missing")
    ToolCls = _import_entry(entry)
    params = tool_meta.get("params", {})
    tool_instance = ToolCls(tool_meta, params)
    if hasattr(tool_instance, "prepare"):
        tool_instance.prepare()
    # If lifecycle eager wants preloaded model it will call LOAD_MODEL explicitly; INIT stays lightweight.
    return {"tool_id": tool_meta.get("id"), "lazy": tool_meta.get("model", {}).get("lazy", True)}


def handle_predict(msg: Dict[str, Any]):
    if tool_instance is None or tool_meta is None:
        raise RuntimeError("Tool not initialized; INIT must be called first")
    # Ensure model loaded (predict path safeguards even if eager preload failed)
    if getattr(tool_instance, "model_loaded", False) is False and hasattr(tool_instance, "load_model"):
        try:
            tool_instance.load_model()
        except Exception as e:  # noqa
            raise RuntimeError(f"Model load failed: {e}")
    request = msg.get("request", {})
    return tool_instance.predict(request)


def handle_shutdown(_msg: Dict[str, Any]):
    global tool_instance
    if tool_instance and hasattr(tool_instance, "shutdown"):
        tool_instance.shutdown()
    return {"bye": True}


def handle_load_model(_msg: Dict[str, Any]):
    if tool_instance is None or tool_meta is None:
        raise RuntimeError("Tool not initialized")
    if hasattr(tool_instance, "load_model") and getattr(tool_instance, "model_loaded", False) is False:
        tool_instance.load_model()
    resp = {"loaded": getattr(tool_instance, "model_loaded", False)}
    # Attach optional telemetry if tool provided it (segmentation tool sets _load_telemetry)
    telem = getattr(tool_instance, "_load_telemetry", None)
    if isinstance(telem, dict):  # minimal filtering to keep JSON-serializable
        safe = {}
        for k, v in telem.items():
            if isinstance(v, (str, int, float, type(None), bool)):
                safe[k] = v
        resp.update(safe)
    return resp


HANDLERS = {"INIT": handle_init, "PREDICT": handle_predict, "LOAD_MODEL": handle_load_model, "SHUTDOWN": handle_shutdown}

# Extend handlers dynamically to include WARMUP if tool supports it after load
def handle_warmup(_msg: Dict[str, Any]):
    if tool_instance is None:
        raise RuntimeError("Tool not initialized")
    if hasattr(tool_instance, "warmup"):
        return tool_instance.warmup()
    return {"warmed_up": False, "reason": "no warmup method"}

HANDLERS["WARMUP"] = handle_warmup


def main():  # pragma: no cover - exercised via higher-level tests
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            cmd = msg.get("cmd")
            if cmd not in HANDLERS:
                raise ValueError(f"Unknown cmd {cmd}")
            data = HANDLERS[cmd](msg)
            sys.stdout.write(json.dumps({"ok": True, "cmd": cmd, "data": data}) + "\n")
            sys.stdout.flush()
            if cmd == "SHUTDOWN":
                break
        except Exception as e:  # noqa: BLE001
            sys.stdout.write(json.dumps({"ok": False, "error": str(e), "trace": traceback.format_exc()}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":  # pragma: no cover
    main()
