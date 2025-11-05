from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:  # optional; fallback limited to JSON
    yaml = None


def _maybe_load_text_fields(d: Dict[str, Any]) -> Dict[str, Any]:
    """Expand fields that look like file references into text.
    Rule: any key ending with `_path` will be loaded and a sibling without `_path` will be set.
    Example: prompts.system_path -> prompts.system (file content).
    """
    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    out[k] = walk(v)
                else:
                    out[k] = v
            # process *_path keys
            for k, v in list(out.items()):
                if isinstance(k, str) and k.endswith("_path") and isinstance(v, str):
                    p = Path(v)
                    if p.exists() and p.is_file():
                        try:
                            text = p.read_text(encoding="utf-8")
                            out[k[:-5]] = text
                        except Exception:
                            pass
            return out
        elif isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(d)


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in {".yaml", ".yml"} and yaml is not None:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    else:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be an object/dict")
    cfg = _maybe_load_text_fields(cfg)
    return cfg
