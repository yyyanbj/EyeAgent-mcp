from __future__ import annotations
import os
import json
from typing import Any, Dict, List, Optional
from loguru import logger

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    yaml = None  # type: ignore
    _HAS_YAML = False


def _read_file_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pipelines_config() -> Dict[str, Any]:
    """Load pipeline profiles from YAML or JSON.

    Search order:
    1) EYEAGENT_PIPELINES_FILE env
    2) config/pipelines.yml
    3) config/pipelines.json
    """
    candidates = []
    env_path = os.getenv("EYEAGENT_PIPELINES_FILE")
    if env_path:
        candidates.append(env_path)
    # repo-relative defaults
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates.append(os.path.join(repo_root, "config", "pipelines.yml"))
    candidates.append(os.path.join(repo_root, "config", "pipelines.json"))

    for p in candidates:
        if not os.path.isfile(p):
            continue
        try:
            text = _read_file_text(p)
            if p.endswith(".yml") or p.endswith(".yaml"):
                if not _HAS_YAML:
                    logger.warning("pipelines file is YAML but PyYAML not installed; please `pip install pyyaml`")
                    continue
                data = yaml.safe_load(text) or {}
            else:
                data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception as e:
            logger.warning(f"failed to load pipelines config from {p}: {e}")
    return {"profiles": {}}


def get_profile_steps(profile: str) -> List[Dict[str, Any]]:
    """Return list of step dicts: [{name, when?}] for the given profile.

    YAML shape:
    profiles:
      default:
        steps:
          - name: orchestrator
          - name: image_analysis
          - name: specialist
            when:
              any:
                - { key: "image_analysis.diseases.DR", op: ">", value: 0.3 }
                - { key: "image_analysis.diseases.AMD", op: ">", value: 0.3 }
          - name: follow_up
          - name: report
    """
    cfg = load_pipelines_config()
    prof = (cfg.get("profiles") or {}).get(profile) if isinstance(cfg, dict) else None
    steps = ((prof or {}).get("steps") or []) if isinstance(prof, dict) else []
    out: List[Dict[str, Any]] = []
    for s in steps:
        if isinstance(s, str):
            out.append({"name": s})
        elif isinstance(s, dict) and "name" in s:
            out.append({"name": s.get("name"), "when": s.get("when")})
    return out


def _get_by_path(state: Dict[str, Any], path: str) -> Any:
    cur: Any = state
    for part in (path or "").split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _eval_condition(cond: Dict[str, Any], state: Dict[str, Any]) -> bool:
    key = cond.get("key")
    op = cond.get("op", "==")
    val = cond.get("value")
    left = _get_by_path(state, key) if isinstance(key, str) else None
    try:
        if op == "==":
            return left == val
        if op == "!=":
            return left != val
        if op == ">":
            return float(left) > float(val)
        if op == ">=":
            return float(left) >= float(val)
        if op == "<":
            return float(left) < float(val)
        if op == "<=":
            return float(left) <= float(val)
        if op == "exists":
            return left is not None
        if op == "not_exists":
            return left is None
    except Exception:
        return False
    return False


def step_should_run(step: Dict[str, Any], state: Dict[str, Any]) -> bool:
    cond = step.get("when")
    if not cond:
        return True
    if "all" in cond:
        arr = cond.get("all") or []
        return all(_eval_condition(c, state) for c in arr if isinstance(c, dict))
    if "any" in cond:
        arr = cond.get("any") or []
        return any(_eval_condition(c, state) for c in arr if isinstance(c, dict))
    if isinstance(cond, dict) and "key" in cond:
        return _eval_condition(cond, state)
    return True
