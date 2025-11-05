from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional
import os as _os
import yaml
from loguru import logger

# In-process runtime overrides to avoid environment variables for control.
_CONFIG_FILE_OVERRIDE: Optional[str] = None
_CONFIG_DIR_OVERRIDE: Optional[str] = None
_WORKFLOW_BACKEND_OVERRIDE: Optional[str] = None
_PIPELINE_PROFILE_OVERRIDE: Optional[str] = None
_MCP_SERVER_URL_OVERRIDE: Optional[str] = None
_ROUTING_STRATEGY_OVERRIDE: Optional[str] = None
_MCP_ADAPTER_BIND_OVERRIDE: Optional[bool] = None
_MAX_CLASSES_OVERRIDE: Optional[int] = None
_CASES_DIR_OVERRIDE: Optional[str] = None
_DATA_DIR_OVERRIDE: Optional[str] = None


def set_overrides(
    *,
    config_file: Optional[str] = None,
    config_dir: Optional[str] = None,
    workflow_backend: Optional[str] = None,
    pipeline_profile: Optional[str] = None,
    mcp_server_url: Optional[str] = None,
    routing_strategy: Optional[str] = None,
    mcp_adapter_bind: Optional[bool] = None,
    max_classes: Optional[int] = None,
    cases_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> None:
    """Set process-local overrides for configuration and runtime behavior.

    These take precedence over environment variables and file defaults.
    """
    global _CONFIG_FILE_OVERRIDE, _CONFIG_DIR_OVERRIDE, _WORKFLOW_BACKEND_OVERRIDE, _PIPELINE_PROFILE_OVERRIDE, _MCP_SERVER_URL_OVERRIDE, _ROUTING_STRATEGY_OVERRIDE, _MCP_ADAPTER_BIND_OVERRIDE, _MAX_CLASSES_OVERRIDE, _CASES_DIR_OVERRIDE, _DATA_DIR_OVERRIDE
    if config_file is not None:
        _CONFIG_FILE_OVERRIDE = config_file
    if config_dir is not None:
        _CONFIG_DIR_OVERRIDE = config_dir
    if workflow_backend is not None:
        _WORKFLOW_BACKEND_OVERRIDE = workflow_backend
    if pipeline_profile is not None:
        _PIPELINE_PROFILE_OVERRIDE = pipeline_profile
    if mcp_server_url is not None:
        _MCP_SERVER_URL_OVERRIDE = mcp_server_url
    if routing_strategy is not None:
        _ROUTING_STRATEGY_OVERRIDE = routing_strategy
    if mcp_adapter_bind is not None:
        _MCP_ADAPTER_BIND_OVERRIDE = bool(mcp_adapter_bind)
    if max_classes is not None:
        try:
            _MAX_CLASSES_OVERRIDE = int(max_classes)
        except Exception:
            _MAX_CLASSES_OVERRIDE = None
    if cases_dir is not None:
        _CASES_DIR_OVERRIDE = str(cases_dir)
    if data_dir is not None:
        _DATA_DIR_OVERRIDE = str(data_dir)


class Settings:
    def __init__(self, base_dir: Optional[str] = None):
        # Find a plausible repo root without importing TraceLogger to avoid cycles
        def _find_repo_root(start: Path) -> Optional[Path]:
            cur = start.resolve()
            parents = [cur] + list(cur.parents)
            git_candidates = [p for p in parents if (p / ".git").exists()]
            if git_candidates:
                return git_candidates[-1]
            py_candidates = [p for p in parents if (p / "pyproject.toml").exists()]
            if py_candidates:
                return py_candidates[-1]
            return None

        repo_root = _find_repo_root(Path(__file__)) or Path.cwd()
        base = Path(base_dir) if base_dir else repo_root
        default_cfg_dir = base / "eyeagent" / "config"
        legacy_cfg_dir = base / "config"
        # Respect explicit config file override
        if _CONFIG_FILE_OVERRIDE:
            self.file_path = Path(_CONFIG_FILE_OVERRIDE)
            return
        # Otherwise, prefer explicit override dir; else default paths (no env)
        cfg_dir = Path(_CONFIG_DIR_OVERRIDE or str(legacy_cfg_dir if legacy_cfg_dir.exists() else default_cfg_dir))
        cfg_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = cfg_dir / "eyeagent.yml"

    def load(self) -> Dict[str, Any]:
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"failed to load settings from {self.file_path}: {e}")
                data = {}
        else:
            data = {}
        return _deep_merge(data)

    def save(self, cfg: Dict[str, Any]) -> None:
        data = _deep_merge(cfg or {})
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def get_llm_config(agent_name: Optional[str]) -> Dict[str, Any]:
    cfg = Settings().load().get("llm", {})
    default = cfg.get("default", {})
    per_agents = cfg.get("agents", {}) or {}
    if agent_name and agent_name in per_agents:
        # merge agent override on top of default
        return _deep_merge(default, per_agents.get(agent_name) or {})
    return default


def build_chat_model(agent_name: Optional[str]):
    # Deferred import to avoid hard dependency in config module
    from langchain_openai import ChatOpenAI
    llm_cfg = get_llm_config(agent_name)
    base_url = llm_cfg.get("base_url")
    model = llm_cfg.get("model")
    temperature = float(llm_cfg.get("temperature", 1.0))
    max_tokens = int(llm_cfg.get("max_tokens", 64000))
    return ChatOpenAI(base_url=base_url, model=model, temperature=temperature, max_tokens=max_tokens)


# Note: legacy get_workflow_mode removed. Use get_workflow_backend() instead.


def get_workflow_backend() -> str:
    """Return the configured backend name.

    Valid values: langgraph | profile | interaction | single | topology
    Resolution order (runtime-evaluated):
    1) Env EYEAGENT_WORKFLOW_BACKEND (highest priority)
    2) workflow.backend from config file
    3) Legacy workflow.mode mapping (graph->langgraph, interaction->interaction, profile->profile)
    4) default: langgraph
    """
    # 1) In-process override
    if isinstance(_WORKFLOW_BACKEND_OVERRIDE, str):
        b = (_WORKFLOW_BACKEND_OVERRIDE or "").strip().lower()
        if b in {"langgraph", "profile", "interaction", "single", "topology"}:
            return b

    # 2) Config file
    cfg = Settings().load()
    wf = cfg.get("workflow") or {}
    backend = (wf.get("backend") or "").strip().lower()
    if backend in {"langgraph", "profile", "interaction", "single", "topology"}:
        return backend

    # 3) Legacy mode mapping
    mode = str((wf.get("mode") or "")).strip().lower()
    if mode == "graph":
        return "langgraph"
    if mode == "interaction":
        return "interaction"
    if mode == "profile":
        return "profile"

    # 4) Default
    return "langgraph"


def get_pipeline_profile() -> str:
    """Return pipeline profile name with in-process override taking precedence."""
    if isinstance(_PIPELINE_PROFILE_OVERRIDE, str) and _PIPELINE_PROFILE_OVERRIDE.strip():
        return _PIPELINE_PROFILE_OVERRIDE.strip()
    # Config file may specify a default profile under pipelines.default_profile
    try:
        cfg = Settings().load()
        pipe = cfg.get("pipelines") or {}
        dp = str(pipe.get("default_profile") or "").strip()
        if dp:
            return dp
    except Exception:
        pass
    return "default"


def get_mcp_server_url(default: str = "http://localhost:8000/mcp/") -> str:
    if isinstance(_MCP_SERVER_URL_OVERRIDE, str) and _MCP_SERVER_URL_OVERRIDE.strip():
        return _MCP_SERVER_URL_OVERRIDE.strip()
    return default


def get_cases_dir(default_repo_subdir: str = "cases") -> str:
    """Resolve cases directory from config.

    Checks Settings().load():
    - paths.cases_dir
    - paths.data_dir + "/cases" if data_dir present
    Fallbacks: repo_root/<default_repo_subdir> or ~/.local/share/eyeagent/cases.
    """
    if isinstance(_CASES_DIR_OVERRIDE, str) and _CASES_DIR_OVERRIDE.strip():
        p = Path(_CASES_DIR_OVERRIDE).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return str(p)
    try:
        cfg = Settings().load()
        paths = (cfg.get("paths") or {}) if isinstance(cfg, dict) else {}
        cd = paths.get("cases_dir") or paths.get("cases")
        if isinstance(cd, str) and cd.strip():
            p = Path(cd).expanduser().resolve()
            p.mkdir(parents=True, exist_ok=True)
            return str(p)
        dd = paths.get("data_dir") or paths.get("data")
        if isinstance(dd, str) and dd.strip():
            p = Path(dd).expanduser().resolve() / "cases"
            p.mkdir(parents=True, exist_ok=True)
            return str(p)
    except Exception:
        pass
    # Fallbacks
    repo_root = Path.cwd()
    try:
        rr = repo_root
        p = (rr / default_repo_subdir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return str(p)
    except Exception:
        p = Path.home() / ".local" / "share" / "eyeagent" / "cases"
        p.mkdir(parents=True, exist_ok=True)
        return str(p)


def get_data_dir(default_repo_subdir: str = "data") -> str:
    if isinstance(_DATA_DIR_OVERRIDE, str) and _DATA_DIR_OVERRIDE.strip():
        p = Path(_DATA_DIR_OVERRIDE).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return str(p)
    try:
        cfg = Settings().load()
        paths = (cfg.get("paths") or {}) if isinstance(cfg, dict) else {}
        dd = paths.get("data_dir") or paths.get("data")
        if isinstance(dd, str) and dd.strip():
            p = Path(dd).expanduser().resolve()
            p.mkdir(parents=True, exist_ok=True)
            return str(p)
    except Exception:
        pass
    rr = Path.cwd()
    p = (rr / default_repo_subdir).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def get_logging_config() -> Dict[str, str]:
    cfg = {}
    try:
        data = Settings().load()
        log = (data.get("logging") or {}) if isinstance(data, dict) else {}
        level = str(log.get("level") or "DEBUG")
        fmt = str(log.get("format") or "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
        file_path = log.get("file")
        if file_path is not None:
            file_path = str(file_path)
        cfg = {"level": level, "format": fmt, "file": file_path}
    except Exception:
        cfg = {"level": "DEBUG", "format": "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", "file": None}
    return cfg


def get_routing_strategy(default: str = "llm") -> str:
    if isinstance(_ROUTING_STRATEGY_OVERRIDE, str) and _ROUTING_STRATEGY_OVERRIDE.strip():
        return _ROUTING_STRATEGY_OVERRIDE.strip().lower()
    try:
        wf = (Settings().load().get("workflow") or {})
        rs = str(wf.get("routing") or default).strip().lower()
        return rs or default
    except Exception:
        return default


def get_mcp_adapter_bind(default: bool = False) -> bool:
    if _MCP_ADAPTER_BIND_OVERRIDE is not None:
        return bool(_MCP_ADAPTER_BIND_OVERRIDE)
    try:
        wf = (Settings().load().get("workflow") or {})
        val = wf.get("mcp_adapter_bind")
        if isinstance(val, bool):
            return val
    except Exception:
        pass
    return default


def get_max_classes(default: int = 8) -> int:
    if isinstance(_MAX_CLASSES_OVERRIDE, int) and _MAX_CLASSES_OVERRIDE is not None:
        return int(_MAX_CLASSES_OVERRIDE)
    try:
        cfg = Settings().load()
        out = cfg.get("outputs") or {}
        mc = out.get("max_classes")
        if mc is not None:
            return int(mc)
    except Exception:
        pass
    return default


def get_specialist_selection_settings() -> Dict[str, Any]:
    cfg = Settings().load()
    wf = cfg.get("workflow") or {}
    sp = wf.get("specialist") or {}
    try:
        th = float(sp.get("candidate_threshold", 0.3))
    except Exception:
        th = 0.3
    try:
        k = int(sp.get("candidate_top_k", 5))
    except Exception:
        k = 5
    return {"candidate_threshold": th, "candidate_top_k": k}


def get_configured_agents() -> Dict[str, Dict[str, Any]]:
    cfg = Settings().load()
    agents = cfg.get("agents") or {}
    # ensure dict-of-dicts
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in agents.items():
        if isinstance(v, dict):
            out[k] = v
    return out


def get_knowledge_config() -> Dict[str, Any]:
    cfg = Settings().load()
    kn = cfg.get("knowledge") or {}
    out = {
        "default_top_k": int(kn.get("default_top_k", 3) or 3),
        "max_calls_per_agent": int(kn.get("max_calls_per_agent", 2) or 2),
    }
    return out
