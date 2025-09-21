from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from loguru import logger


DEFAULT_SETTINGS: Dict[str, Any] = {
    "llm": {
        "default": {
            "base_url": os.getenv("AGENT_LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")),
            "model": os.getenv("AGENT_LLM_MODEL", os.getenv("OPENAI_MODEL", "deepseek-chat")),
            "temperature": float(os.getenv("AGENT_LLM_TEMPERATURE", "1.0")),
            "max_tokens": int(os.getenv("AGENT_LLM_MAX_TOKENS", "64000")),
        },
        "agents": {
            # "OrchestratorAgent": {"model": "gpt-4o-mini"}
        },
    },
    # Workflow + agent wiring is config-driven; default to unified
    "workflow": {
        "mode": os.getenv("EYEAGENT_WORKFLOW_MODE", "unified"),  # unified | graph | interaction | profile
        "specialist": {
            # threshold to select candidate diseases from screening probabilities
            "candidate_threshold": float(os.getenv("EYEAGENT_CANDIDATE_THRESHOLD", "0.3")),
            "candidate_top_k": int(os.getenv("EYEAGENT_CANDIDATE_TOPK", "5")),
        },
    },
    # Agents map: role -> dotted class path; enabled flag allows pruning
    "agents": {
        "unified": {
            "class": "eyeagent.agents.unified_agent.UnifiedAgent",
            "enabled": True,
        },
        "orchestrator": {
            "class": "eyeagent.agents.orchestrator_agent.OrchestratorAgent",
            "enabled": True,
        },
        "image_analysis": {
            "class": "eyeagent.agents.image_analysis_agent.ImageAnalysisAgent",
            "enabled": True,
        },
        "specialist": {
            "class": "eyeagent.agents.specialist_agent.SpecialistAgent",
            "enabled": True,
        },
        "follow_up": {
            "class": "eyeagent.agents.followup_agent.FollowUpAgent",
            "enabled": True,
        },
        "report": {
            "class": "eyeagent.agents.report_agent.ReportAgent",
            "enabled": True,
        },
        "knowledge": {
            "class": "eyeagent.agents.knowledge_agent.KnowledgeAgent",
            "enabled": True,
        },
    },
    # Optional tool filters per agent
    "tools_filter": {
        "OrchestratorAgent": {
            "include": ["classification:(modality|laterality|multidis)"]
        },
        "ImageAnalysisAgent": {
            "include": ["classification:cfp_quality", "segmentation:cfp_.*", "segmentation:oct_.*", "segmentation:ffa_.*"]
        },
        "SpecialistAgent": {
            "include": ["disease_specific_cls:.*"]
        },
        "FollowUpAgent": {
            "include": ["classification:cfp_age"]
        },
        "KnowledgeAgent": {
            "include": ["rag:query", "web_search:.*"]
        }
    },
}


class Settings:
    def __init__(self, base_dir: Optional[str] = None):
        from ..tracing.trace_logger import TraceLogger
        t = TraceLogger()
        cases_dir = Path(t.base_dir)
        repo_root = cases_dir.parent if cases_dir.name == "cases" else Path.cwd()
        base = Path(base_dir) if base_dir else repo_root
        # precedence: EYEAGENT_CONFIG_FILE > EYEAGENT_CONFIG_DIR/eyeagent.yml > repo_root/eyeagent/config/eyeagent.yml > repo_root/config/eyeagent.yml
        cfg_file_env = os.getenv("EYEAGENT_CONFIG_FILE")
        if cfg_file_env:
            self.file_path = Path(cfg_file_env)
        else:
            # Prefer repo-root config/ by default, fallback to eyeagent/config
            default_cfg_dir = base / "eyeagent" / "config"
            legacy_cfg_dir = base / "config"
            cfg_dir = Path(os.getenv("EYEAGENT_CONFIG_DIR", str(legacy_cfg_dir if legacy_cfg_dir.exists() else default_cfg_dir)))
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
        return _deep_merge(DEFAULT_SETTINGS, data)

    def save(self, cfg: Dict[str, Any]) -> None:
        data = _deep_merge(DEFAULT_SETTINGS, cfg or {})
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


def get_workflow_mode() -> str:
    cfg = Settings().load()
    mode = (cfg.get("workflow") or {}).get("mode") or "unified"
    return str(mode).lower()


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
