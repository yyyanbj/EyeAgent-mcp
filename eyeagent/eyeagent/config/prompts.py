from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

DEFAULT_PROMPTS: Dict[str, Any] = {
    "system_prompts": {
        "OrchestratorAgent": (
            "You are the orchestrator. Tasks: (1) infer imaging modality, (2) classify laterality, "
            "(3) plan downstream pipeline (ImageAnalysis -> Specialist -> FollowUp -> Report). "
            "Be concise but explicit; provide reasoning for decisions."
        ),
        "ImageAnalysisAgent": (
            "You are the image analysis agent. Based on inferred modality (CFP/OCT/FFA): (1) run quality assessment if applicable, "
            "(2) perform modality-appropriate lesion segmentation, (3) when modality is CFP, run a multi-disease screening classifier. "
            "Summarize findings in a clinician-friendly paragraph."
        ),
        "SpecialistAgent": (
            "You are the specialist agent. For candidate diseases, invoke disease-specific grading/classification models and "
            "produce grades with confidence. Summarize clinical impression in one paragraph."
        ),
        "FollowUpAgent": (
            "You are the follow-up agent. Combine disease grades and patient age to suggest follow-up interval and referrals. "
            "Return a short plan and a one-paragraph rationale."
        ),
        "ReportAgent": (
            "You are the report agent. Consolidate findings into a final summary paragraph for clinicians, "
            "including diagnoses and management recommendations."
        ),
    },
    "ui": {
        "instruction_presets": [
            "Perform a comprehensive ocular imaging screening and summarize key findings.",
            "Focus on diabetic retinopathy and grade severity.",
            "Check AMD-related findings and suggest follow-up.",
            "Provide a plain-language patient-friendly summary."
        ],
        "default_instruction": "Perform a comprehensive ocular imaging screening and summarize key findings.",
    },
}


class PromptsConfig:
    def __init__(self, base_dir: Optional[str] = None):
        # Determine config path precedence
        # 1) EYEAGENT_CONFIG_DIR, else 2) repo root (cases parent), else 3) CWD
        from eyeagent.tracing.trace_logger import TraceLogger
        t = TraceLogger()
        cases_dir = Path(t.base_dir)
        repo_root = cases_dir.parent if cases_dir.name == "cases" else Path.cwd()
        base = Path(base_dir) if base_dir else repo_root
        self.config_dir = Path(os.getenv("EYEAGENT_CONFIG_DIR", base / "config"))
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.config_dir / "prompts.yml"

    def load(self) -> Dict[str, Any]:
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except Exception:
                data = {}
        else:
            data = {}
        # deep-merge defaults
        return _deep_merge(DEFAULT_PROMPTS, data)

    def save(self, cfg: Dict[str, Any]) -> None:
        # ensure minimal structure
        data = _deep_merge(DEFAULT_PROMPTS, cfg or {})
        with open(self.file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    def get_system_prompt(self, agent_name: str) -> str:
        cfg = self.load()
        return (
            (cfg.get("system_prompts") or {}).get(agent_name)
            or DEFAULT_PROMPTS["system_prompts"].get(agent_name)
            or ""
        )

    def set_system_prompt(self, agent_name: str, prompt: str) -> None:
        cfg = self.load()
        cfg.setdefault("system_prompts", {})[agent_name] = prompt
        self.save(cfg)

    def get_ui_presets(self) -> Dict[str, Any]:
        cfg = self.load()
        return cfg.get("ui") or {}

    def set_ui_presets(self, ui_cfg: Dict[str, Any]) -> None:
        cfg = self.load()
        cfg["ui"] = _deep_merge(DEFAULT_PROMPTS.get("ui", {}), ui_cfg or {})
        self.save(cfg)


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
