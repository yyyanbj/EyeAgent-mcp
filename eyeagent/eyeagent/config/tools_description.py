"""Local MCP tool descriptions registry and sync utility.

Maintains a local cache of tool descriptions independent of server-provided descriptions
to enable prompt optimization while allowing non-destructive sync when the server adds
new tools. Deprecated tools are kept (not deleted) unless explicitly removed by user.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import os
import json
from pathlib import Path


class ToolsDescriptionRegistry:
    def __init__(self, base_dir: Optional[str] = None):
        # default under repo_root/config/tools_descriptions.json
        from eyeagent.tracing.trace_logger import TraceLogger
        t = TraceLogger()
        cases_dir = Path(t.base_dir)
        repo_root = cases_dir.parent if cases_dir.name == "cases" else Path.cwd()
        base = Path(base_dir) if base_dir else repo_root
        self.config_dir = Path(os.getenv("EYEAGENT_CONFIG_DIR", base / "config"))
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.config_dir / "tools_descriptions.json"

    def load(self) -> Dict[str, Any]:
        if self.file_path.exists():
            try:
                return json.load(open(self.file_path, "r", encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def save(self, data: Dict[str, Any]) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get(self, tool_id: str) -> Optional[Dict[str, Any]]:
        return self.load().get(tool_id)

    def set(self, tool_id: str, desc: Dict[str, Any]) -> None:
        data = self.load()
        data[tool_id] = desc
        self.save(data)

    def list_ids(self) -> List[str]:
        return list(self.load().keys())

    def sync_from_server(self, server_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Non-destructive merge: add new tools or update fields; do not delete local entries."""
        local = self.load()
        for meta in server_tools or []:
            tid = meta.get("id") or meta.get("tool_id") or meta.get("name")
            if not tid:
                continue
            cur = local.get(tid, {})
            # Merge basic fields
            merged = {
                **cur,
                **{k: v for k, v in meta.items() if k in ("id", "name", "version", "role", "modalities", "desc", "desc_long")},
            }
            # Preserve any local-only fields (e.g., prompt_hints)
            local[tid] = merged
        self.save(local)
        return local
