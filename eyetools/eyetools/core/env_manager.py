"""Environment management using uv overlay strategy.

MVP Goals:
- Compute environment key (python base + sorted extra requirements).
- Provide run_in_env(meta, args:list[str]) to execute command under uv with --with requirements file or inline specs.
- Cache mapping env_key -> resolved (placeholder, can be extended later).

Assumptions:
- Base python environments (py310, py312, etc.) correspond to interpreters available on PATH or packaged.
- Extra dependencies are lightweight; heavy libs should ideally be in base env.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import subprocess
import shlex
import hashlib


class EnvManager:
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.cache: Dict[str, Dict] = {}

    def _hash_env(self, python_tag: str, extra: List[str]) -> str:
        h = hashlib.sha256()
        h.update(python_tag.encode())
        for dep in sorted(extra):
            h.update(dep.encode())
        return h.hexdigest()[:12]

    def build_env_key(self, python_tag: str, extra_requires: List[str]) -> str:
        return f"{python_tag}-{self._hash_env(python_tag, extra_requires)}"

    def run_in_env(self, meta: Dict, args: List[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
        python_tag = meta.get("runtime", {}).get("python") or meta.get("python") or "py310"
        extra_requires = meta.get("extra_requires") or meta.get("runtime", {}).get("extra_requires") or []
        env_key = self.build_env_key(python_tag, extra_requires)
        # Compose uv command
        with_parts = []
        if extra_requires:
            # inline spec string
            with_parts.append(",".join(extra_requires))
        cmd = ["uv", "run"]
        if with_parts:
            cmd += ["--with", with_parts[0]]
        cmd += [f"--python={python_tag}"]
        cmd += args
        cp = subprocess.run(cmd, cwd=str(cwd or self.workspace_root), capture_output=True, text=True)
        return cp

__all__ = ["EnvManager"]
