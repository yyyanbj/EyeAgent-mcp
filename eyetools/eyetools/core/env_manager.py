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
import hashlib
import re

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore


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

    def _resolve_env_ref(self, meta: Dict) -> tuple[str, List[str]]:
        """Resolve python tag + dependency list from either explicit runtime or environment_ref.

        Precedence:
        1. meta.runtime.python overrides env ref python requirement.
        2. Collect dependencies from envs/<ref>/pyproject.toml [project.dependencies].
        3. Append meta.extra_requires.
        """
        runtime = meta.get("runtime", {})
        env_ref = meta.get("environment_ref") or runtime.get("environment_ref")
        deps: List[str] = []
        python_tag = runtime.get("python") or meta.get("python") or "py310"

        if env_ref:
            env_dir = self.workspace_root / "envs" / env_ref
            pyproj = env_dir / "pyproject.toml"
            if pyproj.exists():
                try:
                    data = tomllib.loads(pyproj.read_text())
                    raw_deps = data.get("project", {}).get("dependencies", [])
                    for d in raw_deps:
                        if isinstance(d, str):
                            deps.append(d)
                    # Attempt to infer python version constraint if not explicitly provided
                    py_req = data.get("project", {}).get("requires-python")
                    if py_req:
                        # naive parse for >=3.12 -> py312
                        m = re.search(r"3\.(\d+)", py_req)
                        if m and not runtime.get("python") and not meta.get("python"):
                            python_tag = f"py3{m.group(1)}"
                except Exception:
                    pass
        extra_requires = runtime.get("extra_requires") or meta.get("extra_requires") or []
        deps.extend(extra_requires)
        return python_tag, deps

    def run_in_env(self, meta: Dict, args: List[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
        python_tag, deps = self._resolve_env_ref(meta)
        # Fallback: map shorthand like py312 to an actual python executable if uv lacks managed install
        shorthand_map = {"py312": "python3.12", "py310": "python3.10", "py311": "python3.11"}
        exec_tag = shorthand_map.get(python_tag, python_tag)
        env_key = self.build_env_key(python_tag, deps)
        cmd = ["uv", "run"]
        # Pass each dependency as its own --with flag so uv resolves them individually
        for dep in deps:
            cmd += ["--with", dep]
        cmd += [f"--python={exec_tag}"]
        cmd += args
        cp = subprocess.run(cmd, cwd=str(cwd or self.workspace_root), capture_output=True, text=True)
        return cp

__all__ = ["EnvManager"]
