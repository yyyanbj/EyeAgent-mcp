"""
Legacy/demo conversation persistence helper.
Note: not required by the core diagnostic agents; retained for experimental flows.
"""

import os
import uuid
import json
import datetime
from pathlib import Path
from typing import Dict, Any

def _default_conv_dir() -> str:
    # Prefer EYEAGENT_CONV_DIR; else EYEAGENT_DATA_DIR/conversations; else repo-root/cases/conversations; else ~/.local/share/eyeagent/conversations
    env_conv = os.getenv("EYEAGENT_CONV_DIR") or os.getenv("AGENT_CONV_DIR")
    if env_conv:
        return env_conv
    data_dir = os.getenv("EYEAGENT_DATA_DIR")
    if data_dir:
        return str(Path(data_dir) / "conversations")
    # try to locate outermost repo root
    def _find_repo_root(start: Path):
        cur = start.resolve()
        parents = [cur] + list(cur.parents)
        git_candidates = [p for p in parents if (p / ".git").exists()]
        if git_candidates:
            return git_candidates[-1]
        py_candidates = [p for p in parents if (p / "pyproject.toml").exists()]
        if py_candidates:
            return py_candidates[-1]
        return None
    repo = _find_repo_root(Path(__file__))
    if repo:
        return str(repo / "cases" / "conversations")
    return str(Path.home() / ".local" / "share" / "eyeagent" / "conversations")

CONV_DIR = _default_conv_dir()

class ConversationManager:
    """Manage conversation persistence. One UUID per case maps to a JSON file."""

    def __init__(self, directory: str = CONV_DIR):
        self.directory = os.path.abspath(directory)
        os.makedirs(self.directory, exist_ok=True)

    def _path(self, case_id: str) -> str:
        return os.path.join(self.directory, f"{case_id}.json")

    def create_case(self, role: str) -> str:
        case_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow().isoformat()
        data = {
            "case_id": case_id,
            "role": role,
            "created_at": now,
            "updated_at": now,
            "messages": []
        }
        with open(self._path(case_id), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return case_id

    def append_message(self, case_id: str, sender: str, content: str):
        path = self._path(case_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Conversation {case_id} not found")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data['messages'].append({
            'type': sender,
            'content': content,
            'ts': datetime.datetime.utcnow().isoformat()
        })
        data['updated_at'] = datetime.datetime.utcnow().isoformat()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_case(self, case_id: str) -> Dict[str, Any]:
        with open(self._path(case_id), 'r', encoding='utf-8') as f:
            return json.load(f)
