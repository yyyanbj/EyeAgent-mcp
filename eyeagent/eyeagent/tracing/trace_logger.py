import os
import json
import uuid
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from threading import RLock

def ISO() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

class TraceLogger:
    """Record all events and final report for a diagnostic workflow.

    Directory layout:
    cases/<case_id>/
        trace.json        # structured JSON with all events (updated incrementally)
        final_report.json # final consolidated report (schema-compliant)

    trace.json structure:
    {
        "case_id": str,
        "created_at": iso,
        "updated_at": iso,
        "patient": {...},
        "images": [...],
        "events": [ {"ts": iso, "type": "agent_step|tool_call|error", ...} ]
    }
    """

    def __init__(self, base_dir: str = None):
        """Initialize TraceLogger.

        Base directory resolution priority:
        1) explicit base_dir argument
        2) EYEAGENT_CASES_DIR (absolute path to the cases directory)
        3) EYEAGENT_DATA_DIR (parent directory; cases will be stored under <DATA_DIR>/cases)
        4) <repo_root>/cases (if a repo root can be detected; prefers the OUTERMOST directory with .git,
           otherwise the outermost directory with a pyproject.toml)
        5) ~/.local/share/eyeagent/cases
        """
        def _find_repo_root(start: Path) -> Optional[Path]:
            """Return the OUTERMOST repo root.

            Preference order:
            - outermost directory containing .git
            - else outermost directory containing pyproject.toml
            - else None
            """
            cur = start.resolve()
            parents = [cur] + list(cur.parents)
            git_candidates = [p for p in parents if (p / ".git").exists()]
            if git_candidates:
                return git_candidates[-1]  # outermost .git
            py_candidates = [p for p in parents if (p / "pyproject.toml").exists()]
            if py_candidates:
                return py_candidates[-1]  # outermost pyproject
            return None

        if base_dir:
            resolved = Path(base_dir)
        else:
            env_cases = os.getenv("EYEAGENT_CASES_DIR")
            env_data = os.getenv("EYEAGENT_DATA_DIR")
            if env_cases:
                resolved = Path(env_cases)
            elif env_data:
                resolved = Path(env_data) / "cases"
            else:
                repo_root = _find_repo_root(Path(__file__))
                repo = repo_root.resolve() if repo_root else None
                if repo:
                    resolved = repo / "cases"
                else:
                    resolved = Path.home() / ".local" / "share" / "eyeagent" / "cases"

        self.base_dir = str(resolved.absolute())
        os.makedirs(self.base_dir, exist_ok=True)
        self._lock = RLock()

    def _case_dir(self, case_id: str) -> str:
        return os.path.join(self.base_dir, case_id)

    def _trace_path(self, case_id: str) -> str:
        return os.path.join(self._case_dir(case_id), "trace.json")

    def _final_path(self, case_id: str) -> str:
        return os.path.join(self._case_dir(case_id), "final_report.json")

    def _conversation_path(self, case_id: str) -> str:
        return os.path.join(self._case_dir(case_id), "conversation.jsonl")

    def create_case(self, patient: Dict[str, Any], images: List[Dict[str, Any]]) -> str:
        case_id = str(uuid.uuid4())
        case_dir = self._case_dir(case_id)
        os.makedirs(case_dir, exist_ok=True)
        doc = {
            "case_id": case_id,
            "created_at": ISO(),
            "updated_at": ISO(),
            "patient": patient,
            "images": images,
            "events": [],
            "next_seq": 1
        }
        self._atomic_write_json(self._trace_path(case_id), doc)
        return case_id

    def append_event(self, case_id: str, event: Dict[str, Any]):
        with self._lock:
            path = self._trace_path(case_id)
            if not os.path.exists(path):
                # Auto-create a minimal case to avoid hard failure; ensure directory exists
                try:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    doc = {
                        "case_id": case_id,
                        "created_at": ISO(),
                        "updated_at": ISO(),
                        "patient": {},
                        "images": [],
                        "events": [],
                        "next_seq": 1,
                        "note": "Auto-created by TraceLogger.append_event because trace.json was missing",
                    }
                    self._atomic_write_json(path, doc)
                except Exception:
                    raise FileNotFoundError(f"trace.json not found and could not be created for case {case_id}")
            # Try load existing; if corrupted, attempt recovery from backup or reconstruct minimal doc
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
            except Exception as e:
                # Preserve corrupted file for inspection
                try:
                    corrupt_path = path + ".corrupt"
                    if not os.path.exists(corrupt_path):
                        try:
                            os.replace(path, corrupt_path)
                        except Exception:
                            pass
                finally:
                    # Reconstruct minimal doc to keep pipeline running
                    doc = {"case_id": case_id, "created_at": ISO(), "updated_at": ISO(), "patient": {}, "images": [], "events": [], "next_seq": 1}
                    # Attach an error event noting the recovery
                    doc["events"].append({"ts": ISO(), "type": "error", "message": f"Recovered from corrupted trace.json: {type(e).__name__}: {e}"})
            event.setdefault("ts", ISO())
            # assign and increment sequence number
            try:
                seq = int(doc.get("next_seq") or 1)
            except Exception:
                seq = 1
            event["seq"] = seq
            doc["next_seq"] = seq + 1
            doc["events"].append(event)
            doc["updated_at"] = ISO()
            self._atomic_write_json(path, doc)

    def append_conversation_message(self, case_id: str, message: Dict[str, Any]):
        """Append a chat message to conversation.jsonl with seq and timestamp.

        Message should be like {"role": "assistant|user|system", "content": str, ...}.
        We'll enrich it with ts and seq from trace.json to keep consistent ordering.
        """
        with self._lock:
            # First read/advance sequence from trace.json for consistent numbering
            path = self._trace_path(case_id)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
            except Exception:
                doc = {"next_seq": 1}
            try:
                seq = int(doc.get("next_seq") or 1)
            except Exception:
                seq = 1
            # bump and persist the next_seq in trace.json (without adding an event)
            doc["updated_at"] = ISO()
            doc["next_seq"] = seq + 1
            self._atomic_write_json(path, doc)

            rec = dict(message or {})
            rec.setdefault("ts", ISO())
            rec["seq"] = seq
            # Write JSONL record
            cpath = self._conversation_path(case_id)
            os.makedirs(os.path.dirname(cpath), exist_ok=True)
            with open(cpath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def get_conversation_path(self, case_id: str) -> str:
        return self._conversation_path(case_id)

    def write_final_report(self, case_id: str, report: Dict[str, Any]):
        with self._lock:
            self._atomic_write_json(self._final_path(case_id), report)
            # 也把 trace.json 中追加 final_report 路径引用
            path = self._trace_path(case_id)
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        doc = json.load(f)
                except Exception:
                    doc = {"case_id": case_id, "created_at": ISO(), "updated_at": ISO(), "patient": {}, "images": [], "events": []}
                doc['final_report_path'] = self._final_path(case_id)
                doc['updated_at'] = ISO()
                self._atomic_write_json(path, doc)

    def load_trace(self, case_id: str) -> Dict[str, Any]:
        with open(self._trace_path(case_id), 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_final(self, case_id: str) -> Dict[str, Any]:
        with open(self._final_path(case_id), 'r', encoding='utf-8') as f:
            return json.load(f)

    # ---- internals --------------------------------------------------
    def _atomic_write_json(self, path: str, data: Dict[str, Any]):
        """Write JSON atomically to avoid partial writes leaving corrupted files.

        Strategy: write to path.tmp then os.replace to target. Also write a .bak backup
        of the previous file once per session if none exists.
        """
        tmp_path = path + ".tmp"
        # Create backup if original exists and no backup present
        try:
            if os.path.exists(path) and not os.path.exists(path + ".bak"):
                try:
                    import shutil
                    shutil.copy2(path, path + ".bak")
                except Exception:
                    pass
        except Exception:
            pass
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except OSError as e:
            # Gracefully handle no-space-left-on-device: attempt to write a tiny marker file,
            # otherwise skip persisting this update to avoid crashing the workflow.
            if e.errno == 28:  # ENOSPC
                try:
                    # Write a minimal marker without fsync to reduce space pressure
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write('{"error":"trace logging skipped due to ENOSPC"}')
                except Exception:
                    pass
            else:
                raise
