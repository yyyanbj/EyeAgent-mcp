from __future__ import annotations
import os
import json
import uuid
import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List


def ISO() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class TraceLogger:
    """Record sync events and final report for a diagnostic workflow.

    Directory layout:
        <base_dir>/<case_id>/
            trace.json
            final_report.json
            conversation.jsonl
    """

    def __init__(self, base_dir: str | None = None):
        base = base_dir or os.environ.get("EYEAGENT_CASES_DIR")
        if not base:
            base = str(Path(__file__).resolve().parents[2] / "cases")
        self.base_dir = base
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

    def create_case(self, patient: Dict[str, Any] | None = None, images: List[Dict[str, Any]] | None = None) -> str:
        case_id = str(uuid.uuid4())
        os.makedirs(self._case_dir(case_id), exist_ok=True)
        doc = {
            "case_id": case_id,
            "created_at": ISO(),
            "updated_at": ISO(),
            "patient": patient or {},
            "images": images or [],
            "events": [],
            "next_seq": 1,
        }
        self._atomic_write_json(self._trace_path(case_id), doc)
        return case_id

    def append_event(self, case_id: str, event: Dict[str, Any]):
        with self._lock:
            path = self._trace_path(case_id)
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self._atomic_write_json(path, {
                    "case_id": case_id,
                    "created_at": ISO(),
                    "updated_at": ISO(),
                    "patient": {},
                    "images": [],
                    "events": [],
                    "next_seq": 1,
                    "note": "Auto-created by TraceLogger.append_event"
                })
            try:
                with open(path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
            except Exception:
                doc = {"case_id": case_id, "events": [], "next_seq": 1}
            event = dict(event or {})
            event.setdefault("ts", ISO())
            try:
                seq = int(doc.get("next_seq") or 1)
            except Exception:
                seq = 1
            event["seq"] = seq
            doc["next_seq"] = seq + 1
            doc.setdefault("events", []).append(event)
            doc["updated_at"] = ISO()
            self._atomic_write_json(path, doc)

    def append_conversation_message(self, case_id: str, message: Dict[str, Any]):
        with self._lock:
            # advance sequence in trace.json
            path = self._trace_path(case_id)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
            except Exception:
                doc = {"next_seq": 1}
            try:
                seq = int(doc.get("next_seq") or 1)
            except Exception:
                seq = 1
            doc["next_seq"] = seq + 1
            doc["updated_at"] = ISO()
            self._atomic_write_json(path, doc)

            rec = dict(message or {})
            rec.setdefault("ts", ISO())
            rec["seq"] = seq
            cpath = self._conversation_path(case_id)
            os.makedirs(os.path.dirname(cpath), exist_ok=True)
            with open(cpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def write_final_report(self, case_id: str, report: Dict[str, Any]):
        with self._lock:
            self._atomic_write_json(self._final_path(case_id), report)
            # also reference in trace.json
            try:
                with open(self._trace_path(case_id), "r", encoding="utf-8") as f:
                    doc = json.load(f)
            except Exception:
                doc = {"case_id": case_id}
            doc["final_report_path"] = self._final_path(case_id)
            doc["updated_at"] = ISO()
            self._atomic_write_json(self._trace_path(case_id), doc)

    def load_trace(self, case_id: str) -> Dict[str, Any]:
        with open(self._trace_path(case_id), "r", encoding="utf-8") as f:
            return json.load(f)

    def list_cases(self) -> list[str]:
        if not os.path.exists(self.base_dir):
            return []
        return [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]

    # internals
    def _atomic_write_json(self, path: str, data: Dict[str, Any]):
        tmp = path + ".tmp"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except OSError as e:
            # ENOSPC safe-guard
            if getattr(e, "errno", None) == 28:
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write('{"error":"trace logging skipped due to ENOSPC"}')
                except Exception:
                    pass
            else:
                raise
