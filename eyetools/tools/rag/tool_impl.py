from __future__ import annotations
import os
import re
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_text_file(path: Path, max_bytes: int = 2_000_000) -> str:
    try:
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff"}:
            return ""
        data = path.read_bytes()
        return data[:max_bytes].decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _split_docs(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    if not text:
        return []
    # simple paragraph based split with fallback sliding window
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: List[str] = []
    for p in paras:
        if len(p) <= chunk_size:
            chunks.append(p)
        else:
            start = 0
            while start < len(p):
                end = min(len(p), start + chunk_size)
                chunks.append(p[start:end])
                if end == len(p):
                    break
                start = max(end - overlap, start + 1)
    return chunks


def _tokenize(s: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9_]+", s.lower()) if t]


def _bm25_like_score(query_tokens: List[str], doc_tokens: List[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    # simple term frequency match
    from collections import Counter
    q = Counter(query_tokens)
    d = Counter(doc_tokens)
    score = 0.0
    for term, qn in q.items():
        dn = d.get(term, 0)
        if dn > 0:
            score += (1.0 + 0.5 * (qn - 1)) * (1.0 + 0.25 * (dn - 1))
    # normalize by doc length a bit
    return score / (1.0 + 0.0005 * len(doc_tokens))


class RAGQueryTool:
    """Lightweight local RAG tool.

    Constructor signature and methods align with eyetools.core.ToolManager expectations:
      - __init__(meta, params)
      - ensure_model_loaded() for lazy prep
      - predict({"inputs": {"query": str, "top_k"?: int}}) -> Dict
      - optional describe_outputs(meta, params)
    """

    def __init__(self, meta: Dict[str, Any], params: Dict[str, Any]):
        self.meta = meta
        self.params = params or {}
        self._prepared = False
        self._index: List[Tuple[str, str, List[str]]] = []  # (source, chunk, tokens)
        # config
        self.default_top_k = int(self.params.get("top_k", 5))
        self.corpus_dirs = self.params.get("corpus_dirs") or []
        # allow env override
        env_dirs = os.getenv("EYETOOLS_RAG_DIRS")
        if env_dirs:
            self.corpus_dirs = [p for p in env_dirs.split(":") if p]

    @staticmethod
    def describe_outputs(meta: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "schema": (meta.get("io") or {}).get("output_schema", {}),
            "fields": {
                "items": "Top-k snippets with source and score",
                "source": "rag",
            },
        }

    def prepare(self):
        # lightweight check
        return True

    def ensure_model_loaded(self):
        if self._prepared:
            return
        self._build_index()
        self._prepared = True

    def _iter_candidate_files(self) -> List[Path]:
        files: List[Path] = []
        for p in self.corpus_dirs:
            root = Path(p).resolve()
            if root.is_file():
                files.append(root)
                continue
            if not root.exists():
                continue
            for ext in ("*.md", "*.markdown", "*.txt", "*.rst", "*.py", "*.yml", "*.yaml"):
                files.extend(root.rglob(ext))
        # de-dup by path string
        out: List[Path] = []
        seen = set()
        for f in files:
            s = str(f)
            if s not in seen:
                out.append(f)
                seen.add(s)
        return out

    def _build_index(self):
        start = time.time()
        self._index.clear()
        for f in self._iter_candidate_files():
            txt = _read_text_file(f)
            if not txt:
                continue
            for chunk in _split_docs(txt):
                toks = _tokenize(chunk)
                if toks:
                    self._index.append((str(f), chunk, toks))
        # Keep a shallow cap to avoid memory blow-up
        if len(self._index) > 50_000:
            self._index = self._index[:50_000]
        # print small telemetry into meta cache if needed
        self._telemetry = {"chunks": len(self._index), "build_sec": round(time.time() - start, 3)}

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if (self.meta.get("model", {}) or {}).get("lazy", True) and not self._prepared:
            self.ensure_model_loaded()
        inputs = request.get("inputs") if isinstance(request, dict) else None
        if not isinstance(inputs, dict):
            inputs = {}
        query = str(inputs.get("query") or "").strip()
        if not query:
            return {"items": [], "source": "rag", "warning": "empty query"}
        top_k = int(inputs.get("top_k") or self.default_top_k)
        q_toks = _tokenize(query)
        scored: List[Tuple[float, int]] = []  # (score, idx)
        for i, (_src, _chunk, toks) in enumerate(self._index):
            s = _bm25_like_score(q_toks, toks)
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        items: List[Dict[str, Any]] = []
        for s, i in scored[:top_k]:
            src, chunk, _ = self._index[i]
            items.append({"title": Path(src).name, "text": chunk[:1200], "source": src, "score": round(float(s), 4)})
        return {"items": items, "source": "rag"}

__all__ = ["RAGQueryTool"]
