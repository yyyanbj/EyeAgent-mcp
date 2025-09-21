from __future__ import annotations
import os
import time
import json
from typing import Any, Dict, List
from urllib.parse import urlencode
from urllib.request import urlopen, Request


def _http_get_json(url: str, headers: Dict[str, str] | None = None, timeout: int = 10) -> Dict[str, Any] | List[Any] | None:
    try:
        req = Request(url, headers=headers or {"User-Agent": "eyetools-websearch/0.1"})
        with urlopen(req, timeout=timeout) as resp:  # nosec - trusted simple GET
            data = resp.read().decode("utf-8", errors="ignore")
            try:
                return json.loads(data)
            except Exception:
                return None
    except Exception:
        return None


class WebSearchTool:
    """Unified web search tool.

    Variants are distinguished by params.provider ("pubmed" or "tavily").
    Output schema: { items: [ { id/title/abstract/url/year? } ], source: provider }
    """

    def __init__(self, meta: Dict[str, Any], params: Dict[str, Any]):
        self.meta = meta
        self.params = params or {}
        self.provider = (self.params.get("provider") or "pubmed").lower()
        self.default_top_k = int(self.params.get("top_k", 3))

    @staticmethod
    def describe_outputs(meta: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        prov = (params or {}).get("provider", "pubmed")
        return {
            "schema": (meta.get("io") or {}).get("output_schema", {}),
            "fields": {"items": f"Top results from {prov}", "source": prov},
        }

    def prepare(self):
        return True

    def ensure_model_loaded(self):
        # no heavy model
        return

    def _search_pubmed(self, query: str, top_k: int) -> Dict[str, Any]:
        # Use Entrez e-utilities: esearch then esummary. No API key required for small requests.
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        q = urlencode({"db": "pubmed", "term": query, "retmax": str(max(1, min(50, top_k)))})
        ids = []
        js = _http_get_json(f"{base}/esearch.fcgi?retmode=json&{q}")
        if isinstance(js, dict):
            try:
                ids = js.get("esearchresult", {}).get("idlist", [])
            except Exception:
                ids = []
        items: List[Dict[str, Any]] = []
        if ids:
            q2 = urlencode({"db": "pubmed", "id": ",".join(ids)})
            js2 = _http_get_json(f"{base}/esummary.fcgi?retmode=json&{q2}")
            if isinstance(js2, dict):
                try:
                    result = js2.get("result", {})
                    for pid in ids:
                        r = result.get(pid)
                        if isinstance(r, dict):
                            title = r.get("title")
                            year = None
                            try:
                                year = int((r.get("pubdate") or "").split(" ")[0].split("-")[0] or 0) or None
                            except Exception:
                                year = None
                            url = f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
                            items.append({"id": pid, "title": title, "abstract": None, "url": url, "year": year})
                except Exception:
                    pass
        return {"items": items[:top_k], "source": "pubmed"}

    def _search_tavily(self, query: str, top_k: int) -> Dict[str, Any]:
        # Tavily API requires api_key; if missing or network blocked, return graceful empty results.
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return {"items": [], "source": "tavily", "warning": "TAVILY_API_KEY not set"}
        params = {"api_key": api_key, "query": query, "max_results": max(1, min(10, top_k))}
        url = "https://api.tavily.com/search?" + urlencode(params)
        js = _http_get_json(url)
        items: List[Dict[str, Any]] = []
        if isinstance(js, dict):
            for r in js.get("results", []) or []:
                if isinstance(r, dict):
                    items.append({
                        "title": r.get("title"),
                        "url": r.get("url"),
                        "content": r.get("content"),
                    })
        return {"items": items[:top_k], "source": "tavily"}

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        inputs = request.get("inputs") if isinstance(request, dict) else None
        if not isinstance(inputs, dict):
            inputs = {}
        query = str(inputs.get("query") or "").strip()
        if not query:
            return {"items": [], "source": self.provider, "warning": "empty query"}
        top_k = int(inputs.get("top_k") or self.default_top_k)
        if self.provider == "tavily":
            return self._search_tavily(query, top_k)
        return self._search_pubmed(query, top_k)

__all__ = ["WebSearchTool"]
