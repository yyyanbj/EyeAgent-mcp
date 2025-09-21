# Web Search Tools (web_search:pubmed, web_search:tavily)

Simple wrappers around PubMed e-utilities and Tavily API.

- Input: { query: string, top_k?: number }
- Output: { items: [...], source: "pubmed"|"tavily" }

Notes:
- PubMed: uses ESearch+ESummary (no key needed for small requests). Network failures return empty items.
- Tavily: requires env var TAVILY_API_KEY; if missing, returns empty items with a warning field.
