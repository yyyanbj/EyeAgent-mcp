# RAG Tool (rag:query)

Lightweight local RAG over markdown/txt sources. Uses simple token scoring to return top-k snippets.

- Input: { query: string, top_k?: number }
- Output: { items: [{ title, text, source, score }], source: "rag" }

Configure corpus via:
- config.yaml params.corpus_dirs
- or environment variable EYETOOLS_RAG_DIRS=path1:path2