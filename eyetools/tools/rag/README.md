# RAG Tool (rag:query)

Lightweight local RAG over markdown/txt/PDF sources with three retrieval modes:

- local: keyword/BM25-like scoring (no vector dependencies)
- qdrant: hybrid vector search (dense + sparse) using local Qdrant
- faiss: local FAISS vector store built from PDFs with (book title, page) metadata

## Tool IDs
- rag:query

## Inputs / Outputs
- Input: `{ query: string, top_k?: number }`
- Output: `{ items: [{ title, text, source, score }], source: "rag", inference_time: number }`

## Configuration

The tool reads a set of corpus directories for files (.md/.txt/.rst/.py/.yml/.yaml) and PDFs, chunks them, and builds an index.

Priority of corpus configuration (high → low):
1. Environment variable `EYETOOLS_RAG_DIRS` (use colon to separate multiple paths, e.g. `/data/docs:/opt/notes`)
2. `config.yaml` → `variants[].params.corpus_dirs`

Note: Relative paths are resolved against the tool root (this folder), not the service CWD.

### Isolated runtime environment

This tool is configured to run in an isolated environment: `envs/py312-rag/pyproject.toml`. In `config.yaml` it sets:

- `shared.environment_ref: py312-rag`
- `runtime.load_mode: subprocess`

The service launches the tool via `uv run` in a subprocess using that environment, so dependencies (LangChain/Qdrant/FAISS) will not pollute the main project.

Requirement: a system Python 3.12 interpreter (`python3.12`). If not available, install it or adjust `requires-python` in `envs/py312-rag/pyproject.toml` to match your system.

To add/pin RAG dependencies, modify `envs/py312-rag/pyproject.toml` dependencies. No need to change `eyetools/pyproject.toml`.

## Parameters (config.yaml → variants[].params)

- mode: `qdrant` | `faiss` | `local`
	- qdrant: hybrid dense+sparse retrieval (requires warmup to vectorize and ingest)
	- faiss: local FAISS vector store (great for smaller PDF collections; requires warmup)
	- local: keyword-only matching (no extra deps; limited semantics)
- top_k: number of results to return
- maxpages: maximum PDF pages to parse during warmup (caps warmup time and storage size)
- collection_name: Qdrant collection name (qdrant mode)
- vector_local_path: Qdrant local storage directory (qdrant mode)
- doc_local_path: Document chunk local store (qdrant mode)
- faiss_index_dir: FAISS index path (faiss mode)
- embedding_model: HuggingFace embedding model (faiss mode)
- bookmeta: `auto` (prefer PDF metadata title, fallback to filename) or `filename`
- chunk_size / chunk_overlap: chunking configuration

## Warmup and retrieval

- warmup:
  - mode=qdrant: scans `corpus_dirs` (.pdf/.md/.txt/...), parses PDFs up to `maxpages`, chunks with `chunk_size`/`chunk_overlap`, embeds via FastEmbed, ingests into local Qdrant, and stores chunks into `doc_local_path`.
  - mode=faiss: scans `corpus_dirs` for PDFs only, loads them per-page (preserving `book` and 1-based `page_number`), chunks preserving metadata, embeds via HuggingFace, builds a FAISS index, and saves it to `faiss_index_dir`.
- predict:
  - mode=qdrant: hybrid similarity search over vector store, returns chunks and sources
  - mode=faiss: vector search over FAISS, returns chunks and sources, titles are formatted as `"<book> p.<page>"`
  - mode=local: keyword/BM25-like matching

### Prebuilding FAISS index via CLI

To reduce server startup latency, you can prebuild the FAISS index using the provided CLI:

```bash
# Example: build index from PDFs under one or more directories
python eyetools/tools/rag/faiss_cli.py ingest \
	--corpus-dir /abs/path/to/books \
	--corpus-dir /abs/path/to/notes \
	--index-dir  /abs/path/to/faiss_index \
	--embedding-model BAAI/bge-small-zh-v1.5 \
	--chunk-size 800 --chunk-overlap 120 \
	--bookmeta auto --maxpages 600
```

Then set `mode: faiss` and the same `faiss_index_dir` in `config.yaml`. The server will reuse the existing index and skip building at startup.

## Start service (MCP)

Make sure `eyetools/tools` is included in tool discovery, e.g.:

```bash
export EYETOOLS_TOOL_PATHS="/home/you/EyeAgent-mcp/eyetools/tools"
# or via CLI arg
eyetools-mcp serve --tools-dir /home/you/EyeAgent-mcp/eyetools/tools
```

Optional: set corpus directories (supports PDFs):

```bash
export EYETOOLS_RAG_DIRS="/path/to/books:/path/to/notes"
```

Note: Basic PDF text extraction is supported (pdfminer.six is included in the isolated rag environment). Warmup time is bounded by `maxpages`.

## Direct invocation (example)

HTTP (simplified example, actual routes may vary):

POST /predict
```json
{
	"tool_id": "rag:query",
	"request": { "inputs": { "query": "retinal detachment signs", "top_k": 5 } }
}
```

Response:
```json
{
	"output": {
		"items": [ { "title": "...", "text": "...", "source": "/abs/path/file.md", "score": 1.23 } ],
		"source": "rag",
		"inference_time": 0.0123
	},
	"__meta__": { "tool_id": "rag:query", ... }
}
```

## Troubleshooting

- Tool not discovered:
  - Ensure `--tools-dir` or `EYETOOLS_TOOL_PATHS` includes `eyetools/tools`.
  - Check `/tools` or `/mcp/tools` to see if `rag:query` is listed.

- Invocation works but items are empty:
  - Verify `EYETOOLS_RAG_DIRS` or `config.yaml` `corpus_dirs` points to real directories/files.
  - For local mode, ensure the query terms actually appear in the corpus (keyword match only).
  - For vector modes (`qdrant`, `faiss`), call `/admin/warmup?tool_id=rag:query` first to build the index.

- Route differences (/ vs /mcp):
  - Set `EYETOOLS_MCP_MOUNT_PATH=/mcp` to align with the client.
