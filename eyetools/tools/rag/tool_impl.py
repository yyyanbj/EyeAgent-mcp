from __future__ import annotations
import os
import re
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import hashlib
import sys
import importlib.util
from loguru import logger


def _read_text_file(path: Path, max_bytes: int = 2_000_000, pdf_max_pages: int | None = 300) -> str:
    try:
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff"}:
            return ""
        # Basic PDF support (requires pdfminer.six in the tool environment)
        if path.suffix.lower() == ".pdf":
            try:
                from pdfminer.high_level import extract_text  # type: ignore
                # Limit pages to avoid excessive warmup time on large books
                # Accept externally provided page cap
                text = extract_text(str(path), maxpages=pdf_max_pages)
                if not text:
                    return ""
                return text[:max_bytes]
            except Exception:
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
        self._index = []  # type: List[Tuple[str, str, List[str]]]  # (source, chunk, tokens)
        # config
        self.default_top_k = int(self.params.get("top_k", 5))
        self.corpus_dirs = self.params.get("corpus_dirs") or []
        # vector mode config
        # Retrieval mode: "local" (keyword), "qdrant" (hybrid vector), or "faiss" (local vector)
        self.mode = str(self.params.get("mode", "local")).lower()
        self.collection_name = str(self.params.get("collection_name", "rag_collection"))
        self.vector_local_path = str(self.params.get("vector_local_path", "../../temp/rag/qdrant")).strip()
        self.doc_local_path = str(self.params.get("doc_local_path", "../../temp/rag/docstore")).strip()
        self.chunk_size = int(self.params.get("chunk_size", 800))
        self.chunk_overlap = int(self.params.get("chunk_overlap", 120))
        self._vector_ready = False
        # FAISS (local vector) specific settings
        self.faiss_index_dir = str(self.params.get("faiss_index_dir", "../../temp/rag/faiss_index")).strip()
        self.embedding_model = str(self.params.get("embedding_model", "BAAI/bge-m3"))
        self.bookmeta = str(self.params.get("bookmeta", "auto")).lower()  # "auto" | "filename"
        self._faiss_ready = False
        # PDF page cap
        try:
            self.pdf_max_pages = int(self.params.get("maxpages", 300))
        except Exception:
            self.pdf_max_pages = 300
        # allow env override
        env_dirs = os.getenv("EYETOOLS_RAG_DIRS")
        if env_dirs:
            self.corpus_dirs = [p for p in env_dirs.split(":") if p]
        # Resolve corpus directories relative to tool root so defaults work regardless of CWD
        try:
            base_dir = Path(self.meta.get("root_dir") or Path(__file__).parent).resolve()
        except Exception:
            base_dir = Path(__file__).parent.resolve()
        resolved: List[Path] = []
        for p in self.corpus_dirs:
            try:
                pp = Path(p)
                if not pp.is_absolute():
                    pp = (base_dir / pp).resolve()
                else:
                    pp = pp.resolve()
                resolved.append(pp)
            except Exception:
                # ignore malformed path entries
                continue
        self._resolved_dirs = resolved

    @staticmethod
    def describe_outputs(meta: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "schema": (meta.get("io") or {}).get("output_schema", {}),
            "fields": {
                "items": "Top-k snippets with source and score",
                "inference_time": "seconds",
            },
        }

    def prepare(self):
        # lightweight check
        return True

    def ensure_model_loaded(self):
        if self._prepared:
            return
        # For vector mode we defer to warmup() for ingestion to avoid duplicate work
        if self.mode in ("qdrant", "faiss"):
            # Do not ingest here; warmup will handle ingestion explicitly when invoked
            pass
        else:
            self._build_index()
        self._prepared = True

    def _iter_candidate_files(self) -> List[Path]:
        files: List[Path] = []
        # prefer pre-resolved dirs; fallback to raw strings if empty
        dirs = self._resolved_dirs if getattr(self, "_resolved_dirs", None) else [Path(p) for p in self.corpus_dirs]
        for p in dirs:
            root = p if isinstance(p, Path) else Path(str(p)).resolve()
            if root.is_file():
                files.append(root)
                continue
            if not root.exists():
                continue
            for ext in ("*.md", "*.markdown", "*.txt", "*.rst", "*.py", "*.yml", "*.yaml", "*.pdf"):
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
            txt = _read_text_file(f, pdf_max_pages=self.pdf_max_pages)
            if not txt:
                continue
            for chunk in _split_docs(txt, chunk_size=self.chunk_size, overlap=self.chunk_overlap):
                toks = _tokenize(chunk)
                if toks:
                    self._index.append((str(f), chunk, toks))
        # Keep a shallow cap to avoid memory blow-up
        if len(self._index) > 50_000:
            self._index = self._index[:50_000]
        # print small telemetry into meta cache if needed
        self._telemetry = {"chunks": len(self._index), "build_sec": round(time.time() - start, 3)}

    def warmup(self) -> Dict[str, Any]:
        """Optional warmup hook: for vector mode perform ingestion here."""
        start = time.time()
        if self.mode not in ("qdrant", "faiss"):
            # For local mode, just ensure index built
            self.ensure_model_loaded()
            return {"warmed_up": True, "mode": self.mode, "chunks": len(self._index), "sec": round(time.time() - start, 3)}
        # Vector mode ingestion
        try:
            if self.mode == "faiss":
                # Build a local FAISS index from PDF documents with per-page metadata
                return self._warmup_faiss(start)
            # Import inside to keep main process free of deps; support both package and path-based loading
            def _load_vectorstore_cls():
                try:
                    from .vectorstore_qdrant import VectorStore  # type: ignore
                    return VectorStore
                except Exception:
                    here = Path(__file__).parent
                    vs_path = here / "vectorstore_qdrant.py"
                    spec = importlib.util.spec_from_file_location("rag_vectorstore_qdrant", vs_path)
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        sys.modules.setdefault("rag_vectorstore_qdrant", mod)
                        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                        return getattr(mod, "VectorStore")
                    raise

            VectorStore = _load_vectorstore_cls()
            # Minimal FastEmbed wrapper (no external keys needed)
            try:
                from fastembed import TextEmbedding  # type: ignore
                from langchain_core.embeddings import Embeddings  # type: ignore
            except Exception as e:
                return {"warmed_up": False, "mode": self.mode, "error": f"fastembed missing: {e}"}

            class _FastEmbedWrapper(Embeddings):
                def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
                    self.model = TextEmbedding(model_name=model_name)
                    self._dim: Optional[int] = None

                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    # fastembed returns a generator of numpy arrays or lists
                    vecs = list(self.model.embed(texts))
                    if vecs and self._dim is None:
                        try:
                            self._dim = len(vecs[0])
                        except Exception:
                            pass
                    # Ensure list of lists (serialize friendly)
                    return [list(map(float, v)) for v in vecs]

                def embed_query(self, text: str) -> List[float]:
                    vecs = list(self.model.embed([text]))
                    if vecs and self._dim is None:
                        try:
                            self._dim = len(vecs[0])
                        except Exception:
                            pass
                    return list(map(float, vecs[0])) if vecs else []

                @property
                def dimension(self) -> int:
                    if self._dim is None:
                        # probe with a dummy
                        _ = self.embed_query("dim")
                    return int(self._dim or 384)

            embedder = _FastEmbedWrapper()
            emb_dim = embedder.dimension
            # Resolve vector/doc store paths relative to tool root
            base_dir = Path(self.meta.get("root_dir") or Path(__file__).parent).resolve()
            vec_path = (base_dir / Path(self.vector_local_path)).resolve()
            doc_path = (base_dir / Path(self.doc_local_path)).resolve()
            vs = VectorStore(
                collection_name=self.collection_name,
                embedding=embedder,
                embedding_dim=emb_dim,
                top_k=self.default_top_k,
                vector_local_path=str(vec_path),
                doc_local_path=str(doc_path),
            )
            added_docs = 0
            added_chunks = 0
            for f in self._iter_candidate_files():
                txt = _read_text_file(f, pdf_max_pages=self.pdf_max_pages)
                if not txt:
                    continue
                chunks = _split_docs(txt, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
                if not chunks:
                    continue
                # Stable IDs to avoid trivial duplicates on repeated warmups
                base = hashlib.sha1(str(f).encode("utf-8")).hexdigest()[:16]
                ids = [f"{base}_{i}" for i in range(len(chunks))]
                try:
                    vs.add_documents(chunks, str(f), ids=ids)
                    added_docs += 1
                    added_chunks += len(chunks)
                except Exception:
                    # Continue on per-file errors
                    continue
            self._vector_ready = True
            # Keep reference config for predict
            self._vs_cfg = {
                "collection_name": self.collection_name,
                "vector_local_path": str(vec_path),
                "doc_local_path": str(doc_path),
                "dim": emb_dim,
            }
            return {
                "warmed_up": True,
                "mode": self.mode,
                "docs": added_docs,
                "chunks": added_chunks,
                "emb_dim": emb_dim,
                "sec": round(time.time() - start, 3),
            }
        except Exception as e:  # noqa
            return {"warmed_up": False, "mode": self.mode, "error": str(e)}

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        if (self.meta.get("model", {}) or {}).get("lazy", True) and not self._prepared:
            self.ensure_model_loaded()
        inputs = request.get("inputs") if isinstance(request, dict) else None
        if not isinstance(inputs, dict):
            inputs = {}
        query = str(inputs.get("query") or "").strip()
        if not query:
            return {"items": [], "warning": "empty query", "inference_time": round(time.time() - start, 4)}
        top_k = int(inputs.get("top_k") or self.default_top_k)

        logger.debug(f"mode={self.mode} query='{query}' top_k={top_k}")

        # Vector mode predict
        if self.mode == "qdrant":
            try:
                def _load_vectorstore_cls():
                    try:
                        from .vectorstore_qdrant import VectorStore  # type: ignore
                        return VectorStore
                    except Exception:
                        here = Path(__file__).parent
                        vs_path = here / "vectorstore_qdrant.py"
                        spec = importlib.util.spec_from_file_location("rag_vectorstore_qdrant", vs_path)
                        if spec and spec.loader:
                            mod = importlib.util.module_from_spec(spec)
                            sys.modules.setdefault("rag_vectorstore_qdrant", mod)
                            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                            return getattr(mod, "VectorStore")
                        raise

                VectorStore = _load_vectorstore_cls()
                # Recreate vectorstore on demand using stored cfg and a fresh embedder
                from fastembed import TextEmbedding  # type: ignore
                from langchain_core.embeddings import Embeddings  # type: ignore

                class _FastEmbedWrapper(Embeddings):
                    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
                        self.model = TextEmbedding(model_name=model_name)
                    def embed_documents(self, texts: List[str]) -> List[List[float]]:
                        return [list(map(float, v)) for v in self.model.embed(texts)]
                    def embed_query(self, text: str) -> List[float]:
                        return list(map(float, list(self.model.embed([text]))[0]))

                cfg = getattr(self, "_vs_cfg", None)
                base_dir = Path(self.meta.get("root_dir") or Path(__file__).parent).resolve()
                vec_path = (base_dir / Path(self.vector_local_path)).resolve()
                doc_path = (base_dir / Path(self.doc_local_path)).resolve()
                emb = _FastEmbedWrapper()
                vs = VectorStore(
                    collection_name=self.collection_name,
                    embedding=emb,
                    embedding_dim=int((cfg or {}).get("dim", 384)),
                    top_k=top_k,
                    vector_local_path=str(vec_path),
                    doc_local_path=str(doc_path),
                )
                results = vs.similarity_search(query)
                items: List[Dict[str, Any]] = []
                for r in results[:top_k]:
                    # Build structured source: material (book/file name without extension), page not available in qdrant path
                    src_path = r.get("source_path") or r.get("source") or ""
                    material = Path(str(src_path)).stem if src_path else (r.get("source") or "")
                    items.append({
                        "title": r.get("source") or Path(str(src_path)).name,
                        "text": (r.get("content") or "")[:1200],
                        "source": {"material": str(material), "page": None},
                        "score": round(float(r.get("score") or 0.0), 4),
                    })
                return {"items": items, "inference_time": round(time.time() - start, 4)}
            except Exception as e:  # noqa
                return {"items": [], "warning": f"vector search failed: {e}", "inference_time": round(time.time() - start, 4)}
        if self.mode == "faiss":
            try:
                # Lazy import to keep dependencies scoped to the tool environment
                from langchain_community.vectorstores import FAISS  # type: ignore
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
                except Exception:
                    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
                # Resolve index directory relative to tool root
                base_dir = Path(self.meta.get("root_dir") or Path(__file__).parent).resolve()
                index_dir = (base_dir / Path(self.faiss_index_dir)).resolve()

                logger.debug(f"FAISS index_dir={index_dir}")

                if not (index_dir / "index.faiss").exists():
                    return {"items": [], "warning": f"FAISS index missing at {index_dir}. Run warmup first.", "inference_time": round(time.time() - start, 4)}
                
                emb = HuggingFaceEmbeddings(model_name=self.embedding_model, show_progress=False)
                
                logger.debug(f"FAISS using embedding model={self.embedding_model}")
                
                vs = FAISS.load_local(str(index_dir), emb, allow_dangerous_deserialization=True)
                logger.debug(f"FAISS loaded index from {index_dir}")

                try:
                    results = vs.similarity_search_with_score(query, k=top_k)
                except AttributeError:
                    # Fallback for environments without *_with_score API
                    docs_only = vs.similarity_search(query, k=top_k)
                    results = [(d, 0.0) for d in docs_only]

                logger.debug(f"FAISS query='{query}' top_k={top_k} results={len(results)}")
                items: List[Dict[str, Any]] = []
                for doc, score in results:
                    md = doc.metadata or {}
                    book = md.get("book") or Path(md.get("source", "")).stem
                    page = md.get("page_number") or md.get("page")
                    title = f"{book} p.{page}" if page is not None else str(book)
                    items.append({
                        "title": title,
                        "text": doc.page_content[:1200],
                        "source": {"material": str(book), "page": int(page) if page is not None else None},
                        "score": round(float(score), 4),
                    })
                return {"items": items, "inference_time": round(time.time() - start, 4)}
            except Exception as e:
                # Log full traceback for diagnostics and return structured warning
                try:
                    logger.exception("FAISS predict error")
                except Exception:
                    pass
                etype = type(e).__name__
                emsg = str(e) or repr(e)
                return {"items": [], "warning": f"faiss search failed: {etype}: {emsg}", "inference_time": round(time.time() - start, 4)}
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
            # Structured source for local mode; page information is not available
            items.append({
                "title": Path(src).name,
                "text": chunk[:1200],
                "source": {"material": Path(src).stem, "page": None},
                "score": round(float(s), 4),
            })
        return {"items": items, "inference_time": round(time.time() - start, 4)}

    # --------------------------
    # FAISS warmup/ingestion impl
    # --------------------------
    def _warmup_faiss(self, start_time: float) -> Dict[str, Any]:
        """Build a local FAISS index from PDFs, preserving (book title, page number) metadata."""
        try:
            from langchain_community.document_loaders import PyPDFLoader  # type: ignore
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
            from langchain_community.vectorstores import FAISS  # type: ignore
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
            except Exception:
                from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
        except Exception as e:
            return {"warmed_up": False, "mode": self.mode, "error": f"missing deps for faiss mode: {e}"}

        def _human_page(zero_based: int) -> int:
            return int(zero_based) + 1

        def _get_pdf_title(pdf_path: Path) -> str:
            """Try reading PDF metadata title; fallback to filename (stem)."""
            try:
                from pypdf import PdfReader  # type: ignore
                reader = PdfReader(str(pdf_path))
                title = None
                if getattr(reader, "metadata", None):
                    meta = reader.metadata
                    title = getattr(meta, "title", None) or (meta.get("/Title") if hasattr(meta, "get") else None)
                if title and str(title).strip():
                    return str(title).strip()
            except Exception:
                pass
            return pdf_path.stem

        # Collect PDF files only for FAISS ingestion
        pdf_files = [p for p in self._iter_candidate_files() if str(p).lower().endswith(".pdf")]
        if not pdf_files:
            return {"warmed_up": False, "mode": self.mode, "error": "no PDF files found in corpus_dirs"}

        # Load per-page docs with metadata (book, page_number)
        page_docs: List[Any] = []  # LangChain Document
        for pdf in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf))
                per_page = loader.load()  # one Document per page; metadata has {source, page}
                # set book title
                if self.bookmeta == "filename":
                    book_title = Path(pdf).stem
                else:
                    book_title = _get_pdf_title(pdf)
                for d in per_page:
                    md = dict(d.metadata or {})
                    md["book"] = book_title
                    md["page_number"] = _human_page(md.get("page", 0)) if "page" in md else None
                    d.metadata = md
                page_docs.extend(per_page)
            except Exception:
                # Continue on individual file failures
                continue

        if not page_docs:
            return {"warmed_up": False, "mode": self.mode, "error": "no pages loaded from PDFs"}

        # Split into chunks while preserving metadata
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )
        split_docs = splitter.split_documents(page_docs)
        if not split_docs:
            return {"warmed_up": False, "mode": self.mode, "error": "no chunks after split"}

        # Build embeddings and FAISS index
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model, show_progress=True)
        vs = FAISS.from_documents(split_docs, embeddings)

        # Save index
        base_dir = Path(self.meta.get("root_dir") or Path(__file__).parent).resolve()
        index_dir = (base_dir / Path(self.faiss_index_dir)).resolve()
        index_dir.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(index_dir))
        self._faiss_ready = True
        return {
            "warmed_up": True,
            "mode": self.mode,
            "pdfs": len(pdf_files),
            "pages": len(page_docs),
            "chunks": len(split_docs),
            "index_dir": str(index_dir),
            "embedding_model": self.embedding_model,
            "sec": round(time.time() - start_time, 3),
        }

__all__ = ["RAGQueryTool"]
