#!/usr/bin/env python3
from __future__ import annotations
"""
FAISS RAG CLI (prebuild index to reduce server startup time)
------------------------------------------------------------
Build a local FAISS index from PDF files with per-page metadata:
  - book title (from PDF metadata when available, fallback to filename)
  - 1-based page number

This CLI mirrors the FAISS ingestion logic used by the rag:query tool's warmup(mode=faiss),
so the server can start faster and reuse the on-disk index.

Dependencies (installed in the rag isolated environment):
  - langchain-community, langchain
  - sentence-transformers (via HuggingFaceEmbeddings)
  - pypdf (for PDF metadata)
  - faiss-cpu

Usage examples:
  # Minimal (corpus required):
  python eyetools/tools/rag/faiss_cli.py ingest \
    --corpus-dir /abs/path/to/books \
    --index-dir  /abs/path/to/faiss_index

  # With options:
  python eyetools/tools/rag/faiss_cli.py ingest \
    --corpus-dir /abs/path/books --corpus-dir /abs/path/notes \
    --index-dir  ../../temp/rag/faiss_index \
    --embedding-model BAAI/bge-small-zh-v1.5 \
    --chunk-size 800 --chunk-overlap 120 \
    --bookmeta auto --maxpages 600
"""

import argparse
from pathlib import Path
from typing import List, Tuple
from loguru import logger


def human_page(page_index_zero_based: int) -> int:
    """Convert 0-based page index to human-friendly 1-based page number."""
    return int(page_index_zero_based) + 1


def get_pdf_title(pdf_path: Path) -> str:
    """Read PDF metadata title; fallback to filename stem."""
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


def find_pdfs(corpus_dirs: List[Path]) -> List[Path]:
    files: List[Path] = []
    for d in corpus_dirs:
        if d.is_file() and d.suffix.lower() == ".pdf":
            files.append(d)
        elif d.is_dir():
            files.extend(d.rglob("*.pdf"))
    # de-dup
    out: List[Path] = []
    seen = set()
    for f in files:
        s = str(f.resolve())
        if s not in seen:
            out.append(f)
            seen.add(s)
    return out


def ingest_faiss(
    corpus_dirs: List[Path],
    index_dir: Path,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    bookmeta: str,
    maxpages: int,
) -> Tuple[int, int]:
    """Build FAISS index from PDFs (+ optional text files); return (pages_loaded, chunks)."""
    from langchain_community.document_loaders import PyPDFLoader  # type: ignore
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
    from langchain_community.vectorstores import FAISS  # type: ignore
    from langchain_core.documents import Document  # type: ignore
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
    except Exception:  # noqa: fallback for older langchain versions
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

    pdfs = find_pdfs(corpus_dirs)

    # Load docs per page and attach metadata
    page_docs = []
    for pdf in pdfs:
        logger.info(f"Loading PDF: {pdf}")
        try:
            loader = PyPDFLoader(str(pdf))
            # PyPDFLoader doesn't support a direct maxpages; we load all pages,
            # and rely on downstream chunking/embedding limits and the CLI maxpages
            # is informational only here. If needed, users can trim their corpus.
            per_page = loader.load()  # one Document per page
            if bookmeta == "filename":
                book_title = pdf.stem
            else:
                book_title = get_pdf_title(pdf)
            for d in per_page:
                md = dict(d.metadata or {})
                md["book"] = book_title
                md["page_number"] = human_page(md.get("page", 0)) if "page" in md else None
                d.metadata = md
            page_docs.extend(per_page)
        except Exception as e:
            logger.error(f"Failed loading PDF {pdf}: {e}")
            # Continue on file-level errors
            continue

    # Also load plain text/markdown files in the same corpus
    text_files: List[Path] = []
    for d in corpus_dirs:
        if d.is_dir():
            for pattern in ("*.md", "*.markdown", "*.txt", "*.rst", "*.py", "*.yml", "*.yaml"):
                text_files.extend(d.rglob(pattern))
        elif d.is_file() and d.suffix.lower() in {".md", ".markdown", ".txt", ".rst", ".py", ".yml", ".yaml"}:
            text_files.append(d)

    text_docs: List[Document] = []
    for tf in text_files:
        try:
            data = tf.read_bytes().decode("utf-8", errors="ignore")
            if not data.strip():
                continue
            md = {"book": tf.stem, "page_number": None, "source": str(tf.resolve())}
            text_docs.append(Document(page_content=data, metadata=md))
        except Exception:
            continue

    if not page_docs and not text_docs:
        raise SystemExit("No documents loaded. PDFs may be scanned without text; consider OCR or include text/markdown files.")

    # Optional soft cap: if user provided maxpages, trim total pages before chunking
    if isinstance(maxpages, int) and maxpages > 0 and len(page_docs) > maxpages:
        page_docs = page_docs[:maxpages]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    split_docs = splitter.split_documents(page_docs + text_docs)
    if not split_docs:
        raise SystemExit("No chunks produced after splitting.")

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model, show_progress=True)
    vs = FAISS.from_documents(split_docs, embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    return (len(page_docs), len(split_docs))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FAISS RAG CLI (prebuild PDF index)")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Ingest PDFs and build FAISS index")
    ing.add_argument("--corpus-dir", action="append", required=True, help="Corpus directory (repeatable)")
    ing.add_argument("--index-dir", required=True, help="Output directory for FAISS index")
    ing.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5", help="HuggingFace embedding model")
    ing.add_argument("--chunk-size", type=int, default=800)
    ing.add_argument("--chunk-overlap", type=int, default=120)
    ing.add_argument("--bookmeta", choices=["auto", "filename"], default="auto", help="Book title source")
    ing.add_argument("--maxpages", type=int, default=600, help="Soft cap for total pages (global)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "ingest":
        corpus_dirs = [Path(c) for c in (args.corpus_dir or [])]
        index_dir = Path(args.index_dir)
        pages, chunks = ingest_faiss(
            corpus_dirs=corpus_dirs,
            index_dir=index_dir,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            bookmeta=args.bookmeta,
            maxpages=args.maxpages,
        )
        print(f"Built FAISS index at {index_dir} (pages={pages}, chunks={chunks})")


if __name__ == "__main__":
    main()
