import os
import logging
from uuid import uuid4
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from langchain_core.documents import Document
from langchain.storage import LocalFileStore
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams


class VectorStore:
    """
    Minimal wrapper for a local Qdrant vector store with hybrid (dense + sparse) search.
    Designed to be used by eyetools RAG tool. Uses on-disk Qdrant and a LocalFileStore
    for chunk payloads.
    """

    def __init__(self, *, collection_name: str, embedding, embedding_dim: int, top_k: int,
                 vector_local_path: str, doc_local_path: str):
        self.logger = logging.getLogger(__name__)
        self.collection_name = collection_name
        self.embedding = embedding
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.vector_local_path = vector_local_path
        self.doc_local_path = doc_local_path
        Path(self.vector_local_path).mkdir(parents=True, exist_ok=True)
        Path(self.doc_local_path).mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=self.vector_local_path)

    def _ensure_collection(self):
        try:
            collections = self.client.get_collections().collections
            if any(c.name == self.collection_name for c in collections):
                return
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": VectorParams(size=self.embedding_dim, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams()},
            )
            self.logger.info(f"Created qdrant collection {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Failed ensuring collection: {e}")
            raise

    def get_vectorstore(self) -> Tuple[QdrantVectorStore, LocalFileStore]:
        self._ensure_collection()
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        vs = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        docstore = LocalFileStore(self.doc_local_path)
        return vs, docstore

    def add_documents(self, document_chunks: List[str], document_path: str, ids: Optional[List[str]] = None) -> List[str]:
        self._ensure_collection()
        vs, docstore = self.get_vectorstore()
        ids = ids or [str(uuid4()) for _ in document_chunks]
        docs = [
            Document(
                page_content=chunk,
                metadata={
                    "source": os.path.basename(document_path),
                    "doc_id": ids[i],
                    "source_path": os.path.abspath(document_path),
                },
            )
            for i, chunk in enumerate(document_chunks)
        ]
        if docs:
            vs.add_documents(documents=docs, ids=ids)
            # Store raw content bytes
            docstore.mset(list(zip(ids, [c.encode("utf-8") for c in document_chunks])))
        return ids

    def similarity_search(self, query: str) -> List[Dict[str, Any]]:
        vs, docstore = self.get_vectorstore()
        results = vs.similarity_search_with_score(query=query, k=self.top_k)
        out: List[Dict[str, Any]] = []
        for doc, score in results:
            content_bytes = docstore.mget([doc.metadata["doc_id"]])[0]
            content = content_bytes.decode("utf-8") if content_bytes else doc.page_content
            out.append({
                "id": doc.metadata.get("doc_id"),
                "content": content,
                "score": float(score),
                "source": doc.metadata.get("source"),
                "source_path": doc.metadata.get("source_path"),
            })
        return out
