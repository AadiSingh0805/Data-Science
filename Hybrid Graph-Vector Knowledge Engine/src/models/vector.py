"""Dense Vector Search Engine using Qdrant and Sentence Transformers.
Implements HNSW indexing and Cosine similarity retrieval over document chunks.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
from src.data.contracts import DocumentChunk


class VectorEngine:
    """Manages dense chunk embeddings and HNSW vector search in Qdrant."""

    def __init__(
        self,
        collection_name: str = "sec_10k_chunks",
        storage_path: Optional[str] = "data/processed/qdrant_db",
        model_name: str = "all-MiniLM-L6-v2",
        vector_dim: int = 384
    ):
        self.collection_name = collection_name
        self.vector_dim = vector_dim

        # In-memory or local directory persistence
        if storage_path:
            Path(storage_path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=storage_path)
        else:
            self.client = QdrantClient(":memory:")

        # Load embedding model
        self.encoder = SentenceTransformer(model_name)
        self._init_collection()

    def _init_collection(self) -> None:
        """Initializes Qdrant collection with Cosine distance metric."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_dim,
                    distance=qmodels.Distance.COSINE
                )
            )

    def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Embeds text chunks and upserts them into Qdrant."""
        if not chunks:
            return 0

        texts = [chunk.content for chunk in chunks]
        embeddings = self.encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        points = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            points.append(
                qmodels.PointStruct(
                    id=i,
                    vector=emb.tolist(),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "content": chunk.content,
                        "metadata": chunk.metadata
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return len(points)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Searches nearest document chunks by cosine similarity."""
        query_emb = self.encoder.encode(query, normalize_embeddings=True).tolist()
        
        # Qdrant client search API
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_emb,
            limit=top_k
        )

        ranked_results = []
        for rank, hit in enumerate(search_results.points, start=1):
            ranked_results.append({
                "rank": rank,
                "score": float(hit.score),
                "chunk_id": hit.payload["chunk_id"],
                "doc_id": hit.payload["doc_id"],
                "content": hit.payload["content"],
                "chunk_key": f"{hit.payload['doc_id']}:{hit.payload['chunk_id']}"
            })

        return ranked_results

