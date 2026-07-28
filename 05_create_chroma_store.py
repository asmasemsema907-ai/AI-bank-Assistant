from importlib import import_module
from pathlib import Path

import chromadb
from chromadb.config import Settings

vectors = import_module("04_vector_representation")

# Reference lab hardcoded an absolute Windows path (E:/Courses/.../chroma_db),
# which only works on that one machine. This uses a path relative to this
# file instead, so it works identically on any machine/deployment target.
DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "banking_kb_docs"


def _collection_name_for_embedding_backend():
    backend = getattr(vectors, "EMBEDDING_BACKEND_ID", "embedding")
    dimension = getattr(vectors, "EMBEDDING_DIMENSION", "unknown")
    return f"{COLLECTION_NAME}_{backend}_{dimension}"


def create_vector_store():
    client = chromadb.PersistentClient(
        path=str(DB_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(_collection_name_for_embedding_backend())

    collection.upsert(
        ids=[chunk["chunk_id"] for chunk in vectors.chunks],
        documents=[chunk["chunk_text"] for chunk in vectors.chunks],
        metadatas=[
            {
                "document_id": chunk["document_id"],
                "title": chunk["title"],
                "is_current": str(chunk["is_current"]),
            }
            for chunk in vectors.chunks
        ],
        embeddings=vectors.chunk_embeddings.tolist(),
    )

    return collection


if __name__ == "__main__":
    create_vector_store()
    print("Chroma vector store created.")
