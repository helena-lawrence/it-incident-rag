
import os
import uuid

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from document_loader import load_documents
from chunking import chunk_documents
from embeddings import generate_embeddings


# Configuration
COLLECTION_NAME = "it_incidents_hybrid"
SPARSE_MODEL_NAME = "Qdrant/bm25"

# Use Docker Qdrant URL when available,
# otherwise use local Qdrant.
QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

client = QdrantClient(url=QDRANT_URL)

sparse_model = SparseTextEmbedding(
    model_name=SPARSE_MODEL_NAME
)


def ingest_hybrid_documents():

    print("Starting hybrid document ingestion...")
    print(f"Connecting to Qdrant: {QDRANT_URL}")

    # Step 1: Load documents
    documents = load_documents()

    # Step 2: Split documents into chunks
    chunks = chunk_documents(documents)

    if not chunks:
        print("No document chunks found.")
        return

    print(f"Loaded {len(chunks)} document chunks.")

    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    # Step 3: Generate dense embeddings
    print("Generating dense embeddings...")

    dense_embeddings = generate_embeddings(chunks)

    # Step 4: Generate sparse BM25 embeddings
    print("Generating sparse embeddings...")

    sparse_embeddings = list(
        sparse_model.embed(texts)
    )

    vector_size = dense_embeddings.shape[1]

    # Step 5: Create Qdrant collection if missing
    if not client.collection_exists(COLLECTION_NAME):

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            }
        )

        print("Hybrid collection created successfully.")

    else:
        print("Hybrid collection already exists.")

    # Step 6: Prepare document points
    points = []

    for chunk, dense, sparse in zip(
        chunks,
        dense_embeddings,
        sparse_embeddings
    ):

        # Stable ID prevents duplicate chunks
        # when the ingestion script runs again.
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                chunk["source"] + ":" + str(chunk["chunk_id"])
            )
        )

        point = models.PointStruct(
            id=point_id,
            vector={
                "dense": dense.tolist(),
                "sparse": models.SparseVector(
                    indices=sparse.indices.tolist(),
                    values=sparse.values.tolist()
                )
            },
            payload={
                "content": chunk["content"],
                "source": chunk["source"],
                "file_type": chunk["file_type"],
                "chunk_id": chunk["chunk_id"]
            }
        )

        points.append(point)

    # Step 7: Upload document embeddings to Qdrant
    print("Uploading document embeddings...")

    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=points,
        batch_size=32,
        wait=True
    )

    print(
        f"Successfully stored {len(points)} "
        f"chunks in collection '{COLLECTION_NAME}'."
    )

    print("Hybrid document ingestion completed.")


if __name__ == "__main__":
    ingest_hybrid_documents()
