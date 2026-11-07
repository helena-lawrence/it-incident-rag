
import uuid

from qdrant_client import QdrantClient, models

from document_loader import load_documents
from chunking import chunk_documents
from embeddings import generate_embeddings


COLLECTION_NAME = "it_incidents"

# Connect to local Qdrant
client = QdrantClient(url="http://localhost:6333")


def store_embeddings(chunks, embeddings):

    if len(chunks) == 0:
        print("No chunks found.")
        return

    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings count mismatch")

    # Find embedding dimensions
    vector_size = embeddings.shape[1]

    # Create collection if it does not exist
    if not client.collection_exists(COLLECTION_NAME):

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )

        print("Qdrant collection created.")

    # Prepare points
    points = []

    for chunk, embedding in zip(chunks, embeddings):

        # Create a stable ID for this chunk
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                chunk["source"] + ":" + str(chunk["chunk_id"])
            )
        )

        point = models.PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload={
                "content": chunk["content"],
                "source": chunk["source"],
                "file_type": chunk["file_type"],
                "chunk_id": chunk["chunk_id"]
            }
        )

        points.append(point)

    # Upload points to Qdrant
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )

    print(f"Stored {len(points)} chunks in Qdrant.")


if __name__ == "__main__":

    documents = load_documents()

    chunks = chunk_documents(documents)

    embeddings = generate_embeddings(chunks)

    store_embeddings(chunks, embeddings)

    collection_info = client.get_collection(COLLECTION_NAME)

    print("Collection:", COLLECTION_NAME)
    print("Stored points:", collection_info.points_count)
