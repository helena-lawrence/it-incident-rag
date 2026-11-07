
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models


COLLECTION_NAME = "it_incidents_hybrid"

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL_NAME = "Qdrant/bm25"


# Load models
dense_model = SentenceTransformer(DENSE_MODEL_NAME)

sparse_model = SparseTextEmbedding(
    model_name=SPARSE_MODEL_NAME
)

# Connect to Qdrant
import os

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

client = QdrantClient(url=QDRANT_URL)


def hybrid_search(question, top_k=3):

    # Step 1: Dense query embedding
    dense_vector = dense_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    # Step 2: Sparse query embedding
    sparse_result = next(
        sparse_model.query_embed(question)
    )

    sparse_vector = models.SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist()
    )

    # Step 3: Hybrid query using RRF
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=20
            ),
            models.Prefetch(
                query=sparse_vector,
                using="sparse",
                limit=20
            )
        ],
        query=models.FusionQuery(
            fusion=models.Fusion.RRF
        ),
        limit=top_k,
        with_payload=True
    )

    # Step 4: Format results
    results = []

    for point in response.points:

        results.append({
            "score": point.score,
            "content": point.payload.get("content", ""),
            "source": point.payload.get("source", ""),
            "chunk_id": point.payload.get("chunk_id")
        })

    return results


if __name__ == "__main__":

    question = input("Enter your IT question: ").strip()

    if not question:
        print("Please enter a question.")

    else:
        results = hybrid_search(question, top_k=3)

        for index, result in enumerate(results, start=1):

            print("\n------------------------")
            print("Result:", index)
            print("Fusion Score:", round(result["score"], 4))
            print("Source:", result["source"])
            print("Chunk ID:", result["chunk_id"])
            print("Content:")
            print(result["content"])
