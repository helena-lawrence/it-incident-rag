
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


# Step 1: Configuration
COLLECTION_NAME = "it_incidents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Step 2: Load embedding model
model = SentenceTransformer(MODEL_NAME)

# Step 3: Connect to Qdrant
client = QdrantClient(url="http://localhost:6333")


# Step 4: Create retrieval function
def retrieve_documents(question, top_k=3):

    # Convert user question into embedding
    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    # Search Qdrant
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=question_embedding.tolist(),
        limit=top_k,
        with_payload=True
    )

    # Prepare results
    results = []

    for point in response.points:

        result = {
            "score": point.score,
            "content": point.payload.get("content", ""),
            "source": point.payload.get("source", ""),
            "chunk_id": point.payload.get("chunk_id")
        }

        results.append(result)

    return results


# Step 5: Test retrieval
if __name__ == "__main__":

    question = "My VPN is not connecting. How can I fix it?"

    print("\nUser Question:", question)

    results = retrieve_documents(question, top_k=3)

    for index, result in enumerate(results, start=1):

        print("\n-------------------------")
        print("Result:", index)
        print("Similarity Score:", round(result["score"], 4))
        print("Source:", result["source"])
        print("Chunk ID:", result["chunk_id"])
        print("Content:")
        print(result["content"])
