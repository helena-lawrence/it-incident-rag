
from sentence_transformers import CrossEncoder

from hybrid_retrieval import hybrid_search


# Step 1: Load CrossEncoder model
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(RERANKER_MODEL)


# Step 2: Create reranking function
def rerank_documents(question, documents, top_k=3):

    if not documents:
        return []

    # Prepare question-document pairs
    pairs = [
        (question, doc["content"])
        for doc in documents
    ]

    # Generate relevance scores
    scores = reranker.predict(pairs)

    # Attach reranking scores
    reranked_results = []

    for doc, score in zip(documents, scores):

        result = doc.copy()

        result["rerank_score"] = float(score)

        reranked_results.append(result)

    # Sort by reranking score (highest first)
    reranked_results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    # Return top results
    return reranked_results[:top_k]


# Step 3: Combine hybrid search and reranking
def retrieve_and_rerank(question, top_k=3):

    # Retrieve 10 candidates using hybrid search
    retrieved_docs = hybrid_search(
        question,
        top_k=10
    )

    # Rerank and return best 3
    reranked_docs = rerank_documents(
        question,
        retrieved_docs,
        top_k=top_k
    )

    return reranked_docs


# Step 4: Test reranking
if __name__ == "__main__":

    question = input("Enter your IT question: ").strip()

    if not question:
        print("Please enter a question.")

    else:

        results = retrieve_and_rerank(
            question,
            top_k=3
        )

        for index, result in enumerate(results, start=1):

            print("\n------------------------")
            print("Rank:", index)
            print("Source:", result["source"])
            print("Chunk ID:", result["chunk_id"])
            print(
                "Rerank Score:",
                round(result["rerank_score"], 4)
            )
            print("Content:")
            print(result["content"])
