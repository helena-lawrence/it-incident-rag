
from hybrid_retrieval import hybrid_search
from reranker import rerank_documents


question = "How do I fix VPN Error 809?"

# Retrieve 10 candidates
hybrid_results = hybrid_search(
    question,
    top_k=10
)

# Rerank the same candidates
reranked_results = rerank_documents(
    question,
    hybrid_results,
    top_k=3
)


print("\n=== BEFORE RERANKING ===")

for index, doc in enumerate(hybrid_results, start=1):

    print(
        index,
        doc["source"],
        "Chunk:",
        doc["chunk_id"]
    )


print("\n=== AFTER RERANKING ===")

for index, doc in enumerate(reranked_results, start=1):

    print(
        index,
        doc["source"],
        "Chunk:",
        doc["chunk_id"],
        "Score:",
        round(doc["rerank_score"], 4)
    )
