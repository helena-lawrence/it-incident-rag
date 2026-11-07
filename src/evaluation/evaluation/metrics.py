
def calculate_metrics(
    retrieved_sources,
    relevant_sources,
    k=3
):
    # Remove duplicate source documents
    unique_sources = list(
        dict.fromkeys(retrieved_sources)
    )

    # Consider only top K documents
    top_k_sources = unique_sources[:k]

    relevant_set = set(relevant_sources)

    # Count relevant retrieved documents
    relevant_count = sum(
        1
        for source in top_k_sources
        if source in relevant_set
    )

    # Precision@K
    precision = relevant_count / k

    # Recall@K
    recall = (
        relevant_count / len(relevant_set)
        if relevant_set
        else 0.0
    )

    # Hit Rate@K
    hit_rate = 1.0 if relevant_count > 0 else 0.0

    # Reciprocal Rank
    reciprocal_rank = 0.0

    for rank, source in enumerate(
        top_k_sources,
        start=1
    ):
        if source in relevant_set:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "precision": precision,
        "recall": recall,
        "hit_rate": hit_rate,
        "reciprocal_rank": reciprocal_rank
    }
