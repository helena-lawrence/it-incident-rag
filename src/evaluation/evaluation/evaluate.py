
import json
import sys
from pathlib import Path

# Find project folders
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from retrieval import retrieve_documents
from hybrid_retrieval import hybrid_search
from reranker import retrieve_and_rerank
from src.evaluation.evaluation.metrics import calculate_metrics


TEST_FILE = PROJECT_ROOT / "src" / "evaluation" / "test_questions.json"

TOP_K = 3
CANDIDATE_COUNT = 10


def load_test_questions():

    with open(TEST_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def get_sources(results):

    return [
        result["source"]
        for result in results
    ]


def evaluate_method(method_name, search_function, test_data):

    all_metrics = []

    print(f"\n========== {method_name} ==========")

    for item in test_data:

        question = item["question"]
        relevant_sources = item["relevant_sources"]

        # Retrieve documents
        results = search_function(question)

        retrieved_sources = get_sources(results)

        # Calculate metrics
        metrics = calculate_metrics(
            retrieved_sources,
            relevant_sources,
            k=TOP_K
        )

        all_metrics.append(metrics)

        print("\nQuestion:", question)
        print("Expected:", relevant_sources)
        print("Retrieved:", retrieved_sources[:TOP_K])
        print("Precision@3:", round(metrics["precision"], 3))
        print("Recall@3:", round(metrics["recall"], 3))
        print("Hit Rate@3:", round(metrics["hit_rate"], 3))
        print("Reciprocal Rank:", round(metrics["reciprocal_rank"], 3))

    # Average metrics across questions
    total = len(all_metrics)

    summary = {
        "method": method_name,
        "precision": sum(
            m["precision"] for m in all_metrics
        ) / total,
        "recall": sum(
            m["recall"] for m in all_metrics
        ) / total,
        "hit_rate": sum(
            m["hit_rate"] for m in all_metrics
        ) / total,
        "mrr": sum(
            m["reciprocal_rank"] for m in all_metrics
        ) / total
    }

    return summary


def main():

    test_data = load_test_questions()

    if not test_data:
        print("No evaluation questions found.")
        return

    # Give each method enough candidates for
    # document-level evaluation.
    methods = {
        "Semantic Search": lambda q: retrieve_documents(
            q, top_k=CANDIDATE_COUNT
        ),

        "Hybrid Search": lambda q: hybrid_search(
            q, top_k=CANDIDATE_COUNT
        ),

        "Hybrid + Reranking": lambda q: retrieve_and_rerank(
            q, top_k=CANDIDATE_COUNT
        )
    }

    summaries = []

    for name, search_function in methods.items():

        summary = evaluate_method(
            name,
            search_function,
            test_data
        )

        summaries.append(summary)

    print("\n========== FINAL COMPARISON ==========")

    for result in summaries:

        print("\nMethod:", result["method"])
        print("Precision@3:", round(result["precision"], 3))
        print("Recall@3:", round(result["recall"], 3))
        print("Hit Rate@3:", round(result["hit_rate"], 3))
        print("MRR@3:", round(result["mrr"], 3))


if __name__ == "__main__":
    main()
