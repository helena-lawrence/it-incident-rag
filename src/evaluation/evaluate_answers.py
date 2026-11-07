
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rag import generate_answer
from src.evaluation.evaluation.answer_judge import evaluate_answer


TEST_FILE = (
    PROJECT_ROOT / "src" / "evaluation" / "evaluation"
    / "answer_test_cases.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "src" / "evaluation" / "evaluation"
    / "answer_results.json"
)


def load_test_cases():

    with open(TEST_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def main():

    test_cases = load_test_cases()

    if not test_cases:
        print("No test cases found.")
        return

    results = []

    for index, test_case in enumerate(test_cases, start=1):

        question = test_case["question"]
        reference_answer = test_case["reference_answer"]

        print(f"\n========== TEST {index} ==========")
        print("Question:", question)

        # Step 1: Generate RAG answer
        rag_result = generate_answer(question)

        generated_answer = rag_result["answer"]

        context = rag_result.get(
            "retrieved_context",
            ""
        )

        # Step 2: Evaluate answer
        scores = evaluate_answer(
            question=question,
            context=context,
            generated_answer=generated_answer,
            reference_answer=reference_answer
        )

        # Step 3: Save result
        result = {
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": generated_answer,
            "sources": rag_result["sources"],
            "scores": scores
        }

        results.append(result)

        print("\nGenerated Answer:")
        print(generated_answer)

        print("\nEvaluation Scores:")
        print("Faithfulness:", scores["faithfulness"])
        print("Relevance:", scores["relevance"])
        print("Correctness:", scores["correctness"])
        print("Reason:", scores["reason"])

    # Step 4: Calculate averages
    total = len(results)

    metrics = [
        "faithfulness",
        "relevance",
        "correctness"
    ]

    averages = {}

    for metric in metrics:

        total_score = sum(
            result["scores"][metric]
            for result in results
        )

        averages[metric] = round(
            total_score / (total * 2) * 100,
            2
        )

    print("\n========== FINAL RESULTS ==========")

    for metric, value in averages.items():
        print(f"{metric}: {value}%")

    # Step 5: Save evaluation results
    report = {
        "total_questions": total,
        "average_scores_percent": averages,
        "results": results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nEvaluation saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
