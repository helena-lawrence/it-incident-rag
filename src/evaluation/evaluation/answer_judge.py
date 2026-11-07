
import json

from ollama import chat


JUDGE_MODEL = "qwen2.5:1.5b"


def evaluate_answer(
    question,
    context,
    generated_answer,
    reference_answer
):

    system_prompt = """
You are an IT support answer evaluator.

Evaluate the generated answer using these criteria:

1. Faithfulness:
Does the answer contain only claims supported
by the retrieved context?

2. Answer relevance:
Does the answer directly address the question?

3. Answer correctness:
Does the answer agree with the verified
reference answer?

For each criterion, give a score:

0 = Poor
1 = Partially correct
2 = Good

Evaluate carefully.
Do not reward unsupported technical claims.

Return only a JSON object with these keys:

faithfulness
relevance
correctness
reason
"""

    user_prompt = f"""
QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{generated_answer}

REFERENCE ANSWER:
{reference_answer}

Evaluate the generated answer.
"""

    response = chat(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        format="json",
        options={
            "temperature": 0
        }
    )

    result = json.loads(response.message.content)

    # Validate the returned scores
    for key in [
        "faithfulness",
        "relevance",
        "correctness"
    ]:
        score = result.get(key)

        if type(score) is not int or score not in [0, 1, 2]:
            raise ValueError(
                f"Invalid {key} score returned by judge: {score}"
            )

    return {
        "faithfulness": result["faithfulness"],
        "relevance": result["relevance"],
        "correctness": result["correctness"],
        "reason": str(result.get("reason", ""))
    }
