

import logging
import time
import uuid

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Request,
    Response
)
from pydantic import BaseModel, Field
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from rag import generate_answer
from auth import verify_api_key
from logging_config import configure_logging
from metrics import HTTP_REQUESTS, HTTP_DURATION


configure_logging()

logger = logging.getLogger(__name__)


app = FastAPI(
    title="IT Incident Resolution RAG API",
    version="1.2.0"
)


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000
    )


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]


# Track all HTTP requests
@app.middleware("http")
async def request_tracking(request: Request, call_next):

    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Unhandled HTTP request error",
            extra={"request_id": request_id}
        )
        raise

    duration = time.perf_counter() - start

    endpoint = request.scope.get("route")

    endpoint_name = (
        endpoint.path
        if endpoint is not None
        else "unmatched"
    )

    HTTP_REQUESTS.labels(
        method=request.method,
        endpoint=endpoint_name,
        status=str(response.status_code)
    ).inc()

    HTTP_DURATION.labels(
        method=request.method,
        endpoint=endpoint_name
    ).observe(duration)

    logger.info(
        "HTTP request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "endpoint": endpoint_name,
            "status_code": response.status_code,
            "duration_seconds": round(duration, 4)
        }
    )

    response.headers["X-Request-ID"] = request_id

    return response


@app.get("/health")
def health_check():

    return {
        "status": "running",
        "service": "IT Incident Resolution RAG API"
    }


@app.get(
    "/metrics",
    dependencies=[Depends(verify_api_key)]
)
def metrics_endpoint():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post(
    "/ask",
    response_model=AnswerResponse,
    dependencies=[Depends(verify_api_key)]
)
def ask_question(request: QuestionRequest, http_request: Request):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question must not be empty."
        )

    try:
        result = generate_answer(
            question,
            request_id=http_request.state.request_id
        )

        return AnswerResponse(
            answer=result["answer"],
            sources=result["sources"]
        )

    except Exception:
        logger.exception(
            "RAG request failed",
            extra={
                "request_id": http_request.state.request_id
            }
        )

        raise HTTPException(
            status_code=503,
            detail="Unable to process your question."
        )
