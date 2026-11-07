
from prometheus_client import Counter, Histogram


# Total HTTP requests
HTTP_REQUESTS = Counter(
    "rag_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)


# HTTP response time
HTTP_DURATION = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"]
)


# RAG pipeline component durations
RAG_STAGE_DURATION = Histogram(
    "rag_stage_duration_seconds",
    "Duration of individual RAG pipeline stages",
    ["stage"]
)


# Total RAG failures
RAG_ERRORS = Counter(
    "rag_pipeline_errors_total",
    "Total RAG pipeline errors",
    ["stage"]
)


# Total completed RAG requests
RAG_COMPLETED = Counter(
    "rag_pipeline_completed_total",
    "Total completed RAG requests"
)
