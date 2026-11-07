
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader


# Find project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load local environment variables
load_dotenv(PROJECT_ROOT / ".env")


# Read API key from environment
EXPECTED_API_KEY = os.getenv("RAG_API_KEY")


# Stop startup if the secret is missing
if not EXPECTED_API_KEY:
    raise RuntimeError(
        "RAG_API_KEY is not configured."
    )


# Define the authentication header
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)


def verify_api_key(
    api_key: str | None = Security(api_key_header)
):

    # Reject missing keys
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API key is required."
        )

    # Compare securely
    if not secrets.compare_digest(
        api_key,
        EXPECTED_API_KEY
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key."
        )

    return True
