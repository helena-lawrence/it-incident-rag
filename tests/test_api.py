from fastapi.testclient import TestClient
from src.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"
    assert data["service"] == "IT Incident Resolution RAG API"
