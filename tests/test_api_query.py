from fastapi.testclient import TestClient

from knowledge_rag.api.main import app


client = TestClient(app)


def test_query_accepts_top_k() -> None:
    response = client.post(
        "/query",
        json={
            "query": "privacy",
            "top_k": 1,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert len(body["results"]) <= 1


def test_query_accepts_note_type_filter() -> None:
    response = client.post(
        "/query",
        json={
            "query": "privacy",
            "note_type": "reference",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert all(
        result["note_type"] == "reference"
        for result in body["results"]
    )


def test_query_accepts_topic_filter() -> None:
    response = client.post(
        "/query",
        json={
            "query": "privacy",
            "topic": "privacy-demo",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert all(
        result["topic"] == "privacy-demo"
        for result in body["results"]
    )


def test_query_rejects_top_k_below_minimum() -> None:
    response = client.post(
        "/query",
        json={
            "query": "privacy",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_query_rejects_top_k_above_maximum() -> None:
    response = client.post(
        "/query",
        json={
            "query": "privacy",
            "top_k": 26,
        },
    )

    assert response.status_code == 422