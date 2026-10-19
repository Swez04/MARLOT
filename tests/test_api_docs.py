"""Tests for the API's published OpenAPI documentation."""

from fastapi.testclient import TestClient

from simulator.api.app import app

client = TestClient(app)


def test_openapi_schema_is_available():
    response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()

    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema


def test_health_and_simulation_routes_are_documented():
    response = client.get("/openapi.json")
    schema = response.json()

    paths = schema["paths"]

    assert "/health" in paths
    assert "get" in paths["/health"]

    assert "/simulate" in paths
    assert "post" in paths["/simulate"]


def test_simulation_request_schema_is_documented():
    schema = client.get("/openapi.json").json()

    operation = schema["paths"]["/simulate"]["post"]

    request_schema = operation["requestBody"]["content"]["application/json"]["schema"]

    assert request_schema["$ref"].endswith("/SimulationRequest")

    components = schema["components"]["schemas"]
    request_model = components["SimulationRequest"]

    assert "scenario" in request_model["properties"]
    assert "count" in request_model["properties"]

    count_schema = request_model["properties"]["count"]

    assert count_schema["minimum"] == 0
    assert count_schema["maximum"] == 10_000


def test_simulation_response_schema_is_documented():
    schema = client.get("/openapi.json").json()

    operation = schema["paths"]["/simulate"]["post"]

    response_schema = operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ]

    assert response_schema["$ref"].endswith("/SimulationResponse")

    components = schema["components"]["schemas"]

    assert "SimulationResponse" in components
    assert "SimulationSummary" in components
    assert "TransactionResponse" in components


def test_documentation_pages_are_available():
    docs_response = client.get("/docs")
    redoc_response = client.get("/redoc")

    assert docs_response.status_code == 200
    assert redoc_response.status_code == 200
