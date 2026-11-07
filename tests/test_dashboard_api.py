"""
Unit tests for PRADAS Automotive Dashboard Endpoints.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from api.server import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_dashboard_page_returns_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "PRADAS" in response.text
    assert "Tableau de Bord" in response.text or "Cockpit" in response.text


def test_api_sources_returns_list(client):
    response = client.get("/api/sources")
    assert response.status_code == 200
    sources = response.json()
    assert isinstance(sources, list)
    assert any(s["id"] == "0" for s in sources)


def test_api_models_returns_list(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    assert len(models) >= 1


def test_api_pipeline_status(client):
    response = client.get("/api/pipeline/status")
    assert response.status_code == 200
    data = response.json()
    assert "running" in data
    assert "fps" in data
    assert "speed_kmh" in data


def test_api_pipeline_stop(client):
    response = client.post("/api/pipeline/stop")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
