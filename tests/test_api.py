"""
Tests for PRADAS FastAPI Server.

Tests REST endpoints and WebSocket connections using mock data.
Does NOT require AI models to be loaded.
"""

import pytest
import json

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from api.server import create_app, update_app_state


@pytest.fixture
def client():
    """Create a test client for the API."""
    app = create_app()
    return TestClient(app)


class TestStatusEndpoint:
    """Tests for GET /status."""

    def test_status_returns_200(self, client):
        response = client.get("/status")
        assert response.status_code == 200

    def test_status_has_required_fields(self, client):
        response = client.get("/status")
        data = response.json()
        assert "status" in data
        assert "api_version" in data
        assert "models_initialized" in data
        assert "active_ws_clients" in data
        assert "pipeline_status" in data

    def test_status_default_values(self, client):
        response = client.get("/status")
        data = response.json()
        assert data["status"] == "running"
        assert data["api_version"] == "1.0.0"

    def test_status_reflects_state_updates(self, client):
        update_app_state(
            models_initialized=True,
            pipeline_status="running",
            device="cpu",
            total_frames=500,
        )
        response = client.get("/status")
        data = response.json()
        assert data["models_initialized"] is True
        assert data["pipeline_status"] == "running"
        assert data["total_frames_processed"] == 500

        # Reset state
        update_app_state(
            models_initialized=False,
            pipeline_status="idle",
            total_frames=0,
        )


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_fields(self, client):
        response = client.get("/health")
        data = response.json()
        assert "healthy" in data
        assert "api_up" in data
        assert data["api_up"] is True


class TestInvalidEndpoints:
    """Tests for invalid requests."""

    def test_nonexistent_endpoint(self, client):
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_wrong_method(self, client):
        response = client.post("/status")
        assert response.status_code == 405


class TestWebSocket:
    """Tests for WebSocket endpoint."""

    def test_websocket_connect(self, client):
        with client.websocket_connect("/ws/alerts") as ws:
            # Should receive welcome message
            data = ws.receive_json()
            assert data["event_type"] == "connection_established"
            assert "client_id" in data

    def test_websocket_ping_pong(self, client):
        with client.websocket_connect("/ws/alerts") as ws:
            # Read welcome
            ws.receive_json()

            # Send ping
            ws.send_text("ping")
            response = ws.receive_text()
            assert response == "pong"

    def test_websocket_disconnect(self, client):
        """Test that disconnection is handled gracefully."""
        with client.websocket_connect("/ws/alerts") as ws:
            ws.receive_json()  # welcome
        # No error should occur after disconnect
