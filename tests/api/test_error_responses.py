"""Tests for API error responses -- FOUN-05."""


class TestStructuredErrorResponses:
    """Verify API returns JSON errors, not HTML fallback."""

    def test_missing_data_returns_json_error(self):
        """FOUN-05: Missing data returns structured JSON, not HTML."""
        from fastapi.testclient import TestClient

        from api.main import app

        client = TestClient(app, raise_server_exceptions=False)
        # Request data that doesn't exist -- should get JSON error, not HTML
        response = client.get("/predictions/week/9999/99")

        assert response.headers.get("content-type", "").startswith(
            "application/json"
        ), f"Expected JSON response, got {response.headers.get('content-type')}"
        data = response.json()
        assert "message" in data or "error" in data or "detail" in data, (
            f"Error response missing message/error/detail field: {data}"
        )

    def test_no_html_fallback_in_services(self):
        """FOUN-05: _generate_fallback_html_report method does not exist."""
        from api import services

        assert not hasattr(services, "_generate_fallback_html_report"), (
            "Fallback HTML report generator should be removed"
        )
        # Check the service class too
        service_classes = [
            attr
            for attr in dir(services)
            if isinstance(getattr(services, attr, None), type)
        ]
        for cls_name in service_classes:
            cls = getattr(services, cls_name)
            assert not hasattr(cls, "_generate_fallback_html_report"), (
                f"{cls_name} still has _generate_fallback_html_report"
            )
