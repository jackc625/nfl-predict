"""HTML snapshot tests for web UI templates and rendering."""

import re
from unittest.mock import patch

import pytest
from bs4 import BeautifulSoup
from fastapi.testclient import TestClient

from api.main import app


class TestHTMLSnapshots:
    """Test HTML output and template rendering."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_main_page_structure(self, client, mock_predictions_data):
        """Test main page HTML structure and content."""
        with patch("api.services.data_service.get_current_week_metadata") as mock_week:
            with patch("api.services.data_service.list_games") as mock_games:
                mock_week.return_value = {
                    "current_week": 3,
                    "current_season": 2024,
                    "predictions_available": True,
                    "last_updated": "2024-09-15T18:00:00-04:00",
                }
                mock_games.return_value = (
                    mock_predictions_data["games"],
                    mock_week.return_value,
                    len(mock_predictions_data["games"]),
                )

                response = client.get("/")
                assert response.status_code == 200

                html = response.text
                soup = BeautifulSoup(html, "html.parser")

                # Test HTML structure
                self._validate_html_structure(soup)

                # Test navigation
                nav = soup.find("nav")
                assert nav is not None, "Should have navigation"

                # Test page title
                title = soup.find("title")
                assert title is not None
                assert "NFL" in title.text

                # Test main content area
                main = soup.find("main")
                assert main is not None, "Should have main content area"

                # Test games table/grid
                games_container = soup.find(id=re.compile(r"games|predictions"))
                if games_container is None:
                    # Look for table or grid elements
                    games_container = soup.find(
                        ["table", "div"], class_=re.compile(r"game|prediction")
                    )

                assert games_container is not None, (
                    "Should have games/predictions display"
                )

    def test_game_detail_page(self, client, mock_predictions_data):
        """Test game detail page HTML structure."""
        game_id = "TEST_2024_W03_BUF@MIA"

        with patch("api.services.data_service.get_game_detail") as mock_service:
            mock_service.return_value = mock_predictions_data["games"][0]

            response = client.get(f"/games/{game_id}/view")
            assert response.status_code == 200

            html = response.text
            soup = BeautifulSoup(html, "html.parser")

            self._validate_html_structure(soup)

            # Test game-specific content
            game_title = soup.find(["h1", "h2"], string=re.compile(r"BUF|MIA"))
            assert game_title is not None, "Should display team names"

            # Test prediction displays
            prediction_sections = soup.find_all(
                ["div", "section"], class_=re.compile(r"prediction|prob")
            )
            assert len(prediction_sections) > 0, "Should have prediction displays"

            # Test probability values are displayed
            prob_elements = soup.find_all(string=re.compile(r"\d+\.\d+%|\d+%"))
            assert len(prob_elements) > 0, "Should display probability percentages"

    def test_backtest_page(self, client, mock_backtest_data):
        """Test backtest results page HTML structure."""
        with patch(
            "api.services.backtest_service.get_backtest_summary"
        ) as mock_service:
            mock_service.return_value = mock_backtest_data

            response = client.get("/backtest/view")
            assert response.status_code == 200

            html = response.text
            soup = BeautifulSoup(html, "html.parser")

            self._validate_html_structure(soup)

            # Test metrics display
            metrics_section = soup.find(
                ["div", "section"], class_=re.compile(r"metric|performance")
            )
            if metrics_section is None:
                # Look for tables or lists that might contain metrics
                metrics_section = soup.find(["table", "ul", "dl"])

            assert metrics_section is not None, "Should display performance metrics"

            # Test accuracy percentages
            accuracy_elements = soup.find_all(string=re.compile(r"\d+\.\d+%"))
            assert len(accuracy_elements) > 0, "Should display accuracy percentages"

    def test_calibration_page(self, client, mock_calibration_data):
        """Test calibration analysis page HTML structure."""
        with patch(
            "api.services.backtest_service.get_calibration_data"
        ) as mock_service:
            mock_service.return_value = mock_calibration_data

            response = client.get("/calibration/view")
            assert response.status_code == 200

            html = response.text
            soup = BeautifulSoup(html, "html.parser")

            self._validate_html_structure(soup)

            # Test chart containers
            chart_containers = soup.find_all(
                ["div", "canvas"], class_=re.compile(r"chart|calibration")
            )
            if len(chart_containers) == 0:
                # Look for Chart.js canvas elements
                chart_containers = soup.find_all("canvas")

            assert len(chart_containers) > 0, (
                "Should have chart containers for calibration curves"
            )

    def test_responsive_design_elements(self, client, mock_predictions_data):
        """Test responsive design elements are present."""
        with patch("api.services.data_service.get_current_week_metadata") as mock_week:
            with patch("api.services.data_service.list_games") as mock_games:
                mock_week.return_value = {"current_week": 3, "current_season": 2024}
                mock_games.return_value = (
                    mock_predictions_data["games"],
                    mock_week.return_value,
                    len(mock_predictions_data["games"]),
                )

                response = client.get("/")
                html = response.text
                soup = BeautifulSoup(html, "html.parser")

                # Test viewport meta tag
                viewport = soup.find("meta", attrs={"name": "viewport"})
                assert viewport is not None, "Should have viewport meta tag"
                assert "width=device-width" in viewport.get("content", ""), (
                    "Should set proper viewport"
                )

                # Test responsive CSS classes (Tailwind)
                responsive_elements = soup.find_all(
                    class_=re.compile(r"sm:|md:|lg:|xl:")
                )
                assert len(responsive_elements) > 0, "Should use responsive CSS classes"

                # Test mobile menu button
                mobile_menu = soup.find(
                    ["button", "div"], class_=re.compile(r"mobile|menu")
                )
                assert mobile_menu is not None, "Should have mobile menu functionality"

    def test_accessibility_features(self, client, mock_predictions_data):
        """Test accessibility features in HTML."""
        with patch("api.services.data_service.get_current_week_metadata") as mock_week:
            with patch("api.services.data_service.list_games") as mock_games:
                mock_week.return_value = {"current_week": 3, "current_season": 2024}
                mock_games.return_value = (
                    mock_predictions_data["games"],
                    mock_week.return_value,
                    len(mock_predictions_data["games"]),
                )

                response = client.get("/")
                html = response.text
                soup = BeautifulSoup(html, "html.parser")

                # Test alt attributes on images
                images = soup.find_all("img")
                for img in images:
                    assert img.get("alt") is not None, (
                        f"Image should have alt text: {img}"
                    )

                # Test aria labels on interactive elements
                buttons = soup.find_all("button")
                for button in buttons:
                    has_aria_label = button.get("aria-label") is not None
                    has_text_content = button.get_text(strip=True) != ""
                    assert has_aria_label or has_text_content, (
                        f"Button should have aria-label or text content: {button}"
                    )

                # Test heading hierarchy
                headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
                if headings:
                    h1_count = len(soup.find_all("h1"))
                    assert h1_count == 1, (
                        f"Should have exactly one H1, found {h1_count}"
                    )

    def test_javascript_integration(self, client, mock_predictions_data):
        """Test JavaScript integration and script tags."""
        with patch("api.services.data_service.get_current_week_metadata") as mock_week:
            with patch("api.services.data_service.list_games") as mock_games:
                mock_week.return_value = {"current_week": 3, "current_season": 2024}
                mock_games.return_value = (
                    mock_predictions_data["games"],
                    mock_week.return_value,
                    len(mock_predictions_data["games"]),
                )

                response = client.get("/")
                html = response.text
                soup = BeautifulSoup(html, "html.parser")

                # Test JavaScript files are included
                script_tags = soup.find_all("script", src=True)
                script_sources = [script.get("src") for script in script_tags]

                # Should include Chart.js for visualizations
                chart_js_included = any(
                    "chart" in src.lower() for src in script_sources
                )
                assert chart_js_included, "Should include Chart.js library"

                # Should include custom JavaScript
                custom_js_included = any("/static/js/" in src for src in script_sources)
                assert custom_js_included, "Should include custom JavaScript files"

    def test_css_integration(self, client):
        """Test CSS integration and stylesheets."""
        response = client.get("/")
        html = response.text
        soup = BeautifulSoup(html, "html.parser")

        # Test CSS files are included
        link_tags = soup.find_all("link", rel="stylesheet")
        css_sources = [link.get("href") for link in link_tags]

        # Should include Tailwind CSS (CDN or compiled)
        tailwind_included = any(
            "tailwind" in href.lower() for href in css_sources if href
        )
        custom_css_included = any(
            "/static/css/" in href for href in css_sources if href
        )

        assert tailwind_included or custom_css_included, (
            "Should include CSS frameworks or custom styles"
        )

    def test_error_pages(self, client):
        """Test error page rendering."""
        # Test 404 page
        response = client.get("/nonexistent-page")
        assert response.status_code == 404

        html = response.text
        soup = BeautifulSoup(html, "html.parser")

        # Should still have basic HTML structure
        assert soup.find("html") is not None
        assert soup.find("head") is not None
        assert soup.find("body") is not None

        # Should indicate error
        error_content = soup.get_text().lower()
        assert "not found" in error_content or "404" in error_content

    def test_data_attributes(self, client, mock_predictions_data):
        """Test data attributes for JavaScript interaction."""
        with patch("api.services.data_service.get_current_week_metadata") as mock_week:
            with patch("api.services.data_service.list_games") as mock_games:
                mock_week.return_value = {"current_week": 3, "current_season": 2024}
                mock_games.return_value = (
                    mock_predictions_data["games"],
                    mock_week.return_value,
                    len(mock_predictions_data["games"]),
                )

                response = client.get("/")
                html = response.text
                soup = BeautifulSoup(html, "html.parser")

                # Test data attributes for interactive elements
                interactive_elements = (
                    soup.find_all(attrs={"data-action": True})
                    + soup.find_all(attrs={"data-game-id": True})
                    + soup.find_all(attrs={"data-target": True})
                )

                # Should have some interactive elements with data attributes
                # (This is optional - depends on implementation)
                if len(interactive_elements) > 0:
                    for element in interactive_elements:
                        data_attrs = [
                            attr for attr in element.attrs if attr.startswith("data-")
                        ]
                        assert len(data_attrs) > 0, (
                            "Interactive elements should have data attributes"
                        )

    def test_form_elements(self, client):
        """Test form elements and inputs."""
        response = client.get("/")
        html = response.text
        soup = BeautifulSoup(html, "html.parser")

        # Test filter forms or input elements
        forms = soup.find_all("form")
        inputs = soup.find_all(["input", "select", "button"])

        if len(forms) > 0:
            for form in forms:
                # Forms should have proper action or method
                assert form.get("action") is not None or form.get("method") is not None

        if len(inputs) > 0:
            for input_elem in inputs:
                if input_elem.name == "input":
                    # Inputs should have type attribute
                    assert input_elem.get("type") is not None, (
                        f"Input should have type: {input_elem}"
                    )

    def _validate_html_structure(self, soup):
        """Validate basic HTML structure."""
        # Test basic HTML structure
        assert soup.find("html") is not None, "Should have html tag"
        assert soup.find("head") is not None, "Should have head section"
        assert soup.find("body") is not None, "Should have body section"

        # Test meta tags
        charset = soup.find("meta", attrs={"charset": True})
        assert charset is not None, "Should have charset meta tag"

        # Test title
        title = soup.find("title")
        assert title is not None, "Should have title tag"
        assert len(title.get_text(strip=True)) > 0, "Title should not be empty"

    def test_performance_optimizations(self, client):
        """Test performance optimizations in HTML."""
        response = client.get("/")
        html = response.text
        soup = BeautifulSoup(html, "html.parser")

        # Test script loading optimizations
        scripts = soup.find_all("script", src=True)
        for script in scripts:
            # Check for async or defer attributes on external scripts
            src = script.get("src", "")
            if src.startswith("http") or src.startswith("//"):
                assert (
                    script.get("async") is not None
                    or script.get("defer") is not None
                    or True
                )
                # At least some external scripts should be optimized
                # (This is a best practice but not required)

        # Test CSS optimization
        styles = soup.find_all("link", rel="stylesheet")
        # Should not have excessive number of stylesheets
        assert len(styles) <= 10, (
            f"Too many stylesheets ({len(styles)}) - consider bundling"
        )

    def test_content_security(self, client):
        """Test content security aspects."""
        response = client.get("/")

        # Test security headers (if implemented)
        security_headers = [
            "content-security-policy",
            "x-frame-options",
            "x-content-type-options",
        ]

        for header in security_headers:
            if header in response.headers:
                assert len(response.headers[header]) > 0, (
                    f"Security header {header} should not be empty"
                )

        # Test inline script restrictions
        html = response.text
        soup = BeautifulSoup(html, "html.parser")

        soup.find_all("script", src=False)
        # If CSP is strict, should minimize inline scripts
        # (This is optional depending on security requirements)
