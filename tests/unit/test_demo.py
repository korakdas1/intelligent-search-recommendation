"""Demo route and static assets. No PostgreSQL required."""

from fastapi.testclient import TestClient


def test_demo_page_ok(client: TestClient) -> None:
    response = client.get("/demo")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "Intelligent Search" in body
    assert "Recommendation Engine" in body
    assert 'id="search-form"' in body
    assert "/static/demo/app.js" in body


def test_demo_stylesheet_ok(client: TestClient) -> None:
    response = client.get("/static/demo/styles.css")
    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]
    assert "--accent" in response.text


def test_demo_script_ok(client: TestClient) -> None:
    response = client.get("/static/demo/app.js")
    assert response.status_code == 200
    assert "fusion_method" in response.text
    assert "body.fusion_method" in response.text
    assert "innerHTML" not in response.text
    assert "Technical details" in response.text
    assert "Find similar" in response.text
    assert "Copy product ID" in response.text
    for field in ("title", "rank", "product_id", "brand", "category", "price", "score", "source"):
        assert field in response.text


def test_demo_sends_measured_winners_explicitly(client: TestClient) -> None:
    html = client.get("/demo").text
    js = client.get("/static/demo/app.js").text
    assert 'value="weighted" selected' in html
    assert 'value="hybrid" selected' in html
    assert "method=${encodeURIComponent(method)}" in js
    assert 'id="use-ltr"' in html
    assert 'id="use-personalization"' in html
