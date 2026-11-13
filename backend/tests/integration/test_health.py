from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "API is running"}


def test_development_cors_allows_only_frontend_origins():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        for origin in ('http://localhost:5173', 'http://127.0.0.1:5173'):
            response = client.options('/api/v1/ask', headers={'Origin': origin,
                'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'authorization,content-type'})
            assert response.status_code == 200 and response.headers['access-control-allow-origin'] == origin
        denied = client.options('/api/v1/ask', headers={'Origin': 'https://untrusted.example', 'Access-Control-Request-Method': 'POST'})
        assert denied.status_code == 400 and 'access-control-allow-origin' not in denied.headers
