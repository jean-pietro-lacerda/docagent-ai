from uuid import uuid4

from fastapi.testclient import TestClient
import jwt
import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import Settings, get_settings
from app.db.models import LocalCredential, User
from app.db.session import get_db
from app.main import app
from app.security.passwords import get_password_service
from app.security.tokens import create_access_token, decode_access_token


@pytest.fixture
def auth_api(auth_config, db_session):
    config = auth_config
    session = db_session
    user = User(id=uuid4(), status="active")
    credential = LocalCredential(
        user=user, username="alice", password_hash=get_password_service().hash("Only a test password 123")
    )
    original = app.dependency_overrides.copy()
    app.dependency_overrides[get_settings] = lambda: config
    app.dependency_overrides[get_db] = lambda: session
    try:
        with TestClient(app) as client:
            yield client, config, session, user, credential
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original)


def test_login_then_me_returns_only_authenticated_identity(auth_api) -> None:
    client, config, session, user, credential = auth_api
    session.scalar.return_value = credential
    login = client.post("/api/v1/auth/login", json={"username": " ALICE ", "password": "Only a test password 123"})
    assert login.status_code == 200
    assert set(login.json()) == {"access_token", "token_type", "expires_in"}
    assert login.json()["token_type"] == "bearer"
    assert login.json()["expires_in"] == 1800
    assert decode_access_token(login.json()["access_token"], config) == user.id
    assert login.headers["cache-control"] == "no-store"
    assert session.scalar.call_args.args[0].whereclause.compare(LocalCredential.username == "alice")
    session.scalar.return_value = user
    another_id = uuid4()
    response = client.get(
        f"/api/v1/auth/me?user_id={another_id}&owner_id={another_id}",
        headers={"Authorization": "Bearer " + login.json()["access_token"]},
    )
    assert response.status_code == 200
    assert response.json() == {"id": str(user.id), "username": "alice", "status": "active"}
    assert session.scalar.call_args.args[0].whereclause.compare(User.id == user.id)
    assert credential.password_hash not in response.text
    assert "Only a test password" not in response.text


@pytest.mark.parametrize("case", ["unknown", "wrong_password", "inactive"])
def test_login_failures_are_indistinguishable(auth_api, case) -> None:
    client, config, session, user, credential = auth_api
    session.scalar.return_value = None if case == "unknown" else credential
    if case == "inactive":
        user.status = "inactive"
    response = client.post("/api/v1/auth/login", json={"username": "alice", "password": "wrong" if case == "wrong_password" else "Only a test password 123"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Credenciais inválidas."}
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("authorization", [None, "Basic secret", "Bearer invalid.token.value"])
def test_me_requires_valid_bearer_token(auth_api, authorization) -> None:
    client, config, session, user, credential = auth_api
    response = client.get("/api/v1/auth/me", headers={"Authorization": authorization} if authorization else {})
    assert response.status_code == 401
    session.scalar.assert_not_called()


@pytest.mark.parametrize("case", ["unknown", "inactive"])
def test_token_owner_must_exist_and_be_active(auth_api, case) -> None:
    client, config, session, user, credential = auth_api
    if case == "inactive":
        user.status = "inactive"
    session.scalar.return_value = None if case == "unknown" else user
    token = create_access_token(user.id, config)
    response = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer " + token})
    assert response.status_code == 401


def test_changed_subject_without_valid_signature_is_rejected(auth_api) -> None:
    client, config, session, user, credential = auth_api
    token = create_access_token(user.id, config)
    claims = jwt.decode(token, options={"verify_signature": False})
    claims["sub"] = str(uuid4())
    unsigned = jwt.encode(claims, key=None, algorithm="none")
    forged = ".".join(unsigned.split(".")[:2] + [token.split(".")[2]])
    response = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer " + forged})
    assert response.status_code == 401
    session.scalar.assert_not_called()


@pytest.mark.parametrize("body", [
    {"username": "alice", "password": "SENSITIVE" * 30},
    {"username": "alice", "password": {"value": "SENSITIVE"}},
    {"username": "!!SENSITIVE!!", "password": "SENSITIVE"},
    {"username": "alice", "password": "SENSITIVE", "owner_id": "SENSITIVE"},
])
def test_validation_errors_do_not_echo_credentials(auth_api, body) -> None:
    client, config, session, user, credential = auth_api
    response = client.post("/api/v1/auth/login", json=body)
    assert response.status_code == 422
    assert response.json() == {"detail": "Dados de autenticação inválidos."}
    assert "SENSITIVE" not in response.text
    session.scalar.assert_not_called()


def test_database_errors_do_not_expose_details(auth_api) -> None:
    client, config, session, user, credential = auth_api
    session.scalar.side_effect = SQLAlchemyError("SENSITIVE password hash and database details")
    response = client.post("/api/v1/auth/login", json={"username": "alice", "password": "Only a test password 123"})
    assert response.status_code == 503
    assert "SENSITIVE" not in response.text
    token = create_access_token(user.id, config)
    response = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer " + token})
    assert response.status_code == 503
    assert "SENSITIVE" not in response.text


def test_auth_disabled_never_queries_database(auth_api) -> None:
    client, config, session, user, credential = auth_api
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, debug=False, auth_enabled=False)
    for response in (
        client.get("/api/v1/auth/me"),
        client.post("/api/v1/auth/login", json={"username": "alice", "password": "Only a test password 123"}),
    ):
        assert response.status_code == 503
    session.scalar.assert_not_called()


def test_public_signup_disabled_and_upload_protected(auth_api) -> None:
    client, config, session, user, credential = auth_api
    assert client.post("/api/v1/auth/register", json={
        "username": "new_reader", "password": "Only a test password 123",
        "password_confirmation": "Only a test password 123",
    }).status_code == 403
    assert client.post("/api/v1/documents", files={"file": ("document.pdf", b"data", "application/pdf")}).status_code == 401
