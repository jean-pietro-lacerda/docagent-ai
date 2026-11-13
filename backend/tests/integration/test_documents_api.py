import asyncio
from datetime import UTC, datetime
import hashlib
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
import httpx
import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.dependencies import get_document_session
from app.api.document_route import MAX_UPLOAD_BODY
from app.api.routes.documents import get_document_service
from app.core.config import get_settings
from app.db.models import Document, User
from app.db.session import get_db
from app.main import app
from app.security.tokens import create_access_token
from app.services.documents import DocumentService
from app.services.pdf_validation import MAX_FILE_BYTES
from app.storage.base import StorageError
from app.storage.local import LocalFileStorage


@pytest.fixture
def documents_api(auth_config, tmp_path):
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, record):
        connection.execute("PRAGMA foreign_keys=ON")

    User.__table__.create(engine)
    Document.__table__.create(engine)
    owner, other_owner = uuid4(), uuid4()
    with engine.begin() as connection:
        connection.execute(User.__table__.insert(), [{"id": owner}, {"id": other_owner}])
    sessions = sessionmaker(bind=engine, autoflush=False)
    auth_session = MagicMock()
    auth_session.in_transaction.return_value = True

    def authenticated_user(statement):
        assert statement.whereclause.compare(User.id == owner)
        return User(id=owner, status="active")

    auth_session.scalar.side_effect = authenticated_user

    def document_session():
        with sessions() as session:
            yield session

    storage = LocalFileStorage(tmp_path / "private")
    service = DocumentService(storage)
    original = app.dependency_overrides.copy()
    app.dependency_overrides.update({
        get_settings: lambda: auth_config, get_db: lambda: auth_session,
        get_document_session: document_session, get_document_service: lambda: service,
    })
    try:
        with TestClient(app) as client:
            yield SimpleNamespace(
                client=client, owner=owner, other_owner=other_owner, engine=engine,
                sessions=sessions, service=service, storage=storage, auth_session=auth_session,
                headers={"Authorization": "Bearer " + create_access_token(owner, auth_config)},
            )
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original)
        engine.dispose()


def register(api, content, owner=None):
    with api.sessions() as session:
        return api.service.register(
            BytesIO(content), filename="original.pdf", authenticated_owner_id=owner or api.owner,
            session=session,
        )


def test_authenticated_document_lifecycle_and_contract(documents_api, make_pdf, monkeypatch):
    api = documents_api
    content = make_pdf(pages=2)
    response = api.client.post(
        "/api/v1/documents", headers=api.headers,
        files={"file": ("../../relatório.pdf", content, "application/octet-stream")},
    )
    assert response.status_code == 201, response.text
    created = response.json()
    assert set(created) == {"id", "filename", "pages", "file_hash", "status", "created_at"}
    assert created["pages"] == 2 and created["status"] == "pending"
    assert created["file_hash"] == hashlib.sha256(content).hexdigest()
    url = "/api/v1/documents/" + created["id"]
    listed, = api.client.get("/api/v1/documents", headers=api.headers).json()["items"]
    assert {key: value for key, value in listed.items() if key != "created_at"} == {
        key: value for key, value in created.items() if key != "created_at"
    }
    assert datetime.fromisoformat(listed["created_at"]).replace(tzinfo=UTC) == datetime.fromisoformat(created["created_at"])
    detail = api.client.get(url, headers=api.headers)
    assert detail.status_code == 200
    assert set(detail.json()) == set(created) | {"updated_at"}
    opened = []
    original_open = api.storage.open

    def track_open(key):
        stream = original_open(key)
        opened.append(stream)
        return stream

    monkeypatch.setattr(api.storage, "open", track_open)
    download = api.client.get(url + "/file", headers=api.headers)
    assert download.status_code == 200 and download.content == content
    assert download.headers["content-type"] == "application/pdf"
    assert download.headers["cache-control"] == "no-store"
    assert download.headers["x-content-type-options"] == "nosniff"
    assert "../" not in download.headers["content-disposition"]
    assert "relat%C3%B3rio.pdf" in download.headers["content-disposition"]
    assert opened and all(stream.closed for stream in opened)
    assert api.client.delete(url, headers=api.headers).status_code == 204
    assert api.client.get(url, headers=api.headers).status_code == 404
    assert list(api.storage.directory.iterdir()) == []


def test_all_document_routes_require_authentication(documents_api):
    api = documents_api
    url = "/api/v1/documents/" + str(uuid4())
    for method, path, kwargs in [
        ("POST", "/api/v1/documents", {"files": {"file": ("test.pdf", b"data")}}),
        ("GET", "/api/v1/documents", {}), ("GET", url, {}),
        ("GET", url + "/file", {}), ("DELETE", url, {}),
    ]:
        response = api.client.request(method, path, **kwargs)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
    api.auth_session.scalar.assert_not_called()


@pytest.mark.parametrize("case, expected", [("invalid", 415), ("oversized", 413), ("encrypted", 422)])
def test_upload_validation_errors(documents_api, make_pdf, case, expected):
    api = documents_api
    content = {"invalid": b"fake PDF", "oversized": b"x" * (MAX_FILE_BYTES + 1)}.get(case)
    if case == "encrypted":
        content = make_pdf(encrypted=True)
    response = api.client.post("/api/v1/documents", headers=api.headers, files={"file": ("fake.pdf", content, "application/pdf")})
    assert response.status_code == expected
    assert "object_key" not in response.text and str(api.storage.directory) not in response.text
    assert list(api.storage.directory.iterdir()) == []


def test_owner_filter_pagination_and_idor(documents_api, make_pdf):
    api = documents_api
    private = register(api, make_pdf(), api.other_owner)
    url = "/api/v1/documents/" + str(private.id)
    assert api.client.get("/api/v1/documents", headers=api.headers).json()["items"] == []
    for method, path in [("GET", url), ("GET", url + "/file"), ("DELETE", url)]:
        assert api.client.request(method, path, headers=api.headers).status_code == 404
    with api.storage.open(private.object_key) as file:
        assert file.read(5) == b"%PDF-"
    now = datetime.now(UTC)
    with api.engine.begin() as connection:
        connection.execute(Document.__table__.insert(), [
            {"id": UUID(int=value), "owner_id": api.owner, "filename": "test.pdf", "created_at": now}
            for value in (1, 2, 3)
        ])
    page = api.client.get("/api/v1/documents?limit=1&offset=1&owner_id=" + str(api.other_owner), headers=api.headers)
    assert page.status_code == 200
    assert [item["id"] for item in page.json()["items"]] == [str(UUID(int=2))]
    assert page.json()["limit"] == page.json()["offset"] == 1
    assert api.client.get("/api/v1/documents?limit=101", headers=api.headers).status_code == 422
    assert api.client.get("/api/v1/documents/not-a-uuid", headers=api.headers).status_code == 422


def test_upload_rejects_client_supplied_owner(documents_api, make_pdf):
    api = documents_api
    response = api.client.post(
        "/api/v1/documents", headers=api.headers, data={"owner_id": str(api.other_owner)},
        files={"file": ("test.pdf", make_pdf())},
    )
    assert response.status_code == 422
    with api.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(Document)) == 0
    assert list(api.storage.directory.iterdir()) == []


def test_missing_file_is_controlled_and_logged(documents_api, make_pdf, caplog):
    api = documents_api
    document = register(api, make_pdf())
    api.storage.delete(document.object_key)
    response = api.client.get(f"/api/v1/documents/{document.id}/file", headers=api.headers)
    assert response.status_code == 500
    assert str(document.id) in caplog.text
    assert str(api.storage.directory) not in response.text + caplog.text


@pytest.mark.parametrize("failure", ["transaction", "storage"])
def test_delete_partial_failures_are_explicit(documents_api, make_pdf, monkeypatch, caplog, failure):
    api = documents_api
    document = register(api, make_pdf())
    if failure == "storage":
        def fail_delete(key):
            raise StorageError("SENSITIVE path")
        monkeypatch.setattr(api.storage, "delete", fail_delete)
    else:
        def failing_session():
            with api.sessions() as session:
                def fail_flush(*args):
                    raise SQLAlchemyError("SENSITIVE SQL")
                event.listen(session, "before_flush", fail_flush)
                yield session
        app.dependency_overrides[get_document_session] = failing_session
    response = api.client.delete(f"/api/v1/documents/{document.id}", headers=api.headers)
    assert response.status_code == 500
    with api.engine.connect() as connection:
        count = connection.scalar(select(func.count()).select_from(Document))
    assert count == (1 if failure == "transaction" else 0)
    assert (api.storage.directory / document.object_key).exists()
    assert str(document.id) in caplog.text and document.object_key in caplog.text
    assert "SENSITIVE" not in response.text + caplog.text


def test_upload_storage_and_transaction_errors_leave_no_orphans(documents_api, make_pdf, monkeypatch):
    api = documents_api
    content = make_pdf()
    with monkeypatch.context() as patch:
        def no_space(source):
            raise StorageError("SENSITIVE path")
        patch.setattr(api.storage, "store", no_space)
        response = api.client.post("/api/v1/documents", headers=api.headers, files={"file": ("test.pdf", content)})
        assert response.status_code == 500 and "SENSITIVE" not in response.text

    def failing_session():
        with api.sessions() as session:
            def fail_commit(session):
                raise SQLAlchemyError("SENSITIVE connection")
            event.listen(session, "before_commit", fail_commit)
            yield session
    app.dependency_overrides[get_document_session] = failing_session
    response = api.client.post("/api/v1/documents", headers=api.headers, files={"file": ("test.pdf", content)})
    assert response.status_code == 500 and "SENSITIVE" not in response.text
    assert list(api.storage.directory.iterdir()) == []
    with api.engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(Document)) == 0


@pytest.mark.parametrize("declared_length", [None, "1"])
def test_body_limit_counts_actual_stream_and_closes_spools(documents_api, monkeypatch, declared_length):
    import starlette.formparsers as parsers

    api = documents_api
    opened = []
    real_spool = parsers.SpooledTemporaryFile

    def spool(*args, **kwargs):
        file = real_spool(*args, **kwargs)
        opened.append(file)
        return file
    monkeypatch.setattr(parsers, "SpooledTemporaryFile", spool)

    async def content():
        yield b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="test.pdf"\r\n\r\n'
        yield b"x" * (1024 * 1024)
        yield b"x" * MAX_UPLOAD_BODY

    async def send():
        headers = {**api.headers, "Content-Type": "multipart/form-data; boundary=boundary"}
        if declared_length is not None:
            headers["Content-Length"] = declared_length
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/api/v1/documents", headers=headers, content=content())

    response = asyncio.run(send())
    assert response.status_code == 413
    assert opened and all(file.closed for file in opened)
    assert list(api.storage.directory.iterdir()) == []
