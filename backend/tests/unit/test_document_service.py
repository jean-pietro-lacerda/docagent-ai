import hashlib
from io import BytesIO
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.services.documents import DocumentRegistrationError, DocumentService
from app.services.pdf_validation import PDFValidationError
from app.storage.base import StorageError
from app.storage.local import LocalFileStorage


def test_registration_metadata_and_malicious_filename(tmp_path, make_pdf, db_session) -> None:
    storage = LocalFileStorage(tmp_path / "private")
    content = make_pdf(pages=2)
    owner = uuid4()
    filename = "../../private\\original.pdf"
    document = DocumentService(storage).register(
        BytesIO(content), filename=filename, authenticated_owner_id=owner, session=db_session,
    )
    assert document.id is not None
    assert document.owner_id == owner
    assert document.filename == filename
    assert document.file_hash == hashlib.sha256(content).hexdigest()
    assert document.pages == 2
    assert document.status == "pending"
    assert document.created_at == document.updated_at
    assert document.created_at.utcoffset().total_seconds() == 0
    assert document.object_key != filename
    with storage.open(document.object_key) as stream:
        assert stream.read() == content
    db_session.add.assert_called_once_with(document)
    db_session.flush.assert_called_once()
    db_session.expunge.assert_called_once_with(document)
    db_session.begin.return_value.__exit__.assert_called_once_with(None, None, None)


@pytest.mark.parametrize("failure", ["add", "flush", "commit"])
def test_database_failure_removes_file(tmp_path, make_pdf, db_session, failure) -> None:
    storage = LocalFileStorage(tmp_path / "private")
    if failure == "commit":
        db_session.begin.return_value.__exit__.side_effect = SQLAlchemyError("secret SQL details")
    else:
        getattr(db_session, failure).side_effect = SQLAlchemyError("secret SQL details")
    with pytest.raises(DocumentRegistrationError, match="arquivo removido") as error:
        DocumentService(storage).register(
            BytesIO(make_pdf()), filename="original.pdf", authenticated_owner_id=uuid4(), session=db_session,
        )
    assert "secret" not in str(error.value)
    assert list(storage.directory.iterdir()) == []
    assert db_session.begin.return_value.__exit__.called


def test_storage_failure_prevents_database_write(tmp_path, make_pdf, db_session) -> None:
    storage = MagicMock()
    storage.store.side_effect = StorageError("Falha de armazenamento")
    with pytest.raises(StorageError):
        DocumentService(storage).register(
            BytesIO(make_pdf()), filename="original.pdf", authenticated_owner_id=uuid4(), session=db_session,
        )
    db_session.begin.assert_not_called()
    db_session.add.assert_not_called()
    storage.delete.assert_not_called()
    assert not storage.store.call_args.args[0].exists()


def test_cleanup_failure_is_explicit(tmp_path, make_pdf, db_session) -> None:
    storage = MagicMock()
    storage.store.return_value = "a" * 32 + ".pdf"
    storage.delete.side_effect = StorageError("secret storage path")
    db_session.flush.side_effect = SQLAlchemyError("secret DB data")
    with pytest.raises(DocumentRegistrationError, match="reconciliação") as error:
        DocumentService(storage).register(
            BytesIO(make_pdf()), filename="original.pdf", authenticated_owner_id=uuid4(), session=db_session,
        )
    assert "secret" not in str(error.value)
    storage.delete.assert_called_once_with(storage.store.return_value)


def test_invalid_pdf_never_reaches_storage_or_database(db_session) -> None:
    storage = MagicMock()
    with pytest.raises(PDFValidationError):
        DocumentService(storage).register(
            BytesIO(b"not PDF"), filename="forged.pdf", authenticated_owner_id=uuid4(), session=db_session,
        )
    storage.store.assert_not_called()
    db_session.add.assert_not_called()


def test_active_transaction_is_preserved(db_session) -> None:
    db_session.in_transaction.return_value = True
    storage = MagicMock()
    with pytest.raises(DocumentRegistrationError, match="transação ativa"):
        DocumentService(storage).register(
            BytesIO(b""), filename="original.pdf", authenticated_owner_id=uuid4(), session=db_session,
        )
    db_session.begin.assert_not_called()
    db_session.rollback.assert_not_called()
    storage.store.assert_not_called()


@pytest.mark.parametrize("filename", ["", "a" * 256, "file\x00.pdf", "file\n.pdf"])
def test_invalid_filename_is_rejected(filename, db_session) -> None:
    storage = MagicMock()
    with pytest.raises(ValueError, match="Nome original"):
        DocumentService(storage).register(
            BytesIO(b""), filename=filename, authenticated_owner_id=uuid4(), session=db_session,
        )
    storage.store.assert_not_called()


def test_owner_requires_uuid_from_internal_context(db_session) -> None:
    storage = MagicMock()
    with pytest.raises(ValueError, match="UUID"):
        DocumentService(storage).register(
            BytesIO(b""), filename="original.pdf", authenticated_owner_id="client-id", session=db_session,
        )
    storage.store.assert_not_called()
