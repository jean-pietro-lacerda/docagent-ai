import logging
import re
from datetime import UTC, datetime
from typing import BinaryIO
from uuid import UUID, uuid4

from sqlalchemy import Select, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import Document
from app.services.pdf_validation import PDFValidator
from app.storage.base import FileStorage

logger = logging.getLogger(__name__)


class DocumentNotFound(LookupError):
    pass


class DocumentInputError(ValueError):
    pass


class DocumentPersistenceError(RuntimeError):
    pass


class DocumentFileError(RuntimeError):
    pass


class DocumentRegistrationError(RuntimeError):
    pass


class DocumentService:
    def __init__(self, storage: FileStorage, validator: PDFValidator | None = None) -> None:
        self.storage = storage
        self.validator = validator if validator is not None else PDFValidator()

    @staticmethod
    def _owned(document_id: UUID, owner_id: UUID) -> Select[tuple[Document]]:
        return select(Document).where(Document.id == document_id, Document.owner_id == owner_id)

    def get(self, document_id: UUID, owner_id: UUID, session: Session) -> Document:
        try:
            document = session.scalar(self._owned(document_id, owner_id))
        except SQLAlchemyError:
            raise DocumentPersistenceError("Não foi possível consultar o documento.") from None

        if document is None:
            raise DocumentNotFound("Documento não encontrado.")

        return document

    def list(self, owner_id: UUID, session: Session, *, limit: int, offset: int) -> list[Document]:
        if not 1 <= limit <= 100 or offset < 0:
            raise DocumentInputError("Paginação inválida.")

        statement = (
            select(Document).where(Document.owner_id == owner_id)
            .order_by(Document.created_at.desc(), Document.id.desc()).limit(limit).offset(offset)
        )

        try:
            return list(session.scalars(statement))
        except SQLAlchemyError:
            raise DocumentPersistenceError("Não foi possível listar os documentos.") from None

    def open_file(self, document: Document) -> BinaryIO:
        try:
            if document.object_key is None:
                raise OSError

            return self.storage.open(document.object_key)
        except OSError:
            logger.error("document_file_unavailable document_id=%s", document.id)
            raise DocumentFileError("O arquivo do documento está indisponível.") from None

    def delete(self, document_id: UUID, owner_id: UUID, session: Session) -> None:
        if session.in_transaction():
            raise DocumentPersistenceError("A exclusão requer uma sessão sem transação ativa.")

        object_key = None

        try:
            with session.begin():
                document = session.scalar(self._owned(document_id, owner_id).with_for_update())

                if document is None:
                    raise DocumentNotFound("Documento não encontrado.")

                object_key = document.object_key
                session.delete(document)
                session.flush()
        except SQLAlchemyError:
            self._log_cleanup("document_delete_transaction_failed", document_id, object_key)
            raise DocumentPersistenceError("Não foi possível confirmar a exclusão do documento.") from None

        if object_key is not None:
            try:
                self.storage.delete(object_key)
            except OSError:
                self._log_cleanup("document_delete_file_cleanup_failed", document_id, object_key)
                raise DocumentFileError(
                    "Metadados excluídos, mas a limpeza do arquivo falhou; necessária reconciliação."
                ) from None

    @staticmethod
    def _log_cleanup(event: str, document_id: UUID, object_key: str | None) -> None:
        safe_key = object_key if object_key and re.fullmatch(r"[0-9a-f]{32}\.pdf", object_key) else "unavailable"
        logger.error("%s document_id=%s object_key=%s", event, document_id, safe_key)

    def register(
        self, source: BinaryIO, *, filename: str, authenticated_owner_id: UUID, session: Session
    ) -> Document:
        """Owns the transaction; owner identity must come from a trusted authentication context."""

        if not isinstance(authenticated_owner_id, UUID):
            raise DocumentInputError("O proprietário autenticado deve ser um UUID.")

        try:
            filename.encode("utf-8")
        except UnicodeEncodeError:
            raise DocumentInputError("O nome original deve conter texto Unicode válido.") from None

        if not filename or len(filename) > 255 or any(ord(char) < 32 for char in filename):
            raise DocumentInputError("Nome original inválido ou maior que 255 caracteres.")

        if session.in_transaction():
            raise DocumentRegistrationError("O registro requer uma sessão sem transação ativa.")

        object_key = None
        document_id = uuid4()

        try:
            with self.validator.validate(source) as pdf:
                object_key = self.storage.store(pdf.path)

            now = datetime.now(UTC)
            document = Document(
                id=document_id, owner_id=authenticated_owner_id, filename=filename,
                file_hash=pdf.sha256, object_key=object_key, pages=pdf.pages,
                status="pending", created_at=now, updated_at=now,
            )

            with session.begin():
                session.add(document)
                session.flush()
                session.expunge(document)
        except Exception:
            if object_key is None:
                raise

            self._log_cleanup("document_register_failed", document_id, object_key)

            try:
                self.storage.delete(object_key)
            except OSError:
                self._log_cleanup("document_register_file_cleanup_failed", document_id, object_key)
                raise DocumentRegistrationError(
                    "Falha no registro e na limpeza do arquivo; necessária reconciliação do armazenamento."
                ) from None

            raise DocumentRegistrationError("Falha ao registrar o documento; arquivo removido.") from None

        return document
