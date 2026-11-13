import logging
from collections.abc import Iterator
from typing import Annotated, BinaryIO
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user, get_document_session
from app.api.document_route import DocumentRoute
from app.db.models import User
from app.schemas.documents import DocumentDetail, DocumentPage, DocumentResponse
from app.services.documents import DocumentFileError, DocumentService
from app.storage.local import LocalFileStorage

router = APIRouter(prefix="/api/v1/documents", tags=["documents"], route_class=DocumentRoute)
logger = logging.getLogger(__name__)
CurrentUser = Annotated[User, Depends(get_current_user)]
DocumentSession = Annotated[Session, Depends(get_document_session)]


def get_document_service() -> DocumentService:
    return DocumentService(LocalFileStorage())


Service = Annotated[DocumentService, Depends(get_document_service)]


@router.post("", status_code=201, response_model=DocumentResponse)
def upload_document(
    user: CurrentUser, session: DocumentSession, service: Service,
    file: Annotated[UploadFile, File(description="Um PDF de até 5 MiB e 20 páginas.")],
) -> DocumentResponse:
    document = service.register(
        file.file, filename=file.filename or "", authenticated_owner_id=user.id, session=session,
    )

    return DocumentResponse.model_validate(document)


@router.get("", response_model=DocumentPage)
def list_documents(
    user: CurrentUser, session: DocumentSession, service: Service,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentPage:
    documents = service.list(user.id, session, limit=limit, offset=offset)

    return DocumentPage(items=[DocumentResponse.model_validate(item) for item in documents], limit=limit, offset=offset)


@router.get("/{document_id}", response_model=DocumentDetail)
def get_document(document_id: UUID, user: CurrentUser, session: DocumentSession, service: Service) -> DocumentDetail:
    return DocumentDetail.model_validate(service.get(document_id, user.id, session))


def file_chunks(stream: BinaryIO, document_id: UUID) -> Iterator[bytes]:
    try:
        while chunk := stream.read(64 * 1024):
            yield chunk
    except OSError:
        logger.error("document_download_interrupted document_id=%s", document_id)
        raise DocumentFileError("A transferência do documento foi interrompida.") from None


class PDFStreamResponse(StreamingResponse):
    def __init__(self, stream: BinaryIO, document_id: UUID, **kwargs):
        self.stream = stream
        self.document_id = document_id
        super().__init__(file_chunks(stream, document_id), **kwargs)

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                self.stream.close()
            except OSError:
                logger.error("document_download_close_failed document_id=%s", self.document_id)


def download_filename(filename: str) -> str:
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(char for char in name if char.isprintable()).strip(" .")[:200]

    return name if name else "document.pdf"


@router.get("/{document_id}/file", response_class=StreamingResponse, responses={200: {"content": {"application/pdf": {}}}})
def download_document(document_id: UUID, user: CurrentUser, session: DocumentSession, service: Service) -> StreamingResponse:
    document = service.get(document_id, user.id, session)
    stream = service.open_file(document)
    disposition = "attachment; filename*=UTF-8''" + quote(download_filename(document.filename), safe="")

    return PDFStreamResponse(
        stream, document.id, media_type="application/pdf",
        headers={"Content-Disposition": disposition},
    )


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: UUID, user: CurrentUser, session: DocumentSession, service: Service) -> Response:
    service.delete(document_id, user.id, session)

    return Response(status_code=204)
