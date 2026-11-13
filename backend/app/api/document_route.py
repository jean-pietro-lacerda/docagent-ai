import logging
from collections.abc import Awaitable, Callable

from fastapi import HTTPException, Request
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException
from starlette.responses import Response

from app.services.documents import (
    DocumentFileError, DocumentInputError, DocumentNotFound,
    DocumentPersistenceError, DocumentRegistrationError,
)
from app.services.pdf_validation import MAX_FILE_BYTES, PDFTooLarge, PDFValidationError, UnsupportedPDF

MAX_UPLOAD_BODY = MAX_FILE_BYTES + 64 * 1024
logger = logging.getLogger(__name__)


class DocumentRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            try:
                response = (
                    await self._upload(request, original)
                    if request.method == "POST" else await original(request)
                )
            except DocumentNotFound:
                raise HTTPException(404, "Documento não encontrado.") from None
            except PDFTooLarge as exc:
                raise HTTPException(413, str(exc)) from None
            except UnsupportedPDF as exc:
                raise HTTPException(415, str(exc)) from None
            except (PDFValidationError, DocumentInputError) as exc:
                raise HTTPException(422, str(exc)) from None
            except DocumentFileError as exc:
                raise HTTPException(500, str(exc)) from None
            except (DocumentPersistenceError, DocumentRegistrationError, OSError):
                logger.error("document_operation_failed operation=%s", request.method)
                raise HTTPException(500, "Não foi possível concluir a operação do documento.") from None
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
            return response

        return handler

    @staticmethod
    async def _upload(request: Request, original: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "multipart/form-data":
            raise HTTPException(415, "Envie um único arquivo usando multipart/form-data.")
        total = 0
        oversized = False

        async def limited_receive():
            nonlocal total, oversized
            message = await request.receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > MAX_UPLOAD_BODY:
                    oversized = True
                    raise MultiPartException("Limite do corpo do upload excedido.")
            return message

        limited = Request(request.scope, receive=limited_receive)
        try:
            async with limited.form(max_files=1, max_fields=0, max_part_size=1024) as form:
                if list(form.keys()) != ["file"]:
                    raise HTTPException(422, "Envie somente o campo file com um PDF.")
                return await original(limited)
        except StarletteHTTPException as exc:
            if oversized:
                raise HTTPException(413, "O corpo do upload excede o limite permitido.") from None
            if exc.status_code == 400:
                raise HTTPException(422, "Formulário multipart inválido; envie somente um arquivo.") from None
            raise
