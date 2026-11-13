from app.db.models import Document
from app.schemas.extraction import ExtractedPage, ExtractionStatus, PDFExtractionResult, PageExtractionStatus
from app.services.pdf_validation import PDFValidationError, PDFValidator, run_pdf_worker
from app.storage.base import FileStorage

MIN_USEFUL_CHARACTERS = 20


class PDFExtractionError(RuntimeError):
    pass


def clean_page_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text = "".join(char for char in text if char in "\n\t" or (ord(char) >= 32 and ord(char) != 127))
    return "\n".join(line.rstrip(" \t") for line in text.split("\n")).strip("\n")


def useful_character_count(text: str) -> int:
    return sum(character.isalnum() for character in text)


class PDFTextExtractor:
    def __init__(self, storage: FileStorage, validator: PDFValidator | None = None) -> None:
        self.storage = storage
        self.validator = validator if validator is not None else PDFValidator()

    def extract(self, document: Document) -> PDFExtractionResult:
        """Accept an internally authorized document, never a client-supplied file path."""
        if document.object_key is None:
            raise PDFExtractionError("O documento não possui arquivo armazenado.")
        try:
            with self.storage.open(document.object_key) as source:
                with self.validator.validate(source) as pdf:
                    if document.file_hash is not None and document.file_hash != pdf.sha256:
                        raise PDFExtractionError("O conteúdo armazenado não corresponde ao hash do documento.")
                    data = run_pdf_worker(pdf.path, extract_text=True)
                    texts = data.get("page_texts")
                    if data.get("pages") != pdf.pages or not isinstance(texts, list) or len(texts) != pdf.pages:
                        raise PDFExtractionError("Resposta de extração incompleta ou inválida.")
                    if any(text is not None and not isinstance(text, str) for text in texts):
                        raise PDFExtractionError("Resposta de extração inválida.")
                    pages = tuple(self._page(number, text) for number, text in enumerate(texts, start=1))
        except OSError:
            raise PDFExtractionError("Não foi possível ler ou preparar o arquivo privado para extração.") from None
        except PDFValidationError as exc:
            raise PDFExtractionError(str(exc)) from None

        useful = sum(useful_character_count(page.text) for page in pages)
        failed = any(page.extraction_status == "failed" for page in pages)
        status: ExtractionStatus
        if not useful:
            status = "failed" if failed else "no_text"
        elif useful < MIN_USEFUL_CHARACTERS:
            status = "insufficient_text"
        elif any(page.extraction_status != "extracted" for page in pages):
            status = "partial"
        else:
            status = "extracted"
        return PDFExtractionResult(
            document_id=document.id, total_pages=len(pages), pages=pages,
            total_characters=sum(page.character_count for page in pages),
            pages_without_text=tuple(page.page_number for page in pages if not page.text),
            extraction_status=status,
        )

    @staticmethod
    def _page(number: int, raw: str | None) -> ExtractedPage:
        status: PageExtractionStatus
        text = clean_page_text(raw) if raw is not None else ""
        if raw is None:
            status = "failed"
        elif not text.strip():
            text, status = "", "empty"
        elif useful_character_count(text) < MIN_USEFUL_CHARACTERS:
            status = "insufficient_text"
        else:
            status = "extracted"
        return ExtractedPage(number, text, len(text), status)
