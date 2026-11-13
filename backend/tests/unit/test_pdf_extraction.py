import hashlib
from io import BytesIO
import json
import subprocess
from types import SimpleNamespace
from uuid import uuid4

import pymupdf
import pytest

from app.db.models import Document
from app.services import pdf_inspector
from app.services.pdf_extraction import PDFExtractionError, PDFTextExtractor, clean_page_text
from app.storage.local import LocalFileStorage


@pytest.fixture
def stored_pdf(tmp_path, monkeypatch):
    import tempfile

    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(staging))
    storage = LocalFileStorage(tmp_path / "private")
    opened = []
    original_open = storage.open

    def tracked_open(key):
        stream = original_open(key)
        opened.append(stream)
        return stream

    monkeypatch.setattr(storage, "open", tracked_open)

    def create(texts):
        with pymupdf.open() as pdf:
            for text in texts:
                page = pdf.new_page()
                if text is None:
                    image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 8, 8), 0)
                    image.clear_with(255)
                    page.insert_image(pymupdf.Rect(20, 20, 100, 100), pixmap=image)
                elif text:
                    page.insert_text((72, 72), text)
            content = pdf.tobytes()
        source = tmp_path / "source.pdf"
        source.write_bytes(content)
        key = storage.store(source)
        document = Document(
            id=uuid4(), owner_id=uuid4(), filename="document.pdf", object_key=key,
            file_hash=hashlib.sha256(content).hexdigest(), pages=len(texts), status="pending",
        )
        return SimpleNamespace(
            document=document, storage=storage, opened=opened, staging=staging,
            path=storage.directory / key,
        )

    return create


def test_digital_pdf_preserves_pages_values_and_resources(stored_pdf, caplog):
    first = "Art. 12 - Prazo de inscricao\nData: 31/12/2026\nValor: R$ 1.234,56"
    last = "Item 2.1 - Documentos obrigatorios para habilitacao."
    pdf = stored_pdf([first, "", last])
    result = PDFTextExtractor(pdf.storage).extract(pdf.document)
    assert result.document_id == pdf.document.id
    assert result.total_pages == 3
    assert [page.page_number for page in result.pages] == [1, 2, 3]
    assert result.pages[0].text == first
    assert result.pages[2].text == last
    assert result.pages[1].extraction_status == "empty"
    assert result.pages_without_text == (2,)
    assert result.extraction_status == "partial"
    assert result.total_characters == sum(len(page.text) for page in result.pages)
    assert all(page.character_count == len(page.text) for page in result.pages)
    assert pdf.document.status == "pending" and pdf.document.pages == 3
    assert pdf.document.file_hash == hashlib.sha256(pdf.path.read_bytes()).hexdigest()
    assert pdf.opened and all(stream.closed for stream in pdf.opened)
    assert list(pdf.staging.iterdir()) == []
    assert first not in repr(result) + caplog.text


@pytest.mark.parametrize("text, status", [
    ("", "no_text"), (None, "no_text"), ("12", "insufficient_text"),
    ("Um documento digital com texto suficiente para extracao.", "extracted"),
])
def test_page_text_availability_is_explicit(stored_pdf, text, status):
    pdf = stored_pdf([text])
    result = PDFTextExtractor(pdf.storage).extract(pdf.document)
    assert result.total_pages == 1 and result.pages[0].page_number == 1
    assert result.extraction_status == status
    if status == "no_text":
        assert result.total_characters == 0 and result.pages_without_text == (1,)
    elif status == "insufficient_text":
        assert result.pages[0].text == "12"
        assert result.pages[0].extraction_status == "insufficient_text"
    else:
        assert result.pages[0].text == text
        assert result.pages[0].extraction_status == "extracted"


def test_text_cleaning_is_conservative():
    raw = "\r\nArt. 12\tItem 2.1  \r\nR$\u00a01.234,56; 31/12/2026\x00\r\n\r\nPrazo: 10 dias\r\n"
    assert clean_page_text(raw) == "Art. 12\tItem 2.1\nR$\u00a01.234,56; 31/12/2026\n\nPrazo: 10 dias"


@pytest.mark.parametrize("failure", ["missing", "read", "hash"])
def test_read_and_integrity_failures_are_sanitized_and_cleaned(stored_pdf, monkeypatch, failure):
    pdf = stored_pdf(["Um documento digital com texto suficiente para extracao."])
    if failure == "missing":
        pdf.storage.delete(pdf.document.object_key)
    elif failure == "hash":
        pdf.document.file_hash = "0" * 64
    else:
        class BrokenReader(BytesIO):
            def read(self, size):
                raise OSError("SENSITIVE private path or content")
        broken = BrokenReader()
        pdf.opened.append(broken)
        monkeypatch.setattr(pdf.storage, "open", lambda key: broken)
    with pytest.raises(PDFExtractionError) as error:
        PDFTextExtractor(pdf.storage).extract(pdf.document)
    assert "SENSITIVE" not in str(error.value)
    assert all(stream.closed for stream in pdf.opened)
    assert list(pdf.staging.iterdir()) == []


@pytest.mark.parametrize("failure", ["timeout", "resource"])
def test_extraction_processing_limits_close_source_and_temporary(stored_pdf, monkeypatch, failure):
    pdf = stored_pdf(["Um documento digital com texto suficiente para extracao."])
    original_run = subprocess.run

    def controlled_run(command, **kwargs):
        if command[-1] == "extract":
            if failure == "timeout":
                raise subprocess.TimeoutExpired(command, 5)
            return subprocess.CompletedProcess(command, -9, stdout=b"")
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", controlled_run)
    with pytest.raises(PDFExtractionError, match="tempo permitido|limites de recursos"):
        PDFTextExtractor(pdf.storage).extract(pdf.document)
    assert pdf.opened and all(stream.closed for stream in pdf.opened)
    assert list(pdf.staging.iterdir()) == []


def test_worker_limits_text_output(stored_pdf, monkeypatch):
    pdf = stored_pdf(["Um documento digital com texto suficiente para extracao."])
    monkeypatch.setattr(pdf_inspector, "MAX_TEXT_CHARACTERS", 10)
    data = pdf_inspector.inspect_pdf(str(pdf.path), extract_text=True)
    assert data == {"error": "text_limit"}
    monkeypatch.setattr(
        "app.services.pdf_validation.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, stdout=json.dumps(data).encode()),
    )
    with pytest.raises(PDFExtractionError, match="limite de 1.000.000"):
        PDFTextExtractor(pdf.storage).extract(pdf.document)
    assert all(stream.closed for stream in pdf.opened)
    assert list(pdf.staging.iterdir()) == []


@pytest.mark.parametrize("failed_pages, expected", [({0}, "partial"), ({0, 1}, "failed")])
def test_page_failures_keep_original_page_numbers(stored_pdf, monkeypatch, failed_pages, expected):
    pdf = stored_pdf(["Texto relevante na primeira pagina para analise.", "Texto relevante na segunda pagina para analise."])
    original = pymupdf.Page.get_text

    def extract(page, *args, **kwargs):
        if page.number in failed_pages:
            raise pymupdf.mupdf.FzErrorGeneric("SENSITIVE parser detail")
        return original(page, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "get_text", extract)
    data = pdf_inspector.inspect_pdf(str(pdf.path), extract_text=True)
    monkeypatch.setattr("app.services.pdf_extraction.run_pdf_worker", lambda *a, **kw: data)
    result = PDFTextExtractor(pdf.storage).extract(pdf.document)
    assert result.extraction_status == expected
    assert [page.page_number for page in result.pages] == [1, 2]
    for page in result.pages:
        assert page.extraction_status == ("failed" if page.page_number - 1 in failed_pages else "extracted")
    assert "SENSITIVE" not in repr(result)
