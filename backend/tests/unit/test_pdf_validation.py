import hashlib
from io import BytesIO
import subprocess

import pytest

from app.services.pdf_validation import MAX_FILE_BYTES, PDFValidationError, PDFValidator


@pytest.fixture
def private_temp(monkeypatch, tmp_path):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


def test_valid_pdf_pages_hash_and_cleanup(make_pdf, private_temp) -> None:
    content = make_pdf(pages=2)
    source = BytesIO(content)
    with PDFValidator().validate(source) as pdf:
        assert pdf.pages == 2
        assert pdf.sha256 == hashlib.sha256(content).hexdigest()
        assert pdf.path.read_bytes() == content
        path = pdf.path
    assert not source.closed
    assert not path.exists()
    assert list(private_temp.iterdir()) == []


@pytest.mark.parametrize(
    ("content", "message"),
    [(b"", "vazio"), (b"not a PDF", "não é um PDF"),
     (b"%PDF-1.7\ninvalid structure\n%%EOF\n", "corrompido"),
     (b"%PDF-1.7\ntruncated", "incompleto")],
)
def test_invalid_content_and_cleanup(content, message, private_temp) -> None:
    with pytest.raises(PDFValidationError, match=message):
        with PDFValidator().validate(BytesIO(content)):
            pytest.fail("Conteúdo inválido aceito")
    assert list(private_temp.iterdir()) == []


@pytest.mark.parametrize("password", ["secret", ""])
def test_encrypted_pdf_is_rejected(make_pdf, private_temp, password) -> None:
    with pytest.raises(PDFValidationError, match="criptografados"):
        with PDFValidator().validate(BytesIO(make_pdf(encrypted=True, user_password=password))):
            pytest.fail("PDF criptografado aceito")
    assert list(private_temp.iterdir()) == []


def test_repaired_pdf_is_rejected(make_pdf, private_temp) -> None:
    import re

    damaged = re.sub(rb"startxref\n\d+", b"startxref\n0", make_pdf())
    with pytest.raises(PDFValidationError, match="corrompido"):
        with PDFValidator().validate(BytesIO(damaged)):
            pytest.fail("PDF reparado aceito")
    assert list(private_temp.iterdir()) == []


def test_page_limit(make_pdf, private_temp) -> None:
    with PDFValidator().validate(BytesIO(make_pdf(pages=20))) as pdf:
        assert pdf.pages == 20
    with pytest.raises(PDFValidationError, match="20 páginas"):
        with PDFValidator().validate(BytesIO(make_pdf(pages=21))):
            pytest.fail("Limite de páginas ignorado")
    assert list(private_temp.iterdir()) == []


def test_size_limit_stops_reading_and_removes_temporary(private_temp) -> None:
    class LargeStream:
        total = 0

        def read(self, size):
            assert 0 < size <= 64 * 1024
            self.total += size
            return b"x" * size

    source = LargeStream()
    with pytest.raises(PDFValidationError, match="5 MiB"):
        with PDFValidator().validate(source):
            pytest.fail("Limite de bytes ignorado")
    assert source.total == MAX_FILE_BYTES + 1
    assert list(private_temp.iterdir()) == []


def test_exact_size_limit_is_accepted(make_pdf, private_temp) -> None:
    content = make_pdf()
    position = content.rfind(b"startxref")
    content = content[:position] + b" " * (MAX_FILE_BYTES - len(content)) + content[position:]
    with PDFValidator().validate(BytesIO(content)) as pdf:
        assert pdf.path.stat().st_size == MAX_FILE_BYTES


def test_read_error_is_sanitized_and_cleaned(private_temp) -> None:
    class BrokenStream:
        def read(self, size):
            raise OSError("sensitive content")

    with pytest.raises(PDFValidationError, match="Falha ao ler") as error:
        with PDFValidator().validate(BrokenStream()):
            pytest.fail("Falha de leitura ignorada")
    assert "sensitive" not in str(error.value)
    assert list(private_temp.iterdir()) == []


def test_timeout_and_resource_failure(make_pdf, private_temp, monkeypatch) -> None:
    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 5
        raise subprocess.TimeoutExpired("worker", 5)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(PDFValidationError, match="tempo permitido"):
        with PDFValidator().validate(BytesIO(make_pdf())):
            pytest.fail("Timeout ignorado")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess([], -9))
    with pytest.raises(PDFValidationError, match="limites de recursos"):
        with PDFValidator().validate(BytesIO(make_pdf())):
            pytest.fail("Falha do worker ignorada")
    assert list(private_temp.iterdir()) == []


def test_temporary_removed_if_consumer_fails(make_pdf, private_temp) -> None:
    with pytest.raises(RuntimeError):
        with PDFValidator().validate(BytesIO(make_pdf())):
            raise RuntimeError("consumer failed")
    assert list(private_temp.iterdir()) == []
