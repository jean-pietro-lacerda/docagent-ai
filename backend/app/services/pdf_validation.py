import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO


MAX_FILE_BYTES = 5 * 1024 * 1024
READ_SIZE = 64 * 1024


class PDFValidationError(ValueError):
    pass


class PDFTooLarge(PDFValidationError):
    pass


class UnsupportedPDF(PDFValidationError):
    pass


@dataclass(frozen=True)
class ValidatedPDF:
    path: Path
    sha256: str
    pages: int


class PDFValidator:
    @contextmanager
    def validate(self, source: BinaryIO) -> Iterator[ValidatedPDF]:
        with TemporaryDirectory(prefix=".pdf-upload-") as temporary:
            path = Path(temporary) / "upload.pdf"
            digest = hashlib.sha256()
            total = 0
            try:
                with path.open("xb") as output:
                    while True:
                        block = source.read(min(READ_SIZE, MAX_FILE_BYTES - total + 1))
                        if not isinstance(block, bytes):
                            raise PDFValidationError("Não foi possível ler o arquivo binário.")
                        if not block:
                            break
                        total += len(block)
                        if total > MAX_FILE_BYTES:
                            raise PDFTooLarge("O PDF excede o limite de 5 MiB.")
                        digest.update(block)
                        output.write(block)
            except OSError:
                raise PDFValidationError("Falha ao ler ou preparar o arquivo temporário.") from None
            if not total:
                raise PDFValidationError("O arquivo está vazio.")
            with path.open("rb") as file:
                header = file.read(5)
                file.seek(max(0, total - 1024))
                ending = file.read(1024)
            if header != b"%PDF-":
                raise UnsupportedPDF("O conteúdo do arquivo não é um PDF.")
            if not re.search(rb"%%EOF\s*\Z", ending):
                raise PDFValidationError("O PDF está corrompido ou incompleto.")
            pages = self._inspect(path)
            yield ValidatedPDF(path, digest.hexdigest(), pages)

    def _inspect(self, path: Path) -> int:
        data = run_pdf_worker(path)
        pages = data.get("pages")
        if type(pages) is not int or not 1 <= pages <= 20:
            raise PDFValidationError("A contagem de páginas do PDF é inválida.")
        return pages


def run_pdf_worker(path: Path, *, extract_text: bool = False) -> dict:
    command = [sys.executable, "-I", str(Path(__file__).with_name("pdf_inspector.py")), str(path)]
    if extract_text:
        command.append("extract")
    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5, check=False,
            env={"PATH": os.defpath, "LANG": "C.UTF-8"},
        )
    except subprocess.TimeoutExpired:
        raise PDFValidationError("O PDF excedeu o tempo permitido para processamento.") from None
    except OSError:
        raise PDFValidationError("Não foi possível iniciar o processamento do PDF.") from None
    if result.returncode != 0:
        raise PDFValidationError("Não foi possível processar o PDF dentro dos limites de recursos.")
    try:
        data = json.loads(result.stdout)
    except (ValueError, TypeError):
        raise PDFValidationError("O processamento do PDF não pôde ser concluído.") from None
    if not isinstance(data, dict):
        raise PDFValidationError("Resposta inválida do processamento do PDF.")
    errors = {
        "invalid": "O conteúdo do arquivo não é um PDF.",
        "encrypted": "PDFs criptografados ou protegidos por senha não são aceitos.",
        "corrupt": "O PDF está corrompido ou precisou de reparo.",
        "pages": "O PDF excede o limite de 20 páginas.",
        "empty_pages": "O PDF não possui páginas.",
        "text_limit": "O texto extraído excede o limite de 1.000.000 de caracteres.",
        "resources": "Não foi possível processar o PDF dentro dos limites de recursos.",
    }
    if "error" in data:
        raise PDFValidationError(errors.get(data["error"], "Não foi possível processar o PDF."))
    return data
