import os
import re
import shutil
from pathlib import Path
from tempfile import mkstemp
from typing import BinaryIO
from uuid import uuid4

from app.core.config import settings
from app.storage.base import StorageError


class LocalFileStorage:
    def __init__(self, directory: Path | None = None) -> None:
        self.directory = (directory if directory is not None else settings.storage_dir).resolve()
        try:
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError:
            raise StorageError("Não foi possível preparar o armazenamento privado.") from None

    def _path(self, object_key: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}\.pdf", object_key):
            raise StorageError("Identificador de arquivo inválido.")
        return self.directory / object_key

    def store(self, source: Path) -> str:
        temporary = None
        published = None
        linked = False
        try:
            descriptor, name = mkstemp(prefix=".storage-", suffix=".tmp", dir=self.directory)
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output, length=64 * 1024)
                output.flush()
                os.fsync(output.fileno())
            object_key = f"{uuid4().hex}.pdf"
            published = self._path(object_key)
            os.link(temporary, published)
            linked = True
            temporary.unlink()
            temporary = None
            directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return object_key
        except OSError:
            if linked:
                try:
                    published.unlink(missing_ok=True)
                except OSError:
                    raise StorageError("Falha ao armazenar e limpar o PDF; necessária reconciliação.") from None
            raise StorageError("Falha ao armazenar o PDF; nenhum arquivo existente foi sobrescrito.") from None
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    raise StorageError("Falha na limpeza do arquivo temporário de armazenamento.") from None

    def open(self, object_key: str) -> BinaryIO:
        path = self._path(object_key)
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            return os.fdopen(descriptor, "rb")
        except OSError:
            raise StorageError("Não foi possível recuperar o PDF.") from None

    def delete(self, object_key: str) -> None:
        path = self._path(object_key)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            raise StorageError("Não foi possível excluir o PDF armazenado.") from None
