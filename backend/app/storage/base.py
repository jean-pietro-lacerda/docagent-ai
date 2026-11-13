from pathlib import Path
from typing import BinaryIO, Protocol


class StorageError(OSError):
    pass


class FileStorage(Protocol):
    def store(self, source: Path) -> str: ...
    def open(self, object_key: str) -> BinaryIO: ...
    def delete(self, object_key: str) -> None: ...
