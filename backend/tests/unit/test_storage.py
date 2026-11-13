from uuid import UUID

import pytest

from app.core.config import ROOT_DIR, Settings
from app.storage.base import StorageError
from app.storage.local import LocalFileStorage


def test_storage_round_trip_permissions_and_delete(tmp_path) -> None:
    storage = LocalFileStorage(tmp_path / "private")
    source = tmp_path / "source"
    source.write_bytes(b"document bytes")
    key = storage.store(source)
    assert key.endswith(".pdf") and len(key) == 36
    with storage.open(key) as stream:
        assert stream.read() == b"document bytes"
    assert (storage.directory / key).stat().st_mode & 0o777 == 0o600
    assert list(storage.directory.iterdir()) == [storage.directory / key]
    storage.delete(key)
    storage.delete(key)
    assert list(storage.directory.iterdir()) == []


@pytest.mark.parametrize("key", ["../secret.pdf", "/etc/passwd", "..\\secret.pdf", "valid.pdf", "a" * 32 + ".pdf/../secret", ""])
def test_path_traversal_is_rejected(tmp_path, key) -> None:
    storage = LocalFileStorage(tmp_path / "private")
    for operation in (storage.open, storage.delete):
        with pytest.raises(StorageError, match="inválido"):
            operation(key)


def test_collision_does_not_overwrite(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.storage.local.uuid4", lambda: UUID(int=1))
    storage = LocalFileStorage(tmp_path / "private")
    source = tmp_path / "source"
    source.write_bytes(b"original")
    key = storage.store(source)
    source.write_bytes(b"replacement")
    with pytest.raises(StorageError, match="sobrescrito"):
        storage.store(source)
    with storage.open(key) as stream:
        assert stream.read() == b"original"
    assert list(storage.directory.iterdir()) == [storage.directory / key]


@pytest.mark.parametrize("failure", ["copy", "fsync", "link", "directory_fsync"])
def test_storage_failures_clean_temporary(tmp_path, monkeypatch, failure) -> None:
    storage = LocalFileStorage(tmp_path / "private")
    source = tmp_path / "source"
    source.write_bytes(b"data")

    def fail(*args, **kwargs):
        raise OSError("no space or sensitive path")

    if failure == "directory_fsync":
        import os
        import stat

        real_fsync = os.fsync

        def fail_directory(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError("directory sync failed after publication")
            real_fsync(fd)

        monkeypatch.setattr("app.storage.local.os.fsync", fail_directory)
    else:
        target = {"copy": "shutil.copyfileobj", "fsync": "os.fsync", "link": "os.link"}[failure]
        monkeypatch.setattr("app.storage.local." + target, fail)
    with pytest.raises(StorageError) as error:
        storage.store(source)
    assert "sensitive" not in str(error.value)
    assert list(storage.directory.iterdir()) == []


def test_storage_rejects_symlink_on_read(tmp_path) -> None:
    storage = LocalFileStorage(tmp_path / "private")
    secret = tmp_path / "secret"
    secret.write_bytes(b"private")
    key = "a" * 32 + ".pdf"
    (storage.directory / key).symlink_to(secret)
    with pytest.raises(StorageError):
        storage.open(key)
    storage.delete(key)
    assert secret.read_bytes() == b"private"


def test_storage_directory_setting(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("STORAGE_DIR", "var/custom")
    assert Settings(_env_file=None).storage_dir == ROOT_DIR / "var/custom"
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
    assert Settings(_env_file=None).storage_dir == tmp_path
