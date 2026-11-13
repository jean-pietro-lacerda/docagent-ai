from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy.engine import make_url

from app.core.config import ROOT_DIR, Settings


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("false", False)],
)
def test_debug_boolean_environment(monkeypatch, value, expected) -> None:
    monkeypatch.setenv("DEBUG", value)

    assert Settings(_env_file=None).debug is expected


def test_invalid_debug_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("DEBUG", "release")

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    assert any(error["loc"] == ("debug",) for error in exc_info.value.errors())


def test_default_database_has_no_password(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DEBUG", raising=False)

    config = Settings(_env_file=None)
    url = make_url(config.database_url)

    assert url.drivername == "postgresql+psycopg"
    assert url.host == "localhost"
    assert url.port == 5432
    assert url.database == "docagent_ai"
    assert url.username == "docagent"
    assert url.password is None
    assert config.debug is False


def test_dotenv_loading_and_environment_precedence(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("DEBUG", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    env_file = tmp_path / ".env"
    url = "postgresql+psycopg://docagent@localhost:5432/test_dotenv"
    env_file.write_text(f"DEBUG=true\nDATABASE_URL={url}\n", encoding="utf-8")

    config = Settings(_env_file=env_file)
    assert config.debug is True
    assert config.database_url == url

    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://docagent@localhost:5432/env_override")
    override = Settings(_env_file=env_file)
    assert override.debug is False
    assert override.database_url.endswith("/env_override")


def test_dotenv_path_is_project_root() -> None:
    assert ROOT_DIR == Path(__file__).resolve().parents[3]
    assert Settings.model_config["env_file"] == ROOT_DIR / ".env"
