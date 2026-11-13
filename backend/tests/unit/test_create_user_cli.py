from unittest.mock import MagicMock

import pytest

from app.cli import create_user
from app.services.auth import UsernameAlreadyExists


@pytest.fixture
def interactive(monkeypatch):
    monkeypatch.setattr(create_user.sys, "argv", ["create_user"])
    monkeypatch.setattr(create_user.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "alice")
    getpass = MagicMock(side_effect=["Only a test password 123", "Only a test password 123"])
    monkeypatch.setattr(create_user, "getpass", getpass)
    session_factory = MagicMock()
    monkeypatch.setattr(create_user, "SessionLocal", session_factory)
    create = MagicMock()
    monkeypatch.setattr(create_user, "create_local_user", create)
    return getpass, session_factory, create


def test_cli_uses_hidden_prompts_and_one_session(interactive, capsys) -> None:
    getpass, factory, create = interactive
    assert create_user.main() == 0
    assert getpass.call_count == 2
    create.assert_called_once_with(factory.return_value.__enter__.return_value, "alice", "Only a test password 123")
    output = capsys.readouterr()
    assert "Only a test password" not in output.out + output.err


def test_cli_rejects_password_arguments(interactive, monkeypatch, capsys) -> None:
    getpass, factory, create = interactive
    monkeypatch.setattr(create_user.sys, "argv", ["create_user", "--password", "SENSITIVE"])
    assert create_user.main() == 2
    assert "SENSITIVE" not in capsys.readouterr().err
    getpass.assert_not_called()
    factory.assert_not_called()


def test_cli_requires_interactive_terminal(interactive, monkeypatch) -> None:
    getpass, factory, create = interactive
    monkeypatch.setattr(create_user.sys.stdin, "isatty", lambda: False)
    assert create_user.main() == 2
    getpass.assert_not_called()
    factory.assert_not_called()


def test_cli_mismatch_creates_nothing(interactive) -> None:
    getpass, factory, create = interactive
    getpass.side_effect = ["first", "second"]
    assert create_user.main() == 1
    factory.assert_not_called()
    create.assert_not_called()


def test_cli_duplicate_is_safe(interactive, capsys) -> None:
    getpass, factory, create = interactive
    create.side_effect = UsernameAlreadyExists("Username já está em uso.")
    assert create_user.main() == 1
    assert "Only a test password" not in capsys.readouterr().err


def test_cli_cancels_if_password_input_cannot_hide_echo(interactive) -> None:
    from getpass import GetPassWarning
    import warnings

    getpass, factory, create = interactive

    def unavailable_echo(prompt):
        warnings.warn("echo unavailable", GetPassWarning)

    getpass.side_effect = unavailable_echo
    assert create_user.main() == 1
    factory.assert_not_called()
    create.assert_not_called()
