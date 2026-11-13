from types import SimpleNamespace

import pytest
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.security.passwords import get_password_service
from app.services.auth import AuthPersistenceError, UsernameAlreadyExists, create_local_user


def test_local_account_creation_is_atomic_and_normalized(db_session) -> None:
    user = create_local_user(db_session, " Alice ", "Only a test password 123")
    credential = user.local_credential
    assert user.id is not None and user.status == "active"
    assert user.auth_provider_id is None
    assert credential.username == "alice"
    assert credential.user is user
    assert get_password_service().verify("Only a test password 123", credential.password_hash)
    db_session.add.assert_called_once_with(user)
    db_session.flush.assert_called_once()
    db_session.begin.return_value.__exit__.assert_called_once_with(None, None, None)
    db_session.expunge.assert_called_once_with(user)


def test_duplicate_username_is_sanitized_and_transaction_rolls_back(db_session) -> None:
    original = Exception("sensitive DB details")
    original.diag = SimpleNamespace(constraint_name="uq_local_credentials_username")
    db_session.flush.side_effect = IntegrityError("secret SQL", {}, original)
    with pytest.raises(UsernameAlreadyExists) as error:
        create_local_user(db_session, "ALICE", "Only a test password 123")
    assert "secret" not in str(error.value)
    assert db_session.begin.return_value.__exit__.call_args.args[0] is IntegrityError


def test_create_failure_is_generic(db_session) -> None:
    db_session.flush.side_effect = SQLAlchemyError("sensitive DB details")
    with pytest.raises(AuthPersistenceError) as error:
        create_local_user(db_session, "alice", "Only a test password 123")
    assert "sensitive" not in str(error.value)


