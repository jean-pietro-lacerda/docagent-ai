from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, joinedload

from app.db.models import LocalCredential, User
from app.security.passwords import get_password_service
from app.security.usernames import normalize_username


class InvalidCredentials(ValueError):
    pass


class AuthPersistenceError(RuntimeError):
    pass


class UsernameAlreadyExists(ValueError):
    pass


def authenticate_local(session: Session, username: str, password: str) -> User:
    username = normalize_username(username)
    try:
        credential = session.scalar(
            select(LocalCredential).options(joinedload(LocalCredential.user))
            .where(LocalCredential.username == username)
        )
    except SQLAlchemyError:
        raise AuthPersistenceError("Autenticação temporariamente indisponível.") from None
    valid = get_password_service().verify(password, credential.password_hash if credential else None)
    if not credential or not valid or credential.user.status != "active":
        raise InvalidCredentials("Credenciais inválidas.")
    return credential.user


def create_local_user(session: Session, username: str, password: str) -> User:
    username = normalize_username(username)
    password_hash = get_password_service().hash(password)
    if session.in_transaction():
        raise AuthPersistenceError("A criação requer uma sessão sem transação ativa.")
    user = User(id=uuid4(), status="active")
    user.local_credential = LocalCredential(username=username, password_hash=password_hash)
    try:
        with session.begin():
            session.add(user)
            session.flush()
            session.expunge(user)
    except IntegrityError as exc:
        if getattr(getattr(exc.orig, "diag", None), "constraint_name", None) == "uq_local_credentials_username":
            raise UsernameAlreadyExists("Username já está em uso.") from None
        raise AuthPersistenceError("Não foi possível criar a conta local.") from None
    except SQLAlchemyError:
        raise AuthPersistenceError("Não foi possível criar a conta local.") from None
    return user
