from collections.abc import Generator
from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, joinedload

from app.core.config import Settings, get_settings
from app.db.models import User
from app.db.session import get_db
from app.security.tokens import InvalidAccessToken, decode_access_token

bearer = HTTPBearer(auto_error=False)


def get_auth_settings(config: Annotated[Settings, Depends(get_settings)]) -> Settings:
    if not config.auth_enabled:
        raise HTTPException(status_code=503, detail="Autenticação indisponível.")

    return config


def unauthorized() -> HTTPException:
    return HTTPException(
        status_code=401, detail="Credenciais inválidas.", headers={"WWW-Authenticate": "Bearer"}
    )


def get_document_session() -> Generator[Session, None, None]:
    """Keep document transactions separate from authentication's read transaction."""
    yield from get_db()


def get_current_user(
    config: Annotated[Settings, Depends(get_auth_settings)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    session: Annotated[Session, Depends(get_db)],
) -> User:
    if credentials is None:
        raise unauthorized()

    try:
        user_id = decode_access_token(credentials.credentials, config)
    except InvalidAccessToken:
        raise unauthorized() from None

    try:
        user = session.scalar(
            select(User).options(joinedload(User.local_credential)).where(User.id == user_id)
        )
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="Autenticação temporariamente indisponível.") from None

    if user is None or user.status != "active":
        raise unauthorized()

    return user
