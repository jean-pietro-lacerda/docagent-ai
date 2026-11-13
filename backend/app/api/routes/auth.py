from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.routing import APIRoute
from sqlalchemy.orm import Session

from app.api.dependencies import get_auth_settings, get_current_user, unauthorized
from app.core.config import Settings, get_settings
from app.db.models import User
from app.db.session import get_db
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from app.security.registration import RegistrationLimiter, get_registration_limiter
from app.security.tokens import create_access_token
from app.services.auth import (
    AuthPersistenceError, InvalidCredentials, UsernameAlreadyExists,
    authenticate_local, create_local_user,
)


class AuthRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request: Request):
            content = bytearray()
            async for part in request.stream():
                content.extend(part)
                if len(content) > 4096:
                    raise HTTPException(413, 'Dados de autenticação excedem o limite.')
            request._body = bytes(content)
            response = await original(request)
            response.headers['Cache-Control'] = 'no-store'
            return response

        return handler


router = APIRouter(prefix="/api/v1/auth", tags=["auth"], route_class=AuthRoute)


@router.get('/config')
def auth_config(config: Annotated[Settings, Depends(get_settings)]):
    return {'registration_enabled': config.auth_enabled and config.registration_enabled}


def registration_allowed(config: Annotated[Settings, Depends(get_auth_settings)]):
    if not config.registration_enabled:
        raise HTTPException(403, 'Cadastro público desativado.')
    return config


def registration_attempt(
    request: Request,
    config: Annotated[Settings, Depends(registration_allowed)],
    limiter: Annotated[RegistrationLimiter, Depends(get_registration_limiter)],
):
    limiter.check(request.client.host if request.client else 'unknown')


@router.post('/register', response_model=UserResponse, status_code=201,
             dependencies=[Depends(registration_attempt)])
def register(body: RegisterRequest, session: Annotated[Session, Depends(get_db)]):
    try:
        user = create_local_user(session, body.username, body.password.get_secret_value())
    except UsernameAlreadyExists:
        raise HTTPException(409, 'Nome de usuário já está em uso.') from None
    except ValueError:
        raise HTTPException(422, 'Dados de cadastro inválidos.') from None
    except AuthPersistenceError:
        raise HTTPException(503, 'Não foi possível criar a conta.') from None

    return UserResponse(id=user.id, username=body.username, status=user.status)


@router.post("/login", response_model=TokenResponse)
def login(
    body: LoginRequest, response: Response,
    config: Annotated[Settings, Depends(get_auth_settings)],
    session: Annotated[Session, Depends(get_db)],
) -> TokenResponse:
    try:
        user = authenticate_local(session, body.username, body.password.get_secret_value())
    except InvalidCredentials:
        raise unauthorized() from None
    except AuthPersistenceError:
        raise HTTPException(status_code=503, detail="Autenticação temporariamente indisponível.") from None

    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"

    return TokenResponse(
        access_token=create_access_token(user.id, config),
        expires_in=config.jwt_access_token_minutes * 60,
    )


@router.get("/me", response_model=UserResponse)
def me(response: Response, user: Annotated[User, Depends(get_current_user)]) -> UserResponse:
    response.headers["Cache-Control"] = "no-store"

    return UserResponse(
        id=user.id, username=user.local_credential.username if user.local_credential else None,
        status=user.status,
    )
