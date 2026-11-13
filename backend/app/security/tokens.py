from datetime import UTC, datetime, timedelta
from uuid import UUID

import jwt

from app.core.config import Settings


class InvalidAccessToken(ValueError):
    pass


def _key(config: Settings) -> str:
    if not config.auth_enabled or config.jwt_secret_key is None:
        raise RuntimeError("Autenticação indisponível.")
    return config.jwt_secret_key.get_secret_value()


def create_access_token(user_id: UUID, config: Settings) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id), "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=config.jwt_access_token_minutes)).timestamp()),
        "iss": config.jwt_issuer,
    }
    if config.jwt_audience is not None:
        payload["aud"] = config.jwt_audience
    return jwt.encode(payload, _key(config), algorithm="HS256")


def decode_access_token(token: str, config: Settings) -> UUID:
    key = _key(config)
    if len(token) > 4096:
        raise InvalidAccessToken("Token inválido.")
    required = ["sub", "iat", "exp", "iss"]
    if config.jwt_audience is not None:
        required.append("aud")
    try:
        payload = jwt.decode(
            token, key, algorithms=["HS256"], issuer=config.jwt_issuer,
            audience=config.jwt_audience, options={"require": required, "strict_aud": True},
        )
        if type(payload["iat"]) is not int or type(payload["exp"]) is not int:
            raise ValueError
        if payload["exp"] <= payload["iat"]:
            raise ValueError
        identity = UUID(payload["sub"])
        if str(identity) != payload["sub"]:
            raise ValueError
        return identity
    except (jwt.InvalidTokenError, ValueError, TypeError, AttributeError):
        raise InvalidAccessToken("Token inválido ou expirado.") from None
