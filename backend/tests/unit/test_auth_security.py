from datetime import UTC, datetime
from uuid import uuid4

import jwt
from pydantic import ValidationError
import pytest

from app.core.config import Settings
from app.security.passwords import get_password_service, validate_password
from app.security.tokens import InvalidAccessToken, create_access_token, decode_access_token
from app.security.usernames import normalize_username


def test_argon2id_hashes_are_salted_and_verified() -> None:
    service = get_password_service()
    password = "Only a test passphrase 123"
    first, second = service.hash(password), service.hash(password)
    assert first.startswith("$argon2id$")
    assert first != second
    assert password not in first
    assert service.verify(password, first)
    assert not service.verify("wrong password", first)
    assert not service.verify(password, None)
    assert not service.verify(password, "invalid hash")
    assert not service.verify("x" * 129, first)


@pytest.mark.parametrize("password", ["", "short", " " * 12, "x" * 129, "long\npassword12", "long\x00password12", "unpaired\ud800password12"])
def test_password_policy(password) -> None:
    with pytest.raises(ValueError) as error:
        validate_password(password)
    assert "long\x00password12" not in str(error.value)


def test_password_verification_rejects_invalid_unicode() -> None:
    assert not get_password_service().verify("unpaired\ud800password12", None)


@pytest.mark.parametrize(("username", "expected"), [(" Alice ", "alice"), ("ALICE.Name-1", "alice.name-1"), ("user_1", "user_1")])
def test_username_normalization(username, expected) -> None:
    assert normalize_username(username) == expected


@pytest.mark.parametrize("username", ["ab", "a" * 65, "name space", "../user", "álice", "@user"])
def test_invalid_username(username) -> None:
    with pytest.raises(ValueError):
        normalize_username(username)


@pytest.mark.parametrize("key", [None, "", "short", "x" * 64, "REPLACE_WITH_A_RANDOM_SECRET_OF_AT_LEAST_32_BYTES", "change_me_0123456789_abcdefghijklmnopqrstuvwxyz", "a secure but whitespace-containing key123456789"])
def test_auth_rejects_bad_secret_without_leaking(key) -> None:
    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None, debug=False, auth_enabled=True, jwt_secret_key=key)
    if key:
        assert key not in str(error.value)


def test_auth_can_be_disabled_without_secret() -> None:
    config = Settings(_env_file=None, debug=False, auth_enabled=False, jwt_secret_key=None)
    assert config.auth_enabled is False
    with pytest.raises(RuntimeError):
        create_access_token(uuid4(), config)


def test_jwt_round_trip_required_claims(auth_config) -> None:
    user_id = uuid4()
    token = create_access_token(user_id, auth_config)
    assert decode_access_token(token, auth_config) == user_id
    payload = jwt.decode(
        token, auth_config.jwt_secret_key.get_secret_value(), algorithms=["HS256"],
        issuer=auth_config.jwt_issuer, audience=auth_config.jwt_audience,
    )
    assert payload["exp"] - payload["iat"] == 1800
    assert payload["sub"] == str(user_id)
    assert payload["iss"] == auth_config.jwt_issuer
    assert payload["aud"] == "test-api"


@pytest.mark.parametrize("missing", ["sub", "iat", "exp", "iss", "aud"])
def test_jwt_rejects_missing_claims(auth_config, missing) -> None:
    now = int(datetime.now(UTC).timestamp())
    payload = {"sub": str(uuid4()), "iat": now, "exp": now + 60, "iss": auth_config.jwt_issuer, "aud": auth_config.jwt_audience}
    del payload[missing]
    token = jwt.encode(payload, auth_config.jwt_secret_key.get_secret_value(), algorithm="HS256")
    with pytest.raises(InvalidAccessToken):
        decode_access_token(token, auth_config)


@pytest.mark.parametrize("case", ["expired", "issuer", "audience", "subject", "future", "float_iat", "float_exp", "reverse_time", "wrong_algorithm", "unsigned", "tampered", "oversized"])
def test_jwt_rejects_invalid_tokens(auth_config, case) -> None:
    now = int(datetime.now(UTC).timestamp())
    payload = {"sub": str(uuid4()), "iat": now, "exp": now + 60, "iss": auth_config.jwt_issuer, "aud": auth_config.jwt_audience}
    if case == "expired":
        payload.update(iat=now - 120, exp=now - 60)
    elif case == "issuer":
        payload["iss"] = "other-issuer"
    elif case == "audience":
        payload["aud"] = "other-api"
    elif case == "subject":
        payload["sub"] = "not-a-uuid"
    elif case == "future":
        payload["iat"] = now + 3600
    elif case == "float_iat":
        payload["iat"] = float(now)
    elif case == "float_exp":
        payload["exp"] = float(now + 60)
    elif case == "reverse_time":
        payload["exp"] = now
    key = auth_config.jwt_secret_key.get_secret_value()
    algorithm = "HS384" if case == "wrong_algorithm" else "HS256"
    if case == "unsigned":
        algorithm, key = "none", None
    token = jwt.encode(payload, key, algorithm=algorithm)
    if case == "tampered":
        parts = token.split(".")
        parts[1] = parts[1][:-2] + "AA"
        token = ".".join(parts)
    elif case == "oversized":
        token = "a" * 4097
    with pytest.raises(InvalidAccessToken):
        decode_access_token(token, auth_config)


def test_optional_audience(auth_config) -> None:
    without_audience = auth_config.model_copy(update={"jwt_audience": None})
    user_id = uuid4()
    assert decode_access_token(create_access_token(user_id, without_audience), without_audience) == user_id
    with pytest.raises(InvalidAccessToken):
        decode_access_token(create_access_token(user_id, auth_config), without_audience)


@pytest.mark.parametrize("options", [{"jwt_access_token_minutes": 0}, {"jwt_access_token_minutes": 61}, {"jwt_issuer": " "}, {"jwt_audience": ""}])
def test_invalid_token_settings(options) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, debug=False, auth_enabled=False, **options)
