import secrets
from functools import lru_cache

from argon2 import PasswordHasher, Type
from argon2.exceptions import InvalidHashError, VerificationError


def validate_password(password: str) -> None:
    try:
        byte_length = len(password.encode("utf-8"))
    except UnicodeEncodeError:
        raise ValueError("A senha deve conter texto Unicode válido.") from None
    if (
        not 12 <= len(password) <= 128 or byte_length > 512
        or password.isspace() or any(ord(char) < 32 or ord(char) == 127 for char in password)
    ):
        raise ValueError("A senha deve ter de 12 a 128 caracteres, sem caracteres de controle, e não pode conter apenas espaços.")


class PasswordService:
    def __init__(self) -> None:
        self.hasher = PasswordHasher(type=Type.ID)
        self.dummy_hash = self.hasher.hash(secrets.token_urlsafe(32))

    def hash(self, password: str) -> str:
        validate_password(password)
        return self.hasher.hash(password)

    def verify(self, password: str, password_hash: str | None) -> bool:
        try:
            byte_length = len(password.encode("utf-8"))
        except UnicodeEncodeError:
            return False
        if not 1 <= len(password) <= 128 or byte_length > 512:
            return False
        candidate = password_hash if password_hash and password_hash.startswith("$argon2id$") else self.dummy_hash
        try:
            verified = self.hasher.verify(candidate, password)
            return verified and candidate != self.dummy_hash
        except (VerificationError, InvalidHashError):
            return False


@lru_cache
def get_password_service() -> PasswordService:
    return PasswordService()
