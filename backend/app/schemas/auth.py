from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from app.security.passwords import validate_password
from app.security.usernames import normalize_username


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    username: str = Field(min_length=3, max_length=128)
    password: SecretStr = Field(min_length=1, max_length=128)

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        return normalize_username(value)


class TokenResponse(BaseModel):
    access_token: str = Field(repr=False)
    token_type: str = "bearer"
    expires_in: int


class RegisterRequest(LoginRequest):
    password_confirmation: SecretStr = Field(min_length=1, max_length=128)

    @field_validator('password')
    @classmethod
    def password_policy(cls, value: SecretStr) -> SecretStr:
        validate_password(value.get_secret_value())
        return value

    @model_validator(mode='after')
    def matching_passwords(self):
        if self.password.get_secret_value() != self.password_confirmation.get_secret_value():
            raise ValueError('As senhas não coincidem.')
        return self


class UserResponse(BaseModel):
    id: UUID
    username: str | None
    status: str
