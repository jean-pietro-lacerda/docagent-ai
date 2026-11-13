from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    app_name: str = "DocAgent AI"
    debug: bool = False
    auth_enabled: bool = False
    registration_enabled: bool = False
    jwt_secret_key: SecretStr | None = None
    jwt_issuer: str = Field(default="docagent-ai", min_length=1, max_length=255)
    jwt_audience: str | None = None
    jwt_access_token_minutes: int = Field(default=30, ge=1, le=60)
    storage_dir: Path = ROOT_DIR / "var/documents"
    embeddings_cache_dir: Path = Path.home() / '.cache/docagent-ai/models'
    groq_api_key: SecretStr | None = None
    groq_model: str | None = Field(default=None, min_length=1, max_length=255, pattern=r'^[a-zA-Z0-9/._-]+$')
    groq_timeout_seconds: int = Field(default=20, ge=1, le=60)
    groq_max_output_tokens: int = Field(default=768, ge=64, le=2048)
    rag_context_characters: int = Field(default=12000, ge=1000, le=20000)
    rag_min_score: float = Field(default=0.35, ge=0, le=1)
    database_url: str = (
        "postgresql+psycopg://docagent@localhost:5432/docagent_ai"
    )

    @field_validator("storage_dir")
    @classmethod
    def resolve_storage_dir(cls, value: Path) -> Path:
        return value if value.is_absolute() else ROOT_DIR / value

    @field_validator("jwt_issuer", "jwt_audience")
    @classmethod
    def validate_jwt_identifier(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("Emissor e audiência devem ser identificadores não vazios, sem espaços nas bordas.")
        return value

    @model_validator(mode="after")
    def validate_auth_configuration(self) -> "Settings":
        if self.auth_enabled:
            key = self.jwt_secret_key.get_secret_value() if self.jwt_secret_key else ""
            placeholders = ("replace", "change_me", "change-me", "changeme", "placeholder", "your-secret", "example")
            if (
                len(key.encode("utf-8")) < 32 or len(set(key)) < 12
                or any(char.isspace() for char in key)
                or any(word in key.lower() for word in placeholders)
            ):
                raise ValueError("JWT_SECRET_KEY deve conter um segredo aleatório forte de pelo menos 32 bytes.")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
