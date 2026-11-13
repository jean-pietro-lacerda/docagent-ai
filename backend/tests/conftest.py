import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import secrets
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from app.core.config import Settings


@pytest.fixture
def db_session():
    session = MagicMock(spec=Session)
    session.in_transaction.return_value = False
    return session


@pytest.fixture
def auth_config():
    return Settings(
        _env_file=None, debug=False, auth_enabled=True,
        jwt_secret_key=secrets.token_urlsafe(48), jwt_audience="test-api",
    )


@pytest.fixture
def make_pdf():
    def create(pages=1, encrypted=False, user_password="secret"):
        import pymupdf

        with pymupdf.open() as document:
            for _ in range(pages):
                document.new_page()
            options = {}
            if encrypted:
                options = {
                    "encryption": pymupdf.PDF_ENCRYPT_AES_256,
                    "owner_pw": "owner-secret", "user_pw": user_password,
                }
            return document.tobytes(**options)
    return create
