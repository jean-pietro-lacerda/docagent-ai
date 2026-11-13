from pathlib import Path
from unittest.mock import Mock, patch

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import CheckConstraint, MetaData, Table, UniqueConstraint, inspect
from sqlalchemy.dialects import postgresql

from app.db.models import LocalCredential, User


def test_local_credential_structure_and_relationship() -> None:
    table = LocalCredential.__table__
    assert table.name == "local_credentials"
    assert list(table.primary_key.columns.keys()) == ["user_id"]
    assert table.c.user_id.type.compile(dialect=postgresql.dialect()) == "UUID"
    foreign_key, = table.c.user_id.foreign_keys
    assert foreign_key.target_fullname == "users.id" and foreign_key.ondelete == "CASCADE"
    assert table.c.created_at.type.timezone
    assert all(not column.nullable for column in table.columns)
    assert {tuple(c.columns.keys()) for c in table.constraints if isinstance(c, UniqueConstraint)} == {("username",)}
    assert {str(c.sqltext) for c in table.constraints if isinstance(c, CheckConstraint)} == {"username ~ '^[a-z0-9][a-z0-9_.-]{2,63}$'"}
    assert not inspect(User).relationships.local_credential.uselist
    assert inspect(User).relationships.local_credential.passive_deletes
    assert "delete-orphan" in inspect(User).relationships.local_credential.cascade
    assert LocalCredential(username=" Alice ").username == "alice"


def test_credential_migration_only_creates_one_table() -> None:
    backend = Path(__file__).resolve().parents[2]
    scripts = ScriptDirectory.from_config(Config(str(backend / "alembic.ini")))
    assert scripts.get_heads() == ["20261112_0004"]
    revision = scripts.get_revision("20261109_0003")
    assert revision.down_revision == "20261109_0002" and revision.dependencies is None
    operations = Mock()
    with patch.object(revision.module, "op", operations):
        revision.module.upgrade()
    assert [call[0] for call in operations.mock_calls] == ["create_table"]
    name, *elements = operations.create_table.call_args.args
    assert name == "local_credentials"
    table = Table(name, MetaData(), *elements)
    dialect = postgresql.dialect()
    assert [(c.name, c.type.compile(dialect=dialect), c.nullable) for c in table.columns] == [
        (c.name, c.type.compile(dialect=dialect), c.nullable) for c in LocalCredential.__table__.columns
    ]
    assert list(table.primary_key.columns.keys()) == ["user_id"]
    foreign_key, = table.c.user_id.foreign_keys
    assert foreign_key.target_fullname == "users.id"
    assert foreign_key.ondelete == "CASCADE"
    assert {str(c.sqltext) for c in table.constraints if isinstance(c, CheckConstraint)} == {
        str(c.sqltext) for c in LocalCredential.__table__.constraints if isinstance(c, CheckConstraint)
    }
    assert str(table.c.created_at.server_default.arg) == "now()"
    assert any(isinstance(c, UniqueConstraint) and list(c.columns.keys()) == ["username"] for c in table.constraints)
    reverse = Mock()
    with patch.object(revision.module, "op", reverse):
        revision.module.downgrade()
    assert [call[0] for call in reverse.mock_calls] == ["drop_table"]
    reverse.drop_table.assert_called_once_with("local_credentials")
