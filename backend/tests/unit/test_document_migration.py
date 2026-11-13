from pathlib import Path
from unittest.mock import Mock, patch

from alembic.config import Config
from alembic.script import ScriptDirectory
import pytest
from sqlalchemy import CheckConstraint, MetaData, Table, UniqueConstraint
from sqlalchemy.dialects import postgresql

from app.db.base import Base
import app.db.models  # noqa: F401


@pytest.fixture
def revision():
    backend = Path(__file__).resolve().parents[2]
    scripts = ScriptDirectory.from_config(Config(str(backend / "alembic.ini")))
    result = scripts.get_revision("20261109_0002")
    assert result.down_revision == "20261107_0001"
    assert result.dependencies is None
    return result.module


def table_signature(table):
    dialect = postgresql.dialect()
    return {
        "columns": [
            (column.name, column.type.compile(dialect=dialect), column.nullable,
             str(column.server_default.arg) if column.server_default else None)
            for column in table.columns
        ],
        "pk": tuple(table.primary_key.columns.keys()),
        "fk": {
            (column.name, fk.target_fullname, fk.ondelete)
            for column in table.columns for fk in column.foreign_keys
        },
        "unique": {
            (constraint.name, tuple(constraint.columns.keys()))
            for constraint in table.constraints if isinstance(constraint, UniqueConstraint)
        },
        "checks": {
            (constraint.name, str(constraint.sqltext))
            for constraint in table.constraints if isinstance(constraint, CheckConstraint)
        },
    }


def test_migration_matches_models_without_database(revision) -> None:
    operations = Mock()
    with patch.object(revision, "op", operations):
        revision.upgrade()

    assert [call[0] for call in operations.mock_calls] == [
        "create_table", "create_table", "create_index", "create_table", "create_table"
    ]
    calls = operations.create_table.call_args_list
    assert [call.args[0] for call in calls] == [
        "users", "documents", "document_chunks", "chunk_embeddings"
    ]
    migration_metadata = MetaData()
    for call in calls:
        name, *elements = call.args
        table = Table(name, migration_metadata, *elements)
        assert table_signature(table) == table_signature(Base.metadata.tables[name])

    index, = operations.create_index.call_args_list
    assert index.args == ("ix_documents_owner_id", "documents", ["owner_id"])


def test_migration_reverse_order_preserves_extension(revision) -> None:
    operations = Mock()
    with patch.object(revision, "op", operations):
        revision.downgrade()

    assert [call[0] for call in operations.mock_calls] == [
        "drop_table", "drop_table", "drop_index", "drop_table", "drop_table"
    ]
    assert [call.args[0] for call in operations.drop_table.call_args_list] == [
        "chunk_embeddings", "document_chunks", "documents", "users"
    ]
    operations.drop_index.assert_called_once_with("ix_documents_owner_id", table_name="documents")
