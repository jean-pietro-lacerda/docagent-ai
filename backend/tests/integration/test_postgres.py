import json
import os
import re

import pytest
from alembic.config import Config
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import CheckConstraint, UniqueConstraint, MetaData, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import ROOT_DIR
from app.db.base import Base
import app.db.models  # noqa: F401
from app.db.session import engine, get_db


@pytest.mark.skipif(
    os.environ.get("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Requer execução explícita contra o PostgreSQL local de desenvolvimento.",
)
def test_local_postgres_read_only() -> None:
    if (
        engine.url.host != "localhost"
        or engine.url.port != 5432
        or engine.url.database != "docagent_ai"
        or engine.url.username != "docagent"
    ):
        pytest.fail("Destino diferente do PostgreSQL local autorizado.", pytrace=False)

    scripts = ScriptDirectory.from_config(Config(str(ROOT_DIR / "backend/alembic.ini")))
    report = {"local_heads": scripts.get_heads()}
    dependency = get_db()
    session = next(dependency)
    try:
        session.connection(execution_options={"postgresql_readonly": True})
        assert session.execute(text("SHOW transaction_read_only")).scalar_one() == "on"
        info = session.execute(
            text(
                "SELECT current_database() AS database, current_user AS username, "
                "current_setting('server_version') AS version"
            )
        ).mappings().one()
        report["postgres"] = dict(info)
        assert info["database"] == "docagent_ai"
        assert info["username"] == "docagent"

        extension = session.execute(
            text(
                "SELECT name, default_version, installed_version "
                "FROM pg_available_extensions WHERE name = 'vector'"
            )
        ).mappings().all()
        report["pgvector"] = [dict(row) for row in extension]

        tables = session.execute(
            text(
                "SELECT n.nspname AS schema, c.relname AS name "
                "FROM pg_catalog.pg_class c "
                "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
                "WHERE c.relname = 'alembic_version' AND c.relkind IN ('r', 'p')"
            )
        ).mappings().all()
        report["alembic"] = []
        quote = engine.dialect.identifier_preparer.quote_identifier
        for table in tables:
            name = quote(table["schema"]) + "." + quote(table["name"])
            revisions = session.execute(
                text("SELECT version_num FROM " + name)
            ).scalars().all()
            report["alembic"].append({"schema": table["schema"], "revisions": revisions})

        print(json.dumps(report, ensure_ascii=False))
        assert extension and extension[0]["installed_version"], "pgvector não ativado."
    except SQLAlchemyError as exc:
        original = getattr(exc, "orig", None)
        message = str(original).lower()
        categories = (
            "password authentication failed",
            "no password supplied",
            "connection refused",
            "permission denied",
            "timeout expired",
            "does not exist",
        )
        category = next((item for item in categories if item in message), "indeterminada")
        pytest.fail(
            f"PostgreSQL: {type(exc).__name__}; categoria: {category}; "
            f"SQLSTATE: {getattr(original, 'sqlstate', None)}. Detalhes omitidos.",
            pytrace=False,
        )
    finally:
        dependency.close()
        engine.dispose()


@pytest.mark.skipif(
    os.environ.get("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Requer execução explícita contra o PostgreSQL local de desenvolvimento.",
)
def test_local_orm_schema_read_only() -> None:
    assert (engine.url.host, engine.url.port, engine.url.database, engine.url.username) == (
        "localhost", 5432, "docagent_ai", "docagent"
    )
    try:
        with engine.connect().execution_options(postgresql_readonly=True) as connection:
            assert connection.execute(text("SHOW transaction_read_only")).scalar_one() == "on"
            assert connection.execute(text("SELECT current_database()")).scalar_one() == "docagent_ai"
            assert connection.execute(text("SELECT current_schema()")).scalar_one() == "public"
            revisions = connection.execute(
                text("SELECT version_num FROM public.alembic_version")
            ).scalars().all()
            scripts = ScriptDirectory.from_config(Config(str(ROOT_DIR / "backend/alembic.ini")))
            assert scripts.get_heads() == ["20261112_0004"]
            assert revisions in (["20261109_0003"], ["20261112_0004"])
            metadata = Base.metadata
            if revisions == ["20261109_0003"]:
                print({"pending_manual_revision": "20261112_0004", "conversations_validated_on_real_database": False})
                metadata = MetaData()
                for table in Base.metadata.sorted_tables:
                    if table.name != 'conversations':
                        table.to_metadata(metadata)

            context = MigrationContext.configure(
                connection, opts={"compare_type": True, "compare_server_default": True}
            )
            assert compare_metadata(context, metadata) == []
            inspector = inspect(connection)
            assert set(metadata.tables) <= set(inspector.get_table_names(schema="public"))

            for table in metadata.sorted_tables:
                assert_table_matches_model(connection, table)

            vector_type = connection.execute(
                text(
                    "SELECT format_type(a.atttypid, a.atttypmod) "
                    "FROM pg_attribute a WHERE a.attrelid = 'public.chunk_embeddings'::regclass "
                    "AND a.attname = 'embedding' AND NOT a.attisdropped"
                )
            ).scalar_one()
            assert vector_type == "vector(384)"
            assert connection.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ).scalar_one() == "0.8.1"
    except SQLAlchemyError as exc:
        pytest.fail(
            f"Verificação do esquema: {type(exc).__name__}. Detalhes omitidos para proteger credenciais.",
            pytrace=False,
        )
    finally:
        engine.dispose()


def assert_table_matches_model(connection, table) -> None:
    inspector = inspect(connection)
    columns = {column["name"]: column for column in inspector.get_columns(table.name, schema="public")}
    assert set(columns) == set(table.columns.keys())
    for column in table.columns:
        actual = columns[column.name]
        assert actual["nullable"] == column.nullable
        assert actual["type"].compile(dialect=connection.dialect) == column.type.compile(dialect=connection.dialect)
        if getattr(column.type, "timezone", False):
            assert actual["type"].timezone

    assert inspector.get_pk_constraint(table.name, schema="public")["constrained_columns"] == list(table.primary_key.columns.keys())
    actual_foreign_keys = {
        (tuple(fk["constrained_columns"]), fk["referred_table"],
         tuple(fk["referred_columns"]), fk["options"].get("ondelete"))
        for fk in inspector.get_foreign_keys(table.name, schema="public")
    }
    expected_foreign_keys = {
        ((column.name,), fk.column.table.name, (fk.column.name,), fk.ondelete)
        for column in table.columns for fk in column.foreign_keys
    }
    assert actual_foreign_keys == expected_foreign_keys
    assert {
        tuple(constraint["column_names"])
        for constraint in inspector.get_unique_constraints(table.name, schema="public")
    } == {
        tuple(constraint.columns.keys()) for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    assert {
        (index["name"], tuple(index["column_names"]), index["unique"])
        for index in inspector.get_indexes(table.name, schema="public")
        if not index.get("duplicates_constraint")
    } == {
        (index.name, tuple(index.columns.keys()), index.unique)
        for index in table.indexes
    }
    checks = {
        check["name"]: check["sqltext"]
        for check in inspector.get_check_constraints(table.name, schema="public")
    }
    expected_checks = {
        constraint.name: str(constraint.sqltext) for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert checks.keys() == expected_checks.keys()
    for name, expression in expected_checks.items():
        if name == "ck_local_credentials_username":
            assert "username" in checks[name]
            assert "~" in checks[name]
            assert "^[a-z0-9][a-z0-9_.-]{2,63}$" in checks[name]
        elif name.endswith("_status"):
            assert re.findall(r"'([^']*)'", checks[name]) == re.findall(r"'([^']*)'", expression)
            assert "= ANY" in checks[name]
            assert "status" in checks[name]
            assert "NOT" not in checks[name]
        else:
            assert re.sub(r"[\s()]", "", checks[name].replace('::text', '')) == re.sub(r"[\s()]", "", expression)
