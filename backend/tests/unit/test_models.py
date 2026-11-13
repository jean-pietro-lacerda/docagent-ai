from uuid import UUID

import pytest
from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, inspect
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import configure_mappers

from app.db.base import Base
from app.db.models import ChunkEmbedding, Document, DocumentChunk, User


def test_models_registered_and_relationships_configured() -> None:
    configure_mappers()
    assert set(Base.metadata.tables) == {
        "users", "documents", "document_chunks", "chunk_embeddings", "local_credentials", "conversations"
    }
    for model in (User, Document, DocumentChunk, ChunkEmbedding):
        assert inspect(model).local_table is Base.metadata.tables[model.__tablename__]


def test_embedding_identity_and_dimension() -> None:
    table = ChunkEmbedding.__table__
    assert list(table.primary_key.columns.keys()) == ["chunk_id", "model_id"]
    assert isinstance(table.c.embedding.type, Vector)
    assert table.c.embedding.type.dim == 384
    assert table.c.embedding.type.compile(dialect=postgresql.dialect()) == "VECTOR(384)"
    assert not table.c.embedding.nullable


@pytest.mark.parametrize(
    ("model", "column", "target"),
    [
        (Document, "owner_id", "users.id"),
        (DocumentChunk, "document_id", "documents.id"),
        (ChunkEmbedding, "chunk_id", "document_chunks.id"),
    ],
)
def test_required_foreign_keys_cascade(model, column, target) -> None:
    field = model.__table__.c[column]
    foreign_key, = field.foreign_keys
    assert foreign_key.target_fullname == target
    assert foreign_key.ondelete == "CASCADE"
    assert not field.nullable


def test_relationships_link_both_directions() -> None:
    user = User()
    document = Document(filename="example.pdf", owner=user)
    chunk = DocumentChunk(
        document=document, chunk_index=0, page_start=1, page_end=2,
        text="Example", hash="a" * 64,
    )
    embedding = ChunkEmbedding(chunk=chunk, model_id="candidate", embedding=[0.0] * 384)
    assert user.documents == [document]
    assert document.chunks == [chunk]
    assert chunk.embeddings == [embedding]
    for model, name in [(User, "documents"), (Document, "chunks"), (DocumentChunk, "embeddings")]:
        relation = inspect(model).relationships[name]
        assert relation.passive_deletes
        assert "delete-orphan" in relation.cascade
        assert relation.back_populates


def test_status_and_page_constraints() -> None:
    def checks(model):
        return {str(c.sqltext) for c in model.__table__.constraints if isinstance(c, CheckConstraint)}

    assert checks(User) == {"status IN ('active', 'inactive')"}
    assert checks(Document) == {
        "status IN ('pending', 'extracting', 'embedding', 'ready', 'failed')", "pages > 0"
    }
    assert checks(DocumentChunk) == {
        "chunk_index >= 0", "page_start >= 1", "page_end >= page_start"
    }
    assert User.__table__.c.status.server_default.arg == "active"
    assert Document.__table__.c.status.server_default.arg == "pending"


def test_document_updated_at_uses_sqlalchemy_onupdate() -> None:
    assert str(Document.__table__.c.updated_at.onupdate.arg) == "now()"


def test_uuid_defaults_are_generated_by_application() -> None:
    for model in (User, Document, DocumentChunk):
        assert isinstance(model.__table__.c.id.default.arg(None), UUID)
