import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from uuid import uuid4

import pymupdf
import pytest
from sqlalchemy import create_engine, delete, event, select, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from app.ai.embeddings.service import EmbeddingService
from app.db.models import ChunkEmbedding, Document, DocumentChunk, User
from app.schemas.embeddings import EmbeddingConfig
from app.schemas.retrieval import SearchRequest
from app.services.documents import DocumentNotFound, DocumentService
from app.services.indexing import IndexConflict, IndexingService, embedding_key
from app.services.search import SearchService
from app.storage.local import LocalFileStorage

pytestmark = pytest.mark.skipif(os.environ.get('RUN_VECTOR_INTEGRATION') != '1',
                                reason='Requer PostgreSQL temporário isolado, execução optativa.')


@pytest.fixture(scope='module')
def vector_database():
    binaries = Path('/usr/lib/postgresql/18/bin')
    if not (binaries / 'initdb').is_file():
        pytest.skip('Binários locais PostgreSQL 18 indisponíveis.')
    import psycopg

    with tempfile.TemporaryDirectory(prefix='docagent-pg-tests-') as directory:
        root = Path(directory)
        data = root / 'data'
        socket = root / 'socket'
        socket.mkdir(mode=0o700)
        subprocess.run([str(binaries / 'initdb'), '-D', str(data), '--auth-local=trust',
                        '--auth-host=reject', '--no-locale', '--encoding=UTF8'],
                       check=True, capture_output=True, timeout=30)
        options = shlex.join(['-F', '-k', str(socket), '-h', '', '-p', '55439'])
        try:
            subprocess.run([str(binaries / 'pg_ctl'), '-D', str(data), '-l', str(root / 'server.log'),
                            '-o', options, '-w', 'start'], check=True, capture_output=True, timeout=30)
            with psycopg.connect(host=str(socket), port=55439, dbname='postgres', autocommit=True) as connection:
                connection.execute('CREATE DATABASE docagent_ai_test')
            url = URL.create('postgresql+psycopg', database='docagent_ai_test',
                             query={'host': str(socket), 'port': '55439'})
            environment = {'PATH': os.environ.get('PATH', ''), 'HOME': str(Path.home()),
                           'AUTH_ENABLED': 'false', 'DEBUG': 'false', 'DATABASE_URL': url.render_as_string(hide_password=False),
                           'PYTHONDONTWRITEBYTECODE': '1'}
            subprocess.run([sys.executable, '-B', '-m', 'alembic', 'upgrade', 'head'],
                           env=environment, cwd=Path(__file__).resolve().parents[2],
                           capture_output=True, check=True, timeout=30)
            engine = create_engine(url, pool_pre_ping=True)
            try:
                with engine.connect() as connection:
                    assert connection.scalar(text('SELECT current_database()')) == 'docagent_ai_test'
                    assert connection.scalar(text("SELECT extversion FROM pg_extension WHERE extname='vector'"))
                yield engine
            finally:
                engine.dispose()
        finally:
            if (data / 'postmaster.pid').exists():
                subprocess.run([str(binaries / 'pg_ctl'), '-D', str(data), '-m', 'immediate', '-w', 'stop'],
                               check=True, capture_output=True, timeout=30)


class FakeEncoder:
    config = EmbeddingConfig()

    def __init__(self):
        self.on_encode = lambda: None

    def token_counts(self, texts):
        return [len(text.split()) + 2 for text in texts]

    def encode(self, texts):
        self.on_encode()
        return [[1.0 if i == (1 if 'banana' in text.lower() else 0) else 0.0 for i in range(384)]
                for text in texts]


@pytest.fixture
def pipeline(vector_database, tmp_path):
    sessions = sessionmaker(vector_database, expire_on_commit=False)
    storage = LocalFileStorage(tmp_path / 'files')
    backend = FakeEncoder()
    embeddings = EmbeddingService(backend=backend)
    with sessions.begin() as session:
        owner, other = User(id=uuid4(), status='active'), User(id=uuid4(), status='active')
        session.add_all([owner, other])
    def document(user=owner, content='Bolsa mensal de R$ 1.234,56. Prazo: 31/12/2026.'):
        with pymupdf.open() as pdf:
            for _ in range(2):
                pdf.new_page().insert_text((50, 50), content)
            data = pdf.tobytes()
        source = tmp_path / (uuid4().hex + '.pdf')
        source.write_bytes(data)
        item = Document(id=uuid4(), owner_id=user.id, filename='edital.pdf', object_key=storage.store(source),
                        file_hash=hashlib.sha256(data).hexdigest(), pages=2, status='pending')
        with sessions.begin() as session:
            session.add(item)
        return item
    try:
        yield SimpleNamespace(sessions=sessions, engine=vector_database, document=document, owner=owner,
                              other=other, backend=backend, embeddings=embeddings,
                              indexer=IndexingService(DocumentService(storage), embeddings), search=SearchService(embeddings))
    finally:
        with sessions.begin() as session:
            session.execute(delete(User))


def rows(p, document):
    with p.sessions() as session:
        return list(session.execute(select(DocumentChunk.id, DocumentChunk.text, ChunkEmbedding.model_id)
                    .join(ChunkEmbedding).where(DocumentChunk.document_id == document.id)).all())


def test_index_reindex_associations_and_no_transaction_during_inference(pipeline):
    p = pipeline
    document = p.document()
    with p.sessions() as session:
        p.backend.on_encode = lambda: pytest.fail('Transação aberta na inferência') if session.in_transaction() else None
        result = p.indexer.index(document.id, p.owner.id, session)
        assert result.chunks == 2 and result.status == 'ready'
        original = rows(p, document)
        p.indexer.index(document.id, p.owner.id, session)
    indexed = rows(p, document)
    assert len(indexed) == len(original) == 2 and set(r.id for r in indexed).isdisjoint(r.id for r in original)
    with p.sessions() as session:
        chunks = list(session.scalars(select(DocumentChunk).where(DocumentChunk.document_id == document.id).order_by(DocumentChunk.chunk_index)))
        assert [c.page_start for c in chunks] == [1, 2]
        assert [c.chunk_index for c in chunks] == [0, 1]
        assert all(c.hash == hashlib.sha256(c.text.encode()).hexdigest() for c in chunks)
        vectors = list(session.scalars(select(ChunkEmbedding)))
        assert all(len(v.embedding) == 384 and v.model_id == embedding_key(p.embeddings) for v in vectors)


def test_late_database_failure_rolls_back_previous_index(pipeline):
    p = pipeline
    document = p.document()
    with p.sessions() as session:
        p.indexer.index(document.id, p.owner.id, session)
        original = rows(p, document)
        def fail(connection, cursor, statement, parameters, context, executemany):
            if statement.startswith('INSERT INTO chunk_embeddings'):
                raise SQLAlchemyError('simulated failure')
        event.listen(p.engine, 'before_cursor_execute', fail)
        try:
            with pytest.raises(SQLAlchemyError):
                p.indexer.index(document.id, p.owner.id, session)
        finally:
            event.remove(p.engine, 'before_cursor_execute', fail)
        assert not session.in_transaction()
    assert rows(p, document) == original
    with p.sessions() as session:
        assert session.get(Document, document.id).status == 'ready'


def test_owner_integrity_inference_and_concurrent_changes(pipeline):
    p = pipeline
    document = p.document()
    with p.sessions() as session:
        with pytest.raises(DocumentNotFound):
            p.indexer.index(document.id, p.other.id, session)
        def concurrent_change():
            with p.sessions.begin() as other:
                other.get(Document, document.id).filename = 'changed.pdf'
        p.backend.on_encode = concurrent_change
        with pytest.raises(IndexConflict):
            p.indexer.index(document.id, p.owner.id, session)
    assert rows(p, document) == []
    p.backend.on_encode = lambda: (_ for _ in ()).throw(RuntimeError('inference failed'))
    with p.sessions() as session:
        with pytest.raises(RuntimeError):
            p.indexer.index(document.id, p.owner.id, session)
    assert rows(p, document) == []
    path = p.indexer.documents.storage.directory / document.object_key
    path.write_bytes(b'corrupted')
    from app.services.pdf_extraction import PDFExtractionError
    with p.sessions() as session:
        with pytest.raises(PDFExtractionError):
            p.indexer.index(document.id, p.owner.id, session)
    with p.sessions() as session:
        assert session.get(Document, document.id).status == 'pending'


def test_pgvector_cosine_ranking_owner_revision_and_document_filter(pipeline):
    p = pipeline
    one = p.document()
    two = p.document(content='Banana e outras frutas tropicais para uma alimentacao saudavel.')
    private = p.document(user=p.other)
    for document, owner in [(one, p.owner), (two, p.owner), (private, p.other)]:
        with p.sessions() as session:
            p.indexer.index(document.id, owner.id, session)
    request = SearchRequest(question='Qual o valor da bolsa?', top_k=4)
    with p.sessions() as session:
        hits = p.search.search(request, p.owner.id, session)
        assert len(hits) == 4 and hits[0].document_id == one.id and hits[0].score == pytest.approx(1)
        assert hits[-1].document_id == two.id and hits[-1].score == pytest.approx(0)
        assert p.search.search(request.model_copy(update={'document_id': private.id}), p.owner.id, session) == []
        selected = p.search.search(request.model_copy(update={'document_id': two.id, 'top_k': 1}), p.owner.id, session)
        assert len(selected) == 1 and selected[0].document_id == two.id
    with p.sessions.begin() as session:
        chunk_ids = select(DocumentChunk.id).where(DocumentChunk.document_id == one.id)
        for embedding in session.scalars(select(ChunkEmbedding).where(ChunkEmbedding.chunk_id.in_(chunk_ids))):
            embedding.model_id = 'incompatible@revision'
    with p.sessions() as session:
        assert all(h.document_id == two.id for h in p.search.search(request, p.owner.id, session))


def test_index_lock_conflict_preserves_existing_chunks(pipeline):
    p = pipeline
    document = p.document()
    with p.sessions() as session:
        p.indexer.index(document.id, p.owner.id, session)
    original = rows(p, document)
    with p.sessions.begin() as blocker:
        blocker.scalar(select(Document).where(Document.id == document.id).with_for_update())
        with p.sessions() as session:
            with pytest.raises(SQLAlchemyError) as error:
                p.indexer.index(document.id, p.owner.id, session)
            assert error.value.orig.sqlstate == '55P03'
    assert rows(p, document) == original


@pytest.mark.skipif(os.environ.get('RUN_EMBEDDING_INTEGRATION') != '1', reason='Encoder ONNX real optativo.')
def test_real_encoder_authenticated_index_search_and_mocked_groq(pipeline, auth_config):
    from fastapi.testclient import TestClient
    from app.ai.embeddings.backend import ModelUnavailable
    from app.api.dependencies import get_document_session
    from app.api.routes.retrieval import get_indexer, get_rag, get_search
    from app.core.config import get_settings
    from app.db.session import get_db
    from app.main import app
    from app.security.tokens import create_access_token
    from app.services.rag import RAGService
    import json

    p = pipeline
    config = EmbeddingConfig(cache_dir=Path(os.environ.get('EMBEDDINGS_CACHE_DIR', str(EmbeddingConfig().cache_dir))))
    embeddings = EmbeddingService(config)
    try:
        embeddings.backend.token_counts(['bolsa'])
    except ModelUnavailable:
        pytest.skip('Tokenizer real indisponível no cache; nenhum download solicitado.')
    indexer = IndexingService(p.indexer.documents, embeddings)
    search = SearchService(embeddings)
    class Generator:
        calls = 0
        def generate(self, question, sources):
            self.calls += 1
            return json.dumps({'evidence': [{'source_id': 1, 'quote': 'Bolsa mensal de R$ 1.234,56.'}]})
    generator = Generator()
    rag = RAGService(search, generator, auth_config)
    document = p.document()
    private = p.document(user=p.other)
    def sessions():
        with p.sessions() as session:
            yield session
    original = app.dependency_overrides.copy()
    app.dependency_overrides.update({get_settings: lambda: auth_config, get_db: sessions,
        get_document_session: sessions, get_indexer: lambda: indexer,
        get_search: lambda: search, get_rag: lambda: rag})
    headers = {'Authorization': 'Bearer ' + create_access_token(p.owner.id, auth_config)}
    try:
        with TestClient(app) as client:
            indexed = client.post(f'/api/v1/documents/{document.id}/index', headers=headers)
            assert indexed.status_code == 200, indexed.text
            assert indexed.json()['chunks'] == 2
            assert client.post(f'/api/v1/documents/{private.id}/index', headers=headers).status_code == 404
            body = {'question': 'Qual é o valor da bolsa?', 'document_id': str(document.id)}
            response = client.post('/api/v1/search', headers=headers, json=body)
            assert response.status_code == 200 and len(response.json()['items']) == 2
            assert all(item['document_id'] == str(document.id) for item in response.json()['items'])
            answered = client.post('/api/v1/ask', headers=headers, json=body)
            assert answered.status_code == 200 and answered.json()['citations']
            assert 'R$ 1.234,56.' in answered.json()['answer'] and generator.calls == 1
            assert answered.json()['citations'][0]['document_id'] == str(document.id)
            body['document_id'] = str(private.id)
            assert client.post('/api/v1/ask', headers=headers, json=body).json()['citations'] == []
            assert generator.calls == 1
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original)
