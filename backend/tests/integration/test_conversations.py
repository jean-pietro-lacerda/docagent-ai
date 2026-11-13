import json
import os
from uuid import uuid4
from fastapi.testclient import TestClient
import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from tests.integration.test_vector_pipeline import pipeline, vector_database
from app.api.dependencies import get_document_session
from app.api.routes.retrieval import get_rag
from app.core.config import get_settings
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.security.tokens import create_access_token
from app.services.rag import RAGService

pytestmark = pytest.mark.skipif(os.environ.get('RUN_VECTOR_INTEGRATION') != '1', reason='PostgreSQL temporário isolado.')


def test_persistent_conversation_complete_lists_followup_modes_and_isolation(pipeline, auth_config):
    p = pipeline
    text = '1 - Sistemas Embarcados e Eletronica.\n2 - Engenharia Mecanica e Controle e Automacao.\n3 - Inteligencia Artificial.\nRequisitos: matricula ativa e disponibilidade de 20 horas.'
    document, private = p.document(content=text), p.document(user=p.other)
    with p.sessions() as session:
        p.indexer.index(document.id, p.owner.id, session)
    with p.engine.connect() as connection:
        assert compare_metadata(MigrationContext.configure(connection), Base.metadata) == []
    class Generator:
        questions = []
        def generate(self, question, sources):
            self.questions.append(question)
            current = json.loads(question)['question'] if question.startswith('{') else question
            if 'garantia' in current:
                return '{"evidence":[]}'
            item = {'source_id': sources[0]['source_id']}
            if 'requisitos' in current:
                item['quote'] = 'Requisitos: matricula ativa e disponibilidade de 20 horas.'
            return json.dumps({'evidence': [item]})
        def synthesize(self, question, sources):
            text = 'Os requisitos sao matricula ativa e disponibilidade de 20 horas.' if question.startswith('{') else 'Sistemas Embarcados e Eletronica; Engenharia Mecanica e Controle e Automacao.'
            return json.dumps({'answers_question': True, 'supported': True, 'claims': [{'text': text, 'source_ids': [sources[0]['source_id']]}]})
    generator = Generator()
    rag = RAGService(p.search, generator, auth_config)
    def sessions():
        with p.sessions() as session:
            yield session
    old = app.dependency_overrides.copy()
    app.dependency_overrides.update({get_settings: lambda: auth_config, get_db: sessions, get_document_session: sessions, get_rag: lambda: rag})
    headers = {'Authorization': 'Bearer ' + create_access_token(p.owner.id, auth_config)}
    other = {'Authorization': 'Bearer ' + create_access_token(p.other.id, auth_config)}
    try:
        with TestClient(app) as client:
            created = client.post('/api/v1/conversations', headers=headers, json={'document_id': str(document.id)})
            assert created.status_code == 201, created.text
            cid = created.json()['id']
            body = {'question': 'Quais sao as tres areas do processo seletivo do LIVE?', 'conversation_id': cid, 'document_id': str(document.id), 'message_id': str(uuid4())}
            first = client.post('/api/v1/ask', headers=headers, json=body)
            assert first.status_code == 200, first.text
            assert all(area in first.json()['answer'] for area in ['Sistemas Embarcados', 'Engenharia Mecanica', 'Inteligencia Artificial'])
            assert all(c['page_start'] == 1 and c['document_id'] == str(document.id) for c in first.json()['citations'])
            calls = len(generator.questions)
            assert client.post('/api/v1/agent/ask', headers=headers, json=body).json() == first.json()
            assert len(generator.questions) == calls
            second = client.post('/api/v1/agent/ask', headers=headers, json={**body, 'message_id': str(uuid4()), 'question': 'E quais sao os requisitos dessas areas?'})
            assert second.status_code == 200, second.text
            assert 'matricula ativa' in second.json()['answer'] and '20 horas' in second.json()['answer']
            history = json.loads(generator.questions[-1])['history_untrusted']
            assert len(history) == 1 and 'Inteligencia Artificial' in history[0]['answer']
            absent = client.post('/api/v1/ask', headers=headers, json={**body, 'message_id': str(uuid4()), 'question': 'Qual a garantia?'})
            assert absent.status_code == 200 and absent.json()['citations'] == []
            detail = client.get('/api/v1/conversations/' + cid, headers=headers).json()
            assert len(detail['messages']) == 3 and detail['messages'][1]['agent'] is True
            assert client.get('/api/v1/conversations/' + cid, headers=other).status_code == 404
            assert client.get('/api/v1/conversations', headers=other).json() == []
            assert client.post('/api/v1/ask', headers=other, json=body).status_code == 404
            assert client.post('/api/v1/conversations', headers=headers, json={'document_id': str(private.id)}).status_code == 404
            assert client.post('/api/v1/ask', headers=headers, json={**body, 'document_id': str(private.id)}).status_code == 409
        with TestClient(app) as fresh:
            relogin = {'Authorization': 'Bearer ' + create_access_token(p.owner.id, auth_config)}
            assert fresh.get('/api/v1/conversations/' + cid, headers=relogin).json()['messages'] == detail['messages']
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(old)


def test_conversation_history_budget_and_concurrent_write_conflict(pipeline):
    from app.db.models import Conversation
    from app.schemas.conversations import AskRequest, ConversationCreate
    from app.schemas.retrieval import AskResponse
    from app.services.conversations import ConversationService
    from app.services.indexing import IndexConflict
    p = pipeline
    service = ConversationService()
    with p.sessions() as session:
        created = service.create(ConversationCreate(), p.owner.id, session)
    with p.sessions.begin() as session:
        session.get(Conversation, created.id).messages = [
            {'id': str(uuid4()), 'question': 'q' * 500, 'answer': 'a' * 500, 'citations': [], 'agent': False}
            for _ in range(40)]
    class Responder:
        def ask(self, request, owner, session):
            assert not session.in_transaction()
            assert len(request.history) <= 3
            assert len(json.dumps(request.history, ensure_ascii=False).encode()) < 3100
            with p.sessions.begin() as concurrent:
                concurrent.get(Conversation, created.id).version += 1
            return AskResponse(answer='Não foi possível encontrar a resposta nos documentos autorizados.', citations=[])
    with p.sessions() as session:
        with pytest.raises(IndexConflict):
            service.ask(AskRequest(question='Pergunta', conversation_id=created.id), p.owner.id, session, Responder())
    with p.sessions() as session:
        detail = service.get(created.id, p.owner.id, session)
        assert len(detail.messages) == 40 and detail.version == 1
