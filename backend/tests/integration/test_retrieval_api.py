from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.api.dependencies import get_document_session
from app.api.routes.retrieval import get_indexer, get_rag, get_search
from app.core.config import get_settings
from app.db.models import User
from app.db.session import get_db
from app.main import app
from app.schemas.retrieval import AskResponse, IndexResponse
from app.security.tokens import create_access_token
from app.services.documents import DocumentNotFound
from app.services.rag import GenerationRateLimit, GenerationUnavailable


@pytest.fixture
def retrieval_api(auth_config):
    user = User(id=uuid4(), status='active')
    authentication = Mock()
    authentication.scalar.return_value = user
    indexer, search, rag = Mock(), Mock(), Mock()
    original = app.dependency_overrides.copy()
    session = object()
    app.dependency_overrides.update({get_settings: lambda: auth_config, get_db: lambda: authentication,
        get_document_session: lambda: session, get_indexer: lambda: indexer,
        get_search: lambda: search, get_rag: lambda: rag})
    try:
        with TestClient(app) as client:
            yield SimpleNamespace(client=client, user=user, indexer=indexer, search=search, rag=rag, session=session,
                                  headers={'Authorization': 'Bearer ' + create_access_token(user.id, auth_config)})
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original)


def test_retrieval_authentication_identity_and_contracts(retrieval_api):
    api = retrieval_api
    document_id = uuid4()
    paths = [f'/api/v1/documents/{document_id}/index', '/api/v1/search', '/api/v1/ask']
    for path in paths:
        assert api.client.post(path, json={'question': 'Pergunta'}).status_code == 401
    api.indexer.index.return_value = IndexResponse(document_id=document_id, chunks=2, model_id='model@revision')
    response = api.client.post(paths[0], headers=api.headers)
    assert response.status_code == 200 and response.json()['chunks'] == 2
    api.indexer.index.assert_called_once_with(document_id, api.user.id, api.session)
    api.search.search.return_value = []
    response = api.client.post(paths[1], headers=api.headers, json={'question': 'Qual o prazo?', 'top_k': 3})
    assert response.status_code == 200 and response.json() == {'items': []}
    assert api.search.search.call_args.args[1:] == (api.user.id, api.session)
    api.rag.ask.return_value = AskResponse(answer='Sem evidência.', citations=[])
    response = api.client.post(paths[2], headers=api.headers, json={'question': 'Qual o prazo?'})
    assert response.status_code == 200 and response.headers['cache-control'] == 'no-store'
    assert api.rag.ask.call_args.args[1:] == (api.user.id, api.session)


def test_retrieval_input_validation_and_sanitized_errors(retrieval_api):
    api = retrieval_api
    for body in [{'question': ' '}, {'question': 'x' * 1001}, {'question': 'ok', 'top_k': 11},
                 {'question': 'ok', 'owner_id': str(uuid4())}, {'question': 'ok', 'top_k': True}]:
        response = api.client.post('/api/v1/search', headers=api.headers, json=body)
        assert response.status_code == 422 and 'owner_id' not in response.text
    assert api.client.post('/api/v1/search', headers=api.headers, content=b'x' * 16385).status_code == 413
    api.search.search.assert_not_called()
    for error, expected in [(GenerationRateLimit('SENSITIVE'), 429), (GenerationUnavailable('SENSITIVE'), 503),
                            (SQLAlchemyError('SENSITIVE'), 503)]:
        api.rag.ask.side_effect = error
        response = api.client.post('/api/v1/ask', headers=api.headers, json={'question': 'Pergunta'})
        assert response.status_code == expected and 'SENSITIVE' not in response.text
    api.indexer.index.side_effect = DocumentNotFound('SENSITIVE')
    response = api.client.post(f'/api/v1/documents/{uuid4()}/index', headers=api.headers)
    assert response.status_code == 404 and 'SENSITIVE' not in response.text


def test_ask_regression_edital_multiple_areas_with_verified_page_one_sources(retrieval_api, auth_config):
    import json
    from uuid import UUID
    from app.schemas.retrieval import SearchHit
    from app.services.rag import RAGService

    api = retrieval_api
    document_id = UUID('ce8b2ef3-d490-40b8-858d-06af8cfca03d')
    texts = [
        'Edital de seleção de bolsistas.', 'Informações gerais e cronograma.',
        '1 – Sistemas Embarcados e Eletrônica;\n2 – Engenharia Mecânica e Controle e Automação',
        '3 – Inteligência Artificial;', 'Inscrições e documentação necessária.',
    ]
    hits = [SearchHit(chunk_id=uuid4(), document_id=document_id, filename='edital.pdf',
                      page_start=page, page_end=page, text=text, score=score)
            for text, page, score in zip(texts, [1, 3, 1, 1, 2], [0.58985, 0.58788, 0.57257, 0.51105, 0.50971])]
    search, generator = Mock(), Mock()
    search.search.return_value = hits
    generator.generate.return_value = json.dumps({'evidence': [{'source_id': 3}, {'source_id': 4}]})
    service = RAGService(search, generator, auth_config)
    app.dependency_overrides[get_rag] = lambda: service
    body = {'question': 'Para quais areas esse edital estar selecionando?', 'document_id': str(document_id)}
    response = api.client.post('/api/v1/ask', headers=api.headers, json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    for area in ['Sistemas Embarcados e Eletrônica', 'Engenharia Mecânica e Controle e Automação', 'Inteligência Artificial']:
        assert area in result['answer']
    assert len(result['citations']) == 2
    for citation, hit in zip(result['citations'], hits[2:4]):
        assert citation['document_id'] == str(document_id)
        assert citation['page_start'] == citation['page_end'] == 1
        assert citation['chunk_id'] == str(hit.chunk_id) and citation['quote'] == hit.text
    request, owner, session = search.search.call_args.args
    assert request.document_id == document_id and owner == api.user.id and session is api.session
    assert len(generator.generate.call_args.args[1]) == 5


def test_ask_without_retrieved_evidence_abstains_without_groq(retrieval_api, auth_config):
    from app.services.rag import NO_ANSWER, RAGService

    api = retrieval_api
    search, generator = Mock(), Mock()
    search.search.return_value = []
    service = RAGService(search, generator, auth_config)
    app.dependency_overrides[get_rag] = lambda: service
    response = api.client.post('/api/v1/ask', headers=api.headers, json={'question': 'Quais áreas?'})
    assert response.status_code == 200 and response.json() == {'answer': NO_ANSWER, 'citations': []}
    generator.generate.assert_not_called()


def test_agent_endpoint_requires_auth_and_reuses_rag_with_synthesis(retrieval_api, auth_config):
    import json
    from app.schemas.retrieval import SearchHit
    from app.services.rag import RAGService

    api = retrieval_api
    search, generator = Mock(), Mock()
    hit = SearchHit(chunk_id=uuid4(), document_id=uuid4(), filename='edital.pdf', page_start=1, page_end=1,
                    text='O prazo termina em 31/12/2026.', score=0.9)
    search.search.return_value = [hit]
    generator.generate.return_value = json.dumps({'evidence': [{'source_id': 1, 'quote': hit.text}]})
    generator.synthesize.return_value = json.dumps({'answers_question': True, 'supported': True, 'claims': [{'text': 'O prazo é 31/12/2026.', 'source_ids': [1]}]})
    service = RAGService(search, generator, auth_config)
    app.dependency_overrides[get_rag] = lambda: service
    body = {'question': 'Qual o prazo?', 'document_id': str(hit.document_id)}
    assert api.client.post('/api/v1/agent/ask', json=body).status_code == 401
    search.search.assert_not_called()
    response = api.client.post('/api/v1/agent/ask', headers=api.headers, json=body)
    assert response.status_code == 200 and response.json()['answer'] == 'O prazo é 31/12/2026. [1]'
    assert response.json()['citations'][0]['document_id'] == str(hit.document_id)
    assert search.search.call_args.args[1:] == (api.user.id, api.session)
    assert generator.generate.call_count == generator.synthesize.call_count == 1
