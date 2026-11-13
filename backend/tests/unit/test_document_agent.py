import json
from unittest.mock import Mock
from uuid import uuid4

from app.ai.agents.documents import DocumentAgent
from app.schemas.retrieval import AskResponse, Citation, SearchHit, SearchRequest
from app.services.rag import NO_ANSWER
from tests.unit.test_rag import setup_rag


def test_langgraph_reuses_authorized_search_and_bounds_calls_without_memory():
    rag, hit = setup_rag()
    rag.generator.synthesize.return_value = json.dumps({'answers_question': True, 'supported': True, 'claims': [{'text': 'A bolsa é R$ 1.234,56.', 'source_ids': [1]}]})
    request = SearchRequest(question='Qual a bolsa?', document_id=hit.document_id)
    owner, session = uuid4(), object()
    agent = DocumentAgent(rag)
    result = agent.ask(request, owner, session)
    assert result.answer == 'A bolsa é R$ 1.234,56. [1]'
    rag.search.search.assert_called_once_with(request, owner, session)
    assert rag.generator.generate.call_count == rag.generator.synthesize.call_count == 1
    rag.search.search.return_value = []
    assert agent.ask(request, owner, session).answer == NO_ANSWER
    assert rag.generator.generate.call_count == rag.generator.synthesize.call_count == 1


def test_agent_verifier_rejects_citation_outside_retrieved_document():
    rag = Mock()
    document_id = uuid4()
    hit = SearchHit(chunk_id=uuid4(), document_id=document_id, filename='test.pdf',
                    page_start=1, page_end=1, text='A informação é pública neste teste.', score=1)
    rag.search.search.return_value = [hit]
    rag.retrieve.return_value = [hit]
    rag.answer_hits.return_value = AskResponse(answer='Resposta [1]', citations=[Citation(
        source_id=1, chunk_id=hit.chunk_id, document_id=uuid4(), filename='test.pdf',
        page_start=1, page_end=1, quote=hit.text)])
    assert DocumentAgent(rag).ask(SearchRequest(question='Pergunta', document_id=document_id), uuid4(), None).answer == NO_ANSWER
