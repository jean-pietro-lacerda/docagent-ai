import json
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.ai.agents.documents import DocumentAgent
from app.core.config import Settings
from app.schemas.retrieval import SearchHit
from app.services.rag import NO_ANSWER, RAGService

AREAS = ('1 – Sistemas Embarcados e Eletrônica;\n'
         '2 – Engenharia Mecânica e Controle e Automação;\n3 – Inteligência Artificial;')
REQUIREMENTS = ('Área – Inteligência Artificial\nREQUISITOS OBRIGATÓRIOS: '
                'Aprendizagem de Máquina (Python, Numpy, Pandas, ScikitLearn); Conhecimento em Python.')
SELECTION = 'A seleção será realizada mediante análise curricular e entrevista pessoal.'


@pytest.mark.parametrize('agent', [False, True])
def test_current_question_changes_evidence_without_losing_followup_context(agent):
    owner, document = uuid4(), uuid4()
    hits = [SearchHit(chunk_id=uuid4(), document_id=document, filename='edital.pdf',
                      page_start=page, page_end=page, text=text, score=0.8)
            for page, text in [(1, AREAS), (3, REQUIREMENTS), (1, SELECTION)]]
    search = Mock()
    search.search.return_value = hits
    search.embeddings.backend.token_counts.return_value = [60]
    search.embeddings.config.max_tokens = 128
    generator = Mock()
    rag = RAGService(search, generator, Settings(_env_file=None, debug=False, auth_enabled=False))
    service = DocumentAgent(rag) if agent else rag
    history = []
    cases = [('Quais são as áreas do edital?', 1, AREAS),
             ('Quais conhecimentos são exigidos para Inteligência Artificial?', 2, REQUIREMENTS),
             ('Como funciona a seleção?', 3, SELECTION),
             ('E quais são as etapas dessa seleção?', 3, SELECTION),
             ('Qual é a cor do uniforme?', None, NO_ANSWER)]
    for question, source_id, expected in cases:
        generator.generate.return_value = json.dumps({'evidence': [{'source_id': source_id}] if source_id else []})
        generator.synthesize.return_value = json.dumps({
            'answers_question': True, 'supported': True,
            'claims': [{'text': expected, 'source_ids': [source_id]}] if source_id else []})
        request = SimpleNamespace(question=question, document_id=document, top_k=5, history=history[:])
        result = service.ask(request, owner, None)
        if source_id:
            assert expected in result.answer
            assert {c.chunk_id for c in result.citations} == {hits[source_id - 1].chunk_id}
            assert result.citations[0].page_start == hits[source_id - 1].page_start
        else:
            assert result.answer == NO_ANSWER and not result.citations
        if question.startswith('E '):
            assert 'Contexto:' in search.search.call_args.args[0].question
            assert 'history_untrusted' in generator.generate.call_args.args[0]
        else:
            assert search.search.call_args.args[0].question == question
            assert generator.generate.call_args.args[0] == question
        history.append({'question': question, 'answer': result.answer})


@pytest.mark.parametrize('agent', [False, True])
def test_existing_citation_to_areas_cannot_answer_a_requirements_question(agent):
    hit = SearchHit(chunk_id=uuid4(), document_id=uuid4(), filename='edital.pdf',
                    page_start=1, page_end=1, text=AREAS, score=0.9)
    generator = Mock()
    generator.generate.return_value = '{"evidence":[{"source_id":1}]}'
    rag = RAGService(Mock(), generator, Settings(_env_file=None, debug=False, auth_enabled=False))
    rag.search.search.return_value = [hit]
    request = SimpleNamespace(question='Quais os conteúdos que alguém que vai se candidatar pra vaga de inteligência artificial tem que saber?',
                              document_id=hit.document_id, top_k=5, history=[])
    result = (DocumentAgent(rag) if agent else rag).ask(request, uuid4(), None)
    assert result.answer == NO_ANSWER and not result.citations
    generator.synthesize.assert_not_called()


@pytest.mark.parametrize('verdict', [{'answers_question': False, 'supported': True},
                                    {'answers_question': True, 'supported': False}])
def test_semantic_rejection_never_returns_extractive_fallback(verdict):
    hit = SearchHit(chunk_id=uuid4(), document_id=uuid4(), filename='edital.pdf',
                    page_start=3, page_end=3, text=REQUIREMENTS, score=0.9)
    generator = Mock()
    generator.generate.return_value = '{"evidence":[{"source_id":1}]}'
    generator.synthesize.return_value = json.dumps({**verdict, 'claims': [{'text': REQUIREMENTS, 'source_ids': [1]}]})
    rag = RAGService(Mock(), generator, Settings(_env_file=None, debug=False, auth_enabled=False))
    response = rag.answer_hits(SimpleNamespace(question='Quais conhecimentos são exigidos para IA?', document_id=hit.document_id), [hit])
    assert response.answer == NO_ANSWER and response.citations == []


def test_concise_technical_answer_does_not_need_literal_requirements_label():
    hit = SearchHit(chunk_id=uuid4(), document_id=uuid4(), filename='edital.pdf',
                    page_start=3, page_end=3, text=REQUIREMENTS, score=0.9)
    generator = Mock()
    generator.generate.return_value = '{"evidence":[{"source_id":1}]}'
    generator.synthesize.return_value = json.dumps({
        'answers_question': True, 'supported': True,
        'claims': [{'text': 'Python, Numpy, Pandas e ScikitLearn.', 'source_ids': [1]}]})
    rag = RAGService(Mock(), generator, Settings(_env_file=None, debug=False, auth_enabled=False))
    response = rag.answer_hits(SimpleNamespace(question='Quais conhecimentos são exigidos para IA?', document_id=hit.document_id), [hit])
    assert response.answer == 'Python, Numpy, Pandas e ScikitLearn. [1]'
