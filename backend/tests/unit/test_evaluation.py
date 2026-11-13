from pathlib import Path
from uuid import uuid4

import pytest

from app.evaluation.benchmark import answer_metrics, load_cases, retrieval_metrics
from app.schemas.retrieval import AskResponse, Citation, SearchHit


def test_benchmark_metrics_have_known_denominators_and_detect_invalid_citations():
    document = uuid4()
    hits = [SearchHit(chunk_id=uuid4(), document_id=document, filename='test.pdf', page_start=p,
                      page_end=p, text='Prazo: 31/12/2026.', score=1) for p in [3, 2]]
    assert retrieval_metrics(hits, document, [1, 2]) == (0.5, 0.5)
    assert retrieval_metrics(hits, document, []) == (None, None)
    citation = Citation(source_id=1, document_id=document, chunk_id=hits[1].chunk_id,
                        filename='test.pdf', page_start=2, page_end=2, quote=hits[1].text)
    response = AskResponse(answer='Prazo: 31/12/2026. [1]\n\nUma afirmação sem citação.', citations=[citation])
    assert answer_metrics(response, hits, ['31/12/2026']) == {
        'abstained': False, 'citation_coverage': 0.5, 'expected_term_coverage': 1.0}
    invalid = response.model_copy(update={'citations': [citation.model_copy(update={'document_id': uuid4()})]})
    assert answer_metrics(invalid, hits, [])['citation_coverage'] == 0
    data = load_cases(Path(__file__).resolve().parents[2] / 'app/evaluation/cases.json')
    assert len(data['questions']) == 8 and any(not q['expected'] for q in data['questions'])


def test_answer_outcomes_distinguish_errors_and_undue_abstentions():
    from app.evaluation.benchmark import answer_outcome
    assert answer_outcome({'abstained': True}, ['3,3 V']) == 'undue_abstention'
    assert answer_outcome({'abstained': True}, []) == 'correct_abstention'
    assert answer_outcome({'abstained': False, 'expected_term_coverage': 1, 'citation_coverage': 1}, ['3,3 V']) == 'correct_answer'
    assert answer_outcome({'abstained': False, 'expected_term_coverage': 1, 'citation_coverage': 0}, ['3,3 V']) == 'incorrect_answer'
    assert answer_outcome({'abstained': False}, []) == 'incorrect_answer'


@pytest.mark.parametrize('case_id,score', [('areas', 0.25313), ('tensao', 0.25564)])
def test_low_cosine_does_not_discard_verified_benchmark_evidence(case_id, score):
    from unittest.mock import Mock
    from app.core.config import Settings
    from app.evaluation.benchmark import SimulatedGenerator
    from app.schemas.retrieval import SearchRequest
    from app.services.rag import RAGService

    data = load_cases(Path(__file__).resolve().parents[2] / 'app/evaluation/cases.json')
    case = next(q for q in data['questions'] if q['id'] == case_id)
    document = next(d for d in data['documents'] if d['key'] == case['document'])
    hit = SearchHit(chunk_id=uuid4(), document_id=uuid4(), filename='test.pdf', page_start=1,
                    page_end=1, text=document['pages'][0], score=score)
    search = Mock()
    search.search.return_value = [hit]
    service = RAGService(search, SimulatedGenerator(case), Settings(_env_file=None, debug=False, auth_enabled=False))
    response = service.ask(SearchRequest(question=case['question'], document_id=hit.document_id), uuid4(), None)
    assert response.citations and all(term in response.answer for term in case['expected'])
    assert answer_metrics(response, [hit], case['expected'])['expected_term_coverage'] == 1
