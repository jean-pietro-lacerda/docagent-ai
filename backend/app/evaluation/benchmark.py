import argparse
import hashlib
import json
import math
import re
import statistics
import tempfile
import time
import unicodedata
from io import BytesIO
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pymupdf
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from app.ai.embeddings.service import EmbeddingService
from app.core.config import get_settings
from app.db.models import User
from app.evaluation.postgres import temporary_postgres
from app.schemas.embeddings import EmbeddingConfig
from app.schemas.retrieval import SearchRequest
from app.services.documents import DocumentService
from app.services.indexing import IndexingService
from app.services.rag import GroqGenerator, NO_ANSWER, RAGService
from app.services.search import SearchService
from app.storage.local import LocalFileStorage


def normalized(value):
    return ' '.join(''.join(c for c in unicodedata.normalize('NFD', value.casefold()) if not unicodedata.combining(c)).split())


def load_cases(path):
    data = json.loads(Path(path).read_text())
    questions = data['questions']
    if not 1 <= len(questions) <= 60 or len({q['id'] for q in questions}) != len(questions):
        raise ValueError('Use de 1 a 60 perguntas com identificadores únicos.')
    documents = {d['key']: d for d in data['documents']}
    if len(documents) != len(data['documents']):
        raise ValueError('Documentos duplicados.')
    for question in questions:
        pages = documents[question['document']]['pages']
        if any(type(p) is not int or not 1 <= p <= len(pages) for p in question['relevant_pages']):
            raise ValueError('Página esperada inválida.')
        if bool(question['expected']) != bool(question['relevant_pages']):
            raise ValueError('Expectativas e páginas relevantes inconsistentes.')
        SearchRequest(question=question['question'])
    return data


def retrieval_metrics(hits, document_id, pages):
    relevant = set(pages)
    if not relevant:
        return None, None
    found = set()
    reciprocal = 0.0
    for rank, hit in enumerate(hits, 1):
        matching = relevant.intersection(range(hit.page_start, hit.page_end + 1)) if hit.document_id == document_id else set()
        if matching and not reciprocal:
            reciprocal = 1.0 / rank
        found.update(matching)
    return len(found) / len(relevant), reciprocal


def answer_metrics(response, hits, expected):
    lookup = {hit.chunk_id: hit for hit in hits}
    valid = set()
    for citation in response.citations:
        hit = lookup.get(citation.chunk_id)
        if (hit and citation.document_id == hit.document_id and citation.page_start == hit.page_start
                and citation.page_end == hit.page_end and citation.quote in hit.text):
            valid.add(citation.source_id)
    abstained = response.answer == NO_ANSWER and not response.citations
    units = [part for part in response.answer.split('\n\n') if part.strip()] if not abstained else []
    covered = sum(bool(ids := set(map(int, re.findall(r'\[(\d+)\]', unit)))) and ids <= valid for unit in units)
    terms = sum(normalized(term) in normalized(response.answer) for term in expected)
    return {'abstained': abstained, 'citation_coverage': covered / len(units) if units else None,
            'expected_term_coverage': terms / len(expected) if expected else None}


def answer_outcome(metrics, expected):
    if metrics['abstained']:
        return 'undue_abstention' if expected else 'correct_abstention'
    if expected and metrics['expected_term_coverage'] == 1 and metrics['citation_coverage'] == 1:
        return 'correct_answer'
    return 'incorrect_answer'


class SimulatedGenerator:
    """Label-aware oracle for plumbing checks, never a measurement of Groq quality."""
    def __init__(self, case):
        self.case = case

    def generate(self, question, sources):
        expected = self.case['expected']
        matching = [s for s in sources if any(normalized(term) in normalized(s['text']) for term in expected)]
        if not all(any(normalized(term) in normalized(s['text']) for s in matching) for term in expected):
            matching = []
        return json.dumps({'evidence': [{'source_id': s['source_id']} for s in matching]})

    def synthesize(self, question, sources):
        return json.dumps({'answers_question': True, 'supported': True, 'claims': [{'text': '; '.join(self.case['expected']),
                                       'source_ids': [s['source_id'] for s in sources]}]})


def run_benchmark(engine, storage, embeddings, config, data, *, real_groq=False, top_k=5):
    with engine.connect() as connection:
        if connection.scalar(text('SELECT current_database()')) != 'docagent_ai_evaluation':
            raise ValueError('O benchmark exige seu banco temporário dedicado.')
    sessions = sessionmaker(engine)
    owner_id = uuid5(NAMESPACE_URL, 'docagent-evaluation-owner')
    with sessions.begin() as session:
        session.add(User(id=owner_id, status='active'))
    documents = DocumentService(storage)
    indexer = IndexingService(documents, embeddings)
    ids = {}
    for document in data['documents']:
        with pymupdf.open() as pdf:
            for page_text in document['pages']:
                page = pdf.new_page()
                page.insert_textbox(pymupdf.Rect(40, 40, 550, 790), page_text, fontsize=11)
            content = pdf.tobytes()
        with sessions() as session:
            item = documents.register(BytesIO(content), filename=document['key'] + '.pdf',
                                      authenticated_owner_id=owner_id, session=session)
            ids[document['key']] = item.id
            indexer.index(item.id, owner_id, session)
    search = SearchService(embeddings)
    records = []
    for case in data['questions']:
        document_id = ids[case['document']]
        request = SearchRequest(question=case['question'], document_id=document_id, top_k=top_k)
        start = time.perf_counter()
        with sessions() as session:
            hits = search.search(request, owner_id, session)
        retrieval_seconds = time.perf_counter() - start
        recall, mrr = retrieval_metrics(hits, document_id, case['relevant_pages'])
        records.append({'question_id': case['id'], 'mode': 'semantic', 'recall_at_k': recall, 'mrr': mrr,
                        'latency_seconds': retrieval_seconds, 'abstained': None,
                        'citation_coverage': None, 'expected_term_coverage': None})
        for mode in ('extractive', 'synthesis'):
            generator = GroqGenerator(config) if real_groq else SimulatedGenerator(case)
            rag = RAGService(search, generator, config)
            start = time.perf_counter()
            response = rag.answer_hits(request, hits, synthesize=mode == 'synthesis')
            record = {'question_id': case['id'], 'mode': mode, 'recall_at_k': recall, 'mrr': mrr,
                      'latency_seconds': retrieval_seconds + time.perf_counter() - start,
                      **answer_metrics(response, hits, case['expected'])}
            record['outcome'] = answer_outcome(record, case['expected'])
            records.append(record)
    summaries = {}
    for mode in ('semantic', 'extractive', 'synthesis'):
        rows = [row for row in records if row['mode'] == mode]
        def average(field):
            values = [r[field] for r in rows if r[field] is not None]
            return statistics.mean(values) if values else None
        latencies = sorted(r['latency_seconds'] for r in rows)
        summaries[mode] = {key: average(key) for key in ('recall_at_k', 'mrr', 'citation_coverage', 'expected_term_coverage')}
        summaries[mode].update(abstention_rate=average('abstained'), latency_mean_seconds=average('latency_seconds'),
                               latency_p95_seconds=latencies[math.ceil(.95 * len(latencies)) - 1])
        summaries[mode]['outcomes'] = {name: sum(r.get('outcome') == name for r in rows) for name in
                                      ('correct_answer', 'incorrect_answer', 'correct_abstention', 'undue_abstention')} if mode != 'semantic' else None
    return {'dataset_version': data['version'], 'dataset_sha256': hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
            'questions': len(data['questions']), 'top_k': top_k, 'model': embeddings.config.model_id,
            'answerable_questions': sum(bool(q['expected']) for q in data['questions']),
            'rag_min_score': config.rag_min_score, 'rag_context_characters': config.rag_context_characters,
            'cosine_policy': 'ranking only; legacy rag_min_score does not filter evidence candidates',
            'revision': embeddings.config.revision, 'groq': 'real' if real_groq else 'simulated_oracle',
            'groq_model': config.groq_model if real_groq else None, 'retrieval': 'real_onnx_pgvector',
            'citation_coverage_definition': 'answered units with valid source markers; not semantic entailment',
            'relevance_definition': 'gold document and page; recall/MRR exclude unanswerable questions',
            'expected_term_coverage_definition': 'lexical normalized matches, not factual correctness',
            'outcome_definition': 'reference-term matches plus citation coverage; not a semantic correctness judge',
            'latency_definition': 'warm retrieval plus generation; indexing/model load excluded; shared retrieval per question',
            'summary': summaries, 'records': records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, default=Path(__file__).with_name('cases.json'))
    parser.add_argument('--groq-real', action='store_true', help='Autoriza chamadas reais à Groq para esta avaliação.')
    parser.add_argument('--top-k', type=int, choices=range(1, 11), default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = get_settings()
    if args.groq_real and (not config.groq_api_key or not config.groq_model):
        parser.error('Configure GROQ_API_KEY e GROQ_MODEL para avaliação real.')
    embeddings = EmbeddingService(EmbeddingConfig(cache_dir=config.embeddings_cache_dir))
    with tempfile.TemporaryDirectory(prefix='docagent-benchmark-') as directory, temporary_postgres() as engine:
        report = run_benchmark(engine, LocalFileStorage(Path(directory) / 'pdfs'), embeddings, config,
                               load_cases(args.cases), real_groq=args.groq_real, top_k=args.top_k)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
