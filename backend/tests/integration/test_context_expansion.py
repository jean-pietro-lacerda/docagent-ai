import os
from unittest.mock import Mock
from uuid import uuid4

import pytest

from tests.integration.test_vector_pipeline import pipeline, vector_database
from app.ai.agents.documents import DocumentAgent
from app.db.models import ChunkEmbedding, Document, DocumentChunk
from app.schemas.retrieval import SearchHit, SearchRequest
from app.services.indexing import embedding_key
from app.services.rag import RAGService

pytestmark = pytest.mark.skipif(os.environ.get('RUN_VECTOR_INTEGRATION') != '1', reason='PostgreSQL temporário isolado.')


def context_rows(pipeline):
    p = pipeline
    document = p.document()
    private = p.document(user=p.other)
    another = p.document()
    texts = [
        'Trecho distante em outra página.',
        '1 – Sistemas Embarcados e Eletrônica;\n2 – Engenharia',
        '1 – Sistemas Embarcados e Eletrônica;\n2 – Engenharia Mecânica e Controle e Automação\n3 – Inteligência Artificial;',
        'Requisitos: matrícula ativa e disponibilidade.',
    ]
    chunks = [DocumentChunk(id=uuid4(), document_id=document.id, chunk_index=i,
                            page_start=page, page_end=page, text=text, hash='a' * 64)
              for i, (text, page) in enumerate(zip(texts, [4, 1, 1, 2]))]
    incompatible = DocumentChunk(id=uuid4(), document_id=document.id, chunk_index=4,
                                  page_start=2, page_end=2, text='Modelo incompatível.', hash='b' * 64)
    private_chunk = DocumentChunk(id=uuid4(), document_id=private.id, chunk_index=1,
                                   page_start=1, page_end=1, text='SEGREDO de outro usuário.', hash='c' * 64)
    another_chunk = DocumentChunk(id=uuid4(), document_id=another.id, chunk_index=1,
                                   page_start=1, page_end=1, text='Outro documento do mesmo usuário.', hash='d' * 64)
    with p.sessions.begin() as session:
        session.get(Document, document.id).status = 'ready'
        session.get(Document, document.id).pages = 4
        session.get(Document, private.id).status = 'ready'
        session.get(Document, another.id).status = 'ready'
        session.add_all([*chunks, incompatible, private_chunk, another_chunk])
        session.flush()
        for chunk in [*chunks, incompatible, private_chunk, another_chunk]:
            vector = [1.0, 0.0] if chunk.id == chunks[1].id else [0.0, 1.0]
            session.add(ChunkEmbedding(chunk_id=chunk.id,
                        model_id='incompatible' if chunk.id == incompatible.id else embedding_key(p.embeddings),
                        embedding=vector + [0.0] * 382))
    return document, chunks, incompatible, private_chunk, another_chunk


def test_top_one_partial_chunk_expands_for_rag_and_agent_without_reembedding(pipeline, auth_config, monkeypatch):
    p = pipeline
    document, chunks, _, _, _ = context_rows(p)
    request = SearchRequest(question='Quais são as áreas deste edital?', document_id=document.id, top_k=1)
    with p.sessions() as session:
        hits = p.search.search(request, p.owner.id, session)
        assert len(hits) == 1 and hits[0].text.endswith('Engenharia')
    generator = Mock()
    encode = Mock(wraps=p.backend.encode)
    monkeypatch.setattr(p.backend, 'encode', encode)
    generator.generate.return_value = '{"evidence":[{"source_id":1}]}'
    generator.synthesize.return_value = '{"answers_question":true,"supported":true,"claims":[{"text":"Sistemas Embarcados e Eletrônica e Engenharia.","source_ids":[1]}]}'
    rag = RAGService(p.search, generator, auth_config)
    with p.sessions() as session:
        for responder in (rag, DocumentAgent(rag)):
            response = responder.ask(request, p.owner.id, session)
            assert all(area in response.answer for area in [
                'Sistemas Embarcados e Eletrônica', 'Engenharia Mecânica e Controle e Automação', 'Inteligência Artificial'])
            assert all(c.document_id == document.id and c.page_start == c.page_end == 1 for c in response.citations)
            sources = generator.generate.call_args.args[1]
            assert any('Inteligência Artificial' in source['text'] for source in sources)
            assert all('Trecho distante' not in source['text'] for source in sources)
    assert generator.generate.call_count == generator.synthesize.call_count == 2
    assert encode.call_count == 2
    assert all(call.args[0] == [request.question] for call in encode.call_args_list)


def test_context_expansion_checks_owner_document_revision_pages_and_budget(pipeline):
    p = pipeline
    document, chunks, incompatible, private, another = context_rows(p)
    request = SearchRequest(question='Áreas', document_id=document.id, top_k=1)
    def hit(chunk):
        return SearchHit(chunk_id=chunk.id, document_id=document.id, filename='test.pdf',
                         page_start=1, page_end=1, text=chunk.text, score=1)
    with p.sessions() as session:
        assert p.search.expand_context(request, p.owner.id, session, [hit(private)], max_characters=10000) == []
        assert p.search.expand_context(request, p.owner.id, session, [hit(another)], max_characters=10000) == []
        assert p.search.expand_context(request, p.owner.id, session, [hit(incompatible)], max_characters=10000) == []
        assert p.search.expand_context(request, p.other.id, session, [hit(chunks[1])], max_characters=10000) == []
        result = p.search.expand_context(request, p.owner.id, session, [hit(chunks[1])], max_characters=10000)
        assert {h.chunk_id for h in result} == {chunks[i].id for i in (1, 2, 3)}
        assert {h.page_start for h in result} == {1, 2}
        budget = len(chunks[1].text)
        bounded = p.search.expand_context(request, p.owner.id, session, [hit(chunks[1])], max_characters=budget)
        assert [h.chunk_id for h in bounded] == [chunks[1].id]
        assert sum(len(h.text) for h in bounded) <= budget
