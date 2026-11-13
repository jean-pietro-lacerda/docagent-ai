import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.schemas.chunking import TokenAwareConfig
from app.schemas.embeddings import EmbeddingConfig
from app.ai.embeddings.service import EmbeddingInputError, EmbeddingService
from app.services.chunking import ChunkingError
from app.services.token_chunking import TokenAwareChunker
from tests.unit.test_chunking import extraction


class CharacterTokenizer:
    config = EmbeddingConfig()

    def token_counts(self, texts):
        return [len(text) + 2 for text in texts]


def test_exact_special_tokens_pages_values_and_determinism():
    text = ('Prazo: 31/12/2026. Valor: R$ 1.234,56.\n\n' * 15) + 'Fim.'
    source = extraction(text, '', 'Segunda página com informação.')
    chunker = TokenAwareChunker(CharacterTokenizer(), TokenAwareConfig(overlap=0, safety_margin=4))
    result = chunker.chunk(source)
    assert result == chunker.chunk(source)
    assert result.processed_pages == (1, 3) and result.ignored_pages == (2,)
    for page in source.pages:
        assert ''.join(c.text for c in result.chunks if c.page_start == page.page_number) == page.text
    assert all(c.page_start == c.page_end and c.token_count == len(c.text) + 2 <= 124 for c in result.chunks)
    assert all(c.hash == hashlib.sha256(c.text.encode()).hexdigest() for c in result.chunks)
    assert result.tokenizer_id == CharacterTokenizer.config.model_id
    assert result.tokenizer_revision == CharacterTokenizer.config.revision


def test_overlap_progress_and_unbroken_fallback():
    text = ' '.join(f'palavra{i:04}' for i in range(90))
    chunker = TokenAwareChunker(CharacterTokenizer(), TokenAwareConfig(overlap=30))
    result = chunker.chunk(extraction(text))
    cursor = end = 0
    overlaps = []
    for chunk in result.chunks:
        start = text.find(chunk.text, cursor)
        assert cursor <= start <= end
        assert start == 0 or text[start - 1].isspace()
        overlaps.append(end - start)
        end = start + len(chunk.text)
        cursor = start + 1
    assert end == len(text) and any(value > 0 for value in overlaps)
    assert max(overlaps) <= 30
    source = extraction('á' * 500)
    result = chunker.chunk(source)
    assert ''.join(c.text for c in result.chunks) == source.pages[0].text
    assert all(c.token_count <= 128 for c in result.chunks)


def test_incompatible_tokenizer_and_invalid_margin_fail_explicitly():
    with pytest.raises(ValueError):
        TokenAwareConfig(safety_margin=True)
    backend = SimpleNamespace(config=EmbeddingConfig(), token_counts=lambda texts: [200] * len(texts))
    with pytest.raises(ChunkingError, match='sem perda'):
        TokenAwareChunker(backend).chunk(extraction('texto'))


def test_embedding_service_revalidates_token_metadata():
    backend = CharacterTokenizer()
    result = TokenAwareChunker(backend).chunk(extraction('Texto de referência.'))
    service = EmbeddingService(backend=backend)
    assert service.validate_chunks(result).incompatible_count == 0
    with pytest.raises(EmbeddingInputError, match='Tokenizer'):
        service.validate_chunks(replace(result, tokenizer_revision='different'))
    changed = replace(result.chunks[0], token_count=1)
    with pytest.raises(EmbeddingInputError, match='metadados'):
        service.validate_chunks(replace(result, chunks=(changed,)))


def test_fitting_list_moves_as_a_whole_after_a_long_introduction():
    intro = 'Informações de contexto para a seleção. ' * 5 + '\n\n'
    areas = '1 – Sistemas Embarcados;\n2 – Engenharia Mecânica;\n3 – Inteligência Artificial;'
    result = TokenAwareChunker(CharacterTokenizer(), TokenAwareConfig(overlap=0)).chunk(extraction(intro + areas))
    assert ''.join(c.text for c in result.chunks) == intro + areas
    assert any(areas in c.text for c in result.chunks)
    assert all(c.token_count <= 128 for c in result.chunks)


def test_long_lists_split_between_fitting_items_and_keep_lossless_fallback():
    items = [f'{i} – Requisito número {i}: experiência e disponibilidade.\n' for i in range(1, 15)]
    text = ''.join(items)
    chunker = TokenAwareChunker(CharacterTokenizer(), TokenAwareConfig(overlap=0))
    result = chunker.chunk(extraction(text))
    assert ''.join(c.text for c in result.chunks) == text
    assert all(any(item in c.text for c in result.chunks) for item in items)
    assert all(c.token_count <= 128 for c in result.chunks)
    overlapped = TokenAwareChunker(CharacterTokenizer(), TokenAwareConfig(overlap=40)).chunk(extraction(text))
    assert all(c.token_count <= 128 for c in overlapped.chunks)
    assert all(any(item in c.text for c in overlapped.chunks) for item in items)
    huge = '1 – ' + 'palavra ' * 100 + '\n2 – Fim.'
    fallback = chunker.chunk(extraction(huge))
    assert ''.join(c.text for c in fallback.chunks) == huge
    assert all(c.token_count <= 128 for c in fallback.chunks)
