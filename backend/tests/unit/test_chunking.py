import hashlib
from uuid import uuid4

import pytest

from app.schemas.chunking import ChunkingConfig
from app.schemas.extraction import ExtractedPage, PDFExtractionResult
from app.services.chunking import ChunkingError, DocumentChunker


def extraction(*texts, statuses=None):
    statuses = statuses or ["extracted" if text else "empty" for text in texts]
    pages = tuple(ExtractedPage(index, text, len(text), status) for index, (text, status) in enumerate(zip(texts, statuses), 1))
    return PDFExtractionResult(
        uuid4(), len(pages), pages, sum(len(text) for text in texts),
        tuple(page.page_number for page in pages if not page.text),
        "partial" if any(status != "extracted" for status in statuses) else "extracted",
    )


def test_page_provenance_indices_hashes_and_repetition():
    repeated = "Bolsa de R$ 1.200,00 durante 12 meses.\nArt. 12 - Prazo: 31/12/2026."
    source = extraction(repeated, "", repeated, "12", "unusable", statuses=["extracted", "empty", "extracted", "insufficient_text", "failed"])
    chunker = DocumentChunker()
    result = chunker.chunk(source)
    assert result == chunker.chunk(source)
    assert result.total_chunks == 3
    assert result.processed_pages == (1, 3, 4) and result.ignored_pages == (2, 5)
    assert [chunk.chunk_index for chunk in result.chunks] == [0, 1, 2]
    assert [chunk.page_start for chunk in result.chunks] == [1, 3, 4]
    for chunk in result.chunks:
        assert chunk.page_start == chunk.page_end
        assert chunk.document_id == source.document_id
        assert chunk.section is None
        assert chunk.hash == hashlib.sha256(chunk.text.encode()).hexdigest()
    assert result.chunks[0].text == repeated == result.chunks[1].text
    assert result.chunks[0].hash == result.chunks[1].hash
    assert result.chunks[2].text == "12"
    assert result.config.max_characters == 1800 and result.config.overlap == 250
    assert result.config.strategy_id == "characters-semantic-v1"
    assert repeated not in repr(result)


def test_semantic_boundaries_and_lossless_content_without_overlap():
    config = ChunkingConfig(max_characters=80, overlap=0)
    paragraphs = "Artigo primeiro com requisitos de habilitacao.\n\nItem 2.1 - Bolsa de R$ 1.200,00 durante 12 meses.\nData: 31/12/2026."
    result = DocumentChunker(config).chunk(extraction(paragraphs))
    assert result.chunks[0].text.endswith("\n\n")
    assert "".join(chunk.text for chunk in result.chunks) == paragraphs
    assert all(0 < len(chunk.text) <= 80 for chunk in result.chunks)
    sentences = "Primeira frase com informacao relevante. Segunda frase para contexto. Terceira frase para completar."
    result = DocumentChunker(config).chunk(extraction(sentences))
    assert result.chunks[0].text.endswith(". ")
    assert "".join(chunk.text for chunk in result.chunks) == sentences
    long_word = "inicio " + "x" * 75
    result = DocumentChunker(config).chunk(extraction(long_word + " final"))
    assert result.chunks[0].text == "inicio "
    assert result.chunks[1].text.startswith("x" * 75)
    assert "".join(chunk.text for chunk in result.chunks) == long_word + " final"


def test_overlap_preserves_whole_words_and_respects_maximum():
    text = " ".join(f"palavra{index:03}" for index in range(100))
    config = ChunkingConfig(max_characters=100, overlap=30, strategy_id="baseline-experiment")
    result = DocumentChunker(config).chunk(extraction(text))
    assert result.config == config
    cursor = 0
    previous_end = 0
    overlaps = []
    for chunk in result.chunks:
        start = text.find(chunk.text, cursor)
        assert start >= cursor
        assert start == 0 or text[start - 1].isspace()
        assert len(chunk.text) <= 100
        if previous_end:
            overlap = previous_end - start
            assert 0 <= overlap <= 30
            overlaps.append(overlap)
        previous_end = start + len(chunk.text)
        cursor = start + 1
    assert previous_end == len(text)
    assert any(overlaps)


def test_unbroken_text_terminates_with_exact_overlap(monkeypatch):
    text = "x" * 231
    result = DocumentChunker(ChunkingConfig(max_characters=64, overlap=10)).chunk(extraction(text))
    assert [len(chunk.text) for chunk in result.chunks] == [64, 64, 64, 64, 15]
    assert len(result.chunks[0].text) + sum(len(chunk.text) - 10 for chunk in result.chunks[1:]) == len(text)
    assert all(chunk.hash == hashlib.sha256(chunk.text.encode()).hexdigest() for chunk in result.chunks)
    near_limit = DocumentChunker(ChunkingConfig(max_characters=64, overlap=63)).chunk(extraction(text))
    assert len(near_limit.chunks) == len(text) - 64 + 1
    monkeypatch.setattr("app.services.chunking.MAX_CHUNKS", 2)
    with pytest.raises(ChunkingError, match="limite de 10.000 chunks"):
        DocumentChunker(ChunkingConfig(max_characters=64, overlap=10)).chunk(extraction(text))


@pytest.mark.parametrize("status", ["no_text", "failed", "insufficient_text"])
def test_no_artificial_chunks_for_unusable_pages(status):
    source = extraction("", " \t\n", "...", statuses=["empty", "failed", "insufficient_text"])
    source = PDFExtractionResult(source.document_id, 3, source.pages, source.total_characters, source.pages_without_text, status)
    result = DocumentChunker().chunk(source)
    assert result.chunks == () and result.total_chunks == 0
    assert result.processed_pages == () and result.ignored_pages == (1, 2, 3)


@pytest.mark.parametrize("options", [
    {"max_characters": 0}, {"max_characters": 10_001}, {"max_characters": True},
    {"overlap": -1}, {"overlap": 1800}, {"strategy_id": ""},
])
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        ChunkingConfig(**options)


def test_invalid_page_order_is_rejected():
    source = extraction("Texto da primeira pagina.", "Texto da segunda pagina.")
    source = PDFExtractionResult(source.document_id, 2, tuple(reversed(source.pages)), source.total_characters, (), "extracted")
    with pytest.raises(ChunkingError, match="em ordem"):
        DocumentChunker().chunk(source)
