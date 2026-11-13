import math
import os
from pathlib import Path
import resource
import time
from uuid import uuid4

import pytest

from app.ai.embeddings.backend import ModelUnavailable
from app.ai.embeddings.service import EmbeddingService, TokenLimitError
from app.schemas.embeddings import EmbeddingConfig
from app.schemas.extraction import ExtractedPage, PDFExtractionResult
from app.services.chunking import DocumentChunker


def chunked(texts):
    pages = tuple(ExtractedPage(index, text, len(text), "extracted") for index, text in enumerate(texts, 1))
    return DocumentChunker().chunk(PDFExtractionResult(uuid4(), len(pages), pages, sum(len(text) for text in texts), (), "extracted"))


@pytest.mark.skipif(os.environ.get("RUN_EMBEDDING_INTEGRATION") != "1", reason="Modelo real: execução optativa, somente com cache local.")
def test_real_portuguese_embeddings_offline(tmp_path):
    for module in ("onnxruntime", "tokenizers", "huggingface_hub", "numpy"):
        pytest.importorskip(module, reason="Instale o extra embeddings para o teste real.")
    cache = Path(os.environ.get("EMBEDDINGS_CACHE_DIR", str(EmbeddingConfig().cache_dir)))
    config = EmbeddingConfig(cache_dir=cache, allow_download=False)
    service = EmbeddingService(config)
    start = time.perf_counter()
    try:
        assert service.backend.token_counts(["", "bolsa"]) == [2, 3]
        assert service.backend._session is None
        first = service.embed_query("Qual é o valor da bolsa?")
    except ModelUnavailable:
        pytest.skip("Modelo não disponível no cache local; nenhum download foi solicitado.")
    loaded = time.perf_counter()
    texts = ["Qual é o valor da bolsa?", "O edital informa uma bolsa mensal para estudantes matriculados na universidade."]
    result = service.embed_chunks(chunked(texts))
    assert all(0 < item.token_count <= 128 for item in result.embeddings)
    vectors = [item.vector for item in result.embeddings]
    for vector in [first.vector, *vectors]:
        assert len(vector) == 384
        assert all(math.isfinite(value) for value in vector)
        assert math.isclose(math.hypot(*vector), 1.0, abs_tol=1e-6)
    assert sum(a * b for a, b in zip(first.vector, vectors[0])) > 0.99999
    warm_time = time.perf_counter() - loaded
    long_chunks = chunked(["bolsa " * 200])
    report = service.validate_chunks(long_chunks)
    assert report.incompatible_count == 1
    with pytest.raises(TokenLimitError):
        service.embed_chunks(long_chunks)
    from app.db.models import Document
    from app.schemas.chunking import TokenAwareConfig
    from app.services.pdf_extraction import PDFTextExtractor
    from app.services.token_chunking import TokenAwareChunker
    from app.storage.local import LocalFileStorage
    import pymupdf

    text = "Prazo: 31/12/2026. Valor: R$ 1.234,56. Bolsa para estudantes."
    page_text = "\n".join([text] * 12)
    with pymupdf.open() as pdf:
        for _ in range(2):
            pdf.new_page().insert_text((50, 50), page_text)
        source = tmp_path / "synthetic.pdf"
        source.write_bytes(pdf.tobytes())
    storage = LocalFileStorage(tmp_path / "private")
    document = Document(id=uuid4(), owner_id=uuid4(), object_key=storage.store(source))
    extraction = PDFTextExtractor(storage).extract(document)
    assert all(page.text == page_text for page in extraction.pages)
    chunk_start = time.perf_counter()
    chunker = TokenAwareChunker(service.backend, TokenAwareConfig(overlap=0))
    compatible = chunker.chunk(extraction)
    chunk_seconds = time.perf_counter() - chunk_start
    assert compatible == chunker.chunk(extraction)
    for page in extraction.pages:
        assert "".join(c.text for c in compatible.chunks if c.page_start == page.page_number) == page.text
    assert all(c.page_start == c.page_end and 0 < c.token_count <= 128 for c in compatible.chunks)
    assert service.validate_chunks(compatible).token_counts == tuple(c.token_count for c in compatible.chunks)
    encoded = service.embed_chunks(compatible)
    assert len(encoded.embeddings) == compatible.total_chunks > 2
    assert all(len(v.vector) == 384 and math.isclose(math.hypot(*v.vector), 1, abs_tol=1e-6) for v in encoded.embeddings)
    overlapped = TokenAwareChunker(service.backend).chunk(extraction)
    assert service.validate_chunks(overlapped).incompatible_count == 0
    assert all(c.token_count <= 128 for c in overlapped.chunks)
    numbered = '\n'.join(f'{i} – Experiência em processamento de dados e disponibilidade para pesquisa.' for i in range(1, 21))
    page = ExtractedPage(1, numbered, len(numbered), 'extracted')
    list_extraction = PDFExtractionResult(uuid4(), 1, (page,), len(numbered), (), 'extracted')
    list_chunks = chunker.chunk(list_extraction)
    assert ''.join(c.text for c in list_chunks.chunks) == numbered
    assert all(any(item in c.text for c in list_chunks.chunks) for item in numbered.splitlines())
    assert all(c.token_count <= 128 for c in list_chunks.chunks)
    session = service.backend._session
    assert len(service.embed_chunks(list_chunks).embeddings) == list_chunks.total_chunks
    assert service.backend._session is session
    print({
        "token_aware_chunking_seconds": round(chunk_seconds, 4),
        "synthetic_pdf_chunks": compatible.total_chunks,
        "load_and_first_inference_seconds": round(loaded - start, 4),
        "warm_batch_seconds": round(warm_time, 4),
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "revision": config.revision,
    })
