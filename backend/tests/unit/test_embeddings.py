from dataclasses import replace
import hashlib
import json
import math
import sys
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.ai.embeddings.backend import EmbeddingBackend, EmbeddingModelError, ONNXEmbeddingBackend
from app.ai.embeddings.service import EmbeddingInputError, EmbeddingService, TokenLimitError
from app.schemas.chunking import ChunkingConfig, ChunkingResult, TextChunk
from app.schemas.embeddings import EmbeddingConfig


def chunks(*texts):
    document_id = uuid4()
    items = tuple(TextChunk(document_id, index, 1, 1, None, text, hashlib.sha256(text.encode()).hexdigest()) for index, text in enumerate(texts))
    return ChunkingResult(document_id, items, (1,), (), ChunkingConfig())


def test_batches_normalization_and_metadata_order():
    backend = Mock(spec=EmbeddingBackend)
    backend.token_counts.side_effect = lambda texts: [len(text.split()) + 2 for text in texts]
    backend.encode.side_effect = lambda texts: [[float(text[-1]) + 1] + [1.0] * 383 for text in texts]
    config = EmbeddingConfig(batch_size=2)
    service = EmbeddingService(config, backend)
    source = chunks("texto 0", "texto 1", "texto 2")
    result = service.embed_chunks(source)
    assert [item.chunk_index for item in result.embeddings] == [0, 1, 2]
    assert [item.chunk_hash for item in result.embeddings] == [item.hash for item in source.chunks]
    assert [call.args[0] for call in backend.encode.call_args_list] == [["texto 0", "texto 1"], ["texto 2"]]
    for item in result.embeddings:
        assert item.document_id == source.document_id
        assert item.dimension == len(item.vector) == 384
        assert item.model_id == config.model_id and item.revision == config.revision
        assert all(math.isfinite(value) for value in item.vector)
        assert math.isclose(math.hypot(*item.vector), 1.0, abs_tol=1e-6)
    assert result.embeddings[0].vector[0] < result.embeddings[1].vector[0] < result.embeddings[2].vector[0]
    question = service.embed_query("pergunta 1")
    assert question.vector == result.embeddings[1].vector
    assert question.config == result.config == config


def test_token_preflight_reports_all_incompatible_chunks_without_inference():
    backend = Mock(spec=EmbeddingBackend)
    backend.token_counts.return_value = [128, 129, 200]
    service = EmbeddingService(backend=backend)
    source = chunks("primeiro texto", "segundo texto", "terceiro texto")
    report = service.validate_chunks(source)
    assert report.total_chunks == 3 and report.incompatible_count == 2
    assert [(item.chunk_index, item.token_count) for item in report.incompatible_chunks] == [(1, 129), (2, 200)]
    with pytest.raises(TokenLimitError) as error:
        service.embed_chunks(source)
    assert error.value.report == report
    backend.encode.assert_not_called()
    assert source.chunks[2].text == "terceiro texto"
    backend.token_counts.return_value = [129]
    with pytest.raises(EmbeddingInputError, match="não houve truncamento"):
        service.embed_query("Qual é o valor da bolsa?")


def test_empty_chunk_collection_does_not_load_model():
    backend = Mock(spec=EmbeddingBackend)
    result = EmbeddingService(backend=backend).embed_chunks(chunks())
    assert result.embeddings == ()
    backend.token_counts.assert_not_called()
    backend.encode.assert_not_called()


@pytest.mark.parametrize("text", ["", " \n", None])
def test_invalid_input_is_rejected_without_loading(text):
    backend = Mock(spec=EmbeddingBackend)
    with pytest.raises(EmbeddingInputError):
        EmbeddingService(backend=backend).embed_query(text)
    backend.token_counts.assert_not_called()


@pytest.mark.parametrize("vector", [[1.0] * 383, [float("nan")] + [1.0] * 383, [float("inf")] + [1.0] * 383, [0.0] * 384])
def test_invalid_model_output_is_rejected(vector):
    backend = Mock(spec=EmbeddingBackend)
    backend.token_counts.return_value = [5]
    backend.encode.return_value = [vector]
    with pytest.raises(EmbeddingModelError):
        EmbeddingService(backend=backend).embed_query("Uma pergunta válida")


def test_model_errors_and_mismatched_chunk_identity_are_not_hidden():
    backend = Mock(spec=EmbeddingBackend)
    source = chunks("texto válido")
    wrong = replace(source, chunks=(replace(source.chunks[0], document_id=uuid4()),))
    with pytest.raises(EmbeddingInputError):
        EmbeddingService(backend=backend).embed_chunks(wrong)
    backend.token_counts.return_value = [5]
    backend.encode.side_effect = EmbeddingModelError("Inferência indisponível.")
    with pytest.raises(EmbeddingModelError, match="Inferência indisponível"):
        EmbeddingService(backend=backend).embed_chunks(source)


def test_lazy_initialization_reuses_cached_tokenizer_and_session(tmp_path, monkeypatch):
    files = {}
    for name, value in {
        "config.json": {"hidden_size": 384, "pad_token_id": 1},
        "sentence_bert_config.json": {"max_seq_length": 128},
        "1_Pooling/config.json": {"pooling_mode_mean_tokens": True},
        "tokenizer.json": {}, "onnx/model.onnx": {},
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        files[name] = str(path)
    download = Mock(side_effect=lambda repo, name, **kwargs: files[name])
    tokenizer = Mock()
    tokenizer.encode_batch.side_effect = lambda texts: [SimpleNamespace(attention_mask=[1, 1, 1]) for text in texts]
    constructor = Mock(return_value=Mock())
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_file=Mock(return_value=tokenizer))))
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        SessionOptions=SimpleNamespace, InferenceSession=constructor, disable_telemetry_events=Mock(),
    ))
    backend = ONNXEmbeddingBackend(EmbeddingConfig(cache_dir=tmp_path))
    download.assert_not_called()
    assert backend.token_counts(["um texto"]) == backend.token_counts(["outro texto"]) == [3]
    backend._load_session()
    backend._load_session()
    constructor.assert_called_once()
    tokenizer.no_truncation.assert_called_once()
    assert all(call.kwargs["local_files_only"] for call in download.call_args_list)
    constructor.side_effect = MemoryError("SENSITIVE detail")
    with pytest.raises(EmbeddingModelError) as error:
        ONNXEmbeddingBackend(EmbeddingConfig(cache_dir=tmp_path))._load_session()
    assert "SENSITIVE" not in str(error.value)
