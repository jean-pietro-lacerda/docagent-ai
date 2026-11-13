from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


@dataclass(frozen=True)
class EmbeddingConfig:
    model_id: str = MODEL_ID
    revision: str = MODEL_REVISION
    dimension: int = 384
    max_tokens: int = 128
    batch_size: int = 8
    cpu_threads: int = 2
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache/docagent-ai/models")
    allow_download: bool = False
    backend: str = "onnx-cpu-fp32"
    pooling: str = "attention-mask-mean"
    normalization: str = "l2"

    def __post_init__(self):
        if (self.model_id, self.revision) != (MODEL_ID, MODEL_REVISION):
            raise ValueError("Esta implementação suporta somente o modelo e a revisão verificados.")
        if type(self.dimension) is not int or type(self.max_tokens) is not int or self.dimension != 384 or self.max_tokens != 128:
            raise ValueError("O modelo exige 384 dimensões e limite de 128 tokens.")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 32:
            raise ValueError("O lote deve conter de 1 a 32 textos.")
        if type(self.cpu_threads) is not int or not 1 <= self.cpu_threads <= 8:
            raise ValueError("Configure de 1 a 8 threads de CPU.")
        if (self.backend, self.pooling, self.normalization) != ("onnx-cpu-fp32", "attention-mask-mean", "l2"):
            raise ValueError("Configuração de inferência incompatível com o backend.")
        if type(self.allow_download) is not bool:
            raise ValueError("allow_download deve ser booleano explícito.")
        cache = Path(self.cache_dir).expanduser().resolve()
        if cache.is_relative_to(Path(__file__).resolve().parents[3]):
            raise ValueError("O cache do modelo deve ficar fora do repositório.")
        object.__setattr__(self, "cache_dir", cache)


@dataclass(frozen=True)
class TokenOverflow:
    document_id: UUID
    chunk_index: int
    token_count: int


@dataclass(frozen=True)
class TokenValidation:
    total_chunks: int
    token_counts: tuple[int, ...]
    incompatible_chunks: tuple[TokenOverflow, ...]
    max_tokens: int

    @property
    def incompatible_count(self) -> int:
        return len(self.incompatible_chunks)


@dataclass(frozen=True)
class ChunkVector:
    document_id: UUID
    chunk_index: int
    chunk_hash: str
    model_id: str
    revision: str
    dimension: int
    vector: tuple[float, ...] = field(repr=False)
    token_count: int


@dataclass(frozen=True)
class EmbeddingResult:
    document_id: UUID
    embeddings: tuple[ChunkVector, ...]
    config: EmbeddingConfig


@dataclass(frozen=True)
class QueryVector:
    vector: tuple[float, ...] = field(repr=False)
    token_count: int
    config: EmbeddingConfig
