import json
from collections.abc import Sequence
from threading import RLock
from typing import Protocol

from app.schemas.embeddings import EmbeddingConfig


class EmbeddingModelError(RuntimeError):
    pass


class ModelUnavailable(EmbeddingModelError):
    pass


class EmbeddingBackend(Protocol):
    def token_counts(self, texts: Sequence[str]) -> list[int]: ...
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class ONNXEmbeddingBackend:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._lock = RLock()
        self._tokenizer = None
        self._session = None

    def _load_tokenizer(self):
        if self._tokenizer is not None:
            return

        try:
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError:
            raise ModelUnavailable("Instale o extra opcional embeddings para carregar o modelo.") from None

        try:
            names = ("tokenizer.json", "config.json", "sentence_bert_config.json", "1_Pooling/config.json")
            files = {
                name: hf_hub_download(
                    self.config.model_id, name, revision=self.config.revision,
                    cache_dir=self.config.cache_dir, local_files_only=not self.config.allow_download,
                    token=False,
                ) for name in names
            }
        except Exception:
            raise ModelUnavailable("Arquivos do modelo indisponíveis; prepare o cache local explicitamente.") from None

        try:
            with open(files["config.json"]) as source:
                model = json.load(source)

            with open(files["sentence_bert_config.json"]) as source:
                sentence = json.load(source)

            with open(files["1_Pooling/config.json"]) as source:
                pooling = json.load(source)

            if model["hidden_size"] != 384 or sentence["max_seq_length"] != 128:
                raise ValueError

            if not pooling["pooling_mode_mean_tokens"] or any(pooling.get(key, False) for key in (
                "pooling_mode_cls_token", "pooling_mode_max_tokens", "pooling_mode_mean_sqrt_len_tokens",
            )):
                raise ValueError

            tokenizer = Tokenizer.from_file(files["tokenizer.json"])
            tokenizer.no_truncation()
            tokenizer.enable_padding(pad_id=model["pad_token_id"], pad_token="<pad>")
            self._tokenizer = tokenizer
        except (OSError, ValueError, KeyError, MemoryError):
            raise EmbeddingModelError("Configuração ou tokenizer do modelo inválido ou sem recursos.") from None

    def _load_session(self):
        if self._session is not None:
            return

        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ModelUnavailable("Instale o extra opcional embeddings para inferência CPU.") from None

        try:
            path = hf_hub_download(
                self.config.model_id, "onnx/model.onnx", revision=self.config.revision,
                cache_dir=self.config.cache_dir, local_files_only=not self.config.allow_download,
                token=False,
            )
        except Exception:
            raise ModelUnavailable("Pesos ONNX indisponíveis; prepare o cache local explicitamente.") from None

        try:
            ort.disable_telemetry_events()
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.config.cpu_threads
            options.inter_op_num_threads = 1
            options.log_severity_level = 3
            self._session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
        except Exception:
            raise EmbeddingModelError("Falha ao carregar o modelo ONNX na CPU; verifique memória e arquivos.") from None

    def token_counts(self, texts: Sequence[str]) -> list[int]:
        with self._lock:
            self._load_tokenizer()

            try:
                return [sum(item.attention_mask) for item in self._tokenizer.encode_batch(list(texts))]
            except Exception:
                raise EmbeddingModelError("Falha ao tokenizar os textos localmente.") from None

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        with self._lock:
            self._load_tokenizer()
            self._load_session()

            try:
                import numpy as np

                encoded = self._tokenizer.encode_batch(list(texts))

                if any(sum(item.attention_mask) > self.config.max_tokens for item in encoded):
                    raise ValueError

                values = {
                    "input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
                    "attention_mask": np.asarray([item.attention_mask for item in encoded], dtype=np.int64),
                    "token_type_ids": np.asarray([item.type_ids for item in encoded], dtype=np.int64),
                }
                inputs = {node.name: values[node.name] for node in self._session.get_inputs()}
                hidden = self._session.run(None, inputs)[0]

                if hidden.shape != (*values["input_ids"].shape, self.config.dimension):
                    raise ValueError

                mask = values["attention_mask"][:, :, None].astype(np.float32)
                pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1e-9)

                return pooled.tolist()
            except Exception:
                raise EmbeddingModelError("Falha na inferência ONNX; verifique entradas e recursos disponíveis.") from None
