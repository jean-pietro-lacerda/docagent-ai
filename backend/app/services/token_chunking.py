import re
from bisect import bisect_left, bisect_right
from dataclasses import replace

from app.ai.embeddings.backend import ONNXEmbeddingBackend
from app.schemas.chunking import ChunkingResult, TokenAwareConfig
from app.schemas.extraction import PDFExtractionResult
from app.services.chunking import DocumentChunker, ChunkingError, PARAGRAPH, SENTENCE, WORD

LIST_ITEM = re.compile(r'(?m)^[ \t]*(\d{1,3})\s*[.)–—-]\s+')


class TokenAwareChunker(DocumentChunker):
    """Share the backend with EmbeddingService; chunking loads only its tokenizer."""

    def __init__(self, backend: ONNXEmbeddingBackend, config: TokenAwareConfig | None = None):
        super().__init__(config if config is not None else TokenAwareConfig())
        self.backend = backend
        self.limit = backend.config.max_tokens - self.config.safety_margin

    def _counts(self, texts):
        values = self.backend.token_counts(texts)

        if len(values) != len(texts) or any(type(value) is not int or value < 1 for value in values):
            raise ChunkingError("Contagem de tokens inválida.")

        return values

    def chunk(self, extraction: PDFExtractionResult) -> ChunkingResult:
        result = super().chunk(extraction)
        chunks = []

        for start in range(0, len(result.chunks), self.backend.config.batch_size):
            batch = result.chunks[start:start + self.backend.config.batch_size]
            counts = self._counts([chunk.text for chunk in batch])

            for chunk, count in zip(batch, counts):
                if count > self.limit:
                    raise ChunkingError("Chunk incompatível com o encoder; nenhum texto foi truncado.")

                chunks.append(replace(chunk, token_count=count))

        return replace(result, chunks=tuple(chunks), tokenizer_id=self.backend.config.model_id,
                       tokenizer_revision=self.backend.config.revision)

    def _split(self, text):
        protected_spans = self._protected_list_spans(text)
        word_boundaries = [match.end() for match in WORD.finditer(text)]
        boundaries = [
            [match.end() for match in PARAGRAPH.finditer(text)],
            sorted({position for span in protected_spans for position in span}),
            [match.end() for match in SENTENCE.finditer(text)],
            word_boundaries,
        ]

        def safe_boundary(position):
            return not any(start < position < end for start, end in protected_spans)

        start = previous_end = 0

        while start < len(text):
            cap = min(len(text), start + self.config.max_characters)
            cache = {}

            def count(end):
                if end not in cache:
                    cache[end] = self._counts([text[start:end]])[0]

                return cache[end]

            end = None

            if cap == len(text) and count(cap) <= self.limit:
                end = cap
            else:
                for positions in boundaries:
                    candidates = positions[bisect_right(positions, previous_end):bisect_right(positions, cap)]

                    for candidate in reversed(candidates):
                        if candidate > start and safe_boundary(candidate) and count(candidate) <= self.limit:
                            end = candidate
                            break

                    if end is not None:
                        break

            if end is None:
                low, high = start, cap

                while low < high:
                    middle = (low + high + 1) // 2

                    if count(middle) <= self.limit:
                        low = middle
                    else:
                        high = middle - 1

                end = low

                for protected_start, protected_end in protected_spans:
                    if protected_start < end < protected_end:
                        end = protected_start
                        break

            if end <= previous_end and start < previous_end:
                start = previous_end
                continue

            if end <= start or count(end) > self.limit or not text[start:end].strip():
                raise ChunkingError("Não foi possível dividir o trecho sem perda de conteúdo.")

            yield text[start:end]

            if end == len(text):
                break

            previous_end = end
            target = max(start + 1, end - self.config.overlap)
            index = bisect_left(word_boundaries, target)

            if index < len(word_boundaries) and word_boundaries[index] <= end:
                target = word_boundaries[index]
            elif self.config.overlap:
                target = end

            for protected_start, protected_end in protected_spans:
                if protected_start < target < protected_end:
                    target = protected_end
                    break

            start = target

    def _protected_list_spans(self, text):
        matches = list(LIST_ITEM.finditer(text))
        groups = []

        for index, match in enumerate(matches):
            next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            paragraph = PARAGRAPH.search(text, match.end())
            end = min(next_start, paragraph.end() if paragraph else len(text))
            item = (match.start(), end)

            if index and int(match[1]) == int(matches[index - 1][1]) + 1:
                groups[-1].append(item)
            else:
                groups.append([item])

        protected = []

        def fits(span):
            start, end = span
            return end - start <= self.config.max_characters and self._counts([text[start:end]])[0] <= self.limit

        for items in groups:
            whole_list = (items[0][0], items[-1][1])

            if fits(whole_list):
                protected.append(whole_list)
            else:
                protected.extend(item for item in items if fits(item))

        return protected
