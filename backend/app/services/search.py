from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.ai.embeddings.service import EmbeddingService
from app.db.models import ChunkEmbedding, Document, DocumentChunk
from app.schemas.retrieval import SearchHit, SearchRequest
from app.services.indexing import embedding_key

CONTEXT_NEIGHBOR_RADIUS = 2
MAX_CONTEXT_CHUNKS = 30


class SearchService:
    def __init__(self, embeddings: EmbeddingService):
        self.embeddings = embeddings

    def search(self, request: SearchRequest, owner_id: UUID, session: Session) -> list[SearchHit]:
        query = self.embeddings.embed_query(request.question)
        distance = ChunkEmbedding.embedding.cosine_distance(list(query.vector))
        statement = (
            select(DocumentChunk, Document.filename, distance.label('distance'))
            .join(ChunkEmbedding, ChunkEmbedding.chunk_id == DocumentChunk.id)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.owner_id == owner_id, Document.status == 'ready',
                   ChunkEmbedding.model_id == embedding_key(self.embeddings))
            .order_by(distance, DocumentChunk.id).limit(request.top_k)
        )

        if request.document_id is not None:
            statement = statement.where(Document.id == request.document_id)

        with session.begin():
            rows = session.execute(statement).all()

            return [SearchHit(
                chunk_id=chunk.id, document_id=chunk.document_id, filename=filename,
                page_start=chunk.page_start, page_end=chunk.page_end, text=chunk.text,
                score=max(-1.0, min(1.0, 1.0 - float(value))),
            ) for chunk, filename, value in rows]

    def expand_context(
        self, request: SearchRequest, owner_id: UUID, session: Session,
        hits: list[SearchHit], *, max_characters: int,
    ) -> list[SearchHit]:
        if not hits:
            return []

        authorized = (
            select(DocumentChunk, Document.filename)
            .join(Document, Document.id == DocumentChunk.document_id)
            .join(ChunkEmbedding, ChunkEmbedding.chunk_id == DocumentChunk.id)
            .where(Document.owner_id == owner_id, Document.status == 'ready',
                   ChunkEmbedding.model_id == embedding_key(self.embeddings))
        )

        if request.document_id is not None:
            authorized = authorized.where(Document.id == request.document_id)

        with session.begin():
            seeds = session.execute(authorized.where(DocumentChunk.id.in_([hit.chunk_id for hit in hits]))).all()

            if not seeds:
                return []

            windows = [and_(
                DocumentChunk.document_id == chunk.document_id,
                DocumentChunk.chunk_index.between(chunk.chunk_index - CONTEXT_NEIGHBOR_RADIUS,
                                                  chunk.chunk_index + CONTEXT_NEIGHBOR_RADIUS),
                DocumentChunk.page_start <= chunk.page_end + 1,
                DocumentChunk.page_end >= chunk.page_start - 1,
            ) for chunk, _ in seeds]
            neighbors = session.execute(authorized.where(or_(*windows)).order_by(
                DocumentChunk.document_id, DocumentChunk.chunk_index)).all()

            seed_by_id = {chunk.id: chunk for chunk, _ in seeds}
            included = {}
            remaining = max_characters

            def include(chunk, filename, score):
                nonlocal remaining
                if chunk.id in included or len(included) >= MAX_CONTEXT_CHUNKS or len(chunk.text) > remaining:
                    return
                included[chunk.id] = SearchHit(
                    chunk_id=chunk.id, document_id=chunk.document_id, filename=filename,
                    page_start=chunk.page_start, page_end=chunk.page_end, text=chunk.text, score=score,
                )
                remaining -= len(chunk.text)

            filenames = {chunk.id: filename for chunk, filename in seeds}
            for hit in hits:
                if hit.chunk_id in seed_by_id:
                    include(seed_by_id[hit.chunk_id], filenames[hit.chunk_id], hit.score)

            for hit in hits:
                seed = seed_by_id.get(hit.chunk_id)
                if seed is None or seed.id not in included:
                    continue

                nearby = [(chunk, filename) for chunk, filename in neighbors
                          if chunk.document_id == seed.document_id
                          and abs(chunk.chunk_index - seed.chunk_index) <= CONTEXT_NEIGHBOR_RADIUS
                          and chunk.page_start <= seed.page_end + 1 and chunk.page_end >= seed.page_start - 1]
                nearby.sort(key=lambda row: (abs(row[0].chunk_index - seed.chunk_index), row[0].chunk_index))
                for chunk, filename in nearby:
                    include(chunk, filename, 0.0)

            positions = {chunk.id: (chunk.document_id, chunk.chunk_index) for chunk, _ in neighbors}
            result, emitted = [], set()
            for hit in hits:
                seed = seed_by_id.get(hit.chunk_id)
                if seed is None:
                    continue
                window = [value for chunk_id, value in included.items()
                          if positions[chunk_id][0] == seed.document_id
                          and abs(positions[chunk_id][1] - seed.chunk_index) <= CONTEXT_NEIGHBOR_RADIUS]
                for value in sorted(window, key=lambda item: positions[item.chunk_id][1]):
                    if value.chunk_id not in emitted:
                        result.append(value)
                        emitted.add(value.chunk_id)

            return result
