import hashlib
from uuid import UUID, uuid4

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.ai.embeddings.service import EmbeddingService
from app.db.models import ChunkEmbedding, DocumentChunk
from app.schemas.retrieval import IndexResponse
from app.services.documents import DocumentService, DocumentNotFound
from app.services.pdf_extraction import PDFTextExtractor
from app.services.token_chunking import TokenAwareChunker


class IndexConflict(ValueError):
    pass


class IndexInputError(ValueError):
    pass


def embedding_key(embeddings: EmbeddingService) -> str:
    return f'{embeddings.config.model_id}@{embeddings.config.revision}'


def fingerprint(document):
    return (document.owner_id, document.object_key, document.file_hash, document.updated_at, document.status)


class IndexingService:
    def __init__(self, documents: DocumentService, embeddings: EmbeddingService):
        self.documents = documents
        self.embeddings = embeddings

    def index(self, document_id: UUID, owner_id: UUID, session: Session) -> IndexResponse:
        if session.in_transaction():
            raise IndexConflict('A indexação requer sessão sem transação ativa.')

        with session.begin():
            document = self.documents.get(document_id, owner_id, session)
            original_fingerprint = fingerprint(document)
            session.expunge(document)

        if document.file_hash is None:
            raise IndexInputError('O documento não possui hash de integridade.')

        extracted = PDFTextExtractor(self.documents.storage).extract(document)

        if any(page.extraction_status == 'failed' for page in extracted.pages):
            raise IndexInputError('A extração contém páginas com falha.')

        chunks = TokenAwareChunker(self.embeddings.backend).chunk(extracted)

        if not chunks.chunks:
            raise IndexInputError('O documento não contém texto indexável.')

        embedding_result = self.embeddings.embed_chunks(chunks)

        if len(embedding_result.embeddings) != len(chunks.chunks):
            raise IndexInputError('Resultado de embeddings incompleto.')

        model_key = embedding_key(self.embeddings)

        with session.begin():
            current_document = session.scalar(self.documents._owned(document_id, owner_id).with_for_update(nowait=True))

            if current_document is None:
                raise DocumentNotFound('Documento não encontrado.')

            if fingerprint(current_document) != original_fingerprint:
                raise IndexConflict('Documento alterado durante a indexação; tente novamente.')

            session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))

            for chunk, vector in zip(chunks.chunks, embedding_result.embeddings):
                if (vector.document_id, vector.chunk_index, vector.chunk_hash) != (document_id, chunk.chunk_index, chunk.hash):
                    raise IndexInputError('Associação de embedding inválida.')

                if chunk.hash != hashlib.sha256(chunk.text.encode()).hexdigest():
                    raise IndexInputError('Hash do chunk inválido.')

                chunk_id = uuid4()
                session.add(DocumentChunk(
                    id=chunk_id, document_id=document_id, chunk_index=chunk.chunk_index,
                    page_start=chunk.page_start, page_end=chunk.page_end, section=chunk.section,
                    text=chunk.text, hash=chunk.hash,
                ))
                session.flush()
                session.add(ChunkEmbedding(chunk_id=chunk_id, model_id=model_key, embedding=list(vector.vector)))

            current_document.pages = extracted.total_pages
            current_document.status = 'ready'
            session.flush()

        return IndexResponse(document_id=document_id, chunks=len(chunks.chunks), model_id=model_key)
