from functools import lru_cache
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.routing import APIRoute
from sqlalchemy.exc import SQLAlchemyError

from app.ai.agents.documents import DocumentAgent
from app.ai.embeddings.backend import EmbeddingModelError
from app.ai.embeddings.service import EmbeddingInputError, EmbeddingService
from app.api.routes.documents import CurrentUser, DocumentSession, get_document_service
from app.core.config import get_settings
from app.schemas.conversations import AskRequest, ConversationCreate, ConversationDetail, ConversationSummary
from app.schemas.embeddings import EmbeddingConfig
from app.schemas.retrieval import AskResponse, IndexResponse, SearchRequest, SearchResponse
from app.services.chunking import ChunkingError
from app.services.conversations import ConversationService
from app.services.documents import DocumentNotFound, DocumentPersistenceError
from app.services.indexing import IndexConflict, IndexInputError, IndexingService
from app.services.pdf_extraction import PDFExtractionError
from app.services.rag import GenerationRateLimit, GenerationUnavailable, GroqGenerator, RAGService
from app.services.search import SearchService


class RetrievalRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request: Request):
            body = bytearray()
            async for part in request.stream():
                body.extend(part)

                if len(body) > 16_384:
                    raise HTTPException(413, 'Corpo da requisição excede o limite.')
            request._body = bytes(body)

            try:
                response = await original(request)
            except DocumentNotFound:
                raise HTTPException(404, 'Documento não encontrado.') from None
            except IndexConflict as exc:
                raise HTTPException(409, str(exc)) from None
            except (IndexInputError, EmbeddingInputError, ChunkingError, PDFExtractionError) as exc:
                raise HTTPException(422, str(exc)) from None
            except GenerationRateLimit:
                raise HTTPException(429, 'Limite temporário de geração atingido.', headers={'Retry-After': '60'}) from None
            except (EmbeddingModelError, GenerationUnavailable, DocumentPersistenceError):
                raise HTTPException(503, 'Serviço indisponível; verifique a configuração local.') from None
            except SQLAlchemyError as exc:
                original_error = getattr(exc, 'orig', None)
                sqlstate = getattr(original_error, 'sqlstate', None)

                if sqlstate == '55P03':
                    raise HTTPException(409, 'Documento ocupado; tente novamente.') from None

                raise HTTPException(503, 'Persistência temporariamente indisponível.') from None

            response.headers['Cache-Control'] = 'no-store'
            response.headers['X-Content-Type-Options'] = 'nosniff'

            return response

        return handler


router = APIRouter(prefix='/api/v1', tags=['retrieval'], route_class=RetrievalRoute)


@lru_cache
def get_embeddings() -> EmbeddingService:
    config = get_settings()

    return EmbeddingService(EmbeddingConfig(cache_dir=config.embeddings_cache_dir, allow_download=False))


def get_indexer() -> IndexingService:
    return IndexingService(get_document_service(), get_embeddings())


def get_search() -> SearchService:
    return SearchService(get_embeddings())


def get_rag() -> RAGService:
    config = get_settings()

    return RAGService(get_search(), GroqGenerator(config), config)


def get_agent(rag: Annotated[RAGService, Depends(get_rag)]) -> DocumentAgent:
    return DocumentAgent(rag)


def get_conversations() -> ConversationService:
    return ConversationService()


Conversations = Annotated[ConversationService, Depends(get_conversations)]


@router.post('/conversations', response_model=ConversationDetail, status_code=201)
def create_conversation(
    body: ConversationCreate,
    user: CurrentUser,
    session: DocumentSession,
    service: Conversations,
):
    return service.create(body, user.id, session)


@router.get('/conversations', response_model=list[ConversationSummary])
def list_conversations(
    user: CurrentUser,
    session: DocumentSession,
    service: Conversations,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    return service.list(user.id, session, offset)


@router.get('/conversations/{conversation_id}', response_model=ConversationDetail)
def get_conversation(
    conversation_id: UUID,
    user: CurrentUser,
    session: DocumentSession,
    service: Conversations,
):
    return service.get(conversation_id, user.id, session)


@router.post('/documents/{document_id}/index', response_model=IndexResponse)
def index_document(
    document_id: UUID,
    user: CurrentUser,
    session: DocumentSession,
    service: Annotated[IndexingService, Depends(get_indexer)],
):
    return service.index(document_id, user.id, session)


@router.post('/search', response_model=SearchResponse)
def search_documents(
    body: SearchRequest,
    user: CurrentUser,
    session: DocumentSession,
    service: Annotated[SearchService, Depends(get_search)],
):
    return SearchResponse(items=service.search(body, user.id, session))


@router.post('/ask', response_model=AskResponse)
def ask_documents(
    body: AskRequest,
    user: CurrentUser,
    session: DocumentSession,
    service: Annotated[RAGService, Depends(get_rag)],
    conversations: Conversations,
):
    if body.conversation_id is not None:
        return conversations.ask(body, user.id, session, service)

    return service.ask(body, user.id, session)


@router.post('/agent/ask', response_model=AskResponse)
def ask_agent(
    body: AskRequest,
    user: CurrentUser,
    session: DocumentSession,
    service: Annotated[DocumentAgent, Depends(get_agent)],
    conversations: Conversations,
):
    if body.conversation_id is not None:
        return conversations.ask(body, user.id, session, service)

    return service.ask(body, user.id, session)
