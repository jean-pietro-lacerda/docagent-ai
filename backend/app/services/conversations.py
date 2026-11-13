import json
from dataclasses import dataclass
from uuid import UUID, uuid4

from sqlalchemy import select

from app.db.models import Conversation, Document
from app.schemas.conversations import ConversationDetail, ConversationSummary
from app.schemas.retrieval import AskResponse
from app.services.documents import DocumentNotFound
from app.services.indexing import IndexConflict


@dataclass(frozen=True)
class ConversationInput:
    question: str
    document_id: UUID | None
    top_k: int
    history: tuple[dict, ...]


class ConversationService:
    @staticmethod
    def owned(conversation_id, owner_id):
        return select(Conversation).where(Conversation.id == conversation_id, Conversation.owner_id == owner_id)

    def create(self, body, owner_id, session):
        with session.begin():
            if body.document_id is not None and session.scalar(
                select(Document.id).where(
                    Document.id == body.document_id,
                    Document.owner_id == owner_id,
                )
            ) is None:
                raise DocumentNotFound()

            conversation = Conversation(owner_id=owner_id, document_id=body.document_id)
            session.add(conversation)
            session.flush()

            return ConversationDetail.model_validate(conversation)

    def list(self, owner_id, session, offset=0):
        with session.begin():
            statement = (
                select(Conversation)
                .where(Conversation.owner_id == owner_id)
                .order_by(Conversation.updated_at.desc(), Conversation.id)
                .offset(offset)
                .limit(20)
            )

            return [
                ConversationSummary.model_validate(conversation)
                for conversation in session.scalars(statement)
            ]

    def get(self, conversation_id, owner_id, session):
        with session.begin():
            conversation = session.scalar(self.owned(conversation_id, owner_id))

            if conversation is None:
                raise DocumentNotFound()

            return ConversationDetail.model_validate(conversation)

    def ask(self, body, owner_id, session, responder):
        conversation_snapshot = self.get(body.conversation_id, owner_id, session)

        if body.document_id is not None and body.document_id != conversation_snapshot.document_id:
            raise IndexConflict('O documento da conversa não pode ser alterado. Inicie uma nova conversa.')

        if body.message_id is not None:
            for message in conversation_snapshot.messages:
                if message['id'] == str(body.message_id):
                    if message['question'] != body.question:
                        raise IndexConflict('Identificador de mensagem já utilizado.')

                    return AskResponse(answer=message['answer'], citations=message['citations'])

        history, remaining_history_bytes = [], 3000

        for message in reversed(conversation_snapshot.messages[-3:]):
            history_entry = {'question': message['question'][:500], 'answer': message['answer'][:500]}
            entry_bytes = len(json.dumps(history_entry, ensure_ascii=False).encode('utf-8'))

            if entry_bytes <= remaining_history_bytes:
                history.insert(0, history_entry)
                remaining_history_bytes -= entry_bytes

        contextual_request = ConversationInput(body.question, conversation_snapshot.document_id, body.top_k, tuple(history))
        response = responder.ask(contextual_request, owner_id, session)

        with session.begin():
            conversation = session.scalar(self.owned(body.conversation_id, owner_id).with_for_update(nowait=True))

            if conversation is None:
                raise DocumentNotFound()

            if conversation.version != conversation_snapshot.version:
                raise IndexConflict('A conversa foi atualizada. Recarregue o histórico antes de continuar.')

            conversation.messages = (
                conversation.messages + [{
                    'id': str(body.message_id or uuid4()),
                    'question': body.question,
                    'answer': response.answer,
                    'citations': [citation.model_dump(mode='json') for citation in response.citations],
                    'agent': responder.__class__.__name__ == 'DocumentAgent',
                }]
            )[-40:]

            if conversation.version == 0:
                conversation.title = body.question[:80]

            conversation.version += 1

        return response
