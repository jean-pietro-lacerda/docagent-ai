import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context

from app.schemas.retrieval import AskResponse, SearchHit
from app.services.rag import NO_ANSWER, addresses_question, source_quote


class AgentState(TypedDict, total=False):
    hits: list[SearchHit]
    response: AskResponse
    tool_calls: int


class DocumentAgent:
    def __init__(self, rag):
        self.rag = rag

    def ask(self, request, owner_id, session):
        def authorized_search(state):
            MAX_TOOL_CALLS = 3
            if state['tool_calls'] >= MAX_TOOL_CALLS:
                raise RuntimeError('Limite de buscas do agente excedido.')

            hits = self.rag.retrieve(request, owner_id, session)

            return {'hits': hits, 'tool_calls': state['tool_calls'] + 1}

        def generate(state):
            return {'response': self.rag.answer_hits(request, state['hits'])}

        def verify(state):
            response = self._verify_response(request, state['hits'], state['response'])

            return {'response': response}

        graph = StateGraph(AgentState)
        graph.add_node('authorized_search', authorized_search)
        graph.add_node('generate', generate)
        graph.add_node('verify', verify)
        graph.add_edge(START, 'authorized_search')
        graph.add_edge('authorized_search', 'generate')
        graph.add_edge('generate', 'verify')
        graph.add_edge('verify', END)

        with tracing_context(enabled=False):
            result = graph.compile().invoke({'tool_calls': 0}, config={'recursion_limit': 4})

        return result['response']

    @staticmethod
    def _verify_response(request, retrieved_hits: list[SearchHit], response: AskResponse) -> AskResponse:
        hits_by_id = {hit.chunk_id: hit for hit in retrieved_hits}
        valid_citations = []

        for citation in response.citations:
            hit = hits_by_id.get(citation.chunk_id)

            if (
                hit is None
                or citation.document_id != hit.document_id
                or (request.document_id is not None and citation.document_id != request.document_id)
                or (citation.page_start, citation.page_end) != (hit.page_start, hit.page_end)
                or citation.filename != hit.filename
                or source_quote(hit.text, citation.quote) is None
            ):
                continue

            valid_citations.append(citation)

        if not valid_citations:
            return AskResponse(answer=NO_ANSWER, citations=[])

        available_source_ids = {citation.source_id for citation in valid_citations}
        supported_paragraphs = []

        for paragraph in response.answer.split('\n\n'):
            paragraph_source_ids = set(map(int, re.findall(r'\[(\d+)\]', paragraph)))

            if (paragraph_source_ids and paragraph_source_ids <= available_source_ids
                    and addresses_question(request.question, paragraph)):
                supported_paragraphs.append(paragraph)

        if not supported_paragraphs:
            return AskResponse(answer=NO_ANSWER, citations=[])

        answer = '\n\n'.join(supported_paragraphs)
        used_source_ids = set(map(int, re.findall(r'\[(\d+)\]', answer)))

        return AskResponse(
            answer=answer,
            citations=[citation for citation in valid_citations if citation.source_id in used_source_ids],
        )
