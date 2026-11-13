import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import ReactMarkdown from 'react-markdown';
import { api, ApiError, errorMessage } from '../api';
import type { Answer, Citation, Document, Hit, ConversationSummary } from '../types';
import { Icon } from './Icon';
function Scope({ documents, value, onChange }: { documents: Document[]; value: string; onChange: (value: string) => void }) {
  return <label className="scope-label"><Icon name="book" size={16} /><span className="sr-only">Documento para consulta</span><select aria-label="Documento para consulta" value={value} onChange={e => onChange(e.target.value)}><option value="">Todos os meus documentos</option>{value && !documents.some(d => d.id === value) && <option value={value}>Documento desta conversa</option>}{documents.map(doc => <option key={doc.id} value={doc.id}>{doc.filename}</option>)}</select></label>;
}
function Sources({ citations, open }: { citations: Citation[]; open: (id: string, name: string, page?: number) => Promise<void> }) {
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  return <div className="sources"><span className="eyebrow">FONTES VERIFICÁVEIS</span><div className="source-list">{citations.map((source, index) => <details key={`${source.chunk_id}-${index}`}><summary><span className="source-number">{source.source_id}</span><span>{source.filename}<small>Página {source.page_start}{source.page_end !== source.page_start ? `–${source.page_end}` : ''}</small></span></summary><blockquote>{source.quote}</blockquote><button className="quiet" disabled={busy} onClick={async () => { setBusy(true); setError(''); try { await open(source.document_id, source.filename, source.page_start); } catch (error) { setError(errorMessage(error)); } finally { setBusy(false); } }}>{busy ? 'Abrindo…' : 'Abrir página no PDF'} <Icon name="arrow" size={14} /></button></details>)}</div>{error && <p className="error" role="alert">{error}</p>}</div>;
}
type Message = { id: string; question: string; scope: string; agent: boolean; response?: Answer; error?: string };
export function Conversation({ documents, open, semantic = false }: { documents: Document[]; open: (id: string, name: string, page?: number) => Promise<void>; semantic?: boolean }) {
  const [question, setQuestion] = useState('');
  const [document, setDocument] = useState('');
  const [agent, setAgent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversationId, setConversationId] = useState<string>();
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [historyNotice, setHistoryNotice] = useState('');
  const [historyBusy, setHistoryBusy] = useState(false);
  const sequence = useRef(0);
  useEffect(() => {
    let active = true;
    if (!semantic) api.conversations().then(items => { if (active) setConversations(items); }).catch(() => { if (active) setHistoryNotice('Histórico indisponível. Verifique se a migração de conversas foi aplicada.'); });
    return () => { active = false; sequence.current++; };
  }, [semantic]);
  function newConversation(scope = document) {
    if (lock.current) return;
    sequence.current++; setConversationId(undefined); setMessages([]); setDocument(scope); setError('');
  }
  async function loadConversation(id: string) {
    if (lock.current || !id) return;
    lock.current = true; setHistoryBusy(true); setError('');
    const current = ++sequence.current;
    try {
      const detail = await api.conversation(id);
      if (sequence.current !== current) return;
      setConversationId(detail.id); setDocument(detail.document_id || '');
      setMessages(detail.messages.map(turn => ({ id: turn.id, question: turn.question, agent: turn.agent,
        scope: detail.document_id ? 'Documento desta conversa' : 'Todos os documentos', response: { answer: turn.answer, citations: turn.citations } })));
      setHistoryNotice('');
    } catch (error) { setError(errorMessage(error)); }
    finally { lock.current = false; setHistoryBusy(false); }
  }
  const [hits, setHits] = useState<Hit[] | null>(null);
  const [lastSearch, setLastSearch] = useState('');
  const [topK, setTopK] = useState(5);
  const lock = useRef(false);
  const end = useRef<HTMLDivElement>(null);
  const selected = document;
  async function submit(event: FormEvent) {
    event.preventDefault(); if (lock.current || !question.trim()) return;
    lock.current = true; setBusy(true); setError('');
    const current = sequence.current;
    const body = { question: question.trim(), ...(selected ? { document_id: selected } : {}), top_k: topK };
    const id = crypto.randomUUID();
    if (!semantic) { setMessages(m => [...m, { id, question: body.question, scope: documents.find(d => d.id === selected)?.filename || 'Todos os documentos', agent }]); setQuestion(''); }
    try {
      if (semantic) {
        const response = await api.search(body);
        if (selected && response.items.some(hit => hit.document_id !== selected)) throw new Error('O servidor retornou um documento fora da consulta. Resultado bloqueado.');
        setHits(response.items); setLastSearch(body.question);
      } else {
        let activeConversation = conversationId;
        if (!activeConversation) {
          try {
            const created = await api.createConversation(selected);
            activeConversation = created.id; setConversationId(created.id);
          } catch (error) {
            if (!(error instanceof ApiError) || ![404, 503].includes(error.status)) throw error;
            setHistoryNotice('Esta conversa não está sendo salva. Aplique a migração de conversas para manter o contexto entre perguntas.');
          }
        }
        const response = await api.ask({ ...body, ...(activeConversation ? { conversation_id: activeConversation, message_id: id } : {}) }, agent);
        if (sequence.current !== current) return;
        if (activeConversation) void api.conversations().then(setConversations).catch(() => {});
        if (selected && response.citations.some(source => source.document_id !== selected)) throw new Error('A resposta contém uma fonte fora do documento selecionado. Resultado bloqueado.');
        setMessages(m => m.map(item => item.id === id ? { ...item, response } : item));
      }
    } catch (error) { if (semantic) { setError(errorMessage(error)); setHits(null); } else setMessages(m => m.map(item => item.id === id ? { ...item, error: errorMessage(error) } : item)); }
    finally { lock.current = false; setBusy(false); setTimeout(() => end.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 40); }
  }
  return <><div className="page-heading"><div><span className="eyebrow">{semantic ? 'EXPLORE AS EVIDÊNCIAS' : 'CONVERSE COM SEU CONHECIMENTO'}</span><h1>{semantic ? 'Encontre o trecho certo.' : 'Uma pergunta abre caminhos.'}</h1><p className="muted">{semantic ? 'Recupere passagens por similaridade, sem gerar uma resposta.' : 'Pergunte aos seus documentos e confira de onde vem cada resposta.'}</p></div><span className="section-mark">{semantic ? '03 / BUSCA' : '02 / CONVERSA'}</span></div>
  {!semantic && <div className="history-toolbar"><label>Histórico de conversas<select aria-label="Histórico de conversas" disabled={busy || historyBusy} value={conversationId || ''} onChange={e => { if (e.target.value) void loadConversation(e.target.value); else newConversation(); }}><option value="">Nova conversa</option>{conversationId && !conversations.some(c => c.id === conversationId) && <option value={conversationId}>Conversa atual</option>}{conversations.map(c => <option key={c.id} value={c.id}>{c.title}</option>)}</select></label><button className="quiet" disabled={busy || historyBusy} onClick={() => newConversation()}>Nova conversa</button><span>Últimas 40 trocas por conversa</span></div>}
  {!semantic && historyNotice && <p className="notice" role="status">{historyNotice}</p>}
  <section className={`conversation-card ${semantic ? 'semantic-card' : ''}`}><div className="conversation-toolbar"><fieldset className="scope-fieldset" disabled={busy || historyBusy}><Scope documents={documents} value={selected} onChange={value => semantic ? setDocument(value) : newConversation(value)} /></fieldset>{semantic ? <label className="top-k">Resultados <select aria-label="Quantidade de resultados" value={topK} onChange={e => setTopK(Number(e.target.value))}>{[1, 3, 5, 10].map(k => <option key={k}>{k}</option>)}</select></label> : <label className="agent-toggle"><input type="checkbox" disabled={busy || historyBusy} checked={agent} onChange={e => setAgent(e.target.checked)} /><span className="toggle-track" />Usar agente<Icon name="bolt" size={14} /></label>}</div>
  <div className="conversation-body" aria-live="polite">
  {!semantic && !messages.length && <div className="conversation-empty"><span className="conversation-symbol"><Icon name="chat" size={32} /></span><span className="eyebrow">DO SEU PDF PARA UMA BOA CONVERSA</span><h2>O que vamos descobrir?</h2><p>Comece por uma dúvida específica.<br />As respostas usam os documentos que você indexou.</p><div className="prompt-suggestions">{['Quais são os pontos principais?', 'Quais são os prazos e requisitos?'].map(prompt => <button key={prompt} className="suggestion" onClick={() => setQuestion(prompt)}>{prompt}<Icon name="arrow" size={14} /></button>)}</div></div>}
  {!semantic && messages.map(message => <article className="exchange" key={message.id}><div className="user-message"><span>{message.scope} · {message.agent ? 'Agente' : 'Resposta direta'}</span><p>{message.question}</p></div><div className="assistant-message"><span className="assistant-avatar"><Icon name="book" size={17} /></span><div className="assistant-content"><strong>DocAgent AI</strong>{message.error ? <p className="error" role="alert">{message.error}</p> : message.response ? <><div className="answer-text"><ReactMarkdown skipHtml disallowedElements={['img', 'iframe']} components={{ a: ({ children }) => <span>{children}</span> }}>{message.response.answer}</ReactMarkdown></div>{message.response.citations.length ? <Sources citations={message.response.citations} open={open} /> : <span className="abstention"><Icon name="search" size={14} />Sem evidências suficientes nos documentos consultados</span>}</> : <p className="thinking"><span className="spinner" />Consultando seus documentos…</p>}</div></div></article>)}
  {semantic && hits === null && !error && <div className="conversation-empty"><span className="conversation-symbol"><Icon name="search" size={32} /></span><h2>A informação pode estar em uma linha.</h2><p>Busque um assunto para encontrar os trechos mais próximos.<br />Os scores indicam similaridade, não certeza da resposta.</p></div>}
  {semantic && hits && <div className="search-results"><p className="result-heading">{hits.length} {hits.length === 1 ? 'trecho encontrado' : 'trechos encontrados'} <span>para “{lastSearch}”</span></p>{!hits.length && <div className="empty-state"><Icon name="search" size={30} /><h3>Nenhum trecho encontrado.</h3><p>Confira se o documento foi indexado ou reformule a busca.</p></div>}{hits.map((hit, i) => <article className="search-hit" key={hit.chunk_id}><header><span className="result-rank">{String(i + 1).padStart(2, '0')}</span><div><strong>{hit.filename}</strong><span>Página {hit.page_start}{hit.page_end !== hit.page_start ? `–${hit.page_end}` : ''}</span></div><span className="score">Cosseno {hit.score.toLocaleString('pt-BR', { maximumFractionDigits: 3 })}</span></header><p>{hit.text}</p><button className="quiet" onClick={async () => { try { await open(hit.document_id, hit.filename, hit.page_start); } catch (error) { setError(errorMessage(error)); } }}>Ver no PDF <Icon name="arrow" size={14} /></button></article>)}</div>}
  {error && <p className="error" role="alert">{error}</p>}<div ref={end} /></div>
  <form className="question-form" onSubmit={submit}><div className="question-input"><textarea aria-label={semantic ? 'Texto para busca semântica' : 'Sua pergunta'} placeholder={semantic ? 'Qual informação você está procurando?' : 'Pergunte algo sobre seus documentos…'} value={question} maxLength={1000} rows={2} onChange={e => setQuestion(e.target.value)} disabled={busy || historyBusy} /><button className="send-button" disabled={busy || historyBusy || !question.trim()} aria-label={semantic ? 'Buscar trechos' : 'Enviar pergunta'}>{busy ? <span className="spinner" /> : <Icon name={semantic ? 'search' : 'arrow'} size={20} />}</button></div><div className="question-foot"><span><Icon name="lock" size={12} />Somente seus documentos · Confira sempre as fontes</span><span>{question.length}/1000</span></div></form></section></>;
}
