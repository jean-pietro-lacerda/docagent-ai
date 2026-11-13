import type { Answer, Document, Hit, Question, User, ConversationSummary, ConversationDetail } from './types';

const BASE = (import.meta.env.VITE_API_URL ?? 'http://localhost:8000').replace(/\/$/, '');
let token: string | null = null;
let epoch = 0;
let timer: ReturnType<typeof setTimeout> | undefined;
const pending = new Set<AbortController>();

export class ApiError extends Error {
  constructor(message: string, public status = 0) { super(message); }
}

export function clearSession(expired = false) {
  token = null;
  epoch++;
  clearTimeout(timer);
  pending.forEach(controller => controller.abort());
  pending.clear();
  window.dispatchEvent(new CustomEvent('docagent-session-ended', { detail: { expired } }));
}

function setSession(value: string, seconds: number) {
  token = value;
  epoch++;
  clearTimeout(timer);
  timer = setTimeout(() => clearSession(true), seconds * 1000);
}

async function request<T>(path: string, options: RequestInit = {}, auth = true, blob = false, timeout = 30000): Promise<T> {
  const controller = new AbortController();
  const version = epoch;

  if (auth && !token) throw new ApiError('Entre na sua conta para continuar.', 401);

  pending.add(controller);
  const deadline = setTimeout(() => controller.abort(), timeout);
  const headers = new Headers(options.headers);
  if (auth && token) headers.set('Authorization', `Bearer ${token}`);
  if (options.body && !(options.body instanceof FormData)) headers.set('Content-Type', 'application/json');

  try {
    const response = await fetch(BASE + path, { ...options, headers, signal: controller.signal, cache: 'no-store', credentials: 'omit' });
    if (version !== epoch) throw new ApiError('Sessão encerrada.');
    if (!response.ok) {
      if (response.status === 401 && auth) clearSession(true);
      const body = await response.json().catch(() => ({}));
      const defaults: Record<number, string> = { 401: 'Usuário ou senha inválidos.', 409: 'Documento ocupado ou alterado. Tente novamente.', 413: 'O PDF excede o limite de 5 MiB.', 422: 'Revise os dados enviados.', 429: 'Limite temporário atingido. Aguarde e tente novamente.', 503: 'Serviço indisponível. Verifique a configuração do backend.' };
      throw new ApiError(typeof body.detail === 'string' ? body.detail : defaults[response.status] || 'Não foi possível concluir a operação.', response.status);
    }
    if (response.status === 204) return undefined as T;
    const result = blob ? await response.blob() : await response.json();
    if (version !== epoch) throw new ApiError('Sessão encerrada.');
    return result as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error instanceof DOMException && error.name === 'AbortError') throw new ApiError('Operação interrompida ou tempo limite atingido. Atualize a lista antes de tentar novamente.');
    throw new ApiError('Não foi possível conectar ao servidor. Verifique se o backend está disponível.');
  } finally {
    clearTimeout(deadline);
    pending.delete(controller);
  }
}

export const api = {
  authConfig: () => request<{ registration_enabled: boolean }>('/api/v1/auth/config', {}, false),
  register: (username: string, password: string, password_confirmation: string) => request<User>('/api/v1/auth/register', { method: 'POST', body: JSON.stringify({ username, password, password_confirmation }) }, false),
  async login(username: string, password: string): Promise<User> {
    const data = await request<{ access_token: string; expires_in: number }>('/api/v1/auth/login', { method: 'POST', body: JSON.stringify({ username, password }) }, false);
    setSession(data.access_token, data.expires_in);
    try {
      return await request<User>('/api/v1/auth/me');
    } catch (error) {
      clearSession();
      throw error;
    }
  },
  list: (offset = 0) => request<{ items: Document[]; limit: number; offset: number }>(`/api/v1/documents?limit=20&offset=${offset}`),
  upload: (file: File) => {
    const body = new FormData();
    body.append('file', file);

    return request<Document>('/api/v1/documents', { method: 'POST', body }, true, false, 60000);
  },
  index: (id: string) => request<{ chunks: number }>(`/api/v1/documents/${id}/index`, { method: 'POST' }, true, false, 180000),
  remove: (id: string) => request<void>(`/api/v1/documents/${id}`, { method: 'DELETE' }),
  file: (id: string) => request<Blob>(`/api/v1/documents/${id}/file`, {}, true, true),
  ask: (body: Question, agent: boolean) => request<Answer>(agent ? '/api/v1/agent/ask' : '/api/v1/ask', { method: 'POST', body: JSON.stringify(body) }, true, false, 90000),
  conversations: () => request<ConversationSummary[]>('/api/v1/conversations'),
  conversation: (id: string) => request<ConversationDetail>(`/api/v1/conversations/${id}`),
  createConversation: (document_id?: string) => request<ConversationDetail>('/api/v1/conversations', { method: 'POST', body: JSON.stringify({ document_id: document_id || null }) }),
  search: (body: Question) => request<{ items: Hit[] }>('/api/v1/search', { method: 'POST', body: JSON.stringify(body) }, true, false, 60000),
};

export function errorMessage(error: unknown) { return error instanceof Error ? error.message : 'Algo deu errado. Tente novamente.'; }
