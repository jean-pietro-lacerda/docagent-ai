export type User = { id: string; username: string | null; status: string };
export type Document = { id: string; filename: string; pages: number | null; status: 'pending' | 'extracting' | 'embedding' | 'ready' | 'failed'; created_at: string; file_hash: string | null };
export type Hit = { chunk_id: string; document_id: string; filename: string; page_start: number; page_end: number; text: string; score: number };
export type Citation = Omit<Hit, 'text' | 'score'> & { source_id: number; quote: string };
export type Answer = { answer: string; citations: Citation[] };
export type Question = { question: string; document_id?: string; top_k?: number; conversation_id?: string; message_id?: string };

export type ConversationSummary = { id: string; title: string; document_id: string | null; version: number; updated_at: string };
export type StoredTurn = { id: string; question: string; answer: string; citations: Citation[]; agent: boolean };
export type ConversationDetail = ConversationSummary & { messages: StoredTurn[] };
