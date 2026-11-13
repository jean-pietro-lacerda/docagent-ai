import { useEffect, useRef, useState } from 'react';
import type { ChangeEvent } from 'react';
import { api, errorMessage } from '../api';
import type { Document } from '../types';
import { Icon } from './Icon';
export const statusLabels: Record<Document['status'], string> = { pending: 'Não indexado', extracting: 'Extraindo', embedding: 'Indexando', ready: 'Pronto para consultar', failed: 'Falha' };
function DeleteConfirmation({ document, cancel, confirm }: { document: Document; cancel: () => void; confirm: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => { dialog.current?.showModal(); }, []);
  return <dialog ref={dialog} className="confirm-dialog" onCancel={cancel} aria-labelledby="delete-title"><span className="delete-symbol"><Icon name="trash" size={26} /></span><h2 id="delete-title">Excluir este documento?</h2><p>“{document.filename}” e sua indexação serão excluídos. Esta ação não pode ser desfeita.</p><div><button className="quiet" autoFocus onClick={cancel}>Cancelar</button><button className="delete-button" onClick={confirm}>Excluir documento</button></div></dialog>;
}
export function Library({ documents, loading, offset, refresh, open, download }: {
  documents: Document[]; loading: boolean; offset: number; refresh: (offset?: number) => Promise<void>;
  open: (id: string, name: string, page?: number) => Promise<void>; download: (doc: Document) => Promise<void>;
}) {
  const [uploading, setUploading] = useState(false);
  const [busy, setBusy] = useState<Record<string, string>>({});
  const [feedback, setFeedback] = useState('');
  const [error, setError] = useState('');
  const [removing, setRemoving] = useState<Document | null>(null);
  const locks = useRef(new Set<string>());
  async function upload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]; event.target.value = '';
    if (!file || locks.current.has('upload')) return;
    locks.current.add('upload'); setUploading(true); setError(''); setFeedback('');
    try {
      if (!/\.pdf$/i.test(file.name) || file.size > 5 * 1024 * 1024 || file.size === 0 || await file.slice(0, 5).text() !== '%PDF-') throw new Error('Selecione um PDF válido de até 5 MiB.');
      await api.upload(file); setFeedback('PDF enviado. Indexe o documento para começar a consultar.'); await refresh(0);
    } catch (error) { setError(errorMessage(error)); }
    finally { locks.current.delete('upload'); setUploading(false); }
  }
  async function action(doc: Document, operation: string, work: () => Promise<unknown>) {
    if (locks.current.has(doc.id)) return;
    locks.current.add(doc.id); setBusy(b => ({ ...b, [doc.id]: operation })); setError(''); setFeedback('');
    try { await work(); if (operation === 'Indexando') setFeedback(`${doc.filename} está pronto para consultar.`); await refresh(offset); }
    catch (error) { setError(errorMessage(error)); }
    finally { locks.current.delete(doc.id); setBusy(b => { const next = { ...b }; delete next[doc.id]; return next; }); }
  }
  return <>
    <div className="page-heading"><div><span className="eyebrow">SUA BIBLIOTECA</span><h1>Conhecimento começa aqui.</h1><p className="muted">Reúna seus PDFs. Encontre respostas. Volte sempre à fonte.</p></div><span className="section-mark">01 / BIBLIOTECA</span></div>
    <div className="stats"><div><span>Documentos nesta página</span><strong>{documents.length.toString().padStart(2, '0')}<Icon name="book" /></strong></div><div><span>Prontos para consultar</span><strong>{documents.filter(d => d.status === 'ready').length.toString().padStart(2, '0')}<span className="stat-dot" /></strong></div><div><span>Páginas nesta página</span><strong>{documents.reduce((n, d) => n + (d.pages || 0), 0).toString().padStart(2, '0')}<small>de conhecimento</small></strong></div></div>
    <label className={`upload-zone ${uploading ? 'disabled' : ''}`}><span className="upload-icon"><Icon name="upload" size={24} /></span><span><strong>{uploading ? 'Enviando e validando seu PDF…' : 'Adicione uma nova perspectiva.'}</strong><span className="muted">Selecione um PDF · até 5 MiB e 20 páginas</span></span><span className="upload-cta">{uploading ? 'Enviando…' : 'Escolher PDF'} <Icon name="arrow" size={16} /></span><input aria-label="Enviar PDF" type="file" accept=".pdf,application/pdf" onChange={upload} disabled={uploading} /></label>
    {feedback && <p className="notice" role="status">{feedback}</p>}{error && <p className="error" role="alert">{error}</p>}
    <section className="library-card"><div className="card-heading"><div><h2>Seus documentos <span className="count">{documents.length}</span></h2><p className="muted">Só você tem acesso aos arquivos desta biblioteca.</p></div><button className="quiet" onClick={() => { void refresh(offset); }} disabled={loading}>Atualizar</button></div>
    <div className="table-scroll"><table><thead><tr><th>DOCUMENTO</th><th>PÁGINAS</th><th>ADICIONADO EM</th><th>STATUS</th><th><span className="sr-only">Ações</span></th></tr></thead><tbody>
    {documents.map(doc => <tr key={doc.id}><td><div className="file-cell"><span className="pdf-icon">PDF</span><div><strong>{doc.filename}</strong><span>Documento privado</span></div></div></td><td>{doc.pages ?? '—'}</td><td>{new Intl.DateTimeFormat('pt-BR', { dateStyle: 'medium' }).format(new Date(doc.created_at))}</td><td><span className={`status ${busy[doc.id] === 'Indexando' ? 'embedding' : doc.status}`}><i />{busy[doc.id] ? `${busy[doc.id]}…` : statusLabels[doc.status]}</span></td><td><div className="row-actions">
    <button className="index-button" disabled={!!busy[doc.id]} onClick={() => { void action(doc, 'Indexando', () => api.index(doc.id)); }} aria-label={`${doc.status === 'ready' ? 'Reindexar' : 'Indexar'} ${doc.filename}`}><Icon name="bolt" size={15} /><span>{doc.status === 'ready' ? 'Reindexar' : 'Indexar'}</span></button>
    <button className="icon-button" aria-label={`Visualizar ${doc.filename}`} disabled={!!busy[doc.id]} onClick={() => { void action(doc, 'Abrindo', () => open(doc.id, doc.filename)); }}><Icon name="eye" size={18} /></button>
    <button className="icon-button" aria-label={`Baixar ${doc.filename}`} disabled={!!busy[doc.id]} onClick={() => { void action(doc, 'Baixando', () => download(doc)); }}><Icon name="download" size={18} /></button>
    <button className="icon-button danger" aria-label={`Excluir ${doc.filename}`} disabled={!!busy[doc.id]} onClick={() => setRemoving(doc)}><Icon name="trash" size={17} /></button></div></td></tr>)}
    </tbody></table></div>
    {!documents.length && <div className="empty-state"><Icon name="book" size={32} /><h3>{loading ? 'Buscando sua biblioteca…' : 'Espaço para novas descobertas.'}</h3><p>Envie seu primeiro PDF e indexe-o para fazer perguntas.</p></div>}
    <footer className="pagination"><span className="muted">{documents.length ? `${offset + 1}–${offset + documents.length}` : 'Nenhum documento nesta página'} · 20 documentos por página</span><div><button className="quiet" disabled={loading || offset === 0} onClick={() => { void refresh(Math.max(0, offset - 20)); }}>Anterior</button><button className="quiet" disabled={loading || documents.length < 20} onClick={() => { void refresh(offset + 20); }}>Próxima</button></div></footer>
    </section>
    {removing && <DeleteConfirmation document={removing} cancel={() => setRemoving(null)} confirm={() => { const doc = removing; setRemoving(null); void action(doc, 'Excluindo', () => api.remove(doc.id)); }} />}
  </>;
}
