import { useEffect, useRef, useState } from 'react';
import { api, clearSession, errorMessage } from './api';
import type { Document, User } from './types';
import { Icon } from './components/Icon';
import type { IconName } from './components/Icon';
import { Login } from './components/Login';
import { Library } from './components/Library';
import { Conversation } from './components/Conversation';
import { PDFViewer } from './components/PDFViewer';
type Tab = 'library' | 'chat' | 'search';
function Workspace({ user }: { user: User }) {
  const [tab, setTab] = useState<Tab>('library');
  const [documents, setDocuments] = useState<Document[]>([]);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [preview, setPreview] = useState<{ url: string; filename: string; page: number } | null>(null);
  const downloads = useRef(new Set<string>());
  const revision = useRef(0);
  useEffect(() => { void refresh(0); return () => { revision.current++; downloads.current.forEach(url => URL.revokeObjectURL(url)); }; }, []);
  async function refresh(nextOffset = offset) {
    const current = ++revision.current;
    setLoading(true); setError('');
    try { const response = await api.list(nextOffset); if (revision.current === current) { setDocuments(response.items); setOffset(nextOffset); } }
    catch (error) { if (revision.current === current) setError(errorMessage(error)); }
    finally { if (revision.current === current) setLoading(false); }
  }
  async function open(id: string, filename: string, page = 1) {
    const blob = await api.file(id);
    const url = URL.createObjectURL(new Blob([blob], { type: 'application/pdf' }));
    setPreview({ url, filename, page });
  }
  async function download(doc: Document) {
    const blob = await api.file(doc.id);
    const url = URL.createObjectURL(blob); downloads.current.add(url);
    const anchor = window.document.createElement('a'); anchor.href = url;
    anchor.download = doc.filename.split(/[\\/]/).pop() || 'document.pdf'; anchor.click();
    setTimeout(() => { URL.revokeObjectURL(url); downloads.current.delete(url); }, 30000);
  }
  const links: { id: Tab; label: string; icon: IconName }[] = [{ id: 'library', label: 'Biblioteca', icon: 'grid' }, { id: 'chat', label: 'Conversar', icon: 'chat' }, { id: 'search', label: 'Busca semântica', icon: 'search' }];
  return <div className="app-layout"><a className="skip-link" href="#content">Pular para o conteúdo</a><aside className="sidebar"><div className="brand"><span className="brand-symbol"><Icon name="book" /></span>DocAgent <span className="ai-tag">AI</span></div><span className="sidebar-section">SEU WORKSPACE</span><nav aria-label="Navegação principal">{links.map(link => <button key={link.id} className={tab === link.id ? 'active' : ''} aria-current={tab === link.id ? 'page' : undefined} onClick={() => setTab(link.id)}><Icon name={link.icon} size={19} />{link.label}{tab === link.id && <span className="nav-dot" />}</button>)}</nav><div className="sidebar-note"><Icon name="lock" size={18} /><strong>Um espaço só seu.</strong><p>Seus documentos são privados. Ao sair, o acesso a este workspace é encerrado.</p></div><div className="user-panel"><span className="avatar">{(user.username || 'U').slice(0, 2).toUpperCase()}</span><div><strong>{user.username || 'Minha conta'}</strong><span>Sessão autenticada</span></div><button aria-label="Sair da conta" className="logout" onClick={() => clearSession()}><Icon name="exit" size={18} /></button></div></aside>
  <div className="workspace"><header className="topbar"><span>Workspace <span className="breadcrumb-separator">/</span> <strong>{links.find(l => l.id === tab)?.label}</strong></span><span className="private-indicator"><span />Acesso privado</span></header><main id="content" className="main-content">{error && <div className="error" role="alert">{error}<button className="quiet" onClick={() => { void refresh(); }}>Tentar novamente</button></div>}
  <div hidden={tab !== 'library'}><Library documents={documents} loading={loading} offset={offset} refresh={refresh} open={open} download={download} /></div>
  <div hidden={tab !== 'chat'}><Conversation documents={documents} open={open} /></div><div hidden={tab !== 'search'}><Conversation documents={documents} open={open} semantic /></div>
  <footer className="workspace-foot"><span>DocAgent AI</span><span>Conhecimento com origem. Respostas com contexto.</span></footer></main></div>
  {preview && <PDFViewer key={preview.url} {...preview} onClose={() => setPreview(null)} />}</div>;
}
export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [notice, setNotice] = useState('');
  useEffect(() => {
    function ended(event: Event) { setUser(null); setNotice((event as CustomEvent).detail.expired ? 'Sua sessão expirou. Entre novamente para continuar.' : 'Você saiu da sua conta com segurança.'); }
    window.addEventListener('docagent-session-ended', ended);
    return () => window.removeEventListener('docagent-session-ended', ended);
  }, []);
  return user ? <Workspace key={user.id} user={user} /> : <Login notice={notice} onLogin={value => { setNotice(''); setUser(value); }} />;
}
