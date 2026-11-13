import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { api, errorMessage } from '../api';
import type { User } from '../types';
import { Icon } from './Icon';

export function Login({ onLogin, notice }: { onLogin: (user: User) => void; notice: string }) {
  const [registrationEnabled, setRegistrationEnabled] = useState(false);
  const [registering, setRegistering] = useState(false);
  const [success, setSuccess] = useState('');
  const formRef = useRef<HTMLFormElement>(null);
  useEffect(() => {
    let active = true;

    api.authConfig().then(config => {
      if (active) setRegistrationEnabled(config.registration_enabled);
    }).catch(() => {});

    return () => { active = false; };
  }, []);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (busy) return;

    const form = event.currentTarget;
    const values = new FormData(form);
    setBusy(true);
    setError('');
    setSuccess('');

    try {
      const username = String(values.get('username'));
      const password = String(values.get('password'));
      if (registering) {
        const confirmation = String(values.get('confirmation'));
        if (password !== confirmation) throw new Error('As senhas não coincidem.');
        await api.register(username, password, confirmation);
        form.reset();
        setRegistering(false);
        setSuccess('Conta criada. Entre com seu usuário e senha.');
      } else {
        const user = await api.login(username, password);
        form.reset();
        onLogin(user);
      }
    } catch (error) {
      setError(errorMessage(error));
    } finally {
      setBusy(false);
    }
  }

  return <main className="login-layout">
    <section className="login-story"><div className="brand"><span className="brand-symbol"><Icon name="book" /></span>DocAgent <span className="ai-tag">AI</span></div>
      <div className="story-copy"><span className="eyebrow light">SEU CONHECIMENTO, CONECTADO</span><h1>Boas perguntas.<br />Respostas com<br /><em>fundamento.</em></h1><p>Transforme seus PDFs em uma biblioteca que conversa com você. Cada resposta, conectada à sua fonte.</p><div className="story-paper"><span>DA PERGUNTA À FONTE</span><p>Um espaço para entender<br />melhor os seus documentos.</p><div><Icon name="book" size={16} /> PDF <span>→</span> <Icon name="chat" size={16} /> Resposta <span>→</span> <Icon name="check" size={16} /> Fonte</div></div></div>
      <p className="story-foot"><Icon name="lock" size={15} /> Documentos privados. Acesso autenticado.</p></section>
    <section className="login-panel"><div className="login-form"><span className="eyebrow">BEM-VINDO AO SEU WORKSPACE</span><h2>{registering ? 'Seu próximo capítulo.' : 'Entre para explorar.'}</h2><p className="muted">Acesse sua biblioteca e continue de onde a curiosidade levar.</p>
      {success && <p className="notice" role="status">{success}</p>}{notice && <p className="notice" role="status">{notice}</p>}{error && <p className="error" role="alert">{error}</p>}
      <form ref={formRef} onSubmit={submit}><label htmlFor="username">Usuário</label><input id="username" name="username" autoComplete="username" required minLength={3} maxLength={64} placeholder="Seu usuário" disabled={busy} />
      <label htmlFor="password">Senha</label><input id="password" name="password" type="password" autoComplete={registering ? "new-password" : "current-password"} required minLength={registering ? 12 : 1} maxLength={128} placeholder="Sua senha" disabled={busy} />
      {registering && <><p className="muted password-help">Use de 12 a 128 caracteres. Escolha uma senha única.</p><label htmlFor="confirmation">Confirmar senha</label><input id="confirmation" name="confirmation" type="password" autoComplete="new-password" required minLength={12} maxLength={128} disabled={busy} /></>}
      <button className="primary login-submit" disabled={busy}>{busy ? (registering ? 'Criando conta…' : 'Entrando…') : registering ? 'Criar conta' : 'Entrar no workspace'}<Icon name="arrow" size={18} /></button></form>
      {(registrationEnabled || registering) && <button type="button" className="quiet registration-link" disabled={busy} onClick={() => { formRef.current?.reset(); setRegistering(!registering); setError(''); setSuccess(''); }}>{registering ? 'Já tenho conta. Entrar' : 'Não tem conta? Criar conta'}</button>}
      <div className="login-help"><Icon name="lock" size={17} /><p>Utilize uma conta existente.<br /><span>Precisa de acesso? Fale com o administrador.</span></p></div></div><span className="login-caption">DOCAGENT AI · LEIA ALÉM DAS PÁGINAS</span></section>
  </main>;
}
