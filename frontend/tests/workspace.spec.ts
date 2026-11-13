import { test, expect } from '@playwright/test';
import type { Page } from '@playwright/test';
const id = 'ce8b2ef3-d490-40b8-858d-06af8cfca03d';
const doc = { id, filename: 'edital.pdf', pages: 2, status: 'pending', created_at: '2026-11-11T10:00:00Z', file_hash: 'a'.repeat(64) };
const source = { source_id: 1, document_id: id, chunk_id: 'aac6e603-89d4-4c0d-b4d7-0ee26d84c007', filename: doc.filename, page_start: 1, page_end: 1, quote: 'As áreas são Sistemas Embarcados e Inteligência Artificial.' };
async function mockAPI(page: Page, expires = 1800) {
  const state = { document: false, indexed: 0, deleted: 0, queries: [] as { path: string; body: Record<string, unknown> }[], expired: false, threads: new Map<string, any>(), complete: false, registration: false };
  await page.route('http://localhost:8000/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const headers = { 'access-control-allow-origin': 'http://127.0.0.1:5174', 'access-control-allow-headers': 'Authorization,Content-Type', 'access-control-allow-methods': 'GET,POST,DELETE' };
    const reply = (body: unknown, status = 200) => route.fulfill({ status, headers, contentType: 'application/json', body: JSON.stringify(body) });
    if (request.method() === 'OPTIONS') return route.fulfill({ status: 204, headers });
    if (path.endsWith('/auth/config')) return reply({ registration_enabled: state.registration });
    if (path.endsWith('/auth/register')) { expect(request.headers().authorization).toBeUndefined(); return reply({ id: 'new-user', username: request.postDataJSON().username, status: 'active' }, 201); }
    if (path.endsWith('/login')) return reply({ access_token: 'test-memory-token', expires_in: expires, token_type: 'bearer' });
    expect(request.headers().authorization).toBe('Bearer test-memory-token');
    if (state.expired) return reply({ detail: 'Credenciais inválidas.' }, 401);
    if (path === '/api/v1/conversations' && request.method() === 'POST') {
      const item = { id: crypto.randomUUID(), title: 'Nova conversa', document_id: request.postDataJSON().document_id, version: 0, updated_at: new Date().toISOString(), messages: [] };
      state.threads.set(item.id, item); return reply(item, 201);
    }
    if (path === '/api/v1/conversations') return reply([...state.threads.values()].map(({ messages, ...summary }) => summary));
    if (path.startsWith('/api/v1/conversations/')) return reply(state.threads.get(path.split('/').at(-1)!) || {}, state.threads.has(path.split('/').at(-1)!) ? 200 : 404);
    if (path.endsWith('/me')) return reply({ id: 'user-1', username: 'alice', status: 'active' });
    if (path === '/api/v1/documents' && request.method() === 'POST') { state.document = true; return reply(doc, 201); }
    if (path === '/api/v1/documents') return reply({ items: state.document ? [{ ...doc, status: state.indexed ? 'ready' : 'pending' }] : [], limit: 20, offset: 0 });
    if (path.endsWith('/index')) { state.indexed++; await new Promise(r => setTimeout(r, 200)); return reply({ document_id: id, chunks: 2, status: 'ready', model_id: 'test@revision' }); }
    if (path.endsWith('/file')) return route.fulfill({ headers, contentType: 'application/pdf', body: '%PDF-1.4\n%%EOF' });
    if (request.method() === 'DELETE') { state.deleted++; state.document = false; return route.fulfill({ status: 204, headers }); }
    if (['/api/v1/ask', '/api/v1/agent/ask', '/api/v1/search'].includes(path)) {
      const body = request.postDataJSON(); state.queries.push({ path, body });
      if (path.endsWith('/search')) return reply({ items: [{ ...source, text: source.quote, score: 0.58 }] });
      const answer = body.question.includes('garantia') ? { answer: 'Não foi possível encontrar a resposta nos documentos autorizados.', citations: [] } : body.question.includes('requisitos') ? { answer: 'Os requisitos dessas áreas são matrícula ativa e disponibilidade de 20 horas. [1]', citations: [source] } : state.complete ? {
        answer: '**As três áreas são:**\n\n1. Sistemas Embarcados e Eletrônica. [1]\n2. Engenharia Mecânica e Controle e Automação. [1]\n3. Inteligência Artificial. [1]\n\n`' + 'identificador'.repeat(60) + '`', citations: [source]
      } : { answer: 'As áreas são Sistemas Embarcados e Inteligência Artificial. [1]', citations: [source] };
      const thread = state.threads.get(body.conversation_id);
      if (thread) { thread.messages.push({ id: body.message_id, question: body.question, answer: answer.answer, citations: answer.citations, agent: path.includes('/agent/') }); thread.version++; thread.title = thread.messages[0].question.slice(0, 80); }
      return reply(answer);
    }
    return reply({ detail: 'Não encontrado.' }, 404);
  });
  return state;
}
async function login(page: Page) {
  await page.goto('/'); await page.getByLabel('Usuário', { exact: true }).fill('alice');
  await page.getByLabel('Senha', { exact: true }).fill('only-a-test-password');
  await page.getByRole('button', { name: 'Entrar no workspace' }).click();
  await expect(page.getByRole('heading', { name: 'Conhecimento começa aqui.' })).toBeVisible();
}

test('PDF lifecycle, duplicate prevention, authenticated preview and deletion confirmation', async ({ page }) => {
  const state = await mockAPI(page); await login(page);
  await page.getByLabel('Enviar PDF').setInputFiles({ name: 'fake.pdf', mimeType: 'application/pdf', buffer: Buffer.from('invalid') });
  await expect(page.getByRole('alert')).toContainText('PDF válido');
  await page.getByLabel('Enviar PDF').setInputFiles({ name: 'edital.pdf', mimeType: 'application/pdf', buffer: Buffer.from('%PDF-1.4\nsynthetic') });
  await expect(page.getByText('Não indexado', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Indexar edital.pdf', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Indexar edital.pdf', exact: true })).toBeDisabled();
  await expect(page.getByText('Pronto para consultar', { exact: true })).toBeVisible(); expect(state.indexed).toBe(1);
  await page.screenshot({ path: '/tmp/docagent-library.png', fullPage: true });
  await page.getByRole('button', { name: 'Visualizar edital.pdf' }).click();
  await expect(page.locator('dialog iframe')).toHaveAttribute('src', /^blob:.*#page=1$/);
  await page.getByRole('button', { name: 'Fechar visualização' }).click();
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Baixar edital.pdf' }).click();
  expect((await download).suggestedFilename()).toBe('edital.pdf');
  await page.getByRole('button', { name: 'Excluir edital.pdf' }).click();
  await page.getByRole('button', { name: 'Cancelar', exact: true }).click(); expect(state.deleted).toBe(0);
  await page.getByRole('button', { name: 'Excluir edital.pdf' }).click();
  await page.getByRole('button', { name: 'Excluir documento', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Espaço para novas descobertas.' })).toBeVisible(); expect(state.deleted).toBe(1);
});

test('chat, agent, scoped citations and distinct semantic search on a mobile layout', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const state = await mockAPI(page); state.document = true; await login(page);
  await page.getByRole('button', { name: 'Conversar', exact: true }).click();
  const panel = page.locator('.main-content > div').filter({ has: page.getByRole('heading', { name: 'Uma pergunta abre caminhos.' }) });
  await panel.getByLabel('Documento para consulta').selectOption(id);
  await panel.getByLabel('Usar agente').check();
  await panel.getByLabel('Sua pergunta').fill('Quais as áreas?'); await panel.getByRole('button', { name: 'Enviar pergunta' }).click();
  await expect(panel.getByText('As áreas são Sistemas Embarcados e Inteligência Artificial. [1]', { exact: true })).toBeVisible();
  expect(state.queries[0]).toMatchObject({ path: '/api/v1/agent/ask', body: { document_id: id } });
  await panel.locator('summary').click(); await panel.getByRole('button', { name: 'Abrir página no PDF' }).click();
  await expect(page.locator('dialog iframe')).toHaveAttribute('src', /^blob:.*#page=1$/); await page.getByRole('button', { name: 'Fechar visualização' }).click();
  await panel.getByLabel('Usar agente').uncheck();
  await panel.getByLabel('Sua pergunta').fill('Qual a garantia?'); await panel.getByRole('button', { name: 'Enviar pergunta' }).click();
  await expect(panel.getByText('Sem evidências suficientes nos documentos consultados')).toBeVisible();
  expect(state.queries.at(-1)?.path).toBe('/api/v1/ask');
  await page.getByRole('button', { name: 'Busca semântica', exact: true }).click();
  const search = page.locator('.semantic-card');
  await search.getByLabel('Documento para consulta').selectOption(id);
  await search.getByLabel('Texto para busca semântica').fill('Áreas de seleção'); await search.getByRole('button', { name: 'Buscar trechos' }).click();
  await expect(search.getByText('Cosseno 0,58')).toBeVisible(); expect(state.queries.at(-1)?.path).toBe('/api/v1/search');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: '/tmp/docagent-mobile.png', fullPage: true });
});

test('session stays in memory, unauthorized responses and expiry return to login', async ({ page }) => {
  const state = await mockAPI(page); await login(page);
  expect(await page.evaluate(() => [localStorage.length, sessionStorage.length])).toEqual([0, 0]);
  state.expired = true; await page.getByRole('button', { name: 'Atualizar', exact: true }).click();
  await expect(page.getByText('Sua sessão expirou. Entre novamente para continuar.')).toBeVisible();
  state.expired = false; await login(page); await page.reload();
  await expect(page.getByLabel('Usuário', { exact: true })).toBeVisible();
});

test('token expiry clears the workspace without another API request', async ({ page }) => {
  await mockAPI(page, 1); await login(page);
  await expect(page.getByText('Sua sessão expirou. Entre novamente para continuar.')).toBeVisible({ timeout: 5000 });
});


test('complete API Markdown survives rendering, mode switch and history restoration after login', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const state = await mockAPI(page); state.document = true; state.complete = true;
  await login(page); await page.getByRole('button', { name: 'Conversar', exact: true }).click();
  const panel = page.locator('.main-content > div').filter({ has: page.getByRole('heading', { name: 'Uma pergunta abre caminhos.' }) });
  await panel.getByLabel('Documento para consulta').selectOption(id);
  await panel.getByLabel('Sua pergunta').fill('Quais são as três áreas do processo seletivo do LIVE?');
  const apiResponse = page.waitForResponse(r => r.url().endsWith('/api/v1/ask') && r.request().method() === 'POST');
  await panel.getByRole('button', { name: 'Enviar pergunta' }).click();
  const json = await (await apiResponse).json();
  for (const area of ['Sistemas Embarcados e Eletrônica', 'Engenharia Mecânica e Controle e Automação', 'Inteligência Artificial']) {
    expect(json.answer).toContain(area);
    await expect(panel.locator('.answer-text')).toContainText(area);
  }
  await expect(panel.locator('.answer-text ol li')).toHaveCount(3);
  await expect(panel.locator('.answer-text strong')).toHaveText('As três áreas são:');
  expect(await panel.locator('.answer-text').evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
  const cid = state.queries[0].body.conversation_id;
  await panel.getByLabel('Usar agente').check();
  await panel.getByLabel('Sua pergunta').fill('E quais são os requisitos dessas áreas?');
  await panel.getByRole('button', { name: 'Enviar pergunta' }).click();
  await expect(panel.locator('.answer-text').last()).toContainText('matrícula ativa');
  expect(state.queries[1]).toMatchObject({ path: '/api/v1/agent/ask', body: { conversation_id: cid, document_id: id } });
  expect(state.queries[0].body.message_id).not.toBe(state.queries[1].body.message_id);
  await expect(panel.locator('.exchange')).toHaveCount(2);
  await page.getByRole('button', { name: 'Sair da conta' }).click();
  await login(page); await page.getByRole('button', { name: 'Conversar', exact: true }).click();
  await panel.getByLabel('Histórico de conversas', { exact: true }).selectOption(String(cid));
  await expect(panel.locator('.exchange')).toHaveCount(2);
  await expect(panel.locator('.answer-text').first()).toContainText('Inteligência Artificial');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});


test('registration is opt-in, validates confirmation and returns to login', async ({ page }) => {
  const state = await mockAPI(page); state.registration = true;
  await page.goto('/');
  await page.getByRole('button', { name: 'Não tem conta? Criar conta' }).click();
  await page.getByLabel('Usuário', { exact: true }).fill('new_reader');
  await page.getByLabel('Senha', { exact: true }).fill('Synthetic password 123');
  await page.getByLabel('Confirmar senha', { exact: true }).fill('Another password 123');
  await page.getByRole('button', { name: 'Criar conta', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('As senhas não coincidem.');
  await page.getByLabel('Confirmar senha', { exact: true }).fill('Synthetic password 123');
  await page.getByRole('button', { name: 'Criar conta', exact: true }).click();
  await expect(page.getByText('Conta criada. Entre com seu usuário e senha.')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Entrar no workspace' })).toBeVisible();
  await expect(page.getByLabel('Senha', { exact: true })).toHaveValue('');
  const background = await page.locator('html').evaluate(el => getComputedStyle(el).backgroundColor);
  expect(background).toBe('rgb(17, 17, 19)');
  await page.screenshot({ path: '/tmp/docagent-v1-login.png', fullPage: true });
});
