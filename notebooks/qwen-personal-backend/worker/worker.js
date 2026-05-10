// Qwen Personal Backend — CF Worker PoC-003
// WebSocket architecture via Durable Objects with Hibernation API
//
// Key insight: use DO Hibernation API (webSocketMessage/Close/Error methods)
// instead of addEventListener — these survive DO eviction/revival.
// Store pending tasks in state.storage so they survive eviction too.

const ANALYZE_DELIMITERS = {
  section1: '<<<SECTION_1>>>',
  section2: '<<<SECTION_2>>>',
  section3: '<<<SECTION_3>>>',
};

const ANALYZE_FALLBACK_COPY = {
  en: { headings: ['Search queries:', 'Experts & sources:', 'Related topics:'], nextSteps: 'Search queries:\n- Explore this topic further\n\nExperts & sources:\n- Original author / publication\n\nRelated topics:\n- Adjacent topic worth reading next', essenceUnavailable: 'Key idea unavailable in the original response.' },
  ru: { headings: ['Поисковые запросы:', 'Эксперты и источники:', 'Связанные темы:'], nextSteps: 'Поисковые запросы:\n- Изучить тему глубже\n\nЭксперты и источники:\n- Исходный автор / публикация\n\nСвязанные темы:\n- Смежная тема для следующего чтения', essenceUnavailable: 'Не удалось выделить ключевую мысль из исходного ответа.' },
  uk: { headings: ['Пошукові запити:', 'Експерти та джерела:', "Пов'язані теми:"], nextSteps: "Пошукові запити:\n- Дослідити цю тему глибше\n\nЕксперти та джерела:\n- Початковий автор / публікація\n\nПов'язані теми:\n- Суміжна тема для наступного читання", essenceUnavailable: 'Не вдалося виділити ключову думку з початкової відповіді.' },
  es: { headings: ['Consultas de busqueda:', 'Expertos y fuentes:', 'Temas relacionados:'], nextSteps: 'Consultas de busqueda:\n- Explorar este tema con mas profundidad\n\nExpertos y fuentes:\n- Autor o publicacion original\n\nTemas relacionados:\n- Un tema cercano para seguir leyendo', essenceUnavailable: 'No fue posible extraer la idea clave de la respuesta original.' },
  de: { headings: ['Suchanfragen:', 'Experten und Quellen:', 'Verwandte Themen:'], nextSteps: 'Suchanfragen:\n- Dieses Thema weiter vertiefen\n\nExperten und Quellen:\n- Urspruenglicher Autor / Publikation\n\nVerwandte Themen:\n- Angrenzendes Thema zum Weiterlesen', essenceUnavailable: 'Die Kernaussage konnte aus der urspruenglichen Antwort nicht extrahiert werden.' },
  fr: { headings: ['Recherches suggerees:', 'Experts et sources:', 'Sujets connexes:'], nextSteps: "Recherches suggerees:\n- Approfondir ce sujet\n\nExperts et sources:\n- Auteur ou publication d origine\n\nSujets connexes:\n- Sujet voisin a lire ensuite", essenceUnavailable: "Impossible d extraire l idee cle de la reponse d origine." },
  pt: { headings: ['Pesquisas sugeridas:', 'Especialistas e fontes:', 'Topicos relacionados:'], nextSteps: 'Pesquisas sugeridas:\n- Explorar este tema com mais profundidade\n\nEspecialistas e fontes:\n- Autor ou publicacao original\n\nTopicos relacionados:\n- Tema adjacente para ler em seguida', essenceUnavailable: 'Nao foi possivel extrair a ideia principal da resposta original.' },
};

const VALID_MODES = ['explain','analyze','explain_rephrase','explain_example','explain_application','explain_importance'];
const MAX_EXPLAIN_CHARS = 2000;
const MAX_ANALYZE_CHARS = 12000;
function getMaxChars(mode) { return mode === 'analyze' ? MAX_ANALYZE_CHARS : MAX_EXPLAIN_CHARS; }

const SMART_ANALYZE_PROMPT = (text, language) => `The article language is: ${language}. Write your ENTIRE response in this language. No exceptions.

You are R-Searcher — a reading assistant that helps users extract maximum value from articles.

Structure your response using EXACTLY these delimiters, nothing before <<<SECTION_1>>>:

<<<SECTION_1>>>
ESSENCE: 3-5 sentences. What is this article about, what is the main claim or finding, and why does it matter? Be direct — no intro phrases like "This article...". Start with the subject.

<<<SECTION_2>>>
NOTES: Structured digest — not a retelling, not a summary. Capture only insights, decisions, and facts the user would want to reference later. Use ## for main topics, bullet points for key points, **bold** for key terms. Be thorough but ruthlessly cut anything obvious, repetitive or decorative.

<<<SECTION_3>>>
NEXT STEPS:
Keep EVERYTHING inside SECTION_3 in the article language too, including headings, labels, and bullet text.
Use exactly three compact groups in this order:
1. Search queries: 3-5 specific queries to explore the topic further
2. Experts & sources: 2-3 names, authors, publications, or organizations relevant to the topic
3. Related topics: 1-2 adjacent areas worth exploring next

Article:
"""
${text}
"""`;

const SMART_EXPLAIN_PROMPT = (text, language) => `The text language is: ${language}. Write your ENTIRE response in this language. No exceptions.

You are R-Searcher. Explain the following text to a smart person with no background in this field.

- No jargon without explanation
- Max 4-5 sentences
- Be direct, start explaining immediately

Then add EXACTLY this block at the end, in English, no exceptions:

<<<META>>>
type: technical|scientific|historical|legal|medical|general
has_example: true|false
has_application: true|false

Text:
"""
${text}
"""`;

const SMART_EXPLAIN_REPHRASE_PROMPT = (text, language, prev) => `The text language is: ${language}. Write your ENTIRE response in this language. No exceptions.

You are R-Searcher. The previous explanation was:
"""
${prev}
"""

Now explain the SAME text differently. Use a completely different angle, metaphor, or structure.

- No jargon without explanation
- Max 4-5 sentences
- Be direct

Text:
"""
${text}
"""`;

const SMART_EXPLAIN_EXAMPLE_PROMPT = (text, language, prev) => `The text language is: ${language}. Write your ENTIRE response in this language. No exceptions.

You are R-Searcher. Here's the text that was explained:
"""
${prev}
"""

Provide a concrete, real-world example that illustrates the concept.

- Use a relatable scenario
- Max 3-4 sentences
- Be specific and practical

Text:
"""
${text}
"""`;

const SMART_EXPLAIN_APPLICATION_PROMPT = (text, language, prev) => `The text language is: ${language}. Write your ENTIRE response in this language. No exceptions.

You are R-Searcher. Here's what was explained:
"""
${prev}
"""

Where and how is this used in the real world?

- Be practical
- Max 3-4 sentences
- Name specific contexts if possible

Text:
"""
${text}
"""`;

const SMART_EXPLAIN_IMPORTANCE_PROMPT = (text, language, prev) => `The text language is: ${language}. Write your ENTIRE response in this language. No exceptions.

You are R-Searcher. Here's what was explained:
"""
${prev}
"""

Why does this matter?

- Explain the impact or relevance
- Max 3-4 sentences
- Be direct and meaningful

Text:
"""
${text}
"""`;

function buildPrompt(mode, text, language, prev) {
  if (mode === 'explain')             return SMART_EXPLAIN_PROMPT(text, language);
  if (mode === 'explain_rephrase')    return SMART_EXPLAIN_REPHRASE_PROMPT(text, language, prev);
  if (mode === 'explain_example')     return SMART_EXPLAIN_EXAMPLE_PROMPT(text, language, prev);
  if (mode === 'explain_application') return SMART_EXPLAIN_APPLICATION_PROMPT(text, language, prev);
  if (mode === 'explain_importance')  return SMART_EXPLAIN_IMPORTANCE_PROMPT(text, language, prev);
  return SMART_ANALYZE_PROMPT(text, language);
}

function normalizeNewlines(text) { return (text || '').replace(/\r\n?/g, '\n'); }
function getLanguageBase(language) {
  if (!language || typeof language !== 'string') return 'en';
  const [base] = language.trim().toLowerCase().replace('_', '-').split('-');
  return ANALYZE_FALLBACK_COPY[base] ? base : 'en';
}
function cleanAnalyzeSection(text) {
  return normalizeNewlines(text).replace(/^\s*(?:\*\*)?(?:ESSENCE|NOTES|NEXT\s*STEPS)(?:\*\*)?\s*:?\s*/i, '').replace(/\n{3,}/g, '\n\n').trim();
}
function tryParseAnalyzeSections(raw) {
  const n = normalizeNewlines(raw);
  const s1 = n.indexOf(ANALYZE_DELIMITERS.section1);
  const s2 = n.indexOf(ANALYZE_DELIMITERS.section2);
  const s3 = n.indexOf(ANALYZE_DELIMITERS.section3);
  if (s1 === -1 || s2 === -1 || s3 === -1 || !(s1 < s2 && s2 < s3)) return null;
  const essence   = cleanAnalyzeSection(n.slice(s1 + ANALYZE_DELIMITERS.section1.length, s2));
  const notes     = cleanAnalyzeSection(n.slice(s2 + ANALYZE_DELIMITERS.section2.length, s3));
  const nextSteps = cleanAnalyzeSection(n.slice(s3 + ANALYZE_DELIMITERS.section3.length));
  if (!essence || !notes || !nextSteps) return null;
  return { essence, notes, nextSteps };
}
function stripAnalyzeMarkers(raw) {
  return normalizeNewlines(raw).replace(/<<<SECTION_[123]>>>/g, '\n').replace(/^\s*(?:\*\*)?(?:ESSENCE|NOTES|NEXT\s*STEPS)(?:\*\*)?\s*:?\s*/gim, '').replace(/\n{3,}/g, '\n\n').trim();
}
function buildEssenceFallback(text) {
  const sentences = normalizeNewlines(text).replace(/\s+/g, ' ').trim().match(/[^.!?]+[.!?]+(?:["')\]]+)?|[^.!?]+$/g) || [];
  return sentences.slice(0, 5).join(' ').trim() || text.trim();
}
function extractExistingNextSteps(text) {
  const headings = Object.values(ANALYZE_FALLBACK_COPY).flatMap(c => c.headings);
  for (const h of headings) {
    const i = text.toLowerCase().indexOf(h.toLowerCase());
    if (i !== -1) return text.slice(i).trim();
  }
  return '';
}
function normalizeAnalyzeResult(raw, language) {
  const text = typeof raw === 'string' ? raw : String(raw || '');
  const parsed = tryParseAnalyzeSections(text);
  if (parsed) return { raw: text, sections: parsed };
  const fallback  = ANALYZE_FALLBACK_COPY[getLanguageBase(language)];
  const cleaned   = stripAnalyzeMarkers(text);
  const nextSteps = extractExistingNextSteps(cleaned) || fallback.nextSteps;
  const essence   = buildEssenceFallback(cleaned) || fallback.essenceUnavailable;
  return { raw: text, sections: { essence, notes: cleaned || essence, nextSteps } };
}

function getCorsHeaders(request) {
  const origin = request.headers.get('Origin') || '';
  const allowOrigin = origin.startsWith('chrome-extension://') || origin === 'null' ? origin : 'null';
  return {
    'Access-Control-Allow-Origin': allowOrigin,
    'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Vary': 'Origin',
  };
}

function json(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status, headers: { 'Content-Type': 'application/json' },
  });
}

// ─── Durable Object with Hibernation API ─────────────────────────────────────
//
// Instead of ws.addEventListener (lost on eviction), we use DO Hibernation API:
//   webSocketMessage(ws, message) — called by CF when WS message arrives
//   webSocketClose(ws, code, reason, wasClean) — called on close
//   webSocketError(ws, error) — called on error
//
// These methods are called by CF runtime even after DO eviction/revival.
// pending tasks stored in state.storage so they survive eviction too.

export class QwenBackend {
  constructor(state, env) {
    this.state = state;
    // pending is in-memory only — tasks in flight when DO was evicted are lost
    // This is acceptable: the 180s timeout will fire on revival and return error
    this.pending = new Map();
  }

  _getWs() {
    const sockets = this.state.getWebSockets();
    return sockets.length > 0 ? sockets[0] : null;
  }

  // ── DO Hibernation API — called by CF runtime, survives eviction ────────────

  webSocketMessage(ws, message) {
    let msg;
    try { msg = JSON.parse(message); } catch { return; }

    const entry = this.pending.get(msg.id);
    if (!entry) return;

    clearTimeout(entry.timer);
    this.pending.delete(msg.id);

    if (msg.error) {
      entry.resolve({ ok: false, error: `inference_error: ${msg.error}` });
    } else {
      entry.resolve({ ok: true, content: msg.content });
    }
  }

  async webSocketClose(ws, code, reason, wasClean) {
    await this.state.storage.put('disconnectedAt', new Date().toISOString());
    await this.state.storage.delete('connectedAt');

    for (const [, entry] of this.pending) {
      clearTimeout(entry.timer);
      entry.resolve({
        ok:    false,
        error: 'backend_disconnected',
        hint:  'Kaggle notebook was restarted. Please wait ~5 min for model to reload.',
      });
    }
    this.pending.clear();
  }

  async webSocketError(ws, error) {
    await this.state.storage.put('disconnectedAt', new Date().toISOString());
    await this.state.storage.delete('connectedAt');
  }

  // ── /health ──────────────────────────────────────────────────────────────────

  async handleHealth() {
    const ws = this._getWs();
    return json({
      status:          ws ? 'connected' : 'disconnected',
      connected_at:    await this.state.storage.get('connectedAt')    || null,
      disconnected_at: await this.state.storage.get('disconnectedAt') || null,
      pending_tasks:   this.pending.size,
    });
  }

  // ── /connect ─────────────────────────────────────────────────────────────────

  async handleConnect(request) {
    if (request.headers.get('Upgrade') !== 'websocket') {
      return new Response('Expected WebSocket upgrade', { status: 426 });
    }

    const existing = this._getWs();
    if (existing) {
      try { existing.close(1000, 'replaced by new connection'); } catch {}
    }

    const [client, server] = Object.values(new WebSocketPair());
    // acceptWebSocket registers the socket for Hibernation API
    // DO Hibernation API methods (webSocketMessage etc.) will handle events
    this.state.acceptWebSocket(server);

    await this.state.storage.put('connectedAt', new Date().toISOString());

    return new Response(null, { status: 101, webSocket: client });
  }

  // ── /process ─────────────────────────────────────────────────────────────────

  async handleProcess(request) {
    const ws = this._getWs();
    if (!ws) {
      return json({
        error:           'backend_not_connected',
        hint:            'Kaggle notebook is offline. Start the notebook and click Connect.',
        disconnected_at: await this.state.storage.get('disconnectedAt') || null,
      }, 503);
    }

    let body;
    try { body = await request.json(); }
    catch { return json({ error: 'invalid_json' }, 400); }

    const { text, mode, language, previousExplanation } = body;
    if (!text || !mode)              return json({ error: 'missing_fields' }, 400);
    if (!VALID_MODES.includes(mode)) return json({ error: 'invalid_mode' }, 400);
    if (text.length > getMaxChars(mode)) return json({ error: 'request_too_large' }, 413);

    const taskId = crypto.randomUUID();
    const prompt = buildPrompt(mode, text, language || 'en', previousExplanation || '');

    ws.send(JSON.stringify({
      id:          taskId,
      messages:    [{ role: 'user', content: prompt }],
      max_tokens:  mode === 'analyze' ? 2000 : 800,
      temperature: 0.3,
    }));

    const data = await new Promise((resolve) => {
      const timer = setTimeout(() => {
        this.pending.delete(taskId);
        resolve({ ok: false, error: 'timeout', hint: 'Inference took too long. Try reducing text length.' });
      }, 180_000);

      this.pending.set(taskId, { resolve, timer });
    });

    if (!data.ok) {
      return json({ error: data.error, hint: data.hint }, data.error === 'timeout' ? 504 : 503);
    }

    if (mode === 'analyze') {
      return json({ result: normalizeAnalyzeResult(data.content, language) });
    }

    return json({ result: data.content });
  }

  // ── fetch ────────────────────────────────────────────────────────────────────

  async fetch(request) {
    const path = new URL(request.url).pathname;
    if (path === '/connect') return this.handleConnect(request);
    if (path === '/process') return this.handleProcess(request);
    if (path === '/health')  return this.handleHealth();
    return new Response('Not found', { status: 404 });
  }
}

// ─── Worker entry point ───────────────────────────────────────────────────────

export default {
  async fetch(request, env) {
    const cors = getCorsHeaders(request);
    if (request.method === 'OPTIONS') return new Response(null, { headers: cors });

    const path = new URL(request.url).pathname;

    if (path === '/health' || path === '/connect' || path === '/process') {
      const stub = env.BACKEND.get(env.BACKEND.idFromName('singleton'));
      const resp = await stub.fetch(request);
      const out  = new Response(resp.body, resp);
      Object.entries(cors).forEach(([k, v]) => out.headers.set(k, v));
      return out;
    }

    return new Response('Not found', { status: 404, headers: cors });
  }
};
