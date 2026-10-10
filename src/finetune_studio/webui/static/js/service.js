/* Service health: a banner on every page when the app is restarting or broke, and the Settings "Service" card.
 * Source: GET /api/service/status (supervisor component table + plain-language problems). */
(function () {
  'use strict';
  const POLL_MS = 10000, RETRY_MS = 2000, FETCH_TIMEOUT_MS = 6000, CARD_MS = 5000;
  const DISMISS_KEY = 'fts-service-dismissed';
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  let last = null;        // last good status body
  let unreachable = false;
  let timer = null;

  async function fetchStatus() {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), FETCH_TIMEOUT_MS);
    try {
      const r = await fetch('/api/service/status', { cache: 'no-store', signal: ctl.signal });
      const body = await r.json().catch(() => ({}));
      return { ok: r.ok, status: r.status, body };
    } finally { clearTimeout(t); }
  }

  function dismissed() { try { return JSON.parse(localStorage.getItem(DISMISS_KEY) || '[]'); } catch (e) { return []; } }
  function dismiss(id) { try { localStorage.setItem(DISMISS_KEY, JSON.stringify(dismissed().concat([id]).slice(-30))); } catch (e) { /* private mode */ } }

  function renderBanner(problems) {
    const el = $('service-banner');
    if (!el) return;
    const shown = (problems || []).filter((p) => p.severity !== 'notice' || !dismissed().includes(p.id));
    if (!shown.length) { el.hidden = true; el.innerHTML = ''; return; }
    const p = shown[0];
    const more = shown.length > 1 ? ` <span class="dim">(+${shown.length - 1} more)</span>` : '';
    el.className = 'service-banner ' + p.severity;
    el.innerHTML = `<span class="sb-msg"><strong>${esc(p.title)}.</strong> ${esc(p.detail)}${more}</span>`
      + `<a class="btn sm" href="/settings#service-card" data-link>Details</a>`
      + (p.severity === 'notice' ? `<button type="button" class="btn sm" data-dismiss="${esc(p.id)}" aria-label="Dismiss">Dismiss</button>` : '');
    el.hidden = false;
    const b = el.querySelector('[data-dismiss]');
    if (b) b.addEventListener('click', () => { dismiss(p.id); renderBanner(problems); });
  }

  function renderUnreachable(msg) {
    const el = $('service-banner');
    if (!el) return;
    el.className = 'service-banner warning';
    el.innerHTML = `<span class="sb-msg"><strong>Server not reachable.</strong> ${esc(msg)} Retrying…</span>`;
    el.hidden = false;
  }

  async function tick() {
    clearTimeout(timer);
    let next = POLL_MS;
    try {
      const { body } = await fetchStatus();
      if (unreachable && window.fts) window.fts.notify('Server is back.', 'success');
      unreachable = false;
      last = body;
      renderBanner(body.problems);
      renderCard(body);
    } catch (e) {
      unreachable = true;
      renderUnreachable('The server may be restarting.');
      next = RETRY_MS;
    }
    if (document.visibilityState === 'hidden') next = Math.max(next, 30000);
    timer = setTimeout(tick, next);
  }

  /* ── Settings card ─────────────────────────────────────────── */
  const age = (s) => s == null ? '–' : s >= 3600 ? `${Math.floor(s / 3600)}h${String(Math.floor(s % 3600 / 60)).padStart(2, '0')}m`
    : s >= 60 ? `${Math.floor(s / 60)}m${String(Math.floor(s % 60)).padStart(2, '0')}s` : `${Math.floor(s)}s`;
  const stateClass = { ready: 'ok', starting: 'warn', backoff: 'warn', unhealthy: 'warn', failed: 'err', stopped: 'dim' };

  function lastExitText(c) {
    const l = c.last_exit;
    if (!l) return c.health && c.health.detail ? esc(c.health.detail) : '–';
    const how = l.reason === 'requested' ? 'restarted on request' : l.signal ? `killed by ${l.signal}` : l.code != null ? `exit code ${l.code}` : esc(l.error || 'exited');
    const when = l.at ? new Date(l.at * 1000).toLocaleTimeString() : '';
    return `${esc(how)} <span class="dim">${esc(when)}</span>`;
  }

  function renderCard(body) {
    const card = $('service-card');
    if (!card) return;
    const mode = $('service-mode');
    if (!body.managed) {
      mode.innerHTML = `<span class="pill outline">not supervised</span> ${esc(body.hint || '')}`;
      $('service-table').hidden = true; $('service-actions').hidden = true;
      return;
    }
    $('service-table').hidden = false; $('service-actions').hidden = false;
    if (!body.reachable) { mode.innerHTML = `<span class="pill solid-rose">supervisor not answering</span> ${esc(body.error || '')}`; return; }
    const s = body.supervisor;
    mode.innerHTML = `<span class="pill solid-green">supervised</span> started by <strong>${esc(s.launcher || '?')}</strong>`
      + ` · listening on <span class="mono">${esc(s.listen || '?')}</span> · supervisor pid ${s.pid} · up ${age(s.uptime_s)}`;
    $('service-table').tBodies[0].innerHTML = Object.entries(body.components).map(([name, c]) =>
      `<tr><td class="mono">${esc(name)}</td><td><span class="${stateClass[c.state] || ''}">${esc(c.state)}</span></td>`
      + `<td class="mono">${c.pid || '–'}</td><td>${age(c.uptime_s)}</td><td>${c.restarts}</td><td>${lastExitText(c)}</td></tr>`).join('');
  }

  async function loadDetails(kind) {
    const out = $('service-detail');
    out.hidden = false;
    out.textContent = 'Loading…';
    try {
      const r = await fetch(kind === 'log' ? '/api/service/logs/web?lines=150' : '/api/service/events?limit=60', { cache: 'no-store' });
      const d = await r.json();
      if (!r.ok) { out.textContent = d.error || `HTTP ${r.status}`; return; }
      out.textContent = kind === 'log' ? (d.lines.join('\n') || '(no output captured since the last start)')
        : d.events.map((e) => { const { seq, at, component, kind: k, ...rest } = e; delete rest.tail;
          return `${new Date(at * 1000).toLocaleTimeString()}  ${component.padEnd(10)} ${k.padEnd(18)} ${Object.keys(rest).length ? JSON.stringify(rest) : ''}`; }).join('\n') || '(no events)';
    } catch (e) { out.textContent = 'Could not read it: ' + e.message; }
  }

  async function restartWeb() {
    const ok = await window.fts.confirm('Restart the web server? It is back in a few seconds. Running jobs stop and are marked failed; a loaded model is unloaded.', { danger: true, okText: 'Restart' });
    if (!ok) return;
    try {
      const r = await fetch('/api/service/components/web/restart', { method: 'POST' });
      const d = await r.json().catch(() => ({}));
      if (!r.ok) { window.fts.notify(d.error || `Restart failed (HTTP ${r.status})`, 'error'); return; }
      window.fts.notify('Restart requested. Reconnecting…', 'warn');
      clearTimeout(timer); timer = setTimeout(tick, 1500);
    } catch (e) { window.fts.notify('Restart request failed: ' + e.message, 'error'); }
  }

  function initCard() {
    const card = $('service-card');
    if (!card || card.dataset.wired) return;
    card.dataset.wired = '1';
    $('btn-service-restart').addEventListener('click', restartWeb);
    $('btn-service-log').addEventListener('click', () => loadDetails('log'));
    $('btn-service-events').addEventListener('click', () => loadDetails('events'));
    if (last) renderCard(last);
    const loop = setInterval(() => { if (!document.body.contains(card)) { clearInterval(loop); return; } tick(); }, CARD_MS);
  }

  window.ftsService = { refresh: tick, status: () => last };
  document.addEventListener('fts:navigated', initCard);
  const start = () => { initCard(); tick(); };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
