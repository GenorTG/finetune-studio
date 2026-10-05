/* ============================================================
   Settings page — debug info loader + tutorial replay
   ============================================================ */
(async function () {
  const $ = (id) => document.getElementById(id);

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function row(label, value) {
    return `<div class="debug-label">${label}</div><div class="debug-value">${value || "—"}</div>`;
  }

  function humanBytes(n) {
    if (!n) return "—";
    if (n > 1024 * 1024 * 1024) return (n / 1024 / 1024 / 1024).toFixed(1) + " GB";
    if (n > 1024 * 1024) return (n / 1024 / 1024).toFixed(1) + " MB";
    return (n / 1024).toFixed(0) + " KB";
  }

  async function loadDebug() {
    const el = $("debug-info");
    const pathsEl = $("paths-table");
    try {
      const r = await fetch("/api/debug/info");
      if (!r.ok) throw new Error("HTTP " + r.status);
      const d = await r.json();

      let gpuHtml = "";
      if (d.gpus && d.gpus.length > 0) {
        gpuHtml = d.gpus.map((g, i) =>
          `<div class="gpu-line">
             <span class="gpu-idx">[${i}]</span>
             <span class="gpu-name">${esc(g.name)}</span>
             <span class="gpu-vram">${humanBytes(g.vram_total_mb * 1024 * 1024)} (${humanBytes(g.vram_free_mb * 1024 * 1024)} free)</span>
             <span class="gpu-driver dim">driver ${esc(g.driver)}</span>
           </div>`
        ).join("");
      } else {
        gpuHtml = `<div class="dim text-xs">No GPU detected${d.gpu_error ? ` — ${esc(d.gpu_error)}` : ""}.</div>`;
      }
      const ac = d.accelerator || {};
      const acc = ac.accelerator || {};
      const accHtml = acc.kind
        ? `<span class="mono">${esc(acc.torch_device || acc.kind)}</span> · ${esc(acc.name || "")} · ${esc(acc.runtime || "")}`
        : `<span class="dim">unknown${ac.error ? ` — ${esc(ac.error)}` : ""}</span>`;
      const degraded = acc.degraded_reason
        ? `<div class="gpu-degraded" role="alert" data-testid="accel-degraded">${esc(acc.degraded_reason)}</div>` : "";

      const pkgHtml = Object.entries(d.packages || {})
        .map(([k, v]) => `<tr><td class="mono">${esc(k)}</td><td class="mono dim">${esc(v)}</td></tr>`)
        .join("");

      el.innerHTML = `
        ${row("Release", `<span class="text-accent mono">${esc(d.release_channel || "EARLY BETA")}</span> · <span class="mono">v${esc(d.app_version)}</span>`)}
        ${row("Python", `<span class="mono">${esc(d.python)}</span>`)}
        ${row("Platform", esc(d.platform))}
        ${row("Hostname", esc(d.hostname))}
        ${row("GPU", gpuHtml)}
        ${row("Compute", accHtml + degraded)}
        <div class="debug-label">Packages</div>
        <div class="debug-value">
          <table class="pkg-table">${pkgHtml}</table>
        </div>
      `;

      const p = d.paths || {};
      pathsEl.innerHTML = `
        <tr><td class="dim text-xs">Data directory</td><td class="mono text-xs">${esc(p.data_dir)}</td></tr>
        <tr><td class="dim text-xs">HF cache (HF_HOME)</td><td class="mono text-xs">${esc(p.hf_cache)}</td></tr>
        <tr><td class="dim text-xs">HF hub cache</td><td class="mono text-xs">${esc(p.hf_hub_cache)} <span class="dim">(from ${esc(p.hf_cache_source)})</span></td></tr>
        <tr><td class="dim text-xs">Shared models</td><td class="mono text-xs">${esc(p.shared_models)}</td></tr>
      `;
    } catch (e) {
      el.innerHTML = `<div class="text-err">Failed to load debug info: ${esc(e)}</div>`;
    }
  }

  function wireReplay() {
    const btn = $("btn-replay-tutorial");
    if (!btn) return;
    btn.addEventListener("click", () => {
      if (window.ftsTutorial) {
        window.ftsTutorial.resetSeen();
        window.ftsTutorial.start({ force: true });
      }
    });
    // Show last-seen timestamp
    try {
      const at = localStorage.getItem("fts.tutorial.seenAt");
      if (at) $("tutorial-last").textContent = new Date(at).toLocaleString();
    } catch (e) {}
  }

  // ── Updates: trigger the self-healing pipeline + live log tail ──
  // Progress is SSE-first via /api/system/update/events. Silent poll of
  // /api/system/update/latest (fallbackMs ≥ 5s) only when EventSource fails.
  const UPD_BTN = { check: 'btn-update-check', update: 'btn-update-apply', repair: 'btn-update-repair' };
  let stopLive = null;

  const updStatus = () => document.getElementById('update-status');
  const updLog = () => document.getElementById('update-log');

  function updEnable() { Object.values(UPD_BTN).forEach(id => { const b = document.getElementById(id); if (b) b.disabled = false; }); }
  function updDisableAll() { Object.values(UPD_BTN).forEach(id => { const b = document.getElementById(id); if (b) b.disabled = true; }); }

  function updRenderLog(text) {
    const el = updLog();
    if (!el) return;
    if (!text) { el.style.display = 'none'; return; }
    el.style.display = 'block';
    el.textContent = text;
    el.scrollTop = el.scrollHeight;
  }

  async function updHistory() {
    const el = document.getElementById('update-history');
    try {
      const rows = await (await fetch('/api/system/updates?limit=5')).json();
      if (!Array.isArray(rows) || !rows.length) { el.innerHTML = ''; return; }
      el.innerHTML = 'Recent: ' + rows.map(u =>
        `${esc(u.mode)}→<b>${esc(u.status)}</b> ${new Date(((u.finished_at || u.created_at) || 0) * 1000).toLocaleString()}`
      ).join(' · ');
    } catch (e) { /* history is decorative — never block the page */ }
  }

  function updStopLive() {
    if (stopLive) { try { stopLive(); } catch (_e) { /* ignore */ } stopLive = null; }
  }

  async function updShowFinished() {
    updStopLive();
    updEnable();
    try {
      const rows = await (await fetch('/api/system/updates?limit=1')).json();
      const u = Array.isArray(rows) && rows[0];
      if (u) {
        updStatus().textContent = `${u.mode} · ${u.status}${u.status === 'done' ? ' ✓' : ' — see log above'}`;
        if (u.log_text) updRenderLog(String(u.log_text).slice(-4000));
      } else {
        updStatus().textContent = 'Finished ✓';
      }
    } catch (e) { updStatus().textContent = 'Finished ✓'; }
    updHistory();
  }

  function updApplySnapshot(d) {
    if (!d || !d.exists) {
      // No in-progress row — pull final history (covers fast runs + post-restart).
      updShowFinished();
      return;
    }
    updRenderLog(d.log_tail || '');
    if (d.status === 'queued' || d.status === 'running') {
      updStatus().textContent = `${d.mode} · ${d.status}…`;
      return;
    }
    updStopLive();
    updEnable();
    updStatus().textContent = `${d.mode} · ${d.status}${d.status === 'done' ? ' ✓' : ' — see log above'}`;
    updHistory();
  }

  function updStartLive() {
    updStopLive();
    stopLive = window.fts.subscribe('/api/system/update/events', updApplySnapshot, {
      pollUrl: '/api/system/update/latest',
      fallbackMs: 5000,
    });
  }

  async function updTickSilent() {
    try {
      const d = await (await fetch('/api/system/update/latest')).json();
      if (!d.exists) return;
      updDisableAll();
      updApplySnapshot(d);
      updStartLive();
    } catch (e) { /* ignore cold-load errors */ }
  }

  async function updTrigger(mode) {
    updDisableAll();
    updStatus().textContent = mode === 'check'
      ? 'Checking (dry run)…'
      : mode === 'repair'
        ? 'Repairing — recreates the venv, several minutes…'
        : 'Updating — the service will restart shortly…';
    try {
      const r = await fetch('/api/system/update', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode, triggered_by: 'settings-ui' }),
      });
      const d = await r.json().catch(() => ({}));
      if (!r.ok || d.error) throw new Error(d.error || d.detail || 'HTTP ' + r.status);
      updStartLive();
    } catch (e) {
      updStatus().textContent = 'Failed to start: ' + e.message;
      updEnable();
    }
  }

  const bCheck = document.getElementById(UPD_BTN.check);
  const bApply = document.getElementById(UPD_BTN.update);
  const bRepair = document.getElementById(UPD_BTN.repair);
  if (bCheck) bCheck.addEventListener('click', () => updTrigger('check'));
  if (bApply) bApply.addEventListener('click', async () => {
    if (await window.fts.confirm('Apply update? Pulls main, syncs deps, runs migrations and restarts the service (about 10s).', {danger: true, okText: 'Apply update'})) updTrigger('update');
  });
  if (bRepair) bRepair.addEventListener('click', async () => {
    if (await window.fts.confirm('Repair recreates the Python venv from scratch — several minutes, then restarts the service. Continue?', {danger: true, okText: 'Repair'})) updTrigger('repair');
  });
  updHistory();
  updTickSilent();  // resume the live view if an update is already running
  // The SSE stream must not outlive this page after an SPA navigation.
  document.addEventListener('fts:beforeNavigate', updStopLive, { once: true });

  wireReplay();
  await loadDebug();
  wireHosting();
})();

/* ============================================================
   Hosting settings — port, CORS, trusted hosts, proxy
   ============================================================ */
function wireHosting() {
  const $ = (id) => document.getElementById(id);
  const statusEl = $('hosting-status');
  const setStatus = (msg) => { if (statusEl) statusEl.textContent = msg; };

  async function loadHosting() {
    try {
      const r = await fetch('/api/settings');
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const s = await r.json();
      if ($('hosting-port')) $('hosting-port').value = s.port || 7860;
      if ($('hosting-host')) $('hosting-host').value = s.host || '0.0.0.0';
      if ($('hosting-cors')) $('hosting-cors').value = (s.cors_origins || []).join('\n');
      if ($('hosting-cors-creds')) $('hosting-cors-creds').checked = !!s.cors_allow_credentials;
      if ($('hosting-trusted')) $('hosting-trusted').value = (s.trusted_hosts || []).join('\n');
      if ($('hosting-proxy')) $('hosting-proxy').checked = !!s.proxy_headers;
      if ($('hosting-rootpath')) $('hosting-rootpath').value = s.root_path || '';
    } catch (e) {
      setStatus('Failed to load: ' + e.message);
    }
  }

  function parseList(ta) {
    if (!ta) return [];
    return ta.value.split('\n').map(s => s.trim()).filter(Boolean);
  }

  async function saveHosting() {
    const port = parseInt($('hosting-port')?.value, 10);
    if (!port || port < 1 || port > 65535) {
      setStatus('Invalid port (must be 1–65535)');
      return;
    }
    const body = {
      port: port,
      host: $('hosting-host')?.value || '0.0.0.0',
      cors_origins: parseList($('hosting-cors')),
      cors_allow_credentials: !!$('hosting-cors-creds')?.checked,
      trusted_hosts: parseList($('hosting-trusted')),
      proxy_headers: !!$('hosting-proxy')?.checked,
      root_path: $('hosting-rootpath')?.value || '',
    };
    try {
      const r = await fetch('/api/settings', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const d = await r.json();
      if (!r.ok) {
        throw new Error(typeof d.detail === 'string' ? d.detail : 'HTTP ' + r.status);
      }
      setStatus('Saved. Restarting service…');
      try {
        await fetch('/api/settings/reload', { method: 'POST' });
      } catch (e) { /* ignore — the restart will happen via systemd */ }
      setStatus('Saved. Service restarting…');
    } catch (e) {
      setStatus('Failed: ' + e.message);
    }
  }

  const saveBtn = $('btn-hosting-save');
  if (saveBtn) saveBtn.addEventListener('click', saveHosting);
  loadHosting();
}


/* ============================================================
   Benchmark judge settings — key is write-only (set / not set)
   ============================================================ */
function wireJudge() {
  const $ = (id) => document.getElementById(id);
  if (!$('judge-card')) return;
  const setStatus = (m) => { if ($('judge-status')) $('judge-status').textContent = m; };

  function apply(j) {
    $('judge-mode').value = j.mode;
    $('judge-model').value = j.model || '';
    $('judge-api-url').value = j.api_url || '';
    $('judge-api-key').value = '';
    $('judge-key-state').textContent = j.api_key_set ? '(set)' : '(not set)';
  }

  async function load() {
    try {
      const r = await fetch('/api/settings/judge');
      if (!r.ok) throw new Error('HTTP ' + r.status);
      apply(await r.json());
    } catch (e) { setStatus('Failed to load: ' + e.message); }
  }

  async function save(extra) {
    const body = Object.assign({
      judge_mode: $('judge-mode').value,
      judge_model: $('judge-model').value.trim(),
      judge_api_url: $('judge-api-url').value.trim(),
      judge_api_key: $('judge-api-key').value,
    }, extra || {});
    try {
      const r = await fetch('/api/settings', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(typeof d.detail === 'string' ? d.detail : 'HTTP ' + r.status);
      apply(d.judge);
      setStatus('Saved.');
    } catch (e) { setStatus('Failed: ' + e.message); }
  }

  $('btn-judge-save').addEventListener('click', () => save());
  $('btn-judge-clear-key').addEventListener('click', () => save({ judge_api_key: '', judge_api_key_clear: true }));
  load();
}
wireJudge();
