/* ============================================================
   Settings page — debug info loader + tutorial replay
   ============================================================ */
(async function () {
  const $ = (id) => document.getElementById(id);

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
             <span class="gpu-name">${g.name}</span>
             <span class="gpu-vram">${humanBytes(g.vram_total_mb * 1024 * 1024)} (${humanBytes(g.vram_free_mb * 1024 * 1024)} free)</span>
             <span class="gpu-driver dim">driver ${g.driver}</span>
           </div>`
        ).join("");
      } else {
        gpuHtml = `<div class="dim text-xs">No NVIDIA GPU detected${d.gpu_error ? ` — ${d.gpu_error}` : ""}.</div>`;
      }

      const pkgHtml = Object.entries(d.packages || {})
        .map(([k, v]) => `<tr><td class="mono">${k}</td><td class="mono dim">${v}</td></tr>`)
        .join("");

      el.innerHTML = `
        ${row("Release", `<span class="text-accent mono">${d.release_channel || "EARLY BETA"}</span> · <span class="mono">v${d.app_version}</span>`)}
        ${row("Python", `<span class="mono">${d.python}</span>`)}
        ${row("Platform", d.platform)}
        ${row("Hostname", d.hostname)}
        ${row("GPU", gpuHtml)}
        <div class="debug-label">Packages</div>
        <div class="debug-value">
          <table class="pkg-table">${pkgHtml}</table>
        </div>
      `;

      const p = d.paths || {};
      pathsEl.innerHTML = `
        <tr><td class="dim text-xs">Data directory</td><td class="mono text-xs">${p.data_dir}</td></tr>
        <tr><td class="dim text-xs">HF cache</td><td class="mono text-xs">${p.hf_cache}</td></tr>
        <tr><td class="dim text-xs">Shared models</td><td class="mono text-xs">${p.shared_models}</td></tr>
      `;
    } catch (e) {
      el.innerHTML = `<div class="text-err">Failed to load debug info: ${e}</div>`;
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
        `${u.mode}→<b>${u.status}</b> ${new Date(((u.finished_at || u.created_at) || 0) * 1000).toLocaleString()}`
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
    const sub = window.fts && window.fts.subscribe;
    if (sub) {
      stopLive = sub('/api/system/update/events', updApplySnapshot, {
        pollUrl: '/api/system/update/latest',
        fallbackMs: 5000,
      });
      return;
    }
    // No fts.subscribe (very old page load) — slow poll only, never 2s redraw.
    const tick = async () => {
      try {
        const d = await (await fetch('/api/system/update/latest', { cache: 'no-store' })).json();
        updApplySnapshot(d);
      } catch (e) { /* mid-restart */ }
    };
    tick();
    const t = setInterval(tick, 5000);
    stopLive = () => clearInterval(t);
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
      const d = await r.json();
      if (d.error) throw new Error(d.error);
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
  if (bApply) bApply.addEventListener('click', () => {
    if (confirm('Apply update? Pulls main, syncs deps, runs migrations and RESTARTS the service (~10s).')) updTrigger('update');
  });
  if (bRepair) bRepair.addEventListener('click', () => {
    if (confirm('Repair recreates the Python venv from scratch — several minutes, then restarts the service. Continue?')) updTrigger('repair');
  });
  updHistory();
  updTickSilent();  // resume the live view if an update is already running

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
      if (!r.ok) throw new Error(d.detail || 'HTTP ' + r.status);
      setStatus('Saved. Restarting service…');
      // Trigger a restart via the existing update endpoint (but just restart, no pull)
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
