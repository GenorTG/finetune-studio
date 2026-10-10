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
   Test judge — default judge provider + auto-judge toggle
   ============================================================ */
function wireJudge() {
  const $ = (id) => document.getElementById(id);
  if (!$('judge-card')) return;
  const setStatus = (m) => { if ($('judge-status')) $('judge-status').textContent = m; };

  function apply(d) {
    const sel = $('judge-provider');
    sel.innerHTML = '';
    (d.providers || []).forEach((p) => {
      const o = document.createElement('option');
      o.value = p.id;
      o.textContent = p.label + (p.is_helper_seat ? ' (helper seat)' : '') + (p.local ? ' · local GPU' : ' · API');
      sel.appendChild(o);
    });
    sel.value = d.judge_provider_id || d.effective_judge_provider_id || '';
    $('judge-auto').checked = !!d.auto_judge;
  }

  async function load() {
    try {
      const r = await fetch('/api/settings/testing');
      if (!r.ok) throw new Error('HTTP ' + r.status);
      apply(await r.json());
    } catch (e) { setStatus('Failed to load: ' + e.message); }
  }

  async function save() {
    try {
      const r = await fetch('/api/settings/testing', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ judge_provider_id: $('judge-provider').value, auto_judge: $('judge-auto').checked }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(typeof d.detail === 'string' ? d.detail : 'HTTP ' + r.status);
      apply(d);
      setStatus('Saved.');
    } catch (e) { setStatus('Failed: ' + e.message); }
  }

  $('btn-judge-save').addEventListener('click', save);
  load();
}
wireJudge();


/* ============================================================
   Helper model — local GGUF or API provider; key is write-only
   ============================================================ */
function wireHelper() {
  const $ = (id) => document.getElementById(id);
  if (!$('helper-card')) return;
  const setStatus = (m) => { $('helper-status').textContent = m; };
  let presets = [];

  function showFields() {
    $('helper-api-fields').hidden = !$('helper-seat-api').checked;
  }

  function apply(h) {
    presets = h.presets || [];
    const sel = $('helper-api-preset');
    sel.innerHTML = presets.map((p) => `<option value="${p.id}">${p.label}</option>`).join('');
    sel.value = h.api.preset || 'custom';
    $('helper-seat-local').checked = h.seat === 'local';
    $('helper-seat-api').checked = h.seat === 'api';
    $('helper-local-label').textContent = h.local.label ? '— ' + h.local.label : '';
    $('helper-api-url').value = h.api.base_url || '';
    $('helper-api-model').value = h.api.model || '';
    $('helper-api-effort').value = h.api.reasoning_effort || '';
    $('helper-api-key').value = '';
    $('helper-key-state').textContent = h.api.key_set ? '(set)' : '(not set)';
    showFields();
  }

  function body(extra) {
    return Object.assign({
      seat: $('helper-seat-api').checked ? 'api' : 'local',
      preset: $('helper-api-preset').value,
      base_url: $('helper-api-url').value.trim(),
      model: $('helper-api-model').value.trim(),
      api_key: $('helper-api-key').value,
      reasoning_effort: $('helper-api-effort').value,
    }, extra || {});
  }

  async function call(url, method, payload) {
    const r = await fetch(url, {
      method, headers: { 'Content-Type': 'application/json' },
      body: payload ? JSON.stringify(payload) : undefined,
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.error || (typeof d.detail === 'string' ? d.detail : 'HTTP ' + r.status));
    return d;
  }

  async function save(extra) {
    setStatus('Saving…');
    try { apply(await call('/api/settings/helper', 'PUT', body(extra))); setStatus('Saved.'); }
    catch (e) { setStatus('Failed: ' + e.message); }
  }

  $('helper-api-preset').addEventListener('change', () => {
    const p = presets.find((x) => x.id === $('helper-api-preset').value);
    if (p && p.id !== 'custom') {
      $('helper-api-url').value = p.base_url;
      if (!$('helper-api-model').value) $('helper-api-model').value = p.model;
    }
  });
  document.querySelectorAll('input[name="helper-seat"]').forEach((el) => el.addEventListener('change', showFields));
  $('btn-helper-save').addEventListener('click', () => save());
  $('btn-helper-clear-key').addEventListener('click', () => save({ api_key: '', clear_key: true }));
  $('btn-helper-test').addEventListener('click', async () => {
    setStatus('Testing…');
    try {
      const d = await call('/api/settings/helper/test', 'POST', body());
      setStatus(`Works: "${d.reply}" in ${(d.latency_ms / 1000).toFixed(1)} s.`);
    } catch (e) { setStatus('Test failed: ' + e.message); }
  });
  $('btn-helper-models').addEventListener('click', async () => {
    setStatus('Fetching models…');
    try {
      const d = await call('/api/settings/helper/models', 'POST', body());
      $('helper-api-model-list').innerHTML = d.models.map((m) => `<option value="${m}">`).join('');
      setStatus(`${d.models.length} models available — type to filter the Model field.`);
    } catch (e) { setStatus('Failed: ' + e.message); }
  });
  call('/api/settings/helper', 'GET').then(apply).catch((e) => setStatus('Failed to load: ' + e.message));
}
wireHelper();


/* ============================================================
   Compute device — persisted choice, applied at the next start
   ============================================================ */
function wireCompute() {
  const $ = (id) => document.getElementById(id);
  if (!$('compute-card')) return;
  const setStatus = (m) => { $('compute-status').textContent = m; };
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
  const fmtGb = (n) => Number(n).toFixed(1) + ' GB';
  let state = null;

  function choiceValue(c) {
    return c.mode === 'gpu' ? c.device_id : c.mode;
  }

  function renderTable(d) {
    const savedId = d.saved.mode === 'gpu' ? d.saved.device_id : '';
    const rows = d.devices.map((g) => {
      const pills = [];
      if (g.active) pills.push('<span class="pill solid-green">in use</span>');
      if (!g.visible) pills.push('<span class="pill outline" title="Hidden from this process by a visibility setting">masked</span>');
      if (g.id === savedId) pills.push('<span class="pill solid-blue">saved</span>');
      return `<tr data-device-id="${esc(g.id)}">
        <td class="cell-wrap">${esc(g.name)} <span class="dim">· #${g.index}${g.name.toUpperCase().includes(g.vendor.toUpperCase()) ? '' : ' · ' + esc(g.vendor)}</span></td>
        <td>${esc(g.runtime)}</td>
        <td class="num">${fmtGb(g.free_gb)} / ${fmtGb(g.total_gb)}</td>
        <td>${pills.join(' ') || '<span class="dim">—</span>'}</td></tr>`;
    });
    $('compute-table').tBodies[0].innerHTML = rows.join('')
      || '<tr class="empty-row"><td colspan="4" class="dim text-xs">No selectable NVIDIA/AMD GPU detected (nvidia-smi / rocm-smi).</td></tr>';
  }

  function renderSelect(d) {
    const sel = $('compute-select');
    const opts = [['auto', 'Auto (best detected GPU, CPU only if there is none)']]
      .concat(d.devices.map((g) => [g.id, `${g.name} (#${g.index})`]))
      .concat([['cpu', 'CPU only (no GPU is touched)']]);
    if (d.saved.mode === 'gpu' && !d.saved.available) {
      opts.push([d.saved.device_id, `${d.saved.name} (saved, not detected)`]);
    }
    sel.innerHTML = opts.map(([v, label]) => `<option value="${esc(v)}">${esc(label)}</option>`).join('');
    sel.value = choiceValue(d.saved);
    sel.disabled = false;
    $('btn-compute-save').disabled = false;
  }

  function renderBanners(d) {
    const e = d.effective;
    $('compute-effective').innerHTML = `In use now: <span class="mono">${esc(e.device)}</span> · ${esc(e.name)} · ${esc(e.runtime)}`
      + (e.note ? ` — <span class="warn">${esc(e.note)}</span>` : '');
    const env = $('compute-env');
    env.hidden = !d.overridden_by_env;
    if (d.overridden_by_env) {
      env.textContent = `The saved choice is not applied: ${d.env_overrides.join(', ')} ${d.env_overrides.length > 1 ? 'are' : 'is'} set in the service environment and takes precedence. Remove it from the unit or its drop-in, then restart.`;
    }
    const rs = $('compute-restart');
    rs.hidden = !d.restart_required;
    if (d.restart_required) {
      rs.innerHTML = `Restart required — the saved choice takes effect when the service restarts. ${d.restart_command.startsWith('fts ') ? `Press <em>Restart web server</em> in the Service card above, or run <code>${esc(d.restart_command)}</code> on the host.` : `To apply it: ${esc(d.restart_command)}.`}`;
    }
  }

  function render(d) {
    state = d;
    renderTable(d);
    renderSelect(d);
    renderBanners(d);
  }

  async function load() {
    try {
      const r = await fetch('/api/system/compute-device', { cache: 'no-store' });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      render(await r.json());
    } catch (e) {
      setStatus('Failed to load: ' + e.message);
      $('compute-effective').textContent = 'Could not read the compute devices.';
    }
  }

  async function save() {
    const v = $('compute-select').value;
    const body = v === 'auto' || v === 'cpu' ? { mode: v } : { mode: 'gpu', device_id: v };
    if (body.mode === 'cpu' && state && state.devices.length
        && !(await window.fts.confirm('CPU only: training and inference will be much slower and no GPU is used. Save this choice?', { danger: true, okText: 'Use CPU only' }))) {
      return;
    }
    try {
      const r = await fetch('/api/system/compute-device', {
        method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(typeof d.detail === 'string' ? d.detail : 'HTTP ' + r.status);
      render(d);
      setStatus(d.restart_required ? 'Saved — restart required.' : 'Saved.');
    } catch (e) { setStatus('Failed: ' + e.message); }
  }

  $('btn-compute-save').addEventListener('click', save);
  load();
}
wireCompute();
