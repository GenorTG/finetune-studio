/* ============================================================
   Settings page — debug info loader + tutorial replay
   ============================================================ */
// Saved API keys are encrypted at rest; "unreadable" means the vault key was lost and the key must be entered again.
// Top level: the AI assistant card below is wired outside the debug-info closure.
function keyStateText(a) {
  if (a.key_state === 'unreadable') return '(saved key unreadable: enter it again)';
  return a.key_set ? '(saved, encrypted)' : '(not set)';
}

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
      const d = await api.get("/api/debug/info");

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
  if (!document.getElementById('update-status')) {
    wireReplay();
    await loadDebug();
    return;
  }
  const UPD_BTN = { check: 'btn-update-check', update: 'btn-update-apply', repair: 'btn-update-repair' };
  let stopLive = null;

  const updStatus = () => document.getElementById('update-status');
  const updLog = () => document.getElementById('update-log');

  function updEnable() { Object.values(UPD_BTN).forEach(id => { const b = document.getElementById(id); if (b) b.disabled = false; }); }
  function updDisableAll() { Object.values(UPD_BTN).forEach(id => { const b = document.getElementById(id); if (b) b.disabled = true; }); }

  function updRenderLog(text) {
    const el = updLog();
    if (!el) return;
    if (!text) { el.hidden = true; return; }
    el.hidden = false;
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
      await api.post('/api/system/update', { mode, triggered_by: 'settings-ui' });
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
})();



/* ============================================================
   Test judge — default judge provider + auto-judge toggle
   ============================================================ */
function wireJudge() {
  const $ = (id) => document.getElementById(id);
  if (!$('judge-card')) return;
  const setStatus = (m) => { if ($('judge-status')) $('judge-status').textContent = m; };

  async function load() {
    try { $('judge-auto').checked = !!(await api.get('/api/settings/testing')).auto_judge; }
    catch (e) { setStatus('Failed to load: ' + e.message); }
  }

  async function save() {
    try {
      const d = await api.put('/api/settings/testing', { auto_judge: $('judge-auto').checked });
      $('judge-auto').checked = !!d.auto_judge;
      setStatus('Saved.');
    } catch (e) { setStatus('Failed: ' + e.message); }
  }

  $('btn-judge-save').addEventListener('click', save);
  load();
}
wireJudge();


/* ============================================================
   AI assistant — connections (local GGUF + API, keys write-only),
   the main assistant, and the jobs it does (models/assistant.py)
   ============================================================ */
function wireAssistant() {
  const $ = (id) => document.getElementById(id);
  if (!$('assistant-card')) return;
  const setStatus = (m) => { $('assistant-status').textContent = m; };
  const esc = (v) => String(v == null ? '' : v).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  let st = null;

  async function call(method, url, payload) { return api.request(method, url, payload); }
  async function run(label, fn) {
    setStatus(label);
    try { const out = await fn(); return out; }
    catch (e) { setStatus('Failed: ' + e.message); return null; }
  }

  // ── render ──
  function render(d) {
    st = d;
    renderMain(d); renderConnections(d); renderLocal(d); renderJobs(d);
  }

  function renderMain(d) {
    const sel = $('assistant-main');
    sel.innerHTML = d.connections.map((c) => {
      const usable = c.kind !== 'openai_compat' || c.key_set;
      return `<option value="${esc(c.id)}"${usable ? '' : ' disabled'}>${esc(c.label)}${c.kind === 'openai_compat' ? ' · API' : ' · this machine'}${usable ? '' : ' (no key)'}</option>`;
    }).join('');
    sel.value = d.assistant_provider_id;
  }

  function renderConnections(d) {
    const body = $('connections-table').querySelector('tbody');
    body.innerHTML = d.connections.map((c) => {
      const api = c.kind === 'openai_compat';
      const where = api ? esc(c.base_url) : (c.present ? 'this machine' : 'this machine · <span class="sev-warning">file missing</span>');
      const key = api ? esc(keyStateText(c)) : '—';
      const used = [c.is_assistant ? '<b>main assistant</b>' : '', ...(c.used_by || []).map(esc)].filter(Boolean).join(', ') || '—';
      const actions = api
        ? `<button class="btn sm" type="button" data-conn-edit="${esc(c.id)}">Edit</button> ` +
          `<button class="btn sm" type="button" data-conn-test="${esc(c.id)}">Test</button> ` +
          `<button class="btn sm danger" type="button" data-conn-remove="${esc(c.id)}"${c.is_assistant ? ' disabled title="Pick another main assistant first"' : ''}>Remove</button>`
        : `<button class="btn sm" type="button" data-conn-local-test="1">Test</button>`;
      return `<tr><td class="cell-wrap">${esc(c.label)}</td><td class="text-xs cell-wrap">${where}</td><td class="text-xs cell-wrap">${key}</td><td class="text-xs cell-wrap">${used}</td><td class="cell-wrap">${actions}</td></tr>`;
    }).join('');
  }

  let rec = null;
  function renderLocal(d) {
    const row = d.connections.find((c) => c.id === d.local.provider_id) || {};
    const path = row.path || '';
    rec = d.local.recommended || null;
    $('local-path').textContent = path || '(none)';
    $('local-present').textContent = row.present ? '✓ installed' : '✗ file missing';
    $('local-present').className = row.present ? 'sev-ok' : 'sev-warning';
    const cands = d.local.candidates || [];
    $('local-select').innerHTML = cands.length
      ? cands.map((c) => `<option value="${esc(c.path)}"${c.path === path ? ' selected' : ''}>${esc(c.name)}${c.size_gb ? ' · ' + c.size_gb + ' GB' : ''}</option>`).join('')
      : '<option value="">No GGUF installed yet</option>';
    $('btn-local-use').disabled = !cands.length;
    if (!rec) { $('local-recommended').hidden = true; return; }
    $('local-rec-label').textContent = rec.label;
    $('local-rec-meta').textContent = `· ${rec.size_gb} GB · ${rec.license} · ${rec.repo_id}`;
    $('local-rec-why').textContent = rec.why;
    const inUse = rec.installed && path === rec.path;
    const btn = $('btn-local-rec');
    btn.textContent = inUse && rec.vision_installed ? 'In use'
      : rec.installed && !rec.vision_installed ? 'Download vision file and use'
        : rec.installed ? 'Use it' : `Download (${rec.size_gb} GB) and use`;
    btn.disabled = inUse && rec.vision_installed;
  }

  function jobOptions(job, d) {
    const main = d.connections.find((c) => c.id === d.assistant_provider_id) || {};
    const opts = [];
    if (job.id === 'vision') opts.push(['off', 'Off (OCR text only)']);
    const mainLocal = main.kind !== 'openai_compat';
    opts.push(['', `Main assistant (${main.label || d.assistant_provider_id})` + (job.api_only && mainLocal ? ' — needs an API connection' : '')]);
    d.connections.forEach((c) => {
      if (job.api_only && c.kind !== 'openai_compat') return;
      if (c.kind === 'openai_compat' && !c.key_set) return;
      opts.push([c.id, c.label + (c.kind === 'openai_compat' ? ' · API' : ' · this machine')]);
    });
    if (job.id === 'vision') opts.push(['local-file', 'A local vision GGUF file…']);
    return opts;
  }

  function renderJobs(d) {
    const body = $('assistant-jobs').querySelector('tbody');
    body.innerHTML = d.jobs.map((j) => {
      const opts = jobOptions(j, d).map(([v, l]) => `<option value="${esc(v)}"${v === j.setting ? ' selected' : ''}>${esc(l)}</option>`).join('');
      const instr = j.instructions || '';
      return `<tr data-job="${esc(j.id)}">
        <td class="cell-wrap"><b>${esc(j.label)}</b><div class="text-xs dim">${esc(j.does)}</div>
          <details class="mt-1"${instr ? ' open' : ''}><summary class="text-xs">Extra instructions${instr ? ' (set)' : ''}</summary>
            <textarea class="input max-w-420 mt-1" rows="3" maxlength="4000" data-job-instr="${esc(j.id)}" aria-label="Extra instructions for ${esc(j.label)}" placeholder="Optional: added in front of this job's own prompt, e.g. 'Write questions in Polish.'">${esc(instr)}</textarea>
          </details></td>
        <td class="cell-wrap"><select class="select" data-job-select="${esc(j.id)}" aria-label="Connection for ${esc(j.label)}">${opts}</select>
          ${j.api_only && j.provider_kind !== 'openai_compat'
            ? '<div class="text-xs sev-warning mt-1">needs an API connection: pick one here (a local model cannot do this job)</div>'
            : `<div class="text-xs dim mt-1">now: ${esc(j.provider_label || j.provider_id)}</div>`}</td>
        <td class="text-xs cell-wrap">${esc(j.sends)}</td>
        <td><button class="btn sm primary" type="button" data-job-save="${esc(j.id)}">Save</button></td>
      </tr>`;
    }).join('');
    const vision = d.jobs.find((j) => j.id === 'vision') || {};
    $('vision-local-list').innerHTML = (d.vision_candidates || []).map((c) => `<option value="${esc(c.path)}" label="${esc(c.label)}">`).join('');
    if (!$('vision-local-model').value) $('vision-local-model').value = vision.local_model || '';
    $('vision-local-fields').hidden = vision.setting !== 'local-file';
  }

  // ── connection form ──
  function openForm(conn) {
    const c = conn || {};
    $('conn-form').hidden = false;
    $('conn-form-title').textContent = conn ? 'Edit ' + c.name : 'New API connection';
    $('conn-id').value = c.id || '';
    $('conn-name').value = c.name || '';
    $('conn-preset').innerHTML = (st.presets || []).map((p) => `<option value="${esc(p.id)}">${esc(p.label)}</option>`).join('');
    $('conn-preset').value = c.preset || 'custom';
    $('conn-url').value = c.base_url || '';
    $('conn-model').value = c.model || '';
    $('conn-effort').value = c.reasoning_effort || '';
    $('conn-concurrency').value = c.concurrency || '';
    $('conn-key').value = '';
    $('conn-key-state').textContent = conn ? keyStateText(c) : '(not set)';
    $('btn-conn-clear-key').hidden = !conn;
    $('conn-name').focus();
  }

  function formBody(extra) {
    const id = $('conn-id').value;
    const b = {
      name: $('conn-name').value.trim(), preset: $('conn-preset').value, base_url: $('conn-url').value.trim(),
      model: $('conn-model').value.trim(), reasoning_effort: $('conn-effort').value,
    };
    const conc = parseInt($('conn-concurrency').value, 10);
    if (conc) b.concurrency = conc;
    if ($('conn-key').value) b.api_key = $('conn-key').value;
    if (id) b.id = id;
    return Object.assign(b, extra || {});
  }

  async function saveForm(extra) {
    const id = $('conn-id').value;
    const body = formBody(extra);
    delete body.id;
    const d = await run('Saving the connection…', () => id
      ? call('PUT', '/api/settings/connections/' + encodeURIComponent(id), body)
      : call('POST', '/api/settings/connections', body));
    if (!d) return;
    render(d);
    const saved = d.connections.find((c) => c.id === d.id);
    if (saved) openForm(saved);
    setStatus(extra && extra.clear_key ? 'Key removed.' : 'Connection saved. Pick it as main assistant or for a job below.');
  }

  async function testConn(body) {
    const d = await run('Testing the connection…', () => call('POST', '/api/settings/connections/test', body));
    if (d) setStatus(`Works: "${d.reply}" (answered in ${(d.latency_ms / 1000).toFixed(1)} s).`);
  }

  async function testLocal() {
    const d = await run('Loading the local model and asking it… (a 12B takes about a minute)',
      () => call('POST', '/api/settings/helper/test', {}));
    if (d) setStatus(`Local model works: "${d.reply}" (${d.load_ms ? `loaded in ${(d.load_ms / 1000).toFixed(0)} s, ` : ''}answered in ${(d.latency_ms / 1000).toFixed(1)} s)${d.load_ms ? '; unloaded again.' : '.'}`);
  }

  async function useLocal(path) {
    const d = await run('Switching the local model…', () => call('POST', '/api/settings/helper/local', { path }));
    if (d) { render(d); setStatus('Local model set.'); }
  }

  async function downloadRecommended() {
    const status = $('local-rec-status');
    const btn = $('btn-local-rec');
    if (rec && rec.installed && rec.vision_installed) { await useLocal(rec.path); return; }
    btn.disabled = true;
    status.textContent = 'Starting the download…';
    let started;
    try { started = await call('POST', '/api/settings/helper/recommended/download', {}); }
    catch (e) { status.textContent = 'Could not start: ' + e.message; btn.disabled = false; return; }
    // Poll every job until done; the files land in Model library, so leaving the page does not stop them.
    for (;;) {
      let done = 0, total = 0, have = 0, failed = '';
      for (const j of started.jobs) {
        const d = await call('GET', '/api/hf/download/progress?job_id=' + encodeURIComponent(j.job_id)).catch(() => ({}));
        total += Number(d.bytes_total || 0); have += Number(d.bytes_done || 0);
        if (d.status === 'completed') done += 1;
        else if (['error', 'failed', 'cancelled', 'crashed', 'stopped'].includes(d.status)) failed = `${j.file}: ${d.error || d.status}`;
      }
      if (failed) { status.textContent = 'Download failed: ' + failed; btn.disabled = false; return; }
      if (done === started.jobs.length) break;
      status.textContent = total ? `Downloading… ${(have / 1e9).toFixed(1)} of ${(total / 1e9).toFixed(1)} GB` : 'Downloading…';
      await new Promise((r) => setTimeout(r, 2000));
      if (!document.body.contains(btn)) return;   // left the page: Model library keeps the download going
    }
    status.textContent = 'Downloaded.';
    await useLocal(started.path);
  }

  async function saveJob(id) {
    const sel = document.querySelector(`[data-job-select="${id}"]`);
    const instr = document.querySelector(`[data-job-instr="${id}"]`);
    const body = { provider_id: sel.value, instructions: instr ? instr.value : '' };
    if (id === 'vision' && sel.value === 'local-file') body.local_model = $('vision-local-model').value.trim();
    const d = await run('Saving the job…', () => call('PUT', '/api/settings/assistant/jobs/' + encodeURIComponent(id), body));
    if (d) { render(d); setStatus('Saved: ' + ((d.jobs.find((j) => j.id === id) || {}).label || id) + '.'); }
  }

  // ── events ──
  $('btn-assistant-main').addEventListener('click', async () => {
    const d = await run('Changing the main assistant…', () => call('PUT', '/api/settings/assistant', { provider_id: $('assistant-main').value }));
    if (d) { render(d); setStatus('Main assistant: ' + d.assistant_label + '.'); }
  });
  $('btn-conn-add').addEventListener('click', () => openForm(null));
  $('btn-conn-cancel').addEventListener('click', () => { $('conn-form').hidden = true; });
  $('btn-conn-save').addEventListener('click', () => saveForm());
  $('btn-conn-clear-key').addEventListener('click', () => saveForm({ clear_key: true }));
  $('btn-conn-test').addEventListener('click', () => testConn(formBody()));
  $('btn-conn-models').addEventListener('click', async () => {
    const d = await run('Fetching models…', () => call('POST', '/api/settings/connections/models', formBody()));
    if (!d) return;
    $('conn-model-list').innerHTML = d.models.map((m) => `<option value="${esc(m)}">`).join('');
    setStatus(`${d.models.length} models available — type to filter the Model field.`);
  });
  $('conn-preset').addEventListener('change', () => {
    const p = (st.presets || []).find((x) => x.id === $('conn-preset').value);
    if (p && p.id !== 'custom') {
      $('conn-url').value = p.base_url;
      if (!$('conn-model').value) $('conn-model').value = p.model;
      if (!$('conn-name').value) $('conn-name').value = p.label;
    }
  });
  $('connections-table').addEventListener('click', async (ev) => {
    const t = ev.target.closest('button');
    if (!t) return;
    if (t.dataset.connEdit) openForm(st.connections.find((c) => c.id === t.dataset.connEdit));
    else if (t.dataset.connTest) testConn({ id: t.dataset.connTest });
    else if (t.dataset.connLocalTest) testLocal();
    else if (t.dataset.connRemove) {
      const c = st.connections.find((x) => x.id === t.dataset.connRemove);
      const ok = window.fts && window.fts.confirm
        ? await window.fts.confirm(`Remove ${c.label} and its saved key? Jobs that used it go back to the main assistant (reading pictures: off).`, { danger: true, okText: 'Remove connection' })
        : window.confirm(`Remove ${c.label}?`);
      if (!ok) return;
      const d = await run('Removing…', () => call('DELETE', '/api/settings/connections/' + encodeURIComponent(c.id)));
      if (d) { render(d); $('conn-form').hidden = true; setStatus('Connection removed.'); }
    }
  });
  $('assistant-jobs').addEventListener('click', (ev) => {
    const t = ev.target.closest('button[data-job-save]');
    if (t) saveJob(t.dataset.jobSave);
  });
  $('assistant-jobs').addEventListener('change', (ev) => {
    const sel = ev.target.closest('select[data-job-select="vision"]');
    if (sel) $('vision-local-fields').hidden = sel.value !== 'local-file';
  });
  $('btn-local-rec').addEventListener('click', downloadRecommended);
  $('btn-local-use').addEventListener('click', () => { const v = $('local-select').value; if (v) useLocal(v); });
  $('btn-local-test').addEventListener('click', testLocal);
  $('btn-vision-test').addEventListener('click', async () => {
    const d = await run('Reading a test picture (a local model loads first, up to a minute)…', () => call('POST', '/api/settings/vision/test', {}));
    if (d) setStatus(`Reading pictures works: read the code in ${(d.latency_ms / 1000).toFixed(1)} s.`);
  });
  call('GET', '/api/settings/assistant').then(render).catch((e) => setStatus('Failed to load: ' + e.message));
}
wireAssistant();


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
      render(await api.get('/api/system/compute-device'));
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
      const d = await api.put('/api/system/compute-device', body);
      render(d);
      setStatus(d.restart_required ? 'Saved — restart required.' : 'Saved.');
    } catch (e) { setStatus('Failed: ' + e.message); }
  }

  $('btn-compute-save').addEventListener('click', save);
  load();
}
wireCompute();
