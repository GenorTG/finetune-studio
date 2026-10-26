/* Training page: "Start from a use case" card (#uc-card). API: webui/routes/usecases.py.

   Build a use case from an examples file (template-specific options; a 422 lists every unusable row), fill the
   training form with its dataset and proposed settings, start a base-vs-run evaluation on its held-out test, read
   the paired report (per-model metric, gained / lost, McNemar or the style A/B judge). Loaded once; the page's inline
   call runs mount() on every visit (SPA navigation swaps the DOM, a src script is not re-run), and mount() is a no-op
   on an element it already mounted. */
(function () {
  'use strict';

  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const pct = (v) => (v == null ? '—' : `${Math.round(v * 1000) / 10} %`);
  const num = (v, d) => (v == null ? '—' : Number(v).toFixed(d == null ? 2 : d));
  const pval = (p) => (p == null ? '—' : (p < 0.001 ? '< 0.001' : '= ' + Number(p).toFixed(3)));
  const notify = (msg, type) => (window.fts && window.fts.notify ? window.fts.notify(msg, type) : null);
  const OPT_FIELDS = { extraction: ['instruction', 'schema'], classification: ['instruction', 'labels'],
    style: ['instruction', 'style_guide'], distillation: ['instruction', 'teacher_provider_id', 'teacher'] };
  // The plan's overrides map 1:1 onto the training form's field names.
  const FORM_FIELDS = ['num_epochs', 'learning_rate', 'lora_rank', 'lora_alpha', 'batch_size',
    'gradient_accumulation_steps', 'max_seq_length'];

  function mount(root) {
    if (!root || root.dataset.ucMounted === '1') return;
    root.dataset.ucMounted = '1';
    const pid = root.dataset.pid;
    const $ = (id) => root.querySelector('#' + id);
    const base = `/api/projects/${encodeURIComponent(pid)}/usecases`;
    const state = { templates: {}, usecases: [], runs: [], open: null, timer: null };
    const stillHere = () => root.isConnected;

    // ── templates + the "new use case" form ──────────────────────────────
    function templateHelp(t) {
      if (!t) return '';
      const shape = Object.entries(t.row_shape || {}).map(([k, v]) => `<code>${esc(k)}</code> ${esc(v)}`).join('; ');
      const needs = t.needs || {};
      const perLabel = needs.per_label_minimum ? `, at least ${needs.per_label_minimum} per label` : '';
      return `<div>${esc(t.summary)}</div>
        <div class="muted mt-1">Each row: ${shape}. Example: <code>${esc(JSON.stringify(t.example_row))}</code></div>
        <div class="muted mt-1">Needs at least ${t.minimum_rows} rows (${needs.minimum} to train + a held-out test of
          ${Math.round((t.heldout || {}).fraction * 100)} %, at least ${(t.heldout || {}).minimum}${perLabel});
          gains are measured at ${(needs.recommended || []).join('–')}. Scored by: ${esc(t.metric)}.</div>
        ${(t.notes || []).map((n) => `<div class="sev-info mt-1">ℹ ${esc(n)}</div>`).join('')}`;
    }

    function syncTemplate() {
      const id = $('uc-template').value;
      const t = state.templates[id];
      $('uc-template-help').innerHTML = templateHelp(t);
      const wanted = OPT_FIELDS[id] || [];
      root.querySelectorAll('[data-uc-opt]').forEach((el) => { el.hidden = !wanted.includes(el.dataset.ucOpt); });
      $('uc-instruction').placeholder = (t && t.default_instruction) || '';
      syncBuild();
    }

    function syncBuild() {
      const ok = !!$('uc-template').value && !!($('uc-file').files && $('uc-file').files[0]);
      $('uc-build-btn').disabled = !ok;
      $('uc-build-btn').title = ok ? 'Check every example and build the dataset + held-out test'
        : 'Pick a template and a file first';
    }

    async function loadTemplates() {
      const list = await api.get('/api/usecases/templates');
      const sel = $('uc-template');
      sel.innerHTML = '<option value="">— pick what the model should do —</option>';
      for (const t of list) {
        state.templates[t.id] = t;
        const o = document.createElement('option');
        o.value = t.id;
        o.textContent = t.title;
        sel.appendChild(o);
      }
      syncTemplate();
    }

    async function loadTeachers() {
      try {
        const d = await api.get('/api/providers');
        const sel = $('uc-teacher');
        for (const p of (d && d.providers) || []) {
          if (p.kind !== 'openai_compat') continue;   // a local GGUF would replace the model on the card
          const o = document.createElement('option');
          o.value = p.id;
          o.textContent = `${p.name || p.id}${p.model_id ? ' · ' + p.model_id : ''}`;
          sel.appendChild(o);
        }
      } catch (_) { /* the list stays "answers are in the file" */ }
    }

    function buildOptions() {
      const id = $('uc-template').value;
      const o = { template: id, name: $('uc-name').value.trim() };
      const instr = $('uc-instruction').value.trim();
      if (instr) o.instruction = instr;
      if (id === 'classification') {
        const labels = $('uc-labels').value.split(',').map((x) => x.trim()).filter(Boolean);
        if (labels.length) o.labels = labels;
      }
      if (id === 'extraction' && $('uc-schema').value.trim()) {
        o.schema = JSON.parse($('uc-schema').value);   // a parse error is reported before anything is sent
      }
      if (id === 'style' && $('uc-style-guide').value.trim()) o.style_guide = $('uc-style-guide').value.trim();
      if (id === 'distillation') {
        if ($('uc-teacher').value) o.teacher_provider_id = $('uc-teacher').value;
        if ($('uc-teacher-name').value.trim()) o.teacher = $('uc-teacher-name').value.trim();
      }
      return o;
    }

    function rowErrors(errors) {
      if (!errors || !errors.length) return '';
      // Each error is one line naming its row ("row 3: missing label"); bold the row part.
      const items = errors.slice(0, 50).map((e) => {
        const text = typeof e === 'string' ? e : JSON.stringify(e);
        const m = /^(row \d+|label [^:]+):\s*(.*)$/.exec(text);
        return m ? `<li><b>${esc(m[1])}</b>: ${esc(m[2])}</li>` : `<li>${esc(text)}</li>`;
      }).join('');
      const more = errors.length > 50 ? `<li class="muted">… and ${errors.length - 50} more</li>` : '';
      return `<ul class="health-list mt-1">${items}${more}</ul>`;
    }

    async function pollTeacher(stop) {
      while (!stop.done && stillHere()) {
        try {
          const p = await api.get(`${base}/teacher/progress`);
          if (p && p.running) $('uc-build-status').textContent = `The teacher is writing answers: ${p.done} of ${p.total}…`;
        } catch (_) { /* best effort */ }
        await new Promise((r) => setTimeout(r, 2000));
      }
    }

    async function build() {
      const out = $('uc-build-result');
      const status = $('uc-build-status');
      let opts;
      try { opts = buildOptions(); } catch (e) {
        out.hidden = false;
        out.innerHTML = `<span class="sev-error">✕ The JSON schema is not valid JSON: ${esc(e.message)}</span>`;
        return;
      }
      const fd = new FormData();
      fd.append('file', $('uc-file').files[0]);
      fd.append('options', JSON.stringify(opts));
      $('uc-build-btn').disabled = true;
      status.textContent = 'Checking the examples…';
      out.hidden = true;
      const stop = { done: false };
      if (opts.teacher_provider_id) pollTeacher(stop);
      try {
        const m = await api.post(`${base}/upload`, fd);
        stop.done = true;
        status.textContent = '';
        out.hidden = false;
        const warn = (m.warnings || []).map((w) => `<li class="sev-warning">⚠ ${esc(w)}</li>`).join('');
        const dropped = (m.dropped_duplicates || m.dropped_conflicts)
          ? `<li class="muted">dropped ${m.dropped_duplicates || 0} duplicate(s) and ${m.dropped_conflicts || 0} conflicting row(s)</li>` : '';
        out.innerHTML = `<span class="sev-ok">✓ Built <b>${esc(m.name)}</b>: ${m.examples} examples →
          ${m.train_rows} training rows + ${m.heldout} held-out test questions.</span>
          <ul class="health-list mt-1">${warn}${dropped}</ul>
          The form below now uses its dataset and settings; pick the base model and start.`;
        $('uc-file').value = '';
        syncBuild();
        await refresh();
        applyToForm(m);
        notify(`Use case ${m.name} built`, 'success');
      } catch (e) {
        stop.done = true;
        status.textContent = '';
        out.hidden = false;
        const d = (e && e.data) || {};
        out.innerHTML = `<span class="sev-error">✕ ${esc(e.message || e)}</span>${rowErrors(d.errors)}`;
        syncBuild();
      }
    }

    // ── fill the training form ───────────────────────────────────────────
    function applyToForm(m) {
      const form = document.getElementById('train-form');
      const plan = m.training || {};
      if (!form || !window.ftsTraining) return;
      const mode = form.querySelector(`input[name="training_mode"][value="${plan.training_mode || 'sft'}"]`);
      if (mode && !mode.checked) { mode.checked = true; mode.dispatchEvent(new Event('change')); }
      const spm = form.querySelector('[name="system_prompt_mode"]');
      if (spm && plan.system_prompt_mode) { spm.value = plan.system_prompt_mode; spm.dispatchEvent(new Event('change')); }
      const preset = document.getElementById('training-preset');
      if (preset) preset.value = '';   // an effort preset would overwrite the sized settings
      const advisory = document.getElementById('preset-advisory');
      if (advisory) advisory.hidden = true;
      for (const name of FORM_FIELDS) {
        const v = (plan.overrides || {})[name];
        const field = form.querySelector(`[name="${name}"]`);
        if (field && v != null) field.value = v;
      }
      const ds = m.dataset || {};
      // The registered dataset name already ends in "· N rows".
      const label = /\brows\b/.test(ds.name || '') ? `${ds.name} · (use case)` : `${ds.name || m.name} · ${ds.rows} rows · (use case)`;
      if (ds.id) window.ftsTraining.selectDataset(ds.id, ds.path, label);
      const note = document.getElementById('preset-note');
      if (note) {
        note.innerHTML = `<b>Settings from use case ${esc(m.name)}</b>: ` +
          (plan.reasons || []).map((r) => esc(r)).join(' · ');
      }
      notify(`Form filled from ${m.name}: pick the base model and start`, 'success');
    }

    // ── the list of use cases ────────────────────────────────────────────
    function lastEval(m) {
      const ev = (m.evaluations || [])[(m.evaluations || []).length - 1];
      return ev ? `run ${esc(ev.run_id)} · ${new Date(ev.started_at * 1000).toLocaleString()}` : 'not evaluated yet';
    }

    function renderList() {
      const el = $('uc-list');
      if (!state.usecases.length) {
        el.innerHTML = '<div class="empty small">No use cases yet. Open <b>New use case…</b> below to build one from your examples.</div>';
        $('uc-new').open = true;
        return;
      }
      const rows = state.usecases.map((m) => {
        const t = state.templates[m.template] || {};
        return `<tr data-uc="${esc(m.id)}">
          <td><b>${esc(m.name)}</b><div class="text-xs muted">${esc(t.title || m.template)}</div></td>
          <td class="num">${m.train_rows} + ${m.heldout}</td>
          <td class="text-xs">${lastEval(m)}</td>
          <td><div class="flex gap-2 wrap">
            <button type="button" class="btn sm primary" data-uc-act="apply">Use in form</button>
            <button type="button" class="btn sm" data-uc-act="open">Evaluate / results</button>
            <button type="button" class="btn sm danger" data-uc-act="delete">Delete</button>
          </div></td></tr>`;
      }).join('');
      el.innerHTML = `<table class="table compact" id="uc-table">
        <thead><tr><th>Use case</th><th class="num">Train + test rows</th><th>Last evaluation</th><th>Actions</th></tr></thead>
        <tbody>${rows}</tbody></table>`;
    }

    async function refresh() {
      state.usecases = await api.get(base);
      renderList();
    }

    async function loadRuns() {
      try {
        const runs = await api.get(`/api/projects/${encodeURIComponent(pid)}/runs`);
        state.runs = (Array.isArray(runs) ? runs : []).filter((r) => r.status === 'done' && r.output_path);
      } catch (_) { state.runs = []; }
    }

    // ── one use case: evaluate + report ──────────────────────────────────
    function metricRows(template, side, label) {
      const m = side.metrics || {};
      const s = side.scores || {};
      let cells;
      if (template === 'extraction') {
        cells = `<td class="num">${pct(m.exact_match)}</td><td class="num">${num(m.field_f1)}</td><td class="num">${pct(m.json_valid)}</td>`;
      } else if (template === 'classification') {
        cells = `<td class="num">${pct(m.accuracy)}</td><td class="num">${pct(m.label_only)}</td><td class="num">${pct(m.no_label)}</td>`;
      } else {
        cells = `<td class="num">${pct(s.pass_rate)}</td><td class="num">${s.judged == null ? '—' : s.judged}/${s.total == null ? '—' : s.total}</td><td class="num">${s.awaiting == null ? '—' : s.awaiting}</td>`;
      }
      return `<tr><td>${esc(label)}<div class="text-xs muted">${esc(side.status || '')}${side.error ? ' · ' + esc(side.error) : ''}</div></td>${cells}</tr>`;
    }

    const HEADS = {
      extraction: ['Exact match', 'Field F1', 'Valid JSON'],
      classification: ['Accuracy', 'Label only', 'No label'],
      distillation: ['Agrees with teacher', 'Judged', 'Awaiting judge'],
      style: ['Pass rate', 'Judged', 'Awaiting judge'],
    };

    function pairedLine(p) {
      if (!p) return '';
      const cls = p.verdict === 'better' ? 'sev-ok' : (p.verdict === 'worse' ? 'sev-error' : 'sev-warning');
      return `<div class="mt-2"><b>Paired on the same questions</b> (${p.paired}): trained run gained ${p.gained},
        lost ${p.lost}; both right ${p.both_pass}, both wrong ${p.both_fail}; McNemar p ${pval(p.p_value)} →
        <span class="${cls}"><b>${esc(p.verdict)}</b></span></div>`;
    }

    function abBlock(r) {
      const ab = r.ab;
      const running = r.status && r.status.run === 'running';
      if (r.usecase.template !== 'style') return '';
      if (!ab) {
        return running ? '' : `<div class="mt-2"><button type="button" class="btn sm primary" data-uc-act="ab"
          data-gid="${esc(r.evaluation.group_id)}">Run the blind A/B judge</button>
          <span class="text-xs muted">An AI judge (the Test judge job, Settings → AI assistant) sees both answers in both orders without knowing which is which.</span></div>`;
      }
      if (ab.status === 'running') return `<div class="mt-2 text-xs">A/B judge: ${ab.done} of ${ab.total} questions…</div>`;
      if (ab.status === 'failed') return `<div class="mt-2 sev-error">✕ A/B judge failed: ${esc(ab.error)}</div>`;
      const s = ab.summary || {};
      const cls = s.verdict === 'trained preferred' ? 'sev-ok' : (s.verdict === 'base preferred' ? 'sev-error' : 'sev-warning');
      return `<div class="mt-2"><b>Blind A/B</b> (${s.questions} questions): trained preferred ${s.wins}, base preferred
        ${s.losses}, tie or orders disagreed ${s.ties}${s.errors ? `, errors ${s.errors}` : ''}; win rate ${pct(s.win_rate)};
        sign test p ${pval(s.p_value)} → <span class="${cls}"><b>${esc(s.verdict)}</b></span></div>`;
    }

    // Classification: where each expected label went (rows = expected, columns = predicted), base and trained.
    function confusionHtml(r) {
      const sides = [['Untrained base', r.base], ['Trained run', r.tuned]]
        .filter(([, side]) => side.metrics && side.metrics.confusion);
      if (!sides.length) return '';
      const one = ([title, side]) => {
        const c = side.metrics.confusion;
        const head = c.columns.map((col) => `<th class="num">${esc(col)}</th>`).join('');
        const body = c.labels.map((lab) => `<tr><td>${esc(lab)}</td>${c.columns.map((col) => {
          const v = (c.matrix[lab] || {})[col] || 0;
          return `<td class="num">${v && col !== lab ? `<b class="sev-warning">${v}</b>` : v}</td>`;
        }).join('')}</tr>`).join('');
        return `<div class="mt-1"><div class="text-xs muted">${esc(title)}: rows = the right label, columns = the reply</div>
          <table class="table compact"><thead><tr><th>Right label</th>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
      };
      return `<details class="mt-2"><summary class="text-xs">Where the labels went (confusion matrix)</summary>${sides.map(one).join('')}</details>`;
    }

    function reportHtml(r) {
      const t = r.usecase.template;
      const heads = HEADS[t] || HEADS.distillation;
      const st = r.status || {};
      const notes = (r.notes || []).map((n) => `<li class="muted">${esc(n)}</li>`).join('');
      const judgeLink = t === 'distillation'
        ? ` <a href="/projects/${encodeURIComponent(pid)}/compare" data-link>Open Compare to judge or read the answers</a>` : '';
      return `<div class="text-xs muted">Answering: ${esc(st.run || '—')}${st.judge ? ' · judge: ' + esc(st.judge) : ''} ·
        scored by ${esc(r.usecase.metric)}${judgeLink}</div>
        <table class="table compact mt-1" id="uc-report-table"><thead><tr><th>Model</th>${heads.map((h) => `<th class="num">${h}</th>`).join('')}</tr></thead>
        <tbody>${metricRows(t, r.base, 'Untrained base')}${metricRows(t, r.tuned, 'Trained run')}</tbody></table>
        ${pairedLine(r.paired)}${t === 'classification' ? confusionHtml(r) : ''}${abBlock(r)}${notes ? `<ul class="health-list mt-1">${notes}</ul>` : ''}`;
    }

    function stopTimer() { if (state.timer) { clearTimeout(state.timer); state.timer = null; } }

    async function showReport(uid, gid) {
      stopTimer();
      const box = root.querySelector('#uc-eval-result');
      if (!box) return;
      box.innerHTML = '<div class="text-xs muted">Loading the result…</div>';
      try {
        const r = await api.get(`${base}/${encodeURIComponent(uid)}/evaluations/${encodeURIComponent(gid)}`);
        if (!stillHere() || state.open !== uid) return;
        box.innerHTML = reportHtml(r);
        const busy = (r.status && r.status.run === 'running') || (r.ab && r.ab.status === 'running')
          || (r.status && r.status.judge === 'running');
        if (busy) state.timer = setTimeout(() => showReport(uid, gid), 5000);
      } catch (e) {
        box.innerHTML = `<span class="sev-error">✕ ${esc(e.message || e)}</span>`;
      }
    }

    async function openUsecase(uid) {
      stopTimer();
      state.open = uid;
      const m = state.usecases.find((x) => x.id === uid);
      const box = $('uc-report');
      if (!m) { box.hidden = true; return; }
      await loadRuns();
      const t = state.templates[m.template] || {};
      const runOpts = state.runs.map((r) => `<option value="${esc(r.id)}">${esc(r.name || r.id)}${r.final_loss != null ? ' · loss ' + num(r.final_loss, 3) : ''}</option>`).join('');
      const evals = (m.evaluations || []).slice().reverse().map((ev) => `<li><button type="button" class="btn sm" data-uc-act="report"
        data-gid="${esc(ev.group_id)}">Show</button> run ${esc(ev.run_id)} · ${new Date(ev.started_at * 1000).toLocaleString()}</li>`).join('');
      box.hidden = false;
      box.innerHTML = `<div class="card-inner">
        <div class="flex gap-2 items-center wrap"><b>${esc(m.name)}</b><span class="pill outline">${esc(t.title || m.template)}</span>
          <span class="text-xs muted">${m.train_rows} training rows · ${m.heldout} held-out questions${(m.labels || []).length ? ' · labels: ' + m.labels.map(esc).join(', ') : ''}</span></div>
        <div class="text-xs muted mt-1">Instruction: ${esc(m.instruction || '—')}</div>
        <div class="flex gap-2 items-center wrap mt-2">
          <label class="text-xs muted" for="uc-eval-run">Compare a finished run with its untrained base on the held-out test:</label>
          <select id="uc-eval-run" class="select" aria-label="Finished run to evaluate">${runOpts || '<option value="">— no finished run in this project yet —</option>'}</select>
          <button type="button" class="btn sm primary" id="uc-eval-btn" ${runOpts ? '' : 'disabled'}>Start comparison</button>
        </div>
        ${evals ? `<ul class="health-list mt-2">${evals}</ul>` : ''}
        <div id="uc-eval-result" class="mt-2"></div></div>`;
      const last = (m.evaluations || [])[(m.evaluations || []).length - 1];
      if (last) showReport(uid, last.group_id);
    }

    async function startEval(uid) {
      const runId = root.querySelector('#uc-eval-run').value;
      const btn = root.querySelector('#uc-eval-btn');
      if (!runId) return;
      btn.disabled = true;
      try {
        const rec = await api.post(`${base}/${encodeURIComponent(uid)}/evaluate`, { run_id: runId });
        notify('Comparison started: the base and the run answer the held-out questions', 'success');
        await refresh();
        await openUsecase(uid);
        showReport(uid, rec.group_id);
      } catch (e) {
        btn.disabled = false;
        notify(e.message || String(e), 'error');
      }
    }

    async function startAb(uid, gid) {
      try {
        await api.post(`${base}/${encodeURIComponent(uid)}/evaluations/${encodeURIComponent(gid)}/ab`, {});
        notify('Blind A/B judge started', 'success');
        showReport(uid, gid);
      } catch (e) {
        notify(e.message || String(e), 'error');
      }
    }

    async function remove(uid) {
      const m = state.usecases.find((x) => x.id === uid);
      const ask = window.fts && window.fts.confirm ? window.fts.confirm : (msg) => Promise.resolve(window.confirm(msg));
      const ok = await ask(`Delete use case ${m ? m.name : uid}? Its held-out test and A/B results go; the dataset and any trained runs stay.`);
      if (!ok) return;
      try {
        await api.del(`${base}/${encodeURIComponent(uid)}`);
        if (state.open === uid) { stopTimer(); state.open = null; $('uc-report').hidden = true; }
        await refresh();
        notify('Use case deleted', 'success');
      } catch (e) {
        notify(e.message || String(e), 'error');
      }
    }

    // ── wiring ───────────────────────────────────────────────────────────
    $('uc-template').addEventListener('change', syncTemplate);
    $('uc-file').addEventListener('change', syncBuild);
    $('uc-build-btn').addEventListener('click', build);
    root.addEventListener('click', (ev) => {
      const btn = ev.target.closest('[data-uc-act]');
      if (btn) {
        const act = btn.dataset.ucAct;
        const row = btn.closest('[data-uc]');
        const uid = row ? row.dataset.uc : null;   // list rows carry their id; report buttons use the open one
        const current = uid || state.open;
        if (act === 'apply') { const m = state.usecases.find((x) => x.id === uid); if (m) applyToForm(m); }
        else if (act === 'open') openUsecase(uid);
        else if (act === 'delete') remove(uid);
        else if (act === 'report') showReport(current, btn.dataset.gid);
        else if (act === 'ab') startAb(current, btn.dataset.gid);
        return;
      }
      if (ev.target.closest('#uc-eval-btn')) startEval(state.open);
    });
    document.addEventListener('fts:beforeNavigate', stopTimer, { once: true });

    (async () => {
      try {
        await loadTemplates();
        await refresh();
        loadTeachers();
        // Deep link: /training?usecase=<id> fills the form from that use case (Guide, API flows).
        const want = new URLSearchParams(window.location.search).get('usecase');
        const m = want && state.usecases.find((x) => x.id === want);
        if (m) applyToForm(m);
      } catch (e) {
        $('uc-list').innerHTML = `<span class="sev-error">✕ Use cases could not be loaded: ${esc(e.message || e)}</span>`;
      }
    })();
  }

  window.ftsUsecases = { mount };
  mount(document.getElementById('uc-card'));
})();
