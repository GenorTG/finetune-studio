'use strict';
(() => {
  const $ = (id) => document.getElementById(id);
  const setStatus = (message) => { $('status').textContent = message; };
  async function request(path, body) {
    const headers = {'Content-Type': 'application/json'};
    if ($('token').value) headers.Authorization = 'Bearer ' + $('token').value;
    if ($('owner-token').value) headers['X-Usage-Owner'] = $('owner-token').value;
    const response = await fetch(path, {method: body === undefined ? 'GET' : 'POST', headers,
      ...(body === undefined ? {} : {body: JSON.stringify(body)})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
    return result;
  }
  function render(report) {
    $('metrics').replaceChildren();
    for (const [key, label] of Object.entries({questions: 'Questions / operations', sources_found: 'Sources found',
      not_found: 'Not found / not stated', low_confidence: 'Low confidence', thumbs_down: 'Thumbs down', errors: 'Errors',
      latency_mean_ms: 'Mean latency · ms', latency_p95_ms: 'P95 latency · ms'})) {
      const box = document.createElement('div'); box.className = 'metric';
      const value = document.createElement('strong'); value.textContent = report.summary[key];
      const title = document.createElement('span'); title.textContent = label;
      box.append(value, title); $('metrics').append(box);
    }
    $('queries').value = String(report.settings.store_queries); $('retention').value = report.settings.retention_days;
    for (const [id, rows] of [['gap-list', report.gaps], ['events', report.events]]) {
      $(id).replaceChildren();
      for (const row of rows.slice(0, 50)) {
        const li = document.createElement('li');
        li.textContent = `${new Date(row.at * 1000).toLocaleString()} · ${row.operation} · ${row.query || 'Question text not retained'} · ` +
          `${row.sources} source(s) · ${row.latency_ms} ms` + (row.error ? ` · ${row.error}` : '') +
          (row.not_found ? ' · not found / not stated' : '') + (row.weak ? ' · low confidence' : '') +
          (row.feedback === -1 ? ' · thumbs down' : row.feedback === 1 ? ' · helpful' : '');
        $(id).append(li);
      }
      if (!rows.length) { const li = document.createElement('li'); li.textContent = 'No retained activity in this view.'; $(id).append(li); }
    }
    for (const id of ['save', 'clear', 'export', 'gaps']) $(id).disabled = false;
  }
  async function load() { render(await request('usage/report')); }
  async function action(button, work) {
    button.disabled = true;
    try { await work(); } catch (error) { setStatus(error.message); }
    finally { button.disabled = false; }
  }
  $('refresh').addEventListener('click', () => action($('refresh'), async () => { await load(); setStatus('Usage loaded. Totals cover retained events only.'); }));
  $('save').addEventListener('click', () => action($('save'), async () => {
    const days = Number($('retention').value);
    if (!Number.isInteger(days) || days < 1 || days > 365) throw new Error('Retention must be a whole number from 1 to 365.');
    render(await request('usage/settings', {store_queries: $('queries').value === 'true', retention_days: days}));
    setStatus('Privacy settings saved. Retention applied; counts-only removes previously saved question text.');
  }));
  $('clear').addEventListener('click', () => action($('clear'), async () => {
    if (!window.confirm('Delete all retained package activity and feedback?')) return;
    await request('usage/clear', {}); await load(); setStatus('Retained activity deleted. Privacy settings kept.');
  }));
  for (const [id, path, filename] of [['export', 'usage/export', 'package-usage.json'], ['gaps', 'usage/gaps', 'missing-knowledge.json']]) {
    $(id).addEventListener('click', () => action($(id), async () => {
      const report = await request(path);
      const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type: 'application/json'}));
      const link = document.createElement('a'); link.href = url; link.download = filename; document.body.append(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10000); setStatus('Exported ' + filename + '. Review before sharing.');
    }));
  }
})();
