/* Local owner controls only. Never persist access keys or render model HTML. */
'use strict';
(() => {
  const el = id => document.getElementById(id);
  let report = null;
  const headers = () => ({'Content-Type': 'application/json',
    ...(el('token').value ? {Authorization: 'Bearer ' + el('token').value} : {}),
    'X-Usage-Owner': el('owner-token').value});
  async function call(path, body) {
    const response = await fetch(path, {headers: headers(), ...(body === undefined ? {} :
      {method: 'POST', body: JSON.stringify(body)})});
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || 'Package request failed');
    return value;
  }
  function text(tag, value, parent) {
    const node = document.createElement(tag);
    node.textContent = value;
    parent.append(node);
    return node;
  }
  function render(value) {
    report = value;
    el('cases').replaceChildren();
    value.cases.forEach(c => text('li', c.name + ': ' + c.question + ' — expected sources: ' +
      (c.expected_sources.join(', ') || 'abstention (semantic judgment required)'), el('cases')));
    if (!value.cases.length) text('li', 'No question set shipped. Export a package with a reviewed suite.', el('cases'));
    el('baseline').textContent = value.baseline ? 'Pinned baseline: ' + value.baseline.id : 'No pinned baseline.';
    el('gate').textContent = 'Candidate retrieval gate: ' + (value.gate.allowed ? 'ready' :
      'blocked — ' + value.gate.reasons.join(', '));
    el('runs').replaceChildren();
    value.runs.forEach(run => {
      const box = text('article', '', el('runs'));
      text('h3', run.mode + ' · ' + run.status + ' · ' + new Date(run.at * 1000).toLocaleString(), box);
      text('p', 'Run ' + run.id + ' · baseline ' + (run.baseline_id || 'none') +
        ' · regressions: ' + (run.regressions.join(', ') || 'none'), box);
      run.results.forEach(row => {
        text('p', row.name + ': ' + (row.retrieval_hit === null ? 'retrieval not scored (abstention)' :
          row.retrieval_hit ? 'source found' : 'SOURCE MISSED') + ' · answer: ' +
          (row.verdict || 'unjudged') + (row.error ? ' · ERROR: ' + row.error : ''), box);
        const details = text('details', '', box);
        text('summary', 'Evidence and transcript', details);
        text('pre', JSON.stringify(row, null, 2), details);
      });
      if (run.gate) text('p', 'Run gate: ' + (run.gate.allowed ? 'ready' : run.gate.reasons.join(', ')), box);
      const baseline = text('button', 'Accept as retrieval baseline', box);
      baseline.disabled = run.status !== 'complete' || run.suite_digest !== value.suite_digest ||
        run.results.some(r => r.retrieval_hit === false);
      baseline.onclick = () => action('/qc/baseline', {id: run.id}, 'Baseline accepted.');
      if (run.mode === 'answers') {
        const judge = text('button', 'Judge answers (operator LLM)', box);
        judge.onclick = () => action('/qc/judge', {id: run.id}, 'Judging finished. Inspect verdicts and errors.');
      }
    });
    ['retrieval', 'answers'].forEach(id => { el(id).disabled = !value.cases.length; });
    ['download', 'clear'].forEach(id => { el(id).disabled = false; });
  }
  async function action(path, body, success) {
    document.querySelectorAll('button').forEach(b => { b.disabled = true; });
    el('status').textContent = 'Working… LLM operations may take several minutes. Do not close this page.';
    try {
      if (path) await call(path, body);
      render(await call('/qc/report'));
      el('status').textContent = success;
    } catch (error) {
      el('status').textContent = error.message;
    } finally {
      el('load').disabled = false;
      if (report) {
        render(report);
        ['retrieval', 'answers'].forEach(id => { el(id).disabled = !report.cases.length; });
        ['download', 'clear'].forEach(id => { el(id).disabled = false; });
      }
    }
  }
  el('load').onclick = () => action(null, undefined, 'Checks loaded.');
  el('retrieval').onclick = () => action('/qc/run', {}, 'Retrieval checks finished. Inspect source results and regressions.');
  el('answers').onclick = () => action('/qc/run', {answers: true}, 'Answer run saved. Answers remain unjudged until you choose Judge answers.');
  el('clear').onclick = () => action('/qc/clear', {}, 'Local QC history and baseline deleted.');
  el('download').onclick = async () => {
    try {
      const data = await call('/qc/report');
      const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'}));
      const link = document.createElement('a');
      link.href = url; link.download = 'package-qc.json'; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      el('status').textContent = 'QC report exported. It may contain plaintext transcripts.';
    } catch (error) { el('status').textContent = error.message; }
  };
})();
