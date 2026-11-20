/* Local setup generation only: no public probes, storage, credentials or tunnel execution. */
'use strict';
(() => {
  const el = id => document.getElementById('actions-' + id);
  if (!el('builder')) return;
  let plan = null;
  let revision = 0;
  const buttons = ['copy', 'download', 'gemini'];
  const invalidate = () => {
    revision++;
    plan = null;
    buttons.forEach(id => { el(id).disabled = true; });
    el('schema').textContent = el('commands').textContent = '';
    el('status').textContent = 'Setup changed. Generate it again; nothing has been published.';
  };
  for (const field of ['public-url', 'package-url', 'port', 'tunnel']) {
    el(field).addEventListener('input', invalidate);
  }
  el('generate').addEventListener('click', async () => {
    invalidate();
    const current = revision;
    el('generate').disabled = true;
    el('status').textContent = 'Validating setup on this server; no public address is contacted…';
    try {
      const endpoint = el('builder').dataset.endpoint;
      const url = new URL(endpoint, location.href);
      url.search = new URLSearchParams({public_url: el('public-url').value.trim(),
        package_url: el('package-url').value.trim(), relay_port: el('port').value,
        tunnel: el('tunnel').value}).toString();
      const response = await fetch(url, {credentials: 'same-origin', cache: 'no-store',
        redirect: 'error', signal: AbortSignal.timeout(15000)});
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.error || 'Setup validation failed.');
      if (current !== revision) return;
      plan = data;
      el('schema').textContent = JSON.stringify(plan.schema, null, 2);
      el('commands').textContent = '# In the installed package folder, set these privately first:\n' +
        '# RAG_AUTH_TOKEN = the existing package server token\n' +
        '# RAG_ACTIONS_TOKEN = a separate relay bearer token (at least 16 characters)\n' +
        '# Start the package HTTP server separately; no LLM is needed.\n' +
        plan.relay_command + '\n\n# Optional: YOU run this separately to publish ONLY the relay:\n' +
        plan.tunnel_command + '\n\n# Stop the foreground tunnel with Ctrl+C, then stop the relay.\n' +
        '# Verify the public URL no longer responds. No persistent --bg service is generated.';
      buttons.forEach(id => { el(id).disabled = false; });
      el('status').textContent = 'Setup generated, not connected. No tunnel, account, key or public connection was created. Download the schema and test in your vendor account after you choose to publish.';
    } catch (error) {
      if (current === revision) el('status').textContent = 'Setup failed: ' + error.message;
    } finally {
      el('generate').disabled = false;
    }
  });
  el('copy').addEventListener('click', async () => {
    if (!plan) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(plan.schema, null, 2));
      el('status').textContent = 'Schema copied. In the GPT editor add an Action, paste it, and configure API Key → Bearer using the relay token. Vendor connection remains untested.';
    } catch (_) {
      el('status').textContent = 'Clipboard unavailable. Select the schema or use Download OpenAPI JSON.';
    }
  });
  const download = (value, filename) => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2) + '\n'], {type: 'application/json'}));
    const link = document.createElement('a');
    link.href = url; link.download = filename;
    document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  el('download').addEventListener('click', () => {
    if (!plan) return;
    download(plan.schema, 'smart-brain-actions.openapi.json');
    el('status').textContent = 'OpenAPI download requested. This contains no keys, questions, corpus data or account credentials. Vendor import is not tested.';
  });
  el('gemini').addEventListener('click', () => {
    if (!plan) return;
    download(plan.gemini_tool, 'smart-brain-gemini-function.json');
    el('status').textContent = 'Developer function declaration downloaded. Your application must execute retrieveKnowledge against the relay and send its result back to Gemini. This is not a Gemini Gems Actions import or a certified client.';
  });
})();
