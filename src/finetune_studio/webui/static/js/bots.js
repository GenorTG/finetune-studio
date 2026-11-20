/* Credential-free setup only. No vendor contact or persistent browser storage. */
'use strict';
(() => {
  const el = id => document.getElementById('bots-' + id);
  if (!el('builder')) return;
  let plan = null, revision = 0;
  const invalidate = () => {
    revision++; plan = null;
    el('copy').disabled = el('download').disabled = true;
    el('plan').textContent = '';
    el('status').textContent = 'Setup changed. Generate again; vendor connection not checked.';
  };
  ['vendor', 'package-url', 'port'].forEach(id => el(id).addEventListener('input', invalidate));
  el('generate').addEventListener('click', async () => {
    invalidate();
    const current = revision;
    el('generate').disabled = true;
    try {
      const url = new URL(el('builder').dataset.endpoint, location.href);
      url.search = new URLSearchParams({vendor: el('vendor').value,
        package_url: el('package-url').value.trim(), port: el('port').value}).toString();
      const response = await fetch(url, {credentials: 'same-origin', cache: 'no-store',
        redirect: 'error', signal: AbortSignal.timeout(15000)});
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail :
        data.error || 'Choose a valid bridge port and package address.');
      if (current !== revision) return;
      plan = data;
      el('plan').textContent = JSON.stringify(plan, null, 2);
      el('copy').disabled = el('download').disabled = false;
      el('status').textContent = 'Templates generated only. Replace audience IDs and configure credentials privately. Nothing published; vendor connection not checked.';
    } catch (error) {
      if (current === revision) el('status').textContent = 'Setup failed: ' + error.message;
    } finally { el('generate').disabled = false; }
  });
  el('copy').addEventListener('click', async () => {
    if (!plan) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(plan, null, 2));
      el('status').textContent = 'Bridge setup copied. Vendor connection not checked.';
    } catch (_) { el('status').textContent = 'Clipboard unavailable; use Download bot templates.'; }
  });
  el('download').addEventListener('click', () => {
    if (!plan) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(plan, null, 2) + '\n'], {type: 'application/json'}));
    const link = document.createElement('a');
    link.href = url; link.download = 'smart-brain-' + plan.config.vendor + '-templates.json';
    document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    el('status').textContent = 'Bot templates downloaded. No secrets included; vendor connection not checked.';
  });
})();
