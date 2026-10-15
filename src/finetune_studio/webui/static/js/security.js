/* Settings -> Access & network: password, listen address, API tokens, sessions, CORS/proxy.
   API: /api/settings/security* (routes/access.py). Every error shows the server's `detail`. */
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  if (!$('access-card')) return;
  const status = (msg, bad) => { const el = $('access-status'); el.textContent = msg; el.classList.toggle('err', !!bad); };

  const call = (method, url, body) => window.api.request(method, url, body);

  const when = (ts) => ts ? new Date(ts * 1000).toLocaleString() : 'never';

  function render(s) {
    const srv = s.server, reach = s.reach;
    const pw = s.password_set ? 'a password is required' : 'no password';
    let summary = `Reachable now from ${reach.now.join(' + ')} (port ${srv.listening_port}); ${pw}.`;
    if (s.exposed_without_password) summary += ' Anyone who can reach it can use it: set a password.';
    if (srv.restart_needed) summary += ` Saved: ${reach.saved}, port ${srv.port}; applies when the web UI restarts.`;
    $('access-summary').textContent = summary;
    $('access-summary').classList.toggle('err', s.exposed_without_password);
    $('access-pw-note').textContent = s.password_set
      ? `Set ${when(s.password_changed_at)}. Changing it signs out every other browser.`
      : `Off. Set one (${s.password_min_length}+ characters) before opening this instance to the network.`;
    $('access-pw-current').hidden = !s.password_set;
    $('access-pw-off').hidden = !s.password_set;
    $('access-pw-save').textContent = s.password_set ? 'Change password' : 'Set password';
    $('access-signout').hidden = !s.password_set;

    renderReach(s);

    const list = $('access-tokens');
    list.replaceChildren(...s.tokens.map(t => {
      const li = document.createElement('li');
      li.textContent = `${t.name} · created ${when(t.created_at)} · last used ${when(t.last_used_at)} `;
      const b = document.createElement('button');
      b.className = 'btn sm'; b.type = 'button'; b.textContent = 'Revoke';
      b.addEventListener('click', () => run(async () => {
        await call('DELETE', `/api/settings/security/tokens/${encodeURIComponent(t.id)}`);
        return 'Token revoked.';
      }));
      li.append(b);
      return li;
    }));
    if (!s.tokens.length) list.replaceChildren(Object.assign(document.createElement('li'), { textContent: 'No tokens.' }));
    $('access-sessions').textContent = `${s.sessions} signed-in browser(s)`;
  }

  // One switch per kind of device; each row says whether it is in use, saved for the next restart, or unavailable.
  function state(el, saved, live) {
    el.textContent = saved && live ? 'listening' : saved ? 'after restart' : live ? 'off (refused)' : '';
    el.className = 'access-state' + (saved && live ? ' on' : saved ? ' pending' : '');
  }

  function renderReach(s) {
    const r = s.reach, srv = s.server;
    $('access-lan').checked = r.lan.on;
    const ips = r.lan.addresses.map(a => `${a.ip} (${a.interface})`).join(', ');
    $('access-lan-where').textContent = ips ? `devices on your LAN / Wi-Fi · this computer is ${ips}` : 'devices on your LAN / Wi-Fi';
    state($('access-lan-state'), r.lan.on, r.lan.listening);

    const ts = r.tailscale, det = ts.detected;
    $('access-ts').checked = ts.on;
    $('access-ts').disabled = !det && !ts.ip;
    $('access-ts-where').textContent = det
      ? `your tailnet devices · ${det.ip}${det.dns_name ? ' · ' + det.dns_name : ''}`
      : (ts.ip ? `Tailscale is not running (last known ${ts.ip})` : 'Tailscale not detected on this computer');
    state($('access-ts-state'), ts.on, ts.listening);

    $('access-custom').value = r.custom.addresses.join(', ');
    $('access-port').value = srv.port;
    $('access-allowed').value = srv.allowed_hosts.join(', ');
    if (r.custom.addresses.length || srv.allowed_hosts.length || srv.port !== 7860) $('access-advanced').open = true;

    $('access-apply').hidden = !srv.restart_needed;
    let note = '';
    if (srv.listen_error) note = `The last restart could not use the saved setting (${srv.listen_error}); the previous addresses stay in use.`;
    else if (srv.restart_needed && srv.port !== srv.listening_port) note = `The port changes to ${srv.port}: after "Apply now" this page moves there.`;
    else if (det && !ts.on && !r.lan.on) note = 'Tailscale detected: switch it on to open the web UI to your Tailscale devices only.';
    else if (ts.on && det && ts.listening) note = `From your Tailscale devices: http://${det.dns_name || det.ip}:${srv.listening_port}`;
    $('access-reach-note').textContent = note;
    $('access-reach-note').hidden = !note;
  }

  // Restart the web UI so new listening sockets are bound, then come back (on the new port if it changed).
  async function applyNow(port) {
    await call('POST', '/api/service/components/web/restart');
    status('Restarting the web UI…');
    const url = new URL(window.location.href);
    if (port && String(port) !== url.port) url.port = String(port);
    const deadline = Date.now() + 30000;
    while (Date.now() < deadline) {
      await new Promise(r => setTimeout(r, 1000));
      try {
        // Another port is another origin: no-cors only tells "it answers" (an opaque response), which is enough.
        const moved = url.port !== window.location.port;
        const r = await fetch(new URL('/api/health', url), moved ? { mode: 'no-cors', cache: 'no-store' } : { cache: 'no-store' });
        if (moved || r.ok) { window.location.href = url.toString(); return 'Applied.'; }
      } catch (_) { /* still restarting */ }
    }
    throw new Error(`The web UI did not come back on ${url.host} within 30 s: check Settings -> Service, or run "fts status" on this computer.`);
  }

  async function load() { render(await call('GET', '/api/settings/security')); }

  async function run(fn) {
    try { const msg = await fn(); await load(); status(msg || 'Saved.'); }
    catch (e) { status(e.message, true); }
  }

  $('access-pw-form').addEventListener('submit', (ev) => {
    ev.preventDefault();
    run(async () => {
      await call('PUT', '/api/settings/security/password', {
        new_password: $('access-pw-new').value,
        current_password: $('access-pw-current').value || null,
      });
      $('access-pw-new').value = ''; $('access-pw-current').value = '';
      return 'Password saved. This browser stays signed in.';
    });
  });
  $('access-pw-off').addEventListener('click', () => run(async () => {
    const current = $('access-pw-current').value;
    if (!current) throw new Error('Type the current password first.');
    await call('POST', '/api/settings/security/disable', { current_password: current });
    $('access-pw-current').value = '';
    return 'Password removed.';
  }));

  $('access-server-form').addEventListener('submit', (ev) => {
    ev.preventDefault();
    const list = (id) => $(id).value.split(',').map(x => x.trim()).filter(Boolean);
    run(async () => {
      const body = {
        lan: $('access-lan').checked, port: parseInt($('access-port').value, 10),
        custom: list('access-custom'), allowed_hosts: list('access-allowed'),
      };
      if (!$('access-ts').disabled) body.tailscale = $('access-ts').checked;
      let st;
      try { st = await call('PUT', '/api/settings/security/server', body); }
      catch (e) {
        if (!/password/.test(e.message) || !window.confirm(e.message + '\n\nSave anyway?')) throw e;
        st = await call('PUT', '/api/settings/security/server', { ...body, confirm_no_password: true });
      }
      return st.server.restart_needed ? 'Saved. "Apply now" restarts the web UI to use it.' : 'Saved: in effect now.';
    });
  });
  $('access-apply').addEventListener('click', async () => {
    const port = parseInt($('access-port').value, 10);
    const ok = await window.fts.confirm('Restart the web UI to apply the new addresses? It is back in a few seconds; training runs keep going.',
      { okText: 'Restart' });
    if (!ok) return;
    try { status(await applyNow(port)); } catch (e) { status(e.message, true); }
  });

  $('access-token-form').addEventListener('submit', (ev) => {
    ev.preventDefault();
    run(async () => {
      const t = await call('POST', '/api/settings/security/tokens', { name: $('access-token-name').value });
      $('access-token-name').value = '';
      const out = $('access-token-new');
      out.textContent = `New token "${t.name}" (shown once, copy it now): ${t.token}`;
      out.hidden = false;
      return 'Token created.';
    });
  });

  $('access-signout-others').addEventListener('click', () => run(async () => {
    const r = await call('DELETE', '/api/settings/security/sessions');
    return `Signed out ${r.signed_out_elsewhere} other browser(s).`;
  }));
  $('access-signout').addEventListener('click', async () => {
    await call('POST', '/api/auth/logout').catch(() => null);
    window.location.assign('/login');
  });

  // CORS / proxy headers: still stored in settings.json and applied at app start.
  async function loadHosting() {
    const s = await call('GET', '/api/settings');
    $('hosting-cors').value = (s.cors_origins || []).join('\n');
    $('hosting-cors-creds').checked = !!s.cors_allow_credentials;
    $('hosting-proxy').checked = !!s.proxy_headers;
  }
  $('btn-hosting-save').addEventListener('click', () => run(async () => {
    await call('PATCH', '/api/settings', {
      cors_origins: $('hosting-cors').value.split('\n').map(x => x.trim()).filter(Boolean),
      cors_allow_credentials: $('hosting-cors-creds').checked,
      proxy_headers: $('hosting-proxy').checked,
    });
    return 'Saved: applies when the service restarts.';
  }));

  load().catch(e => status(e.message, true));
  loadHosting().catch(e => status(e.message, true));
})();
