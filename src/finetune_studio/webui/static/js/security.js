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
    const srv = s.server;
    const ts = s.tailscale;
    const where = !srv.loopback_only ? 'every device on the network'
      : (ts.listening ? 'this computer and your Tailscale devices' : 'this computer only');
    const pw = s.password_set ? 'a password is required' : 'no password';
    let summary = `Reachable from ${where} (${srv.listening_on}:${srv.listening_port}); ${pw}.`;
    if (s.exposed_without_password) summary += ' Anyone on your network can use it: set a password.';
    if (srv.restart_needed) summary += ` Saved address ${srv.host}:${srv.port} applies after a restart.`;
    $('access-summary').textContent = summary;
    $('access-summary').classList.toggle('err', s.exposed_without_password);

    $('access-pw-note').textContent = s.password_set
      ? `Set ${when(s.password_changed_at)}. Changing it signs out every other browser.`
      : `Off. Set one (${s.password_min_length}+ characters) before opening this instance to the network.`;
    $('access-pw-current').hidden = !s.password_set;
    $('access-pw-off').hidden = !s.password_set;
    $('access-pw-save').textContent = s.password_set ? 'Change password' : 'Set password';
    $('access-signout').hidden = !s.password_set;

    const host = $('access-host');
    if (![...host.options].some(o => o.value === srv.host)) host.add(new Option(srv.host, srv.host));
    host.value = srv.host;
    $('access-port').value = srv.port;
    $('access-allowed').value = srv.allowed_hosts.join(', ');
    renderTailscale(ts, srv);

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

  // Tailscale: offered when this computer is on a tailnet (or was, when it is on but Tailscale is down now).
  function renderTailscale(ts, srv) {
    const det = ts.detected;
    const row = $('access-ts-row'), note = $('access-ts-note');
    row.hidden = !(det || ts.enabled) || ts.covered_by_host;
    $('access-ts').checked = ts.enabled;
    const ip = det ? det.ip : ts.ip;
    const name = det && det.dns_name ? ` · ${det.dns_name}` : '';
    $('access-ts-where').textContent = ip ? `(${ip}${name})` : '';
    let msg = '';
    if (ts.covered_by_host && det) msg = `Tailscale (${det.ip}) is included: this address listens on every network.`;
    else if (ts.enabled && !det) msg = `Tailscale is not running right now; the service keeps listening on ${ts.ip} for when it is back.`;
    else if (det && !ts.enabled) msg = 'Tailscale detected on this computer: tick the box to open the web UI to your Tailscale devices only (needs a password).';
    else if (ts.enabled && det && !ts.listening) msg = 'Saved: applies after a restart.';
    else if (ts.enabled && det) msg = `Open http://${det.dns_name || det.ip}:${srv.listening_port} from any of your Tailscale devices.`;
    note.textContent = msg; note.hidden = !msg;
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
    run(async () => {
      const body = {
        host: $('access-host').value, port: parseInt($('access-port').value, 10),
        allowed_hosts: $('access-allowed').value.split(',').map(x => x.trim()).filter(Boolean),
      };
      if (!$('access-ts-row').hidden) body.tailscale = $('access-ts').checked;
      try { await call('PUT', '/api/settings/security/server', body); }
      catch (e) {
        if (!/password/.test(e.message) || !window.confirm(e.message + '\n\nSave anyway?')) throw e;
        await call('PUT', '/api/settings/security/server', { ...body, confirm_no_password: true });
      }
      return 'Address saved: restart the service to apply it.';
    });
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
