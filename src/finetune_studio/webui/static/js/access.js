/* Loaded first on every page (base.html).
   1. A request answered "sign in" (401 + x-fts-auth) sends the page to /login and back here afterwards:
      a session that expires mid-use must not leave buttons silently failing.
   2. Instances reachable from the network without a password say so on every page. */
(function () {
  'use strict';
  const origFetch = window.fetch.bind(window);
  let leaving = false;
  window.fetch = async function (...args) {
    const r = await origFetch(...args);
    if (r.status === 401 && r.headers.get('x-fts-auth') === 'sign-in' && !leaving) {
      leaving = true;
      window.location.assign('/login?next=' + encodeURIComponent(location.pathname + location.search));
    }
    return r;
  };
  async function exposureBanner() {
    try {
      const r = await origFetch('/api/auth/state', { credentials: 'same-origin' });
      if (!r.ok) return;
      const s = await r.json();
      if (!s.exposed_without_password) return;
      const b = document.createElement('div');
      b.className = 'access-banner';
      b.setAttribute('role', 'alert');
      b.append('Anyone on your network can use this Finetune Studio: no password is set. ');
      const a = Object.assign(document.createElement('a'), { href: '/settings#access-card', textContent: 'Set a password' });
      b.append(a);
      document.body.prepend(b);
    } catch (e) { /* the banner is advisory; the page works without it */ }
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', exposureBanner);
  else exposureBanner();
})();
