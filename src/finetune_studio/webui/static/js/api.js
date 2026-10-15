/* One way to call the API from any page (loaded in base.html before page scripts).

     const run = await api.post(`/api/projects/${pid}/brain/build`, {embedder});   // parsed JSON (null on 204)
     try { await api.del(url); } catch (e) { show(e.message); }                    // e.status, e.code, e.data

   A failed call (any non-2xx) throws ApiError whose message is the server's `detail` (webui/errors.py contract),
   so pages never parse error bodies themselves and never treat a 200 as a failure. A 401 that asks for sign-in
   also triggers the redirect in access.js. */
(function () {
  'use strict';

  class ApiError extends Error {
    constructor(message, status, code, data) {
      super(message);
      this.name = 'ApiError';
      this.status = status;
      this.code = code || '';
      this.data = data || {};
    }
  }

  function messageOf(data, status) {
    if (data && typeof data.detail === 'string' && data.detail) return data.detail;
    if (data && Array.isArray(data.detail) && data.detail.length) {          // a raw FastAPI validation list
      return data.detail.map(e => `${(e.loc || []).slice(1).join('.')}: ${e.msg}`).join('; ');
    }
    if (status === 0) return 'the server did not answer (is Finetune Studio running?)';
    return `request failed (HTTP ${status})`;
  }

  async function request(method, url, body, opts) {
    const o = opts || {};
    const init = { method, credentials: 'same-origin', headers: Object.assign({}, o.headers || {}), signal: o.signal };
    if (body instanceof FormData || body instanceof Blob) {
      init.body = body;
    } else if (body !== undefined && body !== null) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(body);
    }
    let r;
    try {
      r = await fetch(url, init);
    } catch (e) {
      if (e && e.name === 'AbortError') throw e;
      throw new ApiError(messageOf(null, 0), 0, 'network', {});
    }
    if (r.status === 204) return null;
    const text = await r.text();
    let data = null;
    if (text) {
      try { data = JSON.parse(text); } catch (e) { data = { detail: text.slice(0, 300) }; }
    }
    if (!r.ok) throw new ApiError(messageOf(data, r.status), r.status, data && data.code, data);
    return data;
  }

  window.ApiError = ApiError;
  window.api = {
    request,
    get: (url, opts) => request('GET', url, null, opts),
    post: (url, body, opts) => request('POST', url, body === undefined ? {} : body, opts),
    put: (url, body, opts) => request('PUT', url, body === undefined ? {} : body, opts),
    patch: (url, body, opts) => request('PATCH', url, body === undefined ? {} : body, opts),
    del: (url, opts) => request('DELETE', url, null, opts),
  };
})();
