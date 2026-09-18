/**
 * Finetune Studio — global JS bridge
 *
 * Exposed as window.fts:
 *   fts.notify(msg, type)        — toast
 *   fts.api.get/post/upload      — fetch helpers
 *   fts.poll(url, el, field, ms) — refresh an element with a JSON field
 *   fts.init()                   — re-bind global handlers on SPA page change
 *
 * Plus three global delegation listeners for forms, action buttons, and
 * confirm prompts — pages just emit data-action="URL" / data-api="URL"
 * and they wire up automatically.
 */
(function () {
  "use strict";

  // ── Toasts ──────────────────────────────────────────────────────────
  const toastsEl = () => document.getElementById("toasts");
  function notify(msg, type) {
    type = type || "info";
    const div = document.createElement("div");
    div.className = "toast " + type;
    div.textContent = msg;
    toastsEl().appendChild(div);
    setTimeout(() => {
      div.classList.add("leaving");
      setTimeout(() => div.remove(), 280);
    }, 3500);
  }

  // ── Confirm dialog ─────────────────────────────────────────────────
  function confirmDialog(message, opts) {
    opts = opts || {};
    return new Promise((resolve) => {
      const overlay = document.createElement("div");
      overlay.className = "modal-overlay";
      // Use data-fts-modal (not data-action): global [data-action] delegation
      // would otherwise treat "ok"/"cancel" as API URLs. Close the class="…"
      // quote after the interpolated variant or the browser parses
      // class="btn primary data-action=" and breaks the OK handler.
      const variant = opts.danger ? "danger" : "primary";
      overlay.innerHTML = `
        <div class="modal-dialog" role="dialog" aria-modal="true">
          <div class="modal-head">${opts.title || "Confirm"}</div>
          <div class="modal-body">${message}</div>
          <div class="modal-actions">
            <button type="button" class="btn" data-fts-modal="cancel">Cancel</button>
            <button type="button" class="btn ${variant}" data-fts-modal="ok">${opts.okText || "OK"}</button>
          </div>
        </div>
      `;
      document.body.appendChild(overlay);
      const cleanup = () => overlay.remove();
      overlay.addEventListener("click", (e) => {
        if (e.target === overlay) { cleanup(); resolve(false); }
      });
      overlay.querySelector("[data-fts-modal=cancel]").addEventListener("click", (e) => {
        e.stopPropagation();
        cleanup();
        resolve(false);
      });
      overlay.querySelector("[data-fts-modal=ok]").addEventListener("click", (e) => {
        e.stopPropagation();
        cleanup();
        resolve(true);
      });
    });
  }

  function promptDialog(message, initialValue) {
    return new Promise((resolve) => {
      const overlay = document.createElement("div");
      overlay.className = "modal-overlay";
      overlay.innerHTML = `
        <div class="modal-dialog" role="dialog" aria-modal="true">
          <div class="modal-head">Input required</div>
          <div class="modal-body"><p>${message}</p>
            <input class="input" data-fts-prompt-input type="text">
          </div>
          <div class="modal-actions">
            <button type="button" class="btn" data-fts-modal="cancel">Cancel</button>
            <button type="button" class="btn primary" data-fts-modal="ok">OK</button>
          </div>
        </div>`;
      document.body.appendChild(overlay);
      const input = overlay.querySelector("[data-fts-prompt-input]");
      input.value = initialValue || "";
      input.focus();
      const cleanup = (value) => { overlay.remove(); resolve(value); };
      overlay.querySelector("[data-fts-modal=cancel]").onclick = () => cleanup(null);
      overlay.querySelector("[data-fts-modal=ok]").onclick = () => cleanup(input.value);
      input.onkeydown = (event) => {
        if (event.key === "Enter") cleanup(input.value);
        if (event.key === "Escape") cleanup(null);
      };
    });
  }

  // ── API ─────────────────────────────────────────────────────────────
  // ── Fetch helper: throw on non-2xx so callers can handle errors ──
  function _errorMessage(body, status) {
    if (body && typeof body === "object") {
      let detail = body.error || body.detail || "";
      if (Array.isArray(detail)) {
        detail = detail.map((d) => (d && d.msg) || JSON.stringify(d)).join("; ");
      } else if (detail && typeof detail === "object") {
        detail = JSON.stringify(detail);
      }
      if (detail) return String(detail);
    } else if (typeof body === "string" && body) {
      return body;
    }
    return status ? `HTTP ${status}` : "Request failed";
  }

  async function _ok(r) {
    const ct = r.headers.get("content-type") || "";
    let body = ct.includes("application/json") ? await r.json().catch(() => ({})) : await r.text();
    if (!r.ok) {
      const err = new Error(_errorMessage(body, r.status));
      err.status = r.status;
      err.body = body;
      throw err;
    }
    return body;
  }

  const api = {
    get: (url) => fetch(url).then(_ok),
    post: (url, body) => fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    }).then(_ok),
    upload: (url, file) => {
      const fd = new FormData();
      fd.append("file", file);
      return fetch(url, { method: "POST", body: fd }).then(_ok);
    },
    uploadMany: (url, files, fieldName) => {
      const fd = new FormData();
      for (const f of files) fd.append(fieldName || "files", f);
      return fetch(url, { method: "POST", body: fd }).then(_ok);
    },
  };

  // ── Polling helper (dashboard tiles / legacy data-poll) ────────────
  function poll(url, el, field, interval) {
    interval = interval || 3000;
    const fn = () => api.get(url).then((d) => {
      if (!d) return;
      if (field) {
        if (d[field] !== undefined) el.textContent = d[field];
      } else if (typeof d === "string") {
        el.textContent = d;
      }
    });
    fn();
    return setInterval(fn, interval);
  }

  /**
   * Live updates via EventSource (SSE). Falls back to silent polling only
   * when EventSource is unavailable or the stream errors repeatedly.
   *
   * Long-task UIs must prefer an SSE endpoint; use pollUrl + fallbackMs
   * (≥ 5000) as a documented slow fallback — never a 2s full-panel redraw.
   *
   * @param {string} url SSE endpoint
   * @param {(data: object) => void} onEvent called with parsed JSON payloads
   * @param {{ pollUrl?: string, fallbackMs?: number, onError?: Function }} [opts]
   * @returns {() => void} unsubscribe / close
   */
  function subscribe(url, onEvent, opts) {
    opts = opts || {};
    const fallbackMs = opts.fallbackMs || 5000;
    const pollUrl = opts.pollUrl || "";
    let es = null;
    let timer = null;
    let closed = false;
    let failCount = 0;

    function deliver(raw) {
      if (raw == null || raw === "") return;
      try {
        const data = typeof raw === "string" ? JSON.parse(raw) : raw;
        onEvent(data);
      } catch (_e) { /* ignore malformed frames */ }
    }

    function stopFallback() {
      if (timer) { clearInterval(timer); timer = null; }
    }

    function startFallback() {
      if (closed || timer || !pollUrl) return;
      const tick = () => {
        fetch(pollUrl, { cache: "no-store" })
          .then((r) => (r.ok ? r.json() : null))
          .then((d) => { if (d) onEvent(d); })
          .catch(() => { /* ignore */ });
      };
      tick();
      timer = setInterval(tick, fallbackMs);
    }

    function closeEs() {
      if (es) {
        try { es.close(); } catch (_e) { /* ignore */ }
        es = null;
      }
    }

    if (typeof EventSource !== "undefined") {
      try {
        es = new EventSource(url);
        es.onmessage = (ev) => {
          failCount = 0;
          deliver(ev.data);
        };
        es.onerror = () => {
          failCount += 1;
          // After a few consecutive errors, fall back to silent poll.
          if (failCount >= 3) {
            closeEs();
            startFallback();
          }
        };
      } catch (_e) {
        startFallback();
      }
    } else {
      startFallback();
    }

    return function unsubscribe() {
      closed = true;
      closeEs();
      stopFallback();
    };
  }

  // ── Delegation: form data-api / button data-action / data-confirm ──
  // Document-level click listeners must register exactly once; SPA calls
  // fts.init() (= delegate) on every navigation.
  let _docClickListenersBound = false;

  function delegate() {
    // Forms — re-wire per SPA swap, but never double-bind the same element
    document.querySelectorAll("form[data-api]").forEach((form) => {
      if (form.dataset.ftsBound === "1") return;
      form.dataset.ftsBound = "1";
      form.addEventListener("submit", (ev) => {
        ev.preventDefault();
        const url = form.getAttribute("data-api");
        // If the form has file inputs, submit as multipart; otherwise JSON
        const hasFiles = form.querySelector('input[type="file"]');
        if (hasFiles && form.enctype === "multipart/form-data") {
          const fd = new FormData(form);
          fetch(url, { method: "POST", body: fd }).then(_ok).then((d) => {
            if (d && d.error) notify(d.error, "error");
            else notify("Upload done", "success");
            if (form.dataset.reload === "true") setTimeout(() => location.reload(), 600);
          }).catch((err) => notify(err.message || "Upload failed", "error"));
        } else {
          const data = Object.fromEntries(new FormData(form));
          api.post(url, data).then((d) => {
            if (d.error) notify(d.error, "error");
            else if (d.run_id) { notify("Training started · run " + d.run_id.slice(0, 8), "success"); window.ftsActivity?.open(); }
            else notify("Done", "success");
            if (form.dataset.reload === "true") setTimeout(() => location.reload(), 600);
          })
          .catch((err) => notify(err.message || "Request failed", "error"));
        }
      });
    });
    // Buttons — use event delegation on document so dynamically rendered
    // buttons (e.g. HF Explorer cards loaded async after doSearch) still
    // fire data-action handlers.
    if (!_docClickListenersBound) {
      _docClickListenersBound = true;
      document.addEventListener("click", (ev) => {
        const btn = ev.target.closest("[data-action]");
        if (!btn || !document.contains(btn)) return;
        // Modal chrome must never hit the API action path
        if (btn.closest(".modal-overlay")) return;
        // data-confirm buttons are handled by the confirm listener below
        if (btn.hasAttribute("data-confirm")) return;
        ev.preventDefault();
        const url = btn.dataset.action;
        const method = (btn.dataset.method || "POST").toUpperCase();
        const body = btn.dataset.body ? JSON.parse(btn.dataset.body) : {};
        const opts = { method };
        if (method !== "GET") {
          opts.headers = { "Content-Type": "application/json" };
          opts.body = JSON.stringify(body);
        }
        fetch(url, opts)
          .then(_ok)
          .then((d) => {
            if (d && d.error) notify(d.error, "error");
            else if (d && d.job_id) {
              // HF Pull / background jobs — surface in the activity drawer.
              notify("Download queued · job " + String(d.job_id).slice(0, 8), "success");
              if (window.ftsActivity && typeof window.ftsActivity.open === "function") {
                window.ftsActivity.open();
              }
              if (window.ftsActivity && typeof window.ftsActivity.refresh === "function") {
                window.ftsActivity.refresh();
              }
            } else if (d && d.run_id) {
              // Training start / benchmark exec — long op, name it properly
              // (a bare "Done" after starting a 20-minute run is a lie).
              notify("Training started · run " + String(d.run_id).slice(0, 8), "success");
              if (window.ftsActivity && typeof window.ftsActivity.open === "function") {
                window.ftsActivity.open();
              }
            } else notify("Done", "success");
            if (btn.dataset.reload === "true") setTimeout(() => location.reload(), 600);
          })
          .catch((err) => notify(err.message || "Request failed", "error"));
      });
      // data-confirm buttons — use modal dialog (delegated for dynamic buttons)
      document.addEventListener("click", async (ev) => {
        const btn = ev.target.closest("[data-confirm]");
        if (!btn || !document.contains(btn)) return;
        ev.preventDefault();
        const ok = await confirmDialog(btn.dataset.confirm || "Are you sure?", {
          title: btn.dataset.confirmTitle || "Confirm",
          danger: btn.dataset.confirmDanger === "true",
          okText: btn.dataset.confirmOk || "OK",
        });
        if (!ok) return;
        // Fire the action after confirmation
        const url = btn.dataset.action || btn.dataset.api;
        if (!url) return;
        const method = (btn.dataset.method || (btn.dataset.api ? "POST" : "DELETE")).toUpperCase();
        const body = btn.dataset.body ? JSON.parse(btn.dataset.body) : {};
        const opts = { method };
        if (method !== "GET") { opts.headers = { "Content-Type": "application/json" }; opts.body = JSON.stringify(body); }
        fetch(url, opts)
          .then(_ok)
          .then((d) => {
            if (d && d.error) notify(d.error, "error");
            else if (d && d.job_id) {
              notify(
                (btn.dataset.done || "Download queued") + " · job " + String(d.job_id).slice(0, 8),
                "success"
              );
              if (window.ftsActivity && typeof window.ftsActivity.open === "function") {
                window.ftsActivity.open();
              }
              if (window.ftsActivity && typeof window.ftsActivity.refresh === "function") {
                window.ftsActivity.refresh();
              }
            }
            else notify(btn.dataset.done || "Done", "success");
            if (btn.dataset.reload === "true") setTimeout(() => location.reload(), 600);
          })
          .catch((err) => notify(err.message || "Action failed", "error"));
      });
    }
    // data-poll spans
    document.querySelectorAll("[data-poll]").forEach((el) => {
      if (el.dataset.ftsBound === "1") return;
      el.dataset.ftsBound = "1";
      const url = el.getAttribute("data-poll");
      const field = el.getAttribute("data-field");
      const interval = parseInt(el.getAttribute("data-interval") || "3000", 10);
      poll(url, el, field, interval);
    });
    // (data-confirm already handled by delegated click listener above)
  }

  // ── Run once on full load, then re-run on every SPA swap ─────────
  document.addEventListener("DOMContentLoaded", delegate);

  // .js-time formatter: reads data-ts (Unix seconds, possibly float) and
  // writes a localized "YYYY-MM-DD HH:MM:SS" string. Used by training past-runs
  // list + detail to render started_at / finished_at as human time instead of
  // raw Unix timestamps (or "—" when missing).
  function formatJsTime() {
    document.querySelectorAll('.js-time[data-ts]').forEach((el) => {
      const raw = (el.getAttribute('data-ts') || '').trim();
      if (!raw) { el.textContent = '\u2014'; return; }
      const ts = Number(raw);
      if (!Number.isFinite(ts) || ts <= 0) { el.textContent = raw; return; }
      const d = new Date(ts * 1000);
      if (isNaN(d.getTime())) { el.textContent = raw; return; }
      const pad = (n) => String(n).padStart(2, '0');
      el.textContent = d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate())
        + ' ' + pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds());
    });
    document.querySelectorAll('.js-duration[data-secs]').forEach((el) => {
      const raw = (el.getAttribute('data-secs') || '').trim();
      if (!raw) { el.textContent = '\u2014'; return; }
      const s = Number(raw);
      if (!Number.isFinite(s) || s < 0) { el.textContent = raw; return; }
      if (s < 1) { el.textContent = (s * 1000).toFixed(0) + ' ms'; return; }
      const h = Math.floor(s / 3600);
      const m = Math.floor((s % 3600) / 60);
      const sec = (s % 60);
      if (h > 0) el.textContent = h + 'h ' + m + 'm ' + sec.toFixed(1) + 's';
      else if (m > 0) el.textContent = m + 'm ' + sec.toFixed(1) + 's';
      else el.textContent = sec.toFixed(2) + 's';
    });
  }

  // ── Connection status ──────────────────────────────────────────
  const connEl = document.getElementById('conn-status');
  const connLabel = connEl?.querySelector('.conn-label');
  let connTimer = null;
  function connCheck() {
    if (!connEl) return;
    fetch('/api/system/resources', { method: 'GET' })
      .then(r => {
        if (r.ok) { connEl.classList.remove('offline'); if (connLabel) connLabel.textContent = 'connected'; }
        else { connEl.classList.add('offline'); if (connLabel) connLabel.textContent = 'offline'; }
      })
      .catch(() => { connEl.classList.add('offline'); if (connLabel) connLabel.textContent = 'offline'; });
  }
  function connStart() { if (connTimer) return; connTimer = setInterval(connCheck, 5000); connCheck(); }
  function connStop() { if (connTimer) { clearInterval(connTimer); connTimer = null; } }
  // Auto-start connection monitoring
  connStart();

  // ── Theme toggle ────────────────────────────────────────────────
  // QABUG-008: persist under localStorage key `fts-theme`, swap data-theme
  // on <html> so the CSS palette actually changes.
  const THEME_KEY = 'fts-theme';
  function themeGet() {
    try { return localStorage.getItem(THEME_KEY) || 'dark'; }
    catch (e) { return 'dark'; }
  }
  function themeSet(t) {
    if (t !== 'light' && t !== 'dark') t = 'dark';
    try { localStorage.setItem(THEME_KEY, t); } catch (e) { /* ignore */ }
    document.documentElement.setAttribute('data-theme', t);
    const btn = document.getElementById('theme-toggle');
    if (btn) btn.textContent = t === 'light' ? '☀️' : '🌙';
  }
  // Apply saved theme on load (pre-paint script in <head> already set it;
  // this keeps the toggle button icon in sync after SPA swaps).
  themeSet(themeGet());
  function themeToggle() {
    themeSet(themeGet() === 'dark' ? 'light' : 'dark');
    if (window.fts) window.fts.notify('Theme: ' + themeGet(), 'info');
  }
  window.ftsThemeToggle = themeToggle;

  // ── Keyboard navigation ──────────────────────────────────────────
  const _kbShortcuts = [
    { key: 'g', action: () => { window.location.href = '/'; } },  // 'g' then 'd' pattern
    { key: 'p', action: () => { window.location.href = '/projects'; } },
    { key: 'h', action: () => { window.location.href = '/models/explore'; } },
    { key: 'i', action: () => { window.location.href = '/inference'; } },
    { key: 's', action: () => { window.location.href = '/settings'; } },
    { key: '/', action: () => { const el = document.getElementById('proj-search') || document.getElementById('palette-input'); if (el) { el.focus(); el.select(); } } },
    { key: '?', action: () => { const el = document.getElementById('kb-help'); if (el) el.hidden = !el.hidden; } },
  ];
  document.addEventListener('keydown', (e) => {
    // Don't hijack when typing in inputs
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || e.target.isContentEditable) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const match = _kbShortcuts.find(s => s.key === e.key);
    if (match) { e.preventDefault(); match.action(); }
  });

  // Short human type label for a file row. Prefer the filename extension —
  // raw MIME subtypes read as "vnd.openxmlformats-officedocument…" (docx) or
  // "octet-stream" (jsonl) in the file tables.
  function fileTypeLabel(name, mime) {
    const ext = /\.([A-Za-z0-9]{1,10})$/.exec(String(name || ""));
    if (ext) return ext[1].toLowerCase();
    const sub = String(mime || "").split("/").pop() || "";
    return sub.replace(/^(x-|vnd\.)/, "").split(/[.+]/).pop() || "?";
  }

  // Page-swap hook called from spa.js after each navigation
  window.fts = {
    notify, api, poll, subscribe, confirm: confirmDialog, prompt: promptDialog, init: delegate, formatJsTime,
    conn: { check: connCheck, start: connStart, stop: connStop },
    themeToggle, fileTypeLabel,
  };

  // Run the formatter on full load + after every SPA swap
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', formatJsTime);
  } else {
    formatJsTime();
  }
  document.addEventListener('fts:navigated', formatJsTime);
})();
