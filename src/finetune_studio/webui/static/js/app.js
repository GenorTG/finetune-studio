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
      overlay.innerHTML = `
        <div class="modal-dialog" role="dialog" aria-modal="true">
          <div class="modal-head">${opts.title || "Confirm"}</div>
          <div class="modal-body">${message}</div>
          <div class="modal-actions">
            <button class="btn" data-action="cancel">Cancel</button>
            <button class="btn ${opts.danger ? "danger" : "primary"} data-action="ok">${opts.okText || "OK"}</button>
          </div>
        </div>
      `;
      document.body.appendChild(overlay);
      const cleanup = () => overlay.remove();
      overlay.addEventListener("click", (e) => {
        if (e.target === overlay) { cleanup(); resolve(false); }
      });
      overlay.querySelector("[data-action=cancel]").addEventListener("click", () => { cleanup(); resolve(false); });
      overlay.querySelector("[data-action=ok]").addEventListener("click", () => { cleanup(); resolve(true); });
    });
  }

  // ── API ─────────────────────────────────────────────────────────────
  // ── Fetch helper: throw on non-2xx so callers can handle errors ──
  async function _ok(r) {
    const ct = r.headers.get("content-type") || "";
    let body = ct.includes("application/json") ? await r.json().catch(() => ({})) : await r.text();
    if (!r.ok) {
      const detail = (body && typeof body === "object" && body.detail) || (typeof body === "string" ? body : "");
      const err = new Error(detail || `HTTP ${r.status}`);
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

  // ── Polling helper ─────────────────────────────────────────────────
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

  // ── Delegation: form data-api / button data-action / data-confirm ──
  function delegate() {
    // Forms
    document.querySelectorAll("form[data-api]").forEach((form) => {
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
            else notify("Done", "success");
            if (form.dataset.reload === "true") setTimeout(() => location.reload(), 600);
          });
        }
      });
    });
    // Buttons — use event delegation on document so dynamically rendered
    // buttons (e.g. HF Explorer cards loaded async after doSearch) still
    // fire data-action handlers.
    document.addEventListener("click", (ev) => {
      const btn = ev.target.closest("[data-action]");
      if (!btn || !document.contains(btn)) return;
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
        .then((r) => r.json().catch(() => ({})))
        .then((d) => {
          if (d.error) notify(d.error, "error");
          else notify("Done", "success");
          if (btn.dataset.reload === "true") setTimeout(() => location.reload(), 600);
        });
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
        .then((r) => r.json().catch(() => ({})))
        .then((d) => {
          if (d.error) notify(d.error, "error");
          else notify(btn.dataset.done || "Done", "success");
          if (btn.dataset.reload === "true") setTimeout(() => location.reload(), 600);
        })
        .catch((err) => notify(err.message || "Action failed", "error"));
    });
    // data-poll spans
    document.querySelectorAll("[data-poll]").forEach((el) => {
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

  // Page-swap hook called from spa.js after each navigation
  window.fts = {
    notify, api, poll, confirm: confirmDialog, init: delegate, formatJsTime,
  };

  // Run the formatter on full load + after every SPA swap
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', formatJsTime);
  } else {
    formatJsTime();
  }
  document.addEventListener('fts:navigated', formatJsTime);
})();
