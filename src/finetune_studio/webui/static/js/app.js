/**
 * Finetune Studio — global JS bridge
 * Loaded on every page (via base.html <script src="/static/js/app.js">).
 * Adds global event listeners + helpers so pages stay interactive
 * even when their inline scripts haven't loaded yet.
 */
(function () {
  "use strict";

  // ── Utility: show toast / inline error ─────────────────────────────────
  window.notify = function (msg, type) {
    type = type || "info";
    const div = document.createElement("div");
    div.className =
      "fixed top-4 right-4 z-50 px-4 py-2 rounded shadow-lg text-sm " " +
      (type === "error" ? "bg-red-600" : type === "success" ? "bg-emerald-600" : "bg-gray-700");
    div.textContent = msg;
    document.body.appendChild(div);
    setTimeout(() => div.remove(), 3500);
  };

  // ── Utility: small API helper ───────────────────────────────────────────
  window.api = {
    get: (url) => fetch(url).then((r) => r.json().catch(() => ({}))),
    post: (url, body) =>
      fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}),
      }).then((r) => r.json().catch(() => ({}))),
    upload: (url, file) => {
      const fd = new FormData();
      fd.append("file", file);
      return fetch(url, { method: "POST", body: fd }).then((r) => r.json().catch(() => ({})));
    },
  };

  // ── Wire up every <form data-api="POST"> with fetch on submit ──────────
  document.addEventListener("submit", function (ev) {
    const form = ev.target;
    if (!form.matches || !form.matches("form[data-api]")) return;
    ev.preventDefault();
    const url = form.getAttribute("data-api");
    const data = Object.fromEntries(new FormData(form));
    api.post(url, data).then((d) => {
      if (d.error) notify(d.error, "error");
      else notify("Done", "success");
      if (form.dataset.reload === "true") setTimeout(() => location.reload(), 600);
    });
  });

  // ── Wire up every <button data-action="..."> with one-click API calls ──
  document.addEventListener("click", function (ev) {
    const btn = ev.target.closest('[data-action]');
    if (!btn) return;
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

  // ── Refresh <span data-poll="URL" data-field="x"> every N ms ──────────
  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-poll]").forEach((el) => {
      const url = el.getAttribute("data-poll");
      const field = el.getAttribute("data-field");
      const interval = parseInt(el.getAttribute("data-interval") || "3000", 10);
      setInterval(() => {
        api.get(url).then((d) => {
          if (field && d && d[field] !== undefined) el.textContent = d[field];
        });
      }, interval);
    });
  });

  // ── Confirm before destructive actions ─────────────────────────────────
  document.addEventListener("click", function (ev) {
    const el = ev.target.closest("[data-confirm]");
    if (!el) return;
    if (!confirm(el.dataset.confirm)) ev.preventDefault();
  });

  console.log("Finetune Studio ready");
})();