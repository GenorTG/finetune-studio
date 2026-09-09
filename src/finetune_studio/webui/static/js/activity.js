/* ============================================================
   Activity feed — top-right slide-out drawer
   ------------------------------------------------------------
   Each row is collapsed by default (kind + project + status).
   Click the row head to expand inline details: progress bar,
   full message, project_id, started/elapsed, and a prominent
   "GO TO →" button. Multiple rows can be expanded at once;
   expansion pushes siblings down (no overlay).
   Polls /api/activity every 2s.
   ============================================================ */
(function () {
  const $ = (id) => document.getElementById(id);
  const backdrop = $("activity-backdrop");
  const drawer    = $("activity");
  const trigger   = $("sb-activity");
  const countEl   = $("sb-activity-count");
  const bodyEl    = $("activity-body");
  const subEl     = $("activity-sub");
  const closeBtn  = $("activity-close");

  const KIND_ICON = {
    training:   "▣",
    rag_build:  "⌗",
    rag_ready:  "✓",
    data_prep:  "▥",
    inference:  "◉",
    download:   "↓",
  };
  const KIND_LABEL = {
    training:   "Training",
    rag_build:  "RAG build",
    rag_ready:  "RAG ready",
    data_prep:  "Data prep",
    inference:  "Inference",
    download:   "Download",
  };

  function fmtTimeAgo(ts) {
    if (!ts) return "";
    const dt = Math.max(0, Date.now() / 1000 - ts);
    if (dt < 60) return Math.round(dt) + "s ago";
    if (dt < 3600) return Math.round(dt / 60) + "m ago";
    if (dt < 86400) return Math.round(dt / 3600) + "h ago";
    return Math.round(dt / 86400) + "d ago";
  }

  function escapeHtml(s) {
    return String(s || "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;",
      '"': "&quot;", "'": "&#39;"
    }[c]));
  }

  function renderTasks(tasks) {
    if (!tasks || tasks.length === 0) {
      bodyEl.innerHTML =
        '<div class="activity-empty">No activity. Upload a file or start a training run.</div>';
      return;
    }

    bodyEl.innerHTML = tasks
      .map((t, idx) => {
        const icon = KIND_ICON[t.kind] || "›";
        const label = KIND_LABEL[t.kind] || t.kind;
        const pct = Math.round((t.progress || 0) * 100);
        const isDone = t.status === "done" || t.status === "ready";
        const isErr = t.status === "error";
        const isActive = !isDone && !isErr;
        const projectBadge = t.project_name
          ? `<span class="activity-proj">${escapeHtml(t.project_name)}</span>`
          : "";
        const shortMsg = (t.message || "").split("·")[0].trim().slice(0, 80);
        return `
        <div class="activity-row ${isDone ? "done" : ""} ${isErr ? "err" : ""}"
             data-url="${escapeHtml(t.url || "")}" data-idx="${idx}">
          <div class="activity-row-head" aria-expanded="false">
            <span class="ki">${icon}</span>
            <span class="activity-label">${escapeHtml(label)}</span>
            ${projectBadge}
            <span class="activity-status status-${escapeHtml(t.status)}">${escapeHtml(t.status)}</span>
            <span class="activity-time">${escapeHtml(fmtTimeAgo(t.started_at))}</span>
            <span class="activity-chevron" aria-hidden="true">▸</span>
          </div>
          <div class="activity-msg activity-msg-short">${escapeHtml(shortMsg || "")}</div>
          <div class="activity-expand">
            <div class="activity-msg activity-msg-full">${escapeHtml(t.message || "")}</div>
            ${
              isActive || pct > 0
                ? `<div class="activity-bar">
                     <div class="activity-bar-fill" style="width:${pct}%"></div>
                     <span class="activity-bar-text">${pct}%</span>
                   </div>`
                : ""
            }
            <div class="activity-meta">
              ${
                t.project_id
                  ? `<span class="meta-row"><span class="meta-k">project</span><span class="mono">${escapeHtml(t.project_id)}</span></span>`
                  : ""
              }
              ${
                t.kind
                  ? `<span class="meta-row"><span class="meta-k">kind</span><span class="mono">${escapeHtml(t.kind)}</span></span>`
                  : ""
              }
              <span class="meta-row"><span class="meta-k">started</span><span class="mono">${escapeHtml(fmtTimeAgo(t.started_at))}</span></span>
              ${
                typeof t.progress === "number"
                  ? `<span class="meta-row"><span class="meta-k">progress</span><span class="mono">${pct}%</span></span>`
                  : ""
              }
            </div>
            ${
              t.url
                ? `<a href="${escapeHtml(t.url)}" class="activity-goto"
                      data-link data-link-href="${escapeHtml(t.url)}">GO TO →</a>`
                : ""
            }
          </div>
        </div>`;
      })
      .join("");

    // Accordion handlers — each row toggles its expand panel on click.
    bodyEl.querySelectorAll(".activity-row").forEach((row) => {
      const head = row.querySelector(".activity-row-head");
      if (!head) return;
      head.addEventListener("click", (ev) => {
        // Don't toggle when the user clicks the GO TO link.
        if (ev.target.closest(".activity-goto")) return;
        const open = row.classList.toggle("open");
        head.setAttribute("aria-expanded", open ? "true" : "false");
      });
      // The GO TO link itself — stop propagation so click doesn't toggle.
      const goto = row.querySelector(".activity-goto");
      if (goto) {
        goto.addEventListener("click", (ev) => ev.stopPropagation());
      }
    });
  }

  function updateBadge(count) {
    if (count > 0) {
      countEl.textContent = String(count);
      countEl.hidden = false;
      trigger.classList.add("active");
    } else {
      countEl.hidden = true;
      trigger.classList.remove("active");
    }
  }

  let lastTasks = [];

  async function refresh() {
    try {
      const r = await fetch("/api/activity");
      if (!r.ok) return;
      const d = await r.json();
      lastTasks = d.tasks || [];
      updateBadge(d.active_count || 0);

      if (drawer && !drawer.hidden) {
        // Sub-title summarises kinds of active work (or "idle").
        const kinds = Object.entries(d.by_kind || {})
          .map(([k, n]) => `${KIND_LABEL[k] || k}: ${n}`)
          .join(" · ");
        subEl.textContent = kinds || (d.active_count ? `${d.active_count} active` : "idle");
        // Re-render only if the task list changed shape (cheap pointer compare).
        renderTasks(lastTasks);
      }
    } catch (e) {
      console.warn("[activity] refresh failed:", e);
    }
  }

  function open() {
    if (!drawer || !backdrop) return;
    drawer.hidden = false;
    backdrop.hidden = false;
    drawer.getBoundingClientRect();
    drawer.classList.add("open");
    backdrop.classList.add("open");
    refresh();
  }

  function close() {
    if (!drawer || !backdrop) return;
    drawer.classList.remove("open");
    backdrop.classList.remove("open");
    setTimeout(() => {
      drawer.hidden = true;
      backdrop.hidden = true;
    }, 220);
  }

  if (trigger) trigger.addEventListener("click", () => {
    if (drawer && !drawer.hidden) close();
    else open();
  });
  if (closeBtn) closeBtn.addEventListener("click", close);
  if (backdrop) backdrop.addEventListener("click", close);

  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && drawer && !drawer.hidden) {
      ev.preventDefault();
      close();
    }
  });

  setInterval(refresh, 2000);
  refresh();

  window.ftsActivity = { open, close, refresh };
})();
