/* ============================================================
   Activity feed — top-right slide-out drawer
   ------------------------------------------------------------
   Each row is collapsed by default (kind + project + status).
   Click the row head to expand inline details: progress bar,
   full message, project_id, started/elapsed, and a prominent
   "GO TO →" button. Multiple rows can be expanded at once;
   expansion pushes siblings down (no overlay).
   Live updates via SSE (/api/activity/events); silent poll
   fallback only if EventSource fails. Expanded rows and filter
   selects survive re-render, keyed by stable task identity.
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

  /** Stable identity for a task across poll re-renders (not array index). */
  function taskIdentity(t) {
    if (!t || typeof t !== "object") return "";
    if (t.run_id) return String(t.kind || "") + ":run:" + String(t.run_id);
    if (t.id) return String(t.kind || "") + ":id:" + String(t.id);
    // Avoid started_at: training recomputes it every poll from elapsed.
    return [
      t.kind || "",
      t.project_id || "",
      t.url || "",
      t.project_name || "",
    ].join("|");
  }

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

  let lastTasks = [];
  const _filter = { type: "", project: "", status: "" };
  /** @type {Set<string>} taskIdentity keys the user has expanded */
  const _expanded = new Set();

  function syncFilterFromDom() {
    _filter.type = document.getElementById("activity-filter-type")?.value || "";
    _filter.project = document.getElementById("activity-filter-project")?.value || "";
    _filter.status = document.getElementById("activity-filter-status")?.value || "";
  }

  function renderTasks(tasks) {
    // DOM selects are the source of truth across poll re-renders.
    syncFilterFromDom();
    activityPopulateProjectFilter(tasks);
    syncFilterFromDom();

    const filtered = tasks.filter(activityMatchesFilter);
    if (!filtered || filtered.length === 0) {
      bodyEl.innerHTML =
        '<div class="activity-empty">' + (tasks.length ? "No activity matches your filters." : "No activity. Upload a file or start a training run.") + "</div>";
      return;
    }

    const liveKeys = new Set(filtered.map(taskIdentity));
    for (const key of [..._expanded]) {
      if (!liveKeys.has(key)) _expanded.delete(key);
    }

    bodyEl.innerHTML = filtered
      .map((t) => {
        const key = taskIdentity(t);
        const isOpen = _expanded.has(key);
        const icon = KIND_ICON[t.kind] || "›";
        const label = KIND_LABEL[t.kind] || t.kind;
        const pct = Math.round((t.progress || 0) * 100);
        const isDone = t.status === "done" || t.status === "ready" || t.status === "completed";
        const isErr = t.status === "error";
        const isActive = !isDone && !isErr;
        const projectBadge = t.project_name
          ? `<span class="activity-proj">${escapeHtml(t.project_name)}</span>`
          : "";
        const shortMsg = (t.message || "").split("·")[0].trim().slice(0, 80);
        return `
        <div class="activity-row ${isDone ? "done" : ""} ${isErr ? "err" : ""} ${isOpen ? "open" : ""}"
             data-url="${escapeHtml(t.url || "")}" data-task-key="${escapeHtml(key)}">
          <div class="activity-row-head" aria-expanded="${isOpen ? "true" : "false"}">
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
        const key = row.getAttribute("data-task-key") || "";
        const open = row.classList.toggle("open");
        head.setAttribute("aria-expanded", open ? "true" : "false");
        if (key) {
          if (open) _expanded.add(key);
          else _expanded.delete(key);
        }
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

  function activityApplyFilter() {
    syncFilterFromDom();
    renderTasks(lastTasks);
  }

  function activityPopulateProjectFilter(tasks) {
    const sel = document.getElementById("activity-filter-project");
    if (!sel) return;
    const current = sel.value || _filter.project;
    const pids = new Set();
    tasks.forEach((t) => { if (t.project_id) pids.add(t.project_id); });
    sel.innerHTML = '<option value="">All projects</option>';
    [...pids].sort().forEach((pid) => {
      const opt = document.createElement("option");
      opt.value = pid;
      const t = tasks.find((x) => x.project_id === pid);
      opt.textContent = (t?.project_name || pid).substring(0, 24);
      sel.appendChild(opt);
    });
    if (current && pids.has(current)) sel.value = current;
  }

  function activityMatchesFilter(t) {
    if (_filter.type && t.kind !== _filter.type) return false;
    if (_filter.project && t.project_id !== _filter.project) return false;
    if (_filter.status && t.status !== _filter.status) return false;
    return true;
  }

  function applySnapshot(d) {
    if (!d || typeof d !== "object") return;
    lastTasks = d.tasks || [];
    updateBadge(d.active_count || 0);

    // Keep the sub-label accurate even while the drawer is closed so the
    // first open never shows a stale "Loading…". Body re-render stays
    // gated on visibility (no point painting a hidden panel).
    const kinds = Object.entries(d.by_kind || {})
      .map(([k, n]) => `${KIND_LABEL[k] || k}: ${n}`)
      .join(" · ");
    subEl.textContent = kinds || (d.active_count ? `${d.active_count} active` : "idle");
    if (drawer && !drawer.hidden) {
      // Re-render restores expanded rows + filter selects via _expanded / DOM.
      renderTasks(lastTasks);
    }
  }

  async function refresh() {
    try {
      const r = await fetch("/api/activity");
      if (!r.ok) return;
      applySnapshot(await r.json());
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

  // Prefer SSE; fall back to silent /api/activity poll only if needed.
  const subscribe = (window.fts && window.fts.subscribe) || null;
  if (subscribe) {
    subscribe("/api/activity/events", applySnapshot, {
      pollUrl: "/api/activity",
      fallbackMs: 5000,
    });
  } else {
    refresh();
    // Last-resort path when app.js has not loaded yet.
    setInterval(refresh, 5000);
  }

  // base.html filter <select onchange="activityApplyFilter()">
  window.activityApplyFilter = activityApplyFilter;
  window.ftsActivity = { open, close, refresh, applyFilter: activityApplyFilter };
})();
