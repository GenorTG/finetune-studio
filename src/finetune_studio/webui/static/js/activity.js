/* ============================================================
   Activity feed — bottom-right slide-up drawer
   ------------------------------------------------------------
   Shows all running tasks: training, RAG build, data-prep,
   inference loaded, HF downloads. Each row has a "go to" link.
   Polls /api/activity every 2s. The header button shows an
   active-count badge when any task is running.
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
    if (dt < 60) return dt.toFixed(0) + "s ago";
    if (dt < 3600) return (dt / 60).toFixed(0) + "m ago";
    if (dt < 86400) return (dt / 3600).toFixed(0) + "h ago";
    return (dt / 86400).toFixed(0) + "d ago";
  }

  function escapeHtml(s) {
    return String(s || "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;",
      '"': "&quot;", "'": "&#39;"
    }[c]));
  }

  function renderTasks(tasks) {
    if (!tasks || tasks.length === 0) {
      bodyEl.innerHTML = '<div class="activity-empty">No activity. Upload a file or start a training run.</div>';
      return;
    }
    bodyEl.innerHTML = tasks.map((t) => {
      const icon = KIND_ICON[t.kind] || "›";
      const label = KIND_LABEL[t.kind] || t.kind;
      const pct = Math.round((t.progress || 0) * 100);
      const isDone = t.status === "done" || t.status === "ready";
      const isErr = t.status === "error";
      const projectBadge = t.project_name
        ? `<span class="activity-proj">${escapeHtml(t.project_name)}</span>`
        : "";
      return `
        <div class="activity-row ${isDone ? 'done' : ''} ${isErr ? 'err' : ''}" data-url="${escapeHtml(t.url || '')}">
          <div class="activity-row-head">
            <span class="activity-kind"><span class="ki">${icon}</span> ${label}</span>
            ${projectBadge}
            <span class="activity-status status-${t.status}">${escapeHtml(t.status)}</span>
            <span class="activity-time">${fmtTimeAgo(t.started_at)}</span>
          </div>
          <div class="activity-msg">${escapeHtml(t.message || "")}</div>
          ${!isDone ? `
            <div class="activity-bar">
              <div class="activity-bar-fill" style="width:${pct}%"></div>
              <span class="activity-bar-text">${pct}%</span>
            </div>` : ""}
          ${t.url ? `<a href="${escapeHtml(t.url)}" class="activity-goto" data-link data-link-href="${escapeHtml(t.url)}">GO TO →</a>` : ""}
        </div>
      `;
    }).join("");

    // Row click → navigate to url
    bodyEl.querySelectorAll(".activity-row[data-url]").forEach((row) => {
      row.addEventListener("click", (ev) => {
        // Ignore if user clicked the link directly
        if (ev.target.closest(".activity-goto")) return;
        const url = row.dataset.url;
        if (!url) return;
        close();
        if (window.ftsSPA && window.ftsSPA.navigate) window.ftsSPA.navigate(url);
        else location.href = url;
      });
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
        // Show sub-title with breakdown
        const kinds = Object.entries(d.by_kind || {})
          .map(([k, n]) => `${KIND_LABEL[k] || k}: ${n}`)
          .join(" · ");
        subEl.textContent = kinds || (d.active_count ? `${d.active_count} active` : "idle");
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
    }, 200);
  }

  if (trigger) trigger.addEventListener("click", () => {
    if (drawer && !drawer.hidden) close(); else open();
  });
  if (closeBtn) closeBtn.addEventListener("click", close);
  if (backdrop) backdrop.addEventListener("click", close);

  // Escape closes
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && drawer && !drawer.hidden) {
      ev.preventDefault();
      close();
    }
  });

  // Poll every 2s; updates badge regardless of drawer state
  setInterval(refresh, 2000);
  refresh();

  window.ftsActivity = { open, close, refresh };
})();
