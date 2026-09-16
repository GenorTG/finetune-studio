/**
 * Session-bar overflow affordances: scroll buttons, [nav] menu sync,
 * and scroll-active-tab-into-view. Keeps .sb-tabs as the source of truth.
 */
(function () {
  "use strict";

  const SHELL_ID = "sb-nav-shell";
  const TABS_ID = "sb-tabs";
  const MORE_LIST_ID = "sb-nav-more-list";
  const MORE_ID = "sb-nav-more";
  const PREV_ID = "sb-nav-scroll-prev";
  const NEXT_ID = "sb-nav-scroll-next";
  const SCROLL_STEP = 160;

  function $(id) {
    return document.getElementById(id);
  }

  function tabsEl() {
    return $(TABS_ID);
  }

  function shellEl() {
    return $(SHELL_ID);
  }

  function labelFor(a) {
    const lbl = a.querySelector(".sb-tab-label");
    return (lbl ? lbl.textContent : a.textContent || "").trim();
  }

  function syncMoreMenu() {
    const list = $(MORE_LIST_ID);
    const tabs = tabsEl();
    if (!list || !tabs) return;
    const anchors = tabs.querySelectorAll("a.sb-tab[data-link]");
    list.innerHTML = "";
    anchors.forEach((a) => {
      const item = document.createElement("a");
      item.href = a.getAttribute("href") || "#";
      item.className = "sb-nav-more-link" + (a.classList.contains("active") ? " active" : "");
      item.setAttribute("data-link", "");
      const tab = a.getAttribute("data-tab");
      if (tab) item.setAttribute("data-tab", tab);
      item.setAttribute("role", "menuitem");
      if (a.classList.contains("active")) item.setAttribute("aria-current", "page");
      item.textContent = labelFor(a);
      list.appendChild(item);
    });
  }

  function syncMoreActive() {
    const list = $(MORE_LIST_ID);
    const tabs = tabsEl();
    if (!list || !tabs) return;
    const activeTabs = new Set();
    tabs.querySelectorAll("a.sb-tab.active").forEach((a) => {
      const t = a.getAttribute("data-tab");
      if (t) activeTabs.add(t);
    });
    list.querySelectorAll("a.sb-nav-more-link").forEach((a) => {
      const t = a.getAttribute("data-tab") || "";
      const on = activeTabs.has(t);
      a.classList.toggle("active", on);
      if (on) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
  }

  function updateScrollAffordance() {
    const tabs = tabsEl();
    const shell = shellEl();
    const prev = $(PREV_ID);
    const next = $(NEXT_ID);
    if (!tabs || !shell) return;
    const overflow = tabs.scrollWidth > tabs.clientWidth + 2;
    shell.setAttribute("data-overflow", overflow ? "true" : "false");
    shell.classList.toggle("is-overflowing", overflow);
    const atStart = tabs.scrollLeft <= 2;
    const atEnd = tabs.scrollLeft + tabs.clientWidth >= tabs.scrollWidth - 2;
    if (prev) {
      prev.hidden = !overflow;
      prev.disabled = atStart;
    }
    if (next) {
      next.hidden = !overflow;
      next.disabled = atEnd;
    }
    shell.classList.toggle("can-scroll-left", overflow && !atStart);
    shell.classList.toggle("can-scroll-right", overflow && !atEnd);
  }

  function scrollByDir(dir) {
    const tabs = tabsEl();
    if (!tabs) return;
    tabs.scrollBy({ left: dir * SCROLL_STEP, behavior: "smooth" });
  }

  function scrollActiveIntoView() {
    const tabs = tabsEl();
    if (!tabs) return;
    const active = tabs.querySelector("a.sb-tab.active");
    if (!active) return;
    try {
      active.scrollIntoView({ inline: "nearest", block: "nearest", behavior: "smooth" });
    } catch (_e) {
      active.scrollIntoView(false);
    }
    updateScrollAffordance();
  }

  function closeMore() {
    const more = $(MORE_ID);
    if (more) more.open = false;
  }

  function refresh() {
    syncMoreMenu();
    updateScrollAffordance();
    scrollActiveIntoView();
  }

  function wire() {
    const tabs = tabsEl();
    const prev = $(PREV_ID);
    const next = $(NEXT_ID);
    const more = $(MORE_ID);
    if (!tabs) return;

    if (prev) {
      prev.addEventListener("click", function (ev) {
        ev.preventDefault();
        scrollByDir(-1);
      });
    }
    if (next) {
      next.addEventListener("click", function (ev) {
        ev.preventDefault();
        scrollByDir(1);
      });
    }
    tabs.addEventListener("scroll", updateScrollAffordance, { passive: true });
    window.addEventListener("resize", updateScrollAffordance);

    if (more) {
      more.addEventListener("toggle", function () {
        if (more.open) syncMoreMenu();
      });
      // Close after choosing a link (SPA click handler still runs).
      more.addEventListener("click", function (ev) {
        const a = ev.target.closest("a.sb-nav-more-link");
        if (a) closeMore();
      });
      document.addEventListener("click", function (ev) {
        if (!more.open) return;
        if (more.contains(ev.target)) return;
        closeMore();
      });
      document.addEventListener("keydown", function (ev) {
        if (ev.key === "Escape" && more.open) closeMore();
      });
    }

    if (typeof ResizeObserver !== "undefined") {
      const ro = new ResizeObserver(updateScrollAffordance);
      ro.observe(tabs);
      const shell = shellEl();
      if (shell) ro.observe(shell);
    }

    refresh();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", wire);
  } else {
    wire();
  }

  window.ftsNavOverflow = {
    refresh: refresh,
    syncMoreActive: syncMoreActive,
    updateScrollAffordance: updateScrollAffordance,
    scrollActiveIntoView: scrollActiveIntoView,
    closeMore: closeMore,
  };
})();
