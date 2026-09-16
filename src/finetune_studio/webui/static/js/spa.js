/**
 * Finetune Studio — SPA router
 *
 * Intercepts clicks on any <a data-link> inside the sidebar/topbar,
 * fetches the destination as raw HTML, and swaps in only the new
 * <main id="content"> plus <div id="page-scripts">. Base shell
 * (sidebar, topbar, CSS, JS modules) stays loaded. Feels instant
 * after the first page load.
 *
 * Falls back gracefully if the response isn't HTML-shaped, or if
 * page-script injection throws (full navigation via location.href).
 */
(function () {
  "use strict";

  const content = () => document.getElementById("content");
  const pageScriptsEl = () => document.getElementById("page-scripts");
  const app     = () => document.getElementById("app");
  const sessionBar = () => document.getElementById("session-bar");
  const crumbLeaf = () => document.getElementById("crumb-leaf");

  function setActiveNav(url) {
    const path = url.split("?")[0];
    const sb = sessionBar();
    if (!sb) return;
    sb.querySelectorAll(".sb-tab").forEach((a) => {
      const href = a.getAttribute("href") || "";
      if (href === path) a.classList.add("active");
      else if (href && path.startsWith(href + "/")) a.classList.add("active");
      else a.classList.remove("active");
    });
  }

  async function fetchHTML(url) {
    const r = await fetch(url, { headers: { "X-FTS-SPA": "1" } });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.text();
  }

  function extractContent(html) {
    const doc = new DOMParser().parseFromString(html, "text/html");
    const newContent = doc.getElementById("content");
    const newPageScripts = doc.getElementById("page-scripts");
    const newActive  = doc.querySelector(".sb-active-bar");
    const newTitle   = doc.querySelector("title");
    const newCrumb   = doc.getElementById("project-breadcrumb");
    const newWsNav   = doc.getElementById("workspace-subnav");
    return {
      contentHTML: newContent ? newContent.innerHTML : null,
      pageScriptsHTML: newPageScripts ? newPageScripts.innerHTML : "",
      activeHTML:  newActive  ? newActive.innerHTML  : null,
      breadcrumbHTML: newCrumb ? newCrumb.innerHTML : null,
      breadcrumbOuter: newCrumb ? newCrumb.outerHTML : null,
      breadcrumbPresent: !!newCrumb,
      workspaceHTML: newWsNav ? newWsNav.innerHTML : null,
      workspaceOuter: newWsNav ? newWsNav.outerHTML : null,
      workspacePresent: !!newWsNav,
      title:       newTitle   ? newTitle.textContent : null,
      fullHTML:    !newContent,
    };
  }

  /**
   * Sync a base-shell nav that lives outside #content (breadcrumb / workspace).
   * Creates the element when entering a project page from a non-project page;
   * hides it when leaving. Full-page loads already render these in base.html.
   */
  function syncShellNav(id, present, innerHTML, outerHTML, afterId) {
    let el = document.getElementById(id);
    if (present && outerHTML) {
      if (el) {
        if (innerHTML != null) el.innerHTML = innerHTML;
        el.hidden = false;
      } else {
        const after = document.getElementById(afterId);
        if (after) after.insertAdjacentHTML("afterend", outerHTML);
      }
    } else if (el) {
      el.hidden = true;
    }
  }

  /** Rewrite column-0 top-level const/let/class so SPA re-visits don't throw. */
  function rewriteTopLevelDecls(code) {
    return String(code || "")
      .replace(/^(const|let) /gm, "var ")
      .replace(/^class (\w+)/gm, "var $1 = class $1");
  }

  function scriptSrcAlreadyLoaded(srcAttr, exceptEl) {
    if (!srcAttr) return false;
    for (const s of document.querySelectorAll("script[src]")) {
      if (s === exceptEl) continue;
      if (s.getAttribute("src") === srcAttr) return true;
    }
    return false;
  }

  function reexecuteScripts(container) {
    if (!container) return;
    container.querySelectorAll("script").forEach((old) => {
      const srcAttr = old.getAttribute("src");
      if (srcAttr && scriptSrcAlreadyLoaded(srcAttr, old)) {
        old.remove();
        return;
      }
      const s = document.createElement("script");
      for (const attr of old.attributes) s.setAttribute(attr.name, attr.value);
      if (!srcAttr) {
        s.textContent = rewriteTopLevelDecls(old.textContent);
      }
      old.replaceWith(s);
    });
  }

  /**
   * Re-run inline scripts in #content then #page-scripts.
   * On any window error during synchronous injection, fall back to full load.
   */
  function injectScripts(url) {
    let failed = false;
    const onError = function () { failed = true; };
    window.addEventListener("error", onError);
    try {
      reexecuteScripts(content());
      const ps = pageScriptsEl();
      if (ps) reexecuteScripts(ps);
    } finally {
      window.removeEventListener("error", onError);
    }
    if (failed) {
      location.href = url;
      return false;
    }
    return true;
  }

  let inFlight = null;
  async function navigate(url, push = true) {
    if (url === location.href) return;
    if (inFlight) {
      try { await inFlight; } catch (e) {}
    }
    const c = content();
    if (!c) { location.href = url; return; }
    c.classList.add("is-loading");
    const nav_promise = (async () => {
      const html = await fetchHTML(url);
      const ext = extractContent(html);
      if (ext.fullHTML) {
        location.href = url;
        return;
      }
      // Fade-out current
      c.style.transition = "opacity 120ms ease";
      c.style.opacity = "0";
      await new Promise((r) => setTimeout(r, 80));
      // Swap content + page scripts
      c.innerHTML = ext.contentHTML;
      const ps = pageScriptsEl();
      if (ps) ps.innerHTML = ext.pageScriptsHTML;
      const activeBar = document.querySelector(".sb-active-bar");
      if (ext.activeHTML && activeBar) activeBar.innerHTML = ext.activeHTML;
      // QABUG-009 / workspace-subnav: shell chrome outside #content must
      // update (and be created) when SPA-entering a project from /projects.
      syncShellNav(
        "project-breadcrumb",
        ext.breadcrumbPresent,
        ext.breadcrumbHTML,
        ext.breadcrumbOuter,
        "session-bar"
      );
      // Prefer inserting after breadcrumb; fall back to session-bar if the
      // crumb node is still missing (should not happen when pid is set).
      const wsAfter = document.getElementById("project-breadcrumb")
        ? "project-breadcrumb"
        : "session-bar";
      syncShellNav(
        "workspace-subnav",
        ext.workspacePresent,
        ext.workspaceHTML,
        ext.workspaceOuter,
        wsAfter
      );
      if (window.ftsPalette && window.ftsPalette.refresh) window.ftsPalette.refresh();
      if (ext.title) document.title = ext.title;
      // Highlight nav
      setActiveNav(url);
      // Fade-in + re-run page scripts
      c.style.opacity = "1";
      c.classList.remove("is-loading");
      // Re-execute inline <script> tags (#content first, then #page-scripts)
      if (!injectScripts(url)) return;
      // Re-init known page modules (data-poll spans, etc.)
      window.fts && window.fts.init && window.fts.init();
      // Re-mount sprites & animations for the new page
      window.spritesInit && window.spritesInit();
      if (push) history.pushState({}, "", url);
      // Scroll to top on new page
      c.scrollTo({ top: 0, behavior: "instant" });
    })();
    inFlight = nav_promise;
    try { await nav_promise; } finally { inFlight = null; }
  }

  // ── Wire clicks ─────────────────────────────────────────────────────
  document.addEventListener("click", function (ev) {
    const a = ev.target.closest("a[data-link]");
    if (!a) return;
    const href = a.getAttribute("href");
    if (!href || href.startsWith("http") || href.startsWith("#") ||
        a.target === "_blank" || ev.metaKey || ev.ctrlKey || ev.shiftKey) return;
    ev.preventDefault();
    navigate(href);
  });

  // ── Apply persisted collapsed state ──────────────────────────────
  if (app().classList.contains("collapsed-nav")) {
    app().classList.add("collapsed-nav");
  }

  // ── Browser back/forward ──────────────────────────────────────────
  window.addEventListener("popstate", () => {
    // Just re-navigate to the current URL without pushing again.
    navigate(location.href, false);
  });

  // ── Expose for inline page scripts ───────────────────────────────
  window.ftsSPA = { navigate };
})();
