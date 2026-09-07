/**
 * Finetune Studio — SPA router
 *
 * Intercepts clicks on any <a data-link> inside the sidebar/topbar,
 * fetches the destination as raw HTML, and swaps in only the new
 * <main id="content">. Base shell (sidebar, topbar, CSS, JS modules)
 * stays loaded. Feels instant after the first page load.
 *
 * Falls back gracefully if the response isn't HTML-shaped.
 */
(function () {
  "use strict";

  const content = () => document.getElementById("content");
  const app     = () => document.getElementById("app");
  const nav     = () => document.getElementById("nav");
  const crumbs  = () => document.getElementById("crumbs");

  function setActiveNav(url) {
    const path = url.split("?")[0];
    nav().querySelectorAll(".nav-item").forEach((a) => {
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
    const newCrumbs  = doc.getElementById("crumbs");
    const newNav     = doc.getElementById("nav");
    const newTitle   = doc.querySelector("title");
    return {
      contentHTML: newContent ? newContent.innerHTML : null,
      crumbsHTML:  newCrumbs  ? newCrumbs.innerHTML  : null,
      navHTML:     newNav     ? newNav.innerHTML     : null,
      title:       newTitle   ? newTitle.textContent : null,
      fullHTML:    !newContent,
    };
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
      // Swap
      c.innerHTML = ext.contentHTML;
      if (ext.crumbsHTML && crumbs()) crumbs().innerHTML = ext.crumbsHTML;
      if (ext.navHTML && nav()) nav().innerHTML = ext.navHTML;
      if (ext.title) document.title = ext.title;
      // Highlight nav
      setActiveNav(url);
      // Fade-in + re-run page scripts
      c.style.opacity = "1";
      c.classList.remove("is-loading");
      // Re-execute any inline <script> tags in the new content
      c.querySelectorAll("script").forEach((old) => {
        const s = document.createElement("script");
        for (const attr of old.attributes) s.setAttribute(attr.name, attr.value);
        s.textContent = old.textContent;
        old.replaceWith(s);
      });
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

  // ── Nav toggle ────────────────────────────────────────────────────
  document.addEventListener("click", function (ev) {
    const t = ev.target.closest("#nav-toggle");
    if (!t) return;
    const collapsed = !app().classList.contains("collapsed-nav");
    app().classList.toggle("collapsed-nav", collapsed);
    try { localStorage.setItem("fts.nav.collapsed", collapsed ? "1" : ""); } catch (e) {}
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
