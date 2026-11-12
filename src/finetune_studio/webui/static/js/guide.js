/**
 * Finetune Studio — Guide panel
 *
 * A docked assistant that lives in the base shell (outside #content), so SPA navigation
 * (spa.js: data-link clicks swap only #content) never unloads it: the conversation and the
 * panel state survive every in-app page change with no restore step. Hard reloads and the
 * SPA's own full-load fallbacks lose JS state, so history, open state and one pending UI
 * effect are mirrored into sessionStorage (tab-scoped: no stale conversations next day).
 *
 * Talks to POST /api/guide/chat (SSE). Events: start, status, thinking, tool_call,
 * tool_result, ui, final, error, done. `ui` events are the only way the model touches the
 * page, and the server only emits allow-listed ones:
 *   navigate  → ftsSPA.navigate(url)
 *   highlight → pulse one registered control (opens its page first if needed)
 *   prefill   → set registered form fields, visibly flagged, never submitted
 *   action    → a prepared request (method, URL, body) of the page's own endpoint: a card with Run / Skip in
 *               "ask" mode, run at once in "auto" (hands-free) mode; the outcome goes back to the guide as the
 *               next message (ACTION_RESULT / ACTION_SKIPPED) and the conversation continues.
 */
(function () {
  "use strict";

  var STORE = "fts.guide.v1";
  var PENDING = "fts.guide.pending.v1";
  var MAX_STORED = 80;

  var MAX_AUTO_ACTIONS = 4;   // hands-free: actions in a row per user message (guide/actions.py MAX_AUTO_ACTIONS)
  var state = { open: false, messages: [], transcript: [], busy: false, abort: null, mode: "ask",
                autoRuns: 0, continueAfter: false };
  var chain = Promise.resolve();   // UI effects run strictly in order
  var els = {};

  // ── tiny helpers ────────────────────────────────────────────────
  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  function pathOf(url) { return new URL(url, location.href).pathname.replace(/\/$/, "") || "/"; }
  function onPage(url) { return (location.pathname.replace(/\/$/, "") || "/") === pathOf(url); }
  function currentProject() {
    var m = location.pathname.match(/^\/projects\/([^/]+)/);
    return m && m[1] !== "new" ? m[1] : null;
  }
  function waitFor(fn, timeout) {
    return new Promise(function (resolve) {
      var t0 = Date.now();
      (function poll() {
        var v = null;
        try { v = fn(); } catch (e) { v = null; }
        if (v) return resolve(v);
        if (Date.now() - t0 > (timeout || 4000)) return resolve(null);
        setTimeout(poll, 80);
      })();
    });
  }

  // ── persistence ─────────────────────────────────────────────────
  function save() {
    try {
      sessionStorage.setItem(STORE, JSON.stringify({
        open: state.open,
        messages: state.messages.slice(-MAX_STORED),
        transcript: state.transcript.slice(-MAX_STORED),
      }));
    } catch (e) { /* quota — history is a convenience */ }
  }
  function load() {
    try {
      var raw = sessionStorage.getItem(STORE);
      if (!raw) return;
      var d = JSON.parse(raw);
      state.open = !!d.open;
      state.messages = Array.isArray(d.messages) ? d.messages : [];
      state.transcript = Array.isArray(d.transcript) ? d.transcript : [];
    } catch (e) { /* ignore corrupt state */ }
  }

  // ── rendering ───────────────────────────────────────────────────
  function scrollDown() { els.msgs.scrollTop = els.msgs.scrollHeight; }
  function clearEmpty() {
    var e = els.msgs.querySelector(".guide-empty");
    if (e) e.remove();
  }

  function inlineMarkdown(node, text) {
    // Escape-by-construction: only text nodes + <strong>/<code> elements are ever created.
    String(text).split(/(\*\*[^*]+\*\*|`[^`]+`)/g).forEach(function (part) {
      if (!part) return;
      if (part.indexOf("**") === 0 && part.slice(-2) === "**" && part.length > 4) {
        node.appendChild(el("strong", null, part.slice(2, -2)));
      } else if (part[0] === "`" && part.slice(-1) === "`" && part.length > 2) {
        node.appendChild(el("code", null, part.slice(1, -1)));
      } else {
        node.appendChild(document.createTextNode(part));
      }
    });
  }
  function renderText(container, text) {
    var list = null;
    String(text).split("\n").forEach(function (line) {
      var m = line.match(/^\s*(?:[-*]|\d+[.)])\s+(.*)$/);
      if (m) {
        if (!list) { list = el("ul", "guide-list"); container.appendChild(list); }
        var li = el("li"); inlineMarkdown(li, m[1]); list.appendChild(li);
        return;
      }
      list = null;
      if (!line.trim()) return;
      var p = el("p"); inlineMarkdown(p, line); container.appendChild(p);
    });
  }

  function addBubble(role, text, extraClass) {
    clearEmpty();
    var wrap = el("div", "guide-msg guide-" + role + (extraClass ? " " + extraClass : ""));
    wrap.appendChild(el("div", "guide-label", role === "user" ? "You" : role === "assistant" ? "Guide" : role));
    var body = el("div", "guide-body");
    if (role === "assistant") renderText(body, text); else body.textContent = text;
    wrap.appendChild(body);
    els.msgs.appendChild(wrap);
    scrollDown();
    return wrap;
  }

  function argChips(args) {
    var box = el("span", "guide-args");
    Object.keys(args || {}).forEach(function (k) {
      var v = args[k], d;
      if (typeof v === "string") d = v.length > 36 ? '"' + v.slice(0, 36) + '…"' : '"' + v + '"';
      else if (Array.isArray(v)) d = "[" + v.length + " items]";
      else d = JSON.stringify(v);
      box.appendChild(el("span", "guide-arg", k + "=" + String(d).slice(0, 60)));
    });
    return box;
  }

  function summarize(name, r) {
    if (r && r.effect === "action") {
      return "prepared: " + r.label + (r.mode === "auto" ? " · runs now (hands-free)" : " · waiting for Run");
    }
    if (!r || typeof r !== "object") return "done";
    if (r.error) return "Error: " + r.error;
    switch (name) {
      case "app_help":
        return r.results && r.results.length
          ? r.results.length + " KB entr" + (r.results.length === 1 ? "y" : "ies") + ": " + r.results.map(function (x) { return x.title; }).join(" · ")
          : "No matching KB entry";
      case "project_overview":
      case "inspect_project_readiness":
        return r.summary || r.next_step || "overview loaded";
      case "list_datasets":
        return r.count + " dataset(s)" + (r.datasets && r.datasets.length ? ": " + r.datasets.map(function (d) { return d.name + " (" + d.rows + " rows)"; }).join(", ") : "");
      case "list_runs":
        return r.count + " run(s)" + (r.runs && r.runs[0] ? "; latest " + r.runs[0].name + " — " + r.runs[0].status : "");
      case "system_status": {
        var g = (r.gpus || [])[0];
        return (g ? g.name + " " + g.used_gb + "/" + g.total_gb + " GB" : "no GPU reported") +
          " · model: " + (r.loaded_model && r.loaded_model.loaded ? r.loaded_model.name : "none") +
          " · helper: " + (r.helper && r.helper.missing ? "missing" : r.helper && r.helper.configured ? "configured" : "n/a");
      }
      case "recommend_training":
        if (r.route === "brain") return "Smart Brain, not training — " + r.why;
        return r.route + " · " + r.tier + " · " + (r.settings ? r.settings.num_epochs + " epochs, rank " + r.settings.lora_rank : "") +
          " · ~" + r.optimizer_steps + " steps" + (r.too_few_steps ? " ⚠ too few" : "");
      case "dataset_health":
        return r.trainable + "/" + r.examples + " rows usable · " + r.holdout + " held out · " + (r.issues || []).length + " issue(s)";
      case "explain_setting":
        return r.label + ": " + r.meaning;
      case "navigate":
        return "Opening " + r.title;
      case "highlight":
        return "Highlighting “" + r.label + "”" + (r.opens_page_first ? " (opening its page first)" : "");
      case "suggest_settings":
        return "Pre-filling " + Object.keys(r.applied || {}).length + " field(s) — not submitted" +
          (r.rejected ? " · rejected: " + Object.keys(r.rejected).join(", ") : "");
      case "list_sources":
        return (r.sources || []).length ? r.sources.length + " source(s)" : "No sources found";
      case "read_source":
        return "Read " + String(r.text || "").length + " chars of " + (r.filename || "source");
      case "list_qa_pairs":
        return r.count + " Q&A pair(s)";
      case "create_qa_pairs":
        return r.written + " pair(s) saved as pending review";
      default:
        return Object.keys(r).slice(0, 4).join(", ");
    }
  }

  function toolCard(name, args) {
    clearEmpty();
    var card = el("div", "guide-tool guide-tool-pending");
    var head = el("div", "guide-tool-head");
    head.appendChild(el("span", "guide-tool-icon", "…"));
    head.appendChild(el("span", "guide-tool-name", name));
    head.appendChild(argChips(args));
    card.appendChild(head);
    var status = el("div", "guide-tool-status", "running…");
    status.setAttribute("role", "status");
    card.appendChild(status);
    els.msgs.appendChild(card);
    scrollDown();
    return card;
  }

  function finishCard(card, name, args, result) {
    var ok = !(result && result.error) && !(result && result.ok === false);
    card.className = "guide-tool " + (ok ? "guide-tool-ok" : "guide-tool-err");
    card.querySelector(".guide-tool-icon").textContent = ok ? "✓" : "✗";
    card.querySelector(".guide-tool-status").textContent = summarize(name, result);
    var dbg = el("details", "guide-tool-debug");
    dbg.appendChild(el("summary", null, "Debug · raw JSON"));
    var pre = el("pre", "guide-pre");
    pre.textContent = JSON.stringify({ arguments: args || {}, result: result }, null, 2);
    dbg.appendChild(pre);
    card.appendChild(dbg);
    scrollDown();
    return { name: name, args: args, ok: ok, summary: summarize(name, result) };
  }

  function restoreTranscript() {
    state.transcript.forEach(function (it) {
      if (it.k === "user") addBubble("user", it.text);
      else if (it.k === "assistant") addBubble("assistant", it.text);
      else if (it.k === "error") addBubble("error", it.text, "guide-error");
      else if (it.k === "effect") addEffectLine(it.text, true);
      else if (it.k === "action") {
        var a = el("div", "guide-action guide-action-" + (it.state || "skip"));
        a.appendChild(el("div", "guide-action-title", it.label));
        a.appendChild(el("div", "guide-action-status", it.status || ""));
        els.msgs.appendChild(a);
      }
      else if (it.k === "tool") {
        var c = el("div", "guide-tool " + (it.ok ? "guide-tool-ok" : "guide-tool-err"));
        var h = el("div", "guide-tool-head");
        h.appendChild(el("span", "guide-tool-icon", it.ok ? "✓" : "✗"));
        h.appendChild(el("span", "guide-tool-name", it.name));
        h.appendChild(argChips(it.args));
        c.appendChild(h);
        c.appendChild(el("div", "guide-tool-status", it.summary));
        els.msgs.appendChild(c);
      }
    });
    if (state.transcript.length) { clearEmpty(); scrollDown(); }
  }

  function addEffectLine(text, quiet) {
    clearEmpty();
    var line = el("div", "guide-effect" + (quiet ? " guide-effect-old" : ""), text);
    els.msgs.appendChild(line);
    scrollDown();
    return line;
  }
  function note(text, kind) {
    var line = el("div", "guide-note" + (kind ? " guide-note-" + kind : ""), text);
    line.setAttribute("role", "status");
    els.msgs.appendChild(line);
    scrollDown();
    return line;
  }
  function pushT(item) { state.transcript.push(item); save(); }

  function setThinking(text) {
    var t = $("guide-thinking");
    if (!text) { if (t) t.remove(); return; }
    if (!t) { t = el("div", "guide-thinking"); t.id = "guide-thinking"; t.setAttribute("role", "status"); els.msgs.appendChild(t); }
    t.textContent = "● " + text;
    scrollDown();
  }

  // ── UI effects ──────────────────────────────────────────────────
  var activeTag = null;
  function clearTag() {
    if (activeTag) { activeTag.node.remove(); window.removeEventListener("scroll", activeTag.fn, true); activeTag = null; }
    document.querySelectorAll(".guide-pulse").forEach(function (n) { n.classList.remove("guide-pulse"); });
  }

  function spaNavigate(url) {
    return new Promise(function (resolve) {
      var finished = false;
      function done() { if (!finished) { finished = true; document.removeEventListener("fts:navigated", done); resolve(); } }
      document.addEventListener("fts:navigated", done);
      setTimeout(done, 7000);
      if (window.ftsSPA && window.ftsSPA.navigate) window.ftsSPA.navigate(url);
      else location.href = url;
    });
  }

  function findEl(selector, scope) {
    var root = scope ? document.querySelector(scope) : document;
    return root ? root.querySelector(selector) : null;
  }
  function visibleTarget(node) {
    // Checkbox/radio inputs are tiny: pulse their label instead; open any closed <details> above.
    var t = (node.type === "checkbox" || node.type === "radio") && node.closest("label") ? node.closest("label") : node;
    var d = t.closest("details");
    while (d) { d.open = true; d = d.parentElement ? d.parentElement.closest("details") : null; }
    return t;
  }

  async function ensurePage(ev) {
    if (!ev.url || onPage(ev.url)) return true;
    try { sessionStorage.setItem(PENDING, JSON.stringify(ev)); } catch (e) { /* ignore */ }
    await spaNavigate(ev.url);
    return onPage(ev.url);
  }
  function donePending() { try { sessionStorage.removeItem(PENDING); } catch (e) { /* ignore */ } }

  async function doNavigate(ev) {
    if (onPage(ev.url)) { addEffectLine("▶ Already on " + (ev.title || ev.page)); return; }
    await spaNavigate(ev.url);
    var line = "▶ Opened " + (ev.title || ev.page);
    addEffectLine(line); pushT({ k: "effect", text: line });
    var h1 = document.querySelector("#content h1");
    if (h1) { h1.classList.add("guide-pulse"); setTimeout(function () { h1.classList.remove("guide-pulse"); }, 2500); }
  }

  async function doHighlight(ev) {
    if (!(await ensurePage(ev))) { note("Could not open " + (ev.page || ev.url), "err"); return; }
    donePending();
    var node = await waitFor(function () { return findEl(ev.selector); }, 4500);
    if (!node) {
      var miss = "Could not find “" + ev.label + "” on this page.";
      note(miss, "err"); pushT({ k: "effect", text: miss });
      return;
    }
    clearTag();
    var target = visibleTarget(node);
    target.scrollIntoView({ block: "center", behavior: "smooth" });
    await sleep(420);
    target.classList.add("guide-pulse");
    var tag = el("div", "guide-tag", "👉 " + ev.label);
    document.body.appendChild(tag);
    function place() {
      var r = target.getBoundingClientRect();
      tag.style.top = Math.max(8, r.top - 30) + "px";
      tag.style.left = Math.max(8, Math.min(r.left, window.innerWidth - tag.offsetWidth - 8)) + "px";
    }
    place();
    window.addEventListener("scroll", place, true);
    activeTag = { node: tag, fn: place };
    setTimeout(function () { if (activeTag && activeTag.node === tag) clearTag(); }, 9000);
    var line = "◎ Highlighted “" + ev.label + "”";
    addEffectLine(line); pushT({ k: "effect", text: line });
  }

  function applyField(f) {
    var node = findEl(f.selector, f.scope);
    if (!node) return false;
    if (f.kind === "choice" && f.selector.indexOf("training_mode") >= 0) {
      var radio = findEl('[name="training_mode"][value="' + f.value + '"]', f.scope);
      if (!radio) return false;
      radio.checked = true; radio.dispatchEvent(new Event("change", { bubbles: true }));
      flagFilled(radio); return true;
    }
    if (f.kind === "bool") node.checked = !!f.value;
    else if (node.tagName === "SELECT") {
      var has = Array.prototype.some.call(node.options, function (o) { return o.value === String(f.value); });
      if (!has) return false;
      node.value = String(f.value);
    } else node.value = String(f.value);
    node.dispatchEvent(new Event("input", { bubbles: true }));
    node.dispatchEvent(new Event("change", { bubbles: true }));
    flagFilled(node);
    return true;
  }
  function flagFilled(node) {
    var t = visibleTarget(node);
    t.classList.add("guide-prefilled");
    var clear = function () { t.classList.remove("guide-prefilled"); node.removeEventListener("input", clear); node.removeEventListener("change", clear); };
    setTimeout(function () {
      node.addEventListener("input", clear); node.addEventListener("change", clear);
    }, 400);   // ignore the synthetic events dispatched while filling
  }

  async function doPrefill(ev) {
    if (!(await ensurePage(ev))) { note("Could not open " + (ev.page || ev.url), "err"); return; }
    donePending();
    var first = ev.fields[0];
    await waitFor(function () { return findEl(first.selector, first.scope); }, 4500);
    // Route first, then preset (the page's advisor then rewrites its own fields), then the rest.
    var ordered = ev.fields.slice().sort(function (a, b) {
      var rank = function (n) { return n === "training_mode" ? 0 : n === "preset" ? 1 : 2; };
      return rank(a.name) - rank(b.name);
    });
    var done = [], missing = [], firstNode = null;
    for (var i = 0; i < ordered.length; i++) {
      var f = ordered[i];
      if (applyField(f)) {
        done.push(f.label + " = " + f.value);
        if (!firstNode) firstNode = findEl(f.selector, f.scope);
      } else missing.push(f.label);
      if (f.name === "preset" && done.length) {
        await waitFor(function () { var a = $("preset-advisory"); return a && !a.hidden; }, 4000);
      }
    }
    if (firstNode) visibleTarget(firstNode).scrollIntoView({ block: "center", behavior: "smooth" });
    var line = "✎ Pre-filled " + done.length + " field(s): " + done.join(", ") + " — review, then press the button yourself.";
    addEffectLine(line); pushT({ k: "effect", text: line });
    if (missing.length) note("Not found on this page: " + missing.join(", "), "err");
  }

  function applyUi(ev) {
    chain = chain.then(function () {
      if (ev.type === "navigate") return doNavigate(ev);
      if (ev.type === "highlight") return doHighlight(ev);
      if (ev.type === "prefill") return doPrefill(ev);
    }).catch(function (e) { note("UI effect failed: " + (e && e.message || e), "err"); });
    return chain;
  }

  // ── SSE client ──────────────────────────────────────────────────
  // ── actions (guide/actions.py) ──────────────────────────────────
  function compact(d) {
    if (!d || typeof d !== "object") return String(d == null ? "" : d).slice(0, 300);
    var keep = {};
    ["ok", "status", "detail", "code", "id", "run_id", "benchmark_id", "name", "case_count", "examples", "version_number",
     "label", "steps", "total_steps"].forEach(function (k) { if (d[k] != null) keep[k] = d[k]; });
    if (Array.isArray(d.jobs)) keep.jobs = d.jobs.length;
    if (d.build && d.build.build_id) keep.build_id = d.build.build_id;
    if (d.benchmark && d.benchmark.id) keep.benchmark_id = d.benchmark.id;
    return JSON.stringify(Object.keys(keep).length ? keep : d).slice(0, 600);
  }

  async function execute(ev) {
    // Only a server-prepared request of this app's own API, with the user's own session.
    if (!/^\/api\//.test(ev.url) || ["GET", "POST", "PUT", "PATCH"].indexOf(ev.method) < 0) {
      return { ok: false, text: "refused: not an app API request" };
    }
    var opts = { method: ev.method, credentials: "same-origin", headers: {} };
    if (ev.body != null && ev.method !== "GET") { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(ev.body); }
    var r = await fetch(ev.url, opts);
    var type = r.headers.get("content-type") || "";
    var data = null;
    if (type.indexOf("application/json") >= 0) data = await r.json().catch(function () { return null; });
    else {
      await r.blob().catch(function () { return null; });   // e.g. the dataset export: the counts are in the headers
      data = { rows: r.headers.get("X-Rows"), grounded_rows: r.headers.get("X-Grounded-Rows") };
    }
    var text = r.ok ? compact(data) : ((data && data.detail) || ("HTTP " + r.status));
    return { ok: r.ok, http: r.status, text: text };
  }

  function actionCard(ev) {
    clearEmpty();
    var card = el("div", "guide-action");
    card.dataset.action = ev.action;
    card.appendChild(el("div", "guide-action-title", ev.label));
    if (ev.detail) card.appendChild(el("div", "guide-action-detail", ev.detail));
    var buttons = el("div", "guide-action-buttons");
    var run = el("button", "btn sm primary", "Run");
    run.type = "button"; run.dataset.guideRun = ev.id;
    var skip = el("button", "btn sm ghost", "Skip");
    skip.type = "button"; skip.dataset.guideSkip = ev.id;
    buttons.appendChild(run); buttons.appendChild(skip);
    card.appendChild(buttons);
    var status = el("div", "guide-action-status", "");
    status.setAttribute("role", "status");
    card.appendChild(status);
    els.msgs.appendChild(card);
    scrollDown();
    run.addEventListener("click", function () { runAction(ev, card); });
    skip.addEventListener("click", function () { skipAction(ev, card); });
    return card;
  }

  function settle(card, ev, cls, text) {
    card.classList.add("guide-action-" + cls);
    var b = card.querySelector(".guide-action-buttons");
    if (b) b.remove();
    card.querySelector(".guide-action-status").textContent = text;
    pushT({ k: "action", label: ev.label, state: cls, status: text });
  }

  function handBack(content) {
    state.messages.push({ role: "user", content: content });
    save();
    if (state.busy) state.continueAfter = true;   // the reply that proposed it is still streaming
    else turn();
  }

  async function runAction(ev, card) {
    card.querySelector(".guide-action-buttons") && card.querySelectorAll(".guide-action-buttons button").forEach(function (b) { b.disabled = true; });
    card.querySelector(".guide-action-status").textContent = "running…";
    var out;
    try { out = await execute(ev); } catch (e) { out = { ok: false, text: (e && e.message) || String(e) }; }
    settle(card, ev, out.ok ? "ok" : "err", (out.ok ? "✓ done: " : "✗ failed: ") + out.text);
    if (out.ok && window.fts && window.fts.notify) window.fts.notify(ev.label + ": started", "success");
    handBack("ACTION_RESULT " + ev.action + ": " + (out.ok ? "ok" : "failed (HTTP " + (out.http || "?") + ")") + " " + out.text);
  }

  function skipAction(ev, card) {
    settle(card, ev, "skip", "Skipped.");
    handBack("ACTION_SKIPPED " + ev.action + ": the user chose not to run it now.");
  }

  function onAction(ev) {
    var card = actionCard(ev);
    if (state.mode === "auto" && state.autoRuns < MAX_AUTO_ACTIONS) {
      state.autoRuns += 1;
      runAction(ev, card);
    } else if (state.mode === "auto") {
      card.querySelector(".guide-action-status").textContent = "Hands-free stopped after " + MAX_AUTO_ACTIONS +
        " actions in a row: press Run to go on.";
    }
  }

  async function loadMode() {
    try {
      var d = await window.api.get("/api/settings/guide");
      setModeUi(d.action_mode);
    } catch (e) { /* keep "ask" */ }
  }
  function setModeUi(mode) {
    state.mode = mode === "auto" ? "auto" : "ask";
    var sel = $("guide-mode");
    if (sel) sel.value = state.mode;
    var hint = $("guide-hint");
    if (hint) hint.textContent = state.mode === "auto"
      ? "Hands-free: the guide runs actions itself (up to " + MAX_AUTO_ACTIONS + " in a row) and carries on."
      : "Uses the AI assistant. It prepares actions; you press Run.";
  }
  async function saveMode(mode) {
    try {
      var d = await window.api.put("/api/settings/guide", { action_mode: mode });
      setModeUi(d.action_mode);
      note(state.mode === "auto" ? "Hands-free on: the guide will run the actions it prepares." : "The guide will ask before each action.");
    } catch (e) { note("Could not change the mode: " + ((e && e.message) || e), "err"); setModeUi(state.mode); }
  }

  async function stream(body, signal, onEvent) {
    var r = await fetch("/api/guide/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body), signal: signal,
    });
    if (!r.ok) {
      var d = await r.json().catch(function () { return {}; });
      throw new Error(d.detail || ("HTTP " + r.status));
    }
    var reader = r.body.getReader(), dec = new TextDecoder(), buf = "";
    for (;;) {
      var chunk = await reader.read();
      if (chunk.done) break;
      buf += dec.decode(chunk.value, { stream: true });
      var idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        var frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
        frame.split("\n").forEach(function (line) {
          if (line.indexOf("data: ") !== 0) return;
          try { onEvent(JSON.parse(line.slice(6))); } catch (e) { /* skip malformed frame */ }
        });
      }
    }
  }

  function setBusy(b) {
    state.busy = b;
    els.send.disabled = b;
    els.stop.disabled = !b;
    els.input.disabled = b;
    els.panel.classList.toggle("guide-busy", b);
  }

  async function ask(text) {
    text = String(text || "").trim();
    if (!text || state.busy) return;
    open();
    addBubble("user", text);
    state.messages.push({ role: "user", content: text });
    pushT({ k: "user", text: text });
    state.autoRuns = 0;   // hands-free budget is per user message
    await turn();
  }

  // Keep the history to turns that were answered, but never drop an action outcome: the guide must know it ran.
  var lastQuestion = null;
  function dropUnanswered() {
    var last = state.messages[state.messages.length - 1];
    if (last && last.role === "user" && !/^ACTION_(RESULT|SKIPPED) /.test(last.content || "")) {
      lastQuestion = state.messages.pop();
    }
  }

  // One guide reply to state.messages (a user question, or an ACTION_RESULT handed back).
  async function turn() {
    if (state.busy) return;
    setBusy(true);
    setThinking("sending…");
    var ctl = new AbortController();
    state.abort = ctl;
    var cards = {};
    try {
      var body = { messages: state.messages, project_id: currentProject(), page_path: location.pathname, temperature: 0.2 };
      // No provider_id: the server uses the "guide" job's connection (Settings -> AI assistant).
      await stream(body, ctl.signal, function (ev) {
        if (ev.type === "status") setThinking(ev.message);
        else if (ev.type === "thinking") setThinking(ev.forced_final ? "writing the final answer…" : "thinking… (step " + ev.round + ")");
        else if (ev.type === "tool_call") { setThinking(null); cards[ev.index] = toolCard(ev.name, ev.arguments); setThinking("running " + ev.name + "…"); }
        else if (ev.type === "tool_result") {
          var c = cards[ev.index] || toolCard(ev.name, ev.arguments);
          var rec = finishCard(c, ev.name, ev.arguments, ev.result);
          pushT({ k: "tool", name: rec.name, args: rec.args, ok: rec.ok, summary: rec.summary });
          setThinking("thinking…");
        } else if (ev.type === "ui") {
          if (ev.event && ev.event.type === "action") onAction(ev.event);
          else applyUi(ev.event);
        }
        else if (ev.type === "final") {
          setThinking(null);
          addBubble("assistant", ev.reply, ev.fallback ? "guide-fallback" : "");
          state.messages.push({ role: "assistant", content: ev.reply });
          pushT({ k: "assistant", text: ev.reply });
        } else if (ev.type === "error") {
          setThinking(null);
          var msg = ev.error;
          var bubble = addBubble("error", msg, "guide-error");
          pushT({ k: "error", text: msg });
          dropUnanswered();
          if (ev.status === 409 && window.fts && window.fts.offerAssistantLoad) {
            window.fts.offerAssistantLoad(bubble, "guide", function () { state.messages.push(lastQuestion); turn(); });
          }
        }
      });
    } catch (e) {
      setThinking(null);
      if (e && e.name === "AbortError") note("Stopped.");
      else { addBubble("error", (e && e.message) || String(e), "guide-error"); pushT({ k: "error", text: (e && e.message) || String(e) }); dropUnanswered(); }
    } finally {
      setThinking(null);
      setBusy(false);
      state.abort = null;
      save();
      if (state.open) els.input.focus();
      if (state.continueAfter) { state.continueAfter = false; turn(); }
    }
  }

  // ── panel ───────────────────────────────────────────────────────
  function updateContext() {
    var chip = $("guide-ctx");
    if (!chip) return;
    var pid = currentProject();
    chip.textContent = pid ? "project " + pid.slice(0, 8) : "no project";
    chip.title = "The guide sees the current page (" + location.pathname + ")" + (pid ? " and project " + pid : "");
  }
  function open() {
    state.open = true;
    els.panel.hidden = false;
    requestAnimationFrame(function () { els.panel.classList.add("open"); document.body.classList.add("guide-open"); });
    $("sb-guide") && $("sb-guide").setAttribute("aria-expanded", "true");
    updateContext(); save();
    setTimeout(function () { if (els.input && !els.input.disabled) els.input.focus(); }, 120);
  }
  function close() {
    state.open = false;
    els.panel.classList.remove("open");
    document.body.classList.remove("guide-open");
    $("sb-guide") && $("sb-guide").setAttribute("aria-expanded", "false");
    setTimeout(function () { if (!state.open) els.panel.hidden = true; }, 230);
    clearTag(); save();
  }
  function toggle() { state.open ? close() : open(); }
  function clearAll() {
    if (state.abort) state.abort.abort();
    state.messages = []; state.transcript = [];
    els.msgs.innerHTML = "";
    els.msgs.appendChild(emptyState());
    save();
  }

  var SUGGESTIONS = [
    "How do I train a model on my documents?",
    "What should I do next in this project?",
    "Check whether my dataset is good enough to train",
    "Which training settings should I use?",
    "Where do I download a base model?",
  ];
  function emptyState() {
    var box = el("div", "guide-empty");
    box.appendChild(el("p", null, "I can explain any page, check your project, recommend settings and take you there. I can also do the work: mine files, build the dataset or the Smart Brain, train, test, pin a version. I ask before each action, unless you switch on Hands-free above. I never approve pairs, export a model or delete anything."));
    var chips = el("div", "guide-chips");
    SUGGESTIONS.forEach(function (s) {
      var b = el("button", "btn sm guide-chip", s);
      b.type = "button";
      b.addEventListener("click", function () { ask(s); });
      chips.appendChild(b);
    });
    box.appendChild(chips);
    return box;
  }

  function init() {
    els.panel = $("guide-panel");
    if (!els.panel || els.panel.dataset.ready) return;
    els.panel.dataset.ready = "1";
    els.msgs = $("guide-msgs"); els.input = $("guide-input"); els.send = $("guide-send");
    els.stop = $("guide-stop");
    load();
    els.msgs.appendChild(emptyState());
    restoreTranscript();
    var btn = $("sb-guide");
    if (btn) btn.addEventListener("click", toggle);
    $("guide-close").addEventListener("click", close);
    $("guide-clear").addEventListener("click", clearAll);
    els.send.addEventListener("click", function () { var t = els.input.value; els.input.value = ""; ask(t); });
    els.stop.addEventListener("click", function () { if (state.abort) state.abort.abort(); });
    els.input.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); var t = els.input.value; els.input.value = ""; ask(t); }
    });
    document.addEventListener("fts:navigated", updateContext);
    var modeSel = $("guide-mode");
    if (modeSel) modeSel.addEventListener("change", function () { saveMode(modeSel.value); });
    loadMode();
    if (state.open) open();
    // A navigation that fell back to a full page load keeps its effect in sessionStorage.
    try {
      var p = sessionStorage.getItem(PENDING);
      if (p) {
        var ev = JSON.parse(p);
        if (ev && ev.url && onPage(ev.url)) { donePending(); applyUi(ev); }
        else donePending();
      }
    } catch (e) { /* ignore */ }
  }

  window.ftsGuide = { open: open, close: close, toggle: toggle, ask: ask, stream: stream, applyUi: applyUi, summarize: summarize,
                      execute: execute };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
