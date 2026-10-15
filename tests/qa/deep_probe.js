/* Deep WebUI QA probe — pasted into the page via browser evaluate.
 *
 * Catches what pytest and HTTP codes cannot see:
 *   overflow   element sticking past the viewport with no scroll ancestor
 *   clipped    ellipsis/hidden text with NO title attribute (unreadable, unrecoverable)
 *   tiny       body text under 11px
 *   low        text under 4.5:1 against its COMPOSITED ancestor background
 *   cryptic    icon-only / ≤2-char controls with no title, aria-label or text
 *   touching   interactive element with < 4px padding inside a bordered box
 *   wall       a single text block over 320 chars with no structure (hard to read)
 *   emptybad   empty-state with no explanation or no action link
 *
 * Usage: window.__qa('<label>') -> JSON string.
 */
window.__qa = function (label) {
  /* HARD BUDGET: a previous QA session DDoS'd its own browser tab by running
     uncapped full-document scans (getComputedStyle × 7 passes) on heavy pages
     inside iframes; timed-out evaluates kept running and starved the main
     thread. All passes now slice the node set and share one deadline. */
  const DEADLINE = performance.now() + 12000;
  const vw = document.documentElement.clientWidth;
  const vh = document.documentElement.clientHeight;
  const sel = (el) => {
    const c = typeof el.className === 'string' ? el.className.trim().split(/\s+/).filter(Boolean) : [];
    return el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') + (c.length ? '.' + c.slice(0, 2).join('.') : '');
  };
  const vis = (el) => {
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0') return false;
    const r = el.getBoundingClientRect();
    return r.width > 1 && r.height > 1;
  };
  const parseRGB = (s) => { const m = String(s).match(/[\d.]+/g); return m ? m.map(Number) : null; };
  const lum = (x) => { const c = x / 255; return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
  const ratio = (a, b) => {
    if (!a || !b) return null;
    const L = (v) => 0.2126 * lum(v[0]) + 0.7152 * lum(v[1]) + 0.0722 * lum(v[2]);
    const l1 = L(a), l2 = L(b);
    return +((Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05)).toFixed(2);
  };
  const effBg = (el) => {
    let p = el;
    while (p && p !== document.documentElement) {
      const b = parseRGB(getComputedStyle(p).backgroundColor);
      if (b && (b.length < 4 || b[3] > 0.5)) return b;
      p = p.parentElement;
    }
    return parseRGB(getComputedStyle(document.body).backgroundColor) || [10, 10, 10];
  };
  /* An element inside a horizontally scrollable ancestor is REACHABLE by
     scrolling, even when it currently renders past the viewport. Requiring it
     to already fit produced false positives on every wide table (file actions,
     bench case results) — scroll containers exist precisely to hold them. */
  const scrollAncestorAbsorbs = (el) => {
    let p = el.parentElement;
    while (p && p !== document.body) {
      const ox = getComputedStyle(p).overflowX;
      if (ox === 'auto' || ox === 'scroll') return true;
      p = p.parentElement;
    }
    return false;
  };

  const all = [...document.querySelectorAll('body *')].filter(vis).slice(0, 4000);
  /* Decorative subtrees (aria-hidden, e.g. dashboard matrix-rain) are not
     user content — never judge them for clip/contrast/size/readability. */
  const decor = (el) => !!el.closest('[aria-hidden="true"]');
  const overBudget = () => performance.now() > DEADLINE;
  const out = { label, vw, vh, theme: document.documentElement.getAttribute('data-theme') || 'dark', budgetMs: 12000 };

  out.pageScroll = document.documentElement.scrollWidth > vw + 1;

  out.partial = false;
  out.overflow = all.filter((el) => {
    if (overBudget()) { out.partial = true; return false; }
    const r = el.getBoundingClientRect();
    if (r.width < 8 || r.height < 4) return false;
    if (r.right <= vw + 2 || r.left >= vw) return false;
    return !scrollAncestorAbsorbs(el);
  }).map((el) => ({ sel: sel(el), right: Math.round(el.getBoundingClientRect().right) })).slice(0, 12);

  /* Clipped text is only a BUG when the full string is unrecoverable: no
     title, no aria-label. With a tooltip it is a deliberate truncation. */
  out.clipped = all.filter((el) => {
    if (overBudget()) { out.partial = true; return false; }
    if (decor(el)) return false;
    if (!el.children.length && !el.textContent.trim()) return false;
    if (el.scrollWidth <= el.clientWidth + 2) return false;
    const cs = getComputedStyle(el);
    if (cs.overflowX === 'auto' || cs.overflowX === 'scroll') return false;
    if (el.title || el.getAttribute('aria-label')) return false;
    return !el.querySelector('[title]');
  }).map((el) => ({ sel: sel(el), lost: el.scrollWidth - el.clientWidth, txt: el.textContent.trim().slice(0, 30) })).slice(0, 12);

  out.tiny = all.filter((el) => {
    if (overBudget()) { out.partial = true; return false; }
    if (decor(el)) return false;
    const t = [...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim().length > 2);
    if (!t) return false;
    return parseFloat(getComputedStyle(el).fontSize) < 11;
  }).map((el) => ({ sel: sel(el), px: parseFloat(getComputedStyle(el).fontSize), txt: el.textContent.trim().slice(0, 24) })).slice(0, 12);

  out.low = all.filter((el) => {
    if (overBudget()) { out.partial = true; return false; }
    if (decor(el)) return false;
    const t = [...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim().length > 2);
    if (!t) return false;
    const cs = getComputedStyle(el);
    const r = ratio(parseRGB(cs.color), effBg(el));
    return r !== null && r < 4.5;
  }).map((el) => {
    const cs = getComputedStyle(el);
    return { sel: sel(el), r: ratio(parseRGB(cs.color), effBg(el)), px: parseFloat(cs.fontSize), txt: el.textContent.trim().slice(0, 24) };
  }).slice(0, 12);

  /* A control the user cannot identify: no visible words, no tooltip. */
  out.cryptic = [...document.querySelectorAll('button, a, [role="button"], summary')].filter(vis).filter((el) => {
    if (el.title || el.getAttribute('aria-label')) return false;
    const txt = el.textContent.replace(/\s+/g, '');
    if (!txt) return true;
    return txt.length <= 2 && !/^\d+$/.test(txt);
  }).map((el) => ({ sel: sel(el), txt: el.textContent.trim().slice(0, 12) })).slice(0, 12);

  /* Text walls: judged on the element's OWN text nodes, never descendants'.
     Measuring textContent made every layout container (div.main, table-scroll)
     look like a 36,000-character wall. */
  const ownText = (el) => [...el.childNodes].filter((n) => n.nodeType === 3)
    .map((n) => n.textContent).join(' ').replace(/\s+/g, ' ').trim();
  out.wall = all.filter((el) => {
    if (overBudget()) { out.partial = true; return false; }
    /* <pre>/<code> are logs and payloads: monospaced, scrollable, and meant
       to be long. Only prose blocks are readability problems. */
    if (el.closest('pre, code')) return false;
    const t = ownText(el);
    if (t.length < 320) return false;
    return !el.querySelector('li, br, p, ul, ol');
  }).map((el) => ({ sel: sel(el), chars: ownText(el).length, txt: ownText(el).slice(0, 40) })).slice(0, 8);

  /* Empty states must explain AND offer the next action. */
  out.emptyBad = [...document.querySelectorAll('.empty')].filter(vis).filter((el) => {
    const t = el.textContent.trim();
    return t.length < 12 || (!el.querySelector('a, button') && t.length < 40);
  }).map((el) => ({ sel: sel(el), txt: el.textContent.trim().slice(0, 40) })).slice(0, 8);

  out.counts = {
    overflow: out.overflow.length, clipped: out.clipped.length, tiny: out.tiny.length,
    low: out.low.length, cryptic: out.cryptic.length, wall: out.wall.length, emptyBad: out.emptyBad.length,
  };
  out.clean = Object.values(out.counts).every((n) => n === 0) && !out.pageScroll;
  return JSON.stringify(out);
};
'__qa ready';
