/* ============================================================
   FINETUNE STUDIO — SPRITES & ANIMATIONS
   pixel-art inline SVGs + behavior glue
   ============================================================ */

(function () {
  'use strict';

  const SVG = (children, viewBox = '0 0 24 24') =>
    `<svg viewBox="${viewBox}" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" shape-rendering="crispEdges">${children}</svg>`;

  /* ---------- 1. BIN SPRITE ---------- */
  // Pixel-art trash can with chomping mouth
  window.spriteBin = function (size = 80) {
    const wrap = document.createElement('div');
    wrap.className = 'bin-sprite';
    wrap.style.width = size + 'px';
    wrap.style.height = size + 'px';
    wrap.innerHTML = SVG(`
      <!-- handle -->
      <rect x="9" y="2" width="6" height="2" fill="#00ff66"/>
      <!-- lid -->
      <rect class="lid" x="3" y="6" width="18" height="2" fill="#00ff66">
        <animate attributeName="y" values="6;5;6" dur="0.6s" repeatCount="indefinite"/>
      </rect>
      <!-- body -->
      <rect x="5" y="9" width="14" height="2" fill="#00ff66"/>
      <rect x="4" y="11" width="16" height="10" fill="#00ff66" fill-opacity="0.15" stroke="#00ff66" stroke-width="1"/>
      <!-- mouth (chomping rect) -->
      <g class="mouth">
        <rect x="6" y="14" width="12" height="2" fill="#060a07"/>
        <rect x="6" y="14" width="12" height="1" fill="#00ff66" fill-opacity="0.4"/>
      </g>
      <!-- teeth -->
      <rect x="6"  y="13" width="1" height="1" fill="#00ff66"/>
      <rect x="9"  y="13" width="1" height="1" fill="#00ff66"/>
      <rect x="12" y="13" width="1" height="1" fill="#00ff66"/>
      <rect x="15" y="13" width="1" height="1" fill="#00ff66"/>
      <rect x="7"  y="16" width="1" height="1" fill="#00ff66"/>
      <rect x="10" y="16" width="1" height="1" fill="#00ff66"/>
      <rect x="13" y="16" width="1" height="1" fill="#00ff66"/>
      <rect x="16" y="16" width="1" height="1" fill="#00ff66"/>
      <!-- legs -->
      <rect x="6"  y="21" width="2" height="2" fill="#00ff66"/>
      <rect x="16" y="21" width="2" height="2" fill="#00ff66"/>
    `);
    return wrap;
  };

  // animate file being eaten
  window.eatFile = function (container, filename = 'data.txt') {
    if (!container) return;
    const drop = document.createElement('div');
    drop.className = 'bin-drop';
    drop.textContent = '▼ ' + filename;
    container.appendChild(drop);
    container.classList.add('eating');
    setTimeout(() => {
      container.classList.remove('eating');
      container.classList.add('full');
      drop.remove();
      setTimeout(() => container.classList.remove('full'), 300);
    }, 1200);
  };

  /* ---------- 2. TRAINING FORGE ---------- */
  window.spriteForge = function () {
    const wrap = document.createElement('div');
    wrap.className = 'forge';
    wrap.innerHTML = `
      <svg class="anvil" viewBox="0 0 60 20" width="60" height="20" shape-rendering="crispEdges">
        <rect x="22" y="14" width="16" height="4" fill="#ffaa00"/>
        <rect x="18" y="18" width="24" height="2" fill="#ffaa00"/>
      </svg>
      <svg class="hammer" viewBox="0 0 40 40" width="40" height="40" shape-rendering="crispEdges">
        <!-- handle -->
        <rect x="18" y="14" width="2" height="24" fill="#00ff66"/>
        <!-- head -->
        <rect x="10" y="6" width="20" height="8" fill="#00ff66"/>
        <rect x="12" y="4" width="16" height="2" fill="#00ff66"/>
      </svg>
      <span class="spark s1"></span>
      <span class="spark s2"></span>
      <span class="spark s3"></span>
      <span class="spark s4"></span>
    `;
    return wrap;
  };

  /* ---------- 3. WIRE TRANSFER (loading bar with packets) ---------- */
  window.spriteWireTransfer = function (label = 'TRANSMITTING') {
    const wrap = document.createElement('div');
    wrap.className = 'wire-transfer';
    wrap.innerHTML = `
      <span style="color: var(--text-muted);">[${label}]</span>
      <span class="packets">▮▮▮ ▮▮ ▮▮▮▮ ▮ ▮▮▮ ▮▮ ▮▮▮▮ ▮ ▮▮▮ ▮▮ ▮▮▮▮</span>
      <span class="counter" data-counter>000</span>
    `;
    // hacky counter that goes up forever
    let n = 0;
    const tick = () => {
      n += Math.floor(Math.random() * 80) + 40;
      const el = wrap.querySelector('[data-counter]');
      if (el) el.textContent = String(n).padStart(6, '0');
    };
    const iv = setInterval(tick, 200);
    wrap._stop = () => clearInterval(iv);
    return wrap;
  };

  /* ---------- 4. MATRIX RAIN ---------- */
  window.spriteMatrixRain = function (container) {
    if (!container) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const rain = document.createElement('div');
    rain.className = 'matrix-rain';
    const chars = 'アイウエオカキクケコサシスセソタチツテトナニヌネノ0123456789$#@%&';
    for (let i = 0; i < 40; i++) {
      const span = document.createElement('span');
      span.textContent = chars[Math.floor(Math.random() * chars.length)];
      span.style.left = (Math.random() * 100) + '%';
      span.style.animationDuration = (3 + Math.random() * 6) + 's';
      span.style.animationDelay = (Math.random() * 5) + 's';
      rain.appendChild(span);
    }
    container.style.position = 'relative';
    container.prepend(rain);
    return rain;
  };

  /* ---------- 5. RADAR SWEEP (RAG search) ---------- */
  window.spriteRadar = function () {
    const wrap = document.createElement('div');
    wrap.className = 'radar active';
    return wrap;
  };
  window.stopRadar = function (radar) { if (radar) radar.classList.remove('active'); };

  /* ---------- 6. BOOT SEQUENCE ---------- */
  window.spriteBoot = function (lines) {
    const defaults = [
      '> Initializing kernel...',
      '> Loading drivers...',
      '> Mounting /workspace...',
      '> Connecting to GPU...',
      '> System ready.'
    ];
    const msgs = lines || defaults;
    const seq = document.createElement('div');
    seq.className = 'boot-seq';
    let i = 0;
    const tick = () => {
      if (i >= msgs.length) { setTimeout(() => seq.remove(), 400); return; }
      seq.textContent = msgs[i++];
      setTimeout(tick, 600);
    };
    tick();
    document.body.appendChild(seq);
  };

  /* ---------- 7. SCAN BEAM (loading overlay inside any container) ---------- */
  window.spriteScanBeam = function (container) {
    if (!container) return;
    const beam = document.createElement('div');
    beam.className = 'scan-beam';
    container.style.position = 'relative';
    container.appendChild(beam);
    return beam;
  };

  /* ---------- 8. ASSEMBLY LINE (data pipeline) ---------- */
  window.spriteAssembly = function (stages = ['CHUNK', 'EMBED', 'INDEX', 'STORE']) {
    const wrap = document.createElement('div');
    wrap.className = 'assembly';
    wrap.innerHTML = stages.map((s, i) =>
      `<span class="stage ${i === 0 ? 'active' : ''}" data-stage="${i}">${s}</span>${i < stages.length - 1 ? '<span class="arrow">▶</span>' : ''}`
    ).join('');
    const chunk = document.createElement('div');
    chunk.className = 'chunk';
    wrap.appendChild(chunk);
    return wrap;
  };
  window.advanceAssembly = function (assembly, step) {
    if (!assembly) return;
    const stages = assembly.querySelectorAll('.stage');
    stages.forEach((s, i) => s.classList.toggle('active', i === step));
  };

  /* ---------- 9. FLOPPY SAVE ---------- */
  window.spriteFloppy = function () {
    const wrap = document.createElement('span');
    wrap.className = 'floppy';
    wrap.innerHTML = SVG(`
      <rect x="3" y="3" width="18" height="18" fill="#00ff66" fill-opacity="0.15" stroke="#00ff66" stroke-width="1"/>
      <rect x="3" y="3" width="18" height="3" fill="#00ff66"/>
      <rect x="13" y="6" width="6" height="3" fill="#060a07"/>
      <rect x="6" y="11" width="12" height="8" fill="#060a07"/>
      <rect x="8" y="14" width="2" height="3" fill="#00ff66" fill-opacity="0.4"/>
    `);
    return wrap;
  };
  window.floppySave = function (container) {
    if (!container) return;
    const f = container.querySelector('.floppy') || spriteFloppy();
    f.classList.add('saving');
    setTimeout(() => f.classList.remove('saving'), 700);
  };

  /* ---------- 10. CRT FLICKER (route transitions) ---------- */
  window.crtFlicker = function (el) {
    if (!el) return;
    el.classList.remove('flicker');
    void el.offsetWidth;
    el.classList.add('flicker');
    setTimeout(() => el.classList.remove('flicker'), 300);
  };

  /* ============================================================
     NEW SPRITES (P1) — page-context visual effects
     ------------------------------------------------------------
     Each sprite is an inline-SVG pixel-art component plus a
     behavior helper. All respect prefers-reduced-motion.
     ============================================================ */

  /* ---------- ROBOT HEAD (inference / model thinking) ---------- */
  // Eyes scan, blink, and pulse with token-stream activity.
  // Mouth opens/closes as tokens arrive (call `robotFeed(el)` on
  // each streamed chunk to bump activity).
  window.spriteRobotHead = function (size = 96) {
    const wrap = document.createElement('div');
    wrap.className = 'robot-sprite';
    wrap.style.width = size + 'px';
    wrap.style.height = size + 'px';
    wrap.setAttribute('aria-hidden', 'true');
    wrap.innerHTML = SVG(`
      <!-- antenna -->
      <rect class="rh-ant" x="11" y="1" width="2" height="3" fill="#00ff66"/>
      <rect class="rh-ant-dot" x="11" y="0" width="2" height="1" fill="#9cff00"/>
      <!-- head outline -->
      <rect x="5" y="5" width="14" height="13" fill="#0c1810" stroke="#00ff66" stroke-width="1"/>
      <!-- eyes (CRT scanlines inside) -->
      <g class="rh-eyes">
        <rect x="7" y="8" width="3" height="3" fill="#00ff66"/>
        <rect x="14" y="8" width="3" height="3" fill="#00ff66"/>
        <rect class="rh-scan" x="7" y="8" width="3" height="1" fill="#9cff00"/>
        <rect class="rh-scan" x="14" y="8" width="3" height="1" fill="#9cff00"/>
      </g>
      <!-- mouth (LED bar) -->
      <g class="rh-mouth">
        <rect x="8" y="13" width="8" height="1" fill="#00ff66"/>
        <rect x="8" y="15" width="8" height="1" fill="#00ff66" fill-opacity="0.4"/>
      </g>
      <!-- neck/base -->
      <rect x="9" y="18" width="6" height="1" fill="#00ff66"/>
      <rect x="8" y="19" width="8" height="1" fill="#00ff66" fill-opacity="0.5"/>
      <!-- side bolts -->
      <rect x="3" y="9" width="1" height="3" fill="#9cff00"/>
      <rect x="20" y="9" width="1" height="3" fill="#9cff00"/>
    `, '0 0 24 24');

    // CSS-driven behaviors attached via classes already in app.css
    wrap.classList.add('rh-idle');
    return wrap;
  };

  // Bump mouth intensity when a token arrives (caller passes sprite element).
  window.robotFeed = function (el) {
    if (!el) return;
    el.classList.remove('rh-active');
    void el.offsetWidth;
    el.classList.add('rh-active');
    el.classList.remove('rh-idle');
    setTimeout(() => {
      el.classList.remove('rh-active');
      el.classList.add('rh-idle');
    }, 180);
  };

  /* ---------- CPU CHIP (training / GPU active) ---------- */
  // Heat-glow indicator + activity bars flickering at training speed.
  // Pass `progress` (0..1) to fill the central heat meter.
  window.spriteCpuChip = function (size = 96) {
    const wrap = document.createElement('div');
    wrap.className = 'chip-sprite';
    wrap.style.width = size + 'px';
    wrap.style.height = size + 'px';
    wrap.setAttribute('aria-hidden', 'true');
    wrap.innerHTML = SVG(`
      <!-- chip body -->
      <rect x="4" y="4" width="16" height="16" fill="#0c1810" stroke="#00ff66" stroke-width="1"/>
      <!-- pins top -->
      <rect x="6" y="1" width="1" height="3" fill="#00ff66"/>
      <rect x="9" y="1" width="1" height="3" fill="#00ff66"/>
      <rect x="12" y="1" width="1" height="3" fill="#00ff66"/>
      <rect x="15" y="1" width="1" height="3" fill="#00ff66"/>
      <!-- pins bottom -->
      <rect x="6" y="20" width="1" height="3" fill="#00ff66"/>
      <rect x="9" y="20" width="1" height="3" fill="#00ff66"/>
      <rect x="12" y="20" width="1" height="3" fill="#00ff66"/>
      <rect x="15" y="20" width="1" height="3" fill="#00ff66"/>
      <!-- pins left -->
      <rect x="1" y="6" width="3" height="1" fill="#00ff66"/>
      <rect x="1" y="9" width="3" height="1" fill="#00ff66"/>
      <rect x="1" y="12" width="3" height="1" fill="#00ff66"/>
      <rect x="1" y="15" width="3" height="1" fill="#00ff66"/>
      <!-- pins right -->
      <rect x="20" y="6" width="3" height="1" fill="#00ff66"/>
      <rect x="20" y="9" width="3" height="1" fill="#00ff66"/>
      <rect x="20" y="12" width="3" height="1" fill="#00ff66"/>
      <rect x="20" y="15" width="3" height="1" fill="#00ff66"/>
      <!-- core die -->
      <rect class="chip-die" x="7" y="7" width="10" height="10" fill="#00ff66" fill-opacity="0.15" stroke="#9cff00" stroke-width="1"/>
      <!-- activity bars -->
      <g class="chip-bars">
        <rect class="chip-bar cb1" x="8"  y="11" width="1" height="2" fill="#9cff00"/>
        <rect class="chip-bar cb2" x="10" y="10" width="1" height="4" fill="#9cff00"/>
        <rect class="chip-bar cb3" x="12" y="9"  width="1" height="6" fill="#9cff00"/>
        <rect class="chip-bar cb4" x="14" y="11" width="1" height="2" fill="#9cff00"/>
        <rect class="chip-bar cb5" x="16" y="10" width="1" height="4" fill="#9cff00"/>
      </g>
      <!-- heat indicator top-right -->
      <rect class="chip-heat" x="16" y="4" width="2" height="2" fill="#ff3a3a"/>
    `, '0 0 24 24');
    return wrap;
  };

  /* ---------- DATA STREAM (data prep / parser running) ---------- */
  // Bytes flow left-to-right through a pipe with parser arrows.
  window.spriteDataStream = function (size = 120) {
    const wrap = document.createElement('div');
    wrap.className = 'stream-sprite';
    wrap.style.width = size + 'px';
    wrap.style.height = (size * 0.5) + 'px';
    wrap.setAttribute('aria-hidden', 'true');
    wrap.innerHTML = SVG(`
      <!-- source box -->
      <rect class="stream-src" x="1" y="4" width="4" height="4" fill="#0c1810" stroke="#00ff66" stroke-width="1"/>
      <rect class="stream-dot sd1" x="2" y="5" width="1" height="1" fill="#00ff66"/>
      <rect class="stream-dot sd2" x="2" y="6" width="1" height="1" fill="#9cff00"/>
      <!-- pipe -->
      <rect x="5"  y="5" width="14" height="2" fill="#00ff66" fill-opacity="0.15"/>
      <rect class="stream-pkt sp1" x="6"  y="5" width="2" height="2" fill="#00ff66"/>
      <rect class="stream-pkt sp2" x="10" y="5" width="2" height="2" fill="#9cff00"/>
      <rect class="stream-pkt sp3" x="14" y="5" width="2" height="2" fill="#00ff66"/>
      <!-- parser arrow -->
      <rect x="19" y="4" width="1" height="4" fill="#00ff66"/>
      <rect x="20" y="5" width="1" height="2" fill="#9cff00"/>
      <!-- output doc -->
      <rect class="stream-out" x="21" y="3" width="2" height="6" fill="#0c1810" stroke="#00ff66" stroke-width="1"/>
      <rect class="stream-out-line ol1" x="22" y="4" width="1" height="1" fill="#9cff00"/>
      <rect class="stream-out-line ol2" x="22" y="6" width="1" height="1" fill="#9cff00"/>
    `, '0 0 24 12');
    return wrap;
  };

  /* ---------- CORPUS NODE GRAPH (RAG embedding) ---------- */
  // Document nodes on the left, embedding vectors on the right,
  // packet trails flowing inward as chunks get embedded.
  window.spriteCorpusNode = function (size = 120) {
    const wrap = document.createElement('div');
    wrap.className = 'corpus-sprite';
    wrap.style.width = size + 'px';
    wrap.style.height = (size * 0.6) + 'px';
    wrap.setAttribute('aria-hidden', 'true');
    wrap.innerHTML = SVG(`
      <!-- doc nodes (left) -->
      <rect class="corpus-doc cd1" x="1"  y="2"  width="3" height="3" fill="#0c1810" stroke="#00ff66" stroke-width="1"/>
      <rect class="corpus-doc cd2" x="1"  y="7"  width="3" height="3" fill="#0c1810" stroke="#00ff66" stroke-width="1"/>
      <rect class="corpus-doc cd3" x="1"  y="12" width="3" height="3" fill="#0c1810" stroke="#00ff66" stroke-width="1"/>
      <!-- doc text marks -->
      <rect x="2" y="3" width="1" height="1" fill="#9cff00"/>
      <rect x="2" y="8" width="1" height="1" fill="#9cff00"/>
      <rect x="2" y="13" width="1" height="1" fill="#9cff00"/>
      <!-- edges -->
      <line class="corpus-edge" x1="4" y1="3" x2="11" y2="7"  stroke="#00ff66" stroke-width="1"/>
      <line class="corpus-edge" x1="4" y1="8" x2="11" y2="9"  stroke="#00ff66" stroke-width="1"/>
      <line class="corpus-edge" x1="4" y1="13" x2="11" y2="11" stroke="#00ff66" stroke-width="1"/>
      <!-- moving packets on edges -->
      <circle class="corpus-pkt cp1" cx="6"  cy="4.5" r="1" fill="#9cff00"/>
      <circle class="corpus-pkt cp2" cx="6"  cy="8.5" r="1" fill="#9cff00"/>
      <circle class="corpus-pkt cp3" cx="6"  cy="12.5" r="1" fill="#9cff00"/>
      <!-- vector nodes (right) -->
      <rect class="corpus-vec cv1" x="11" y="6"  width="3" height="3" fill="#00ff66" fill-opacity="0.2" stroke="#9cff00" stroke-width="1"/>
      <rect class="corpus-vec cv2" x="11" y="10" width="3" height="3" fill="#00ff66" fill-opacity="0.2" stroke="#9cff00" stroke-width="1"/>
      <!-- vector bits -->
      <rect x="12" y="7"  width="1" height="1" fill="#9cff00"/>
      <rect x="12" y="11" width="1" height="1" fill="#9cff00"/>
      <!-- cosine connection (dashed) -->
      <line class="corpus-cosine" x1="14" y1="7" x2="14" y2="11" stroke="#ff2bd6" stroke-width="1" stroke-dasharray="2 1"/>
      <!-- index dot -->
      <rect class="corpus-index" x="18" y="8" width="3" height="3" fill="#0c1810" stroke="#9cff00" stroke-width="1"/>
      <rect class="corpus-index-dot" x="19" y="9" width="1" height="1" fill="#9cff00"/>
    `, '0 0 24 16');
    return wrap;
  };

  /* ---------- BENCHMARK BARS (live scores) ---------- */
  // Bar chart that fills in real-time. Call `benchUpdate(el, score)`
  // with a 0..1 value; bars grow upward and the score text refreshes.
  window.spriteBenchBars = function (size = 120) {
    const wrap = document.createElement('div');
    wrap.className = 'bench-sprite';
    wrap.style.width = size + 'px';
    wrap.style.height = (size * 0.6) + 'px';
    wrap.setAttribute('aria-hidden', 'true');
    wrap.innerHTML = SVG(`
      <!-- baseline -->
      <rect x="1" y="13" width="22" height="1" fill="#00ff66" fill-opacity="0.4"/>
      <!-- bars (heights set via CSS var --bench-h) -->
      <rect class="bench-bar bb1" x="3"  y="13" width="3" height="1" fill="#00ff66"/>
      <rect class="bench-bar bb2" x="8"  y="13" width="3" height="1" fill="#9cff00"/>
      <rect class="bench-bar bb3" x="13" y="13" width="3" height="1" fill="#00ff66"/>
      <rect class="bench-bar bb4" x="18" y="13" width="3" height="1" fill="#9cff00"/>
      <!-- moving score line -->
      <line class="bench-line" x1="2" y1="13" x2="22" y2="13" stroke="#ff2bd6" stroke-width="1" stroke-dasharray="2 2"/>
      <!-- top label -->
      <rect class="bench-cap" x="9" y="1" width="6" height="1" fill="#00ff66"/>
      <rect x="10" y="2" width="4" height="1" fill="#9cff00"/>
    `, '0 0 24 16');
    wrap.style.setProperty('--bench-h', '12');
    return wrap;
  };

  window.benchUpdate = function (el, score) {
    if (!el) return;
    const h = Math.max(0, Math.min(12, Math.round(score * 12)));
    el.style.setProperty('--bench-h', h + 'px');
  };

  /* ============================================================
     AUTO-WIRING: page-action classes trigger sprites
     ============================================================ */

  function auto() {
    // ---- file-drop on any data prep zone ----
    document.querySelectorAll('[data-drop], .dropzone').forEach((zone) => {
      // pre-mount bin sprite if there's a .bin-host child
      const host = zone.querySelector('.bin-host');
      if (host && !host.querySelector('.bin-sprite')) {
        host.appendChild(spriteBin(60));
      }
      zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dropping'); });
      zone.addEventListener('dragleave', () => zone.classList.remove('dropping'));
      zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dropping');
        const files = Array.from(e.dataTransfer.files || []);
        if (!files.length) return;
        const bin = zone.querySelector('.bin-sprite');
        files.forEach((f, i) => setTimeout(() => eatFile(bin, f.name), i * 600));
      });
      // also handle file-picker
      const fileInput = zone.querySelector('input[type="file"]');
      if (fileInput && !fileInput._binWired) {
        fileInput._binWired = true;
        fileInput.addEventListener('change', () => {
          const bin = zone.querySelector('.bin-sprite');
          Array.from(fileInput.files || []).forEach((f, i) =>
            setTimeout(() => eatFile(bin, f.name), i * 600)
          );
        });
      }
    });

    // ---- training forge auto-mount ----
    document.querySelectorAll('[data-forge]').forEach((el) => {
      el.appendChild(spriteForge());
    });

    // ---- inference loader auto-mount ----
    document.querySelectorAll('[data-wire-transfer]').forEach((el) => {
      el.appendChild(spriteWireTransfer(el.dataset.wireTransfer || 'TRANSMITTING'));
    });

    // ---- matrix rain on dashboard hero ----
    document.querySelectorAll('[data-matrix]').forEach((el) => {
      spriteMatrixRain(el);
    });

    // ---- radar on RAG search ----
    document.querySelectorAll('[data-radar]').forEach((el) => {
      const r = spriteRadar();
      el.appendChild(r);
      el._radar = r;
    });

    // ---- boot sequence on first paint ----
    if (!sessionStorage.getItem('fts.booted')) {
      sessionStorage.setItem('fts.booted', '1');
      setTimeout(() => spriteBoot(), 200);
    }

    // ---- scan beam overlay on .loading cards ----
    document.querySelectorAll('.card.loading, .stat.loading').forEach((el) => {
      spriteScanBeam(el);
    });

    // ---- assembly line on data-prep stages ----
    document.querySelectorAll('[data-assembly]').forEach((el) => {
      const stages = (el.dataset.assembly || 'CHUNK,EMBED,INDEX,STORE').split(',');
      el.appendChild(spriteAssembly(stages));
    });

    // ---- floppy save on save buttons ----
    document.querySelectorAll('[data-save], .btn-save').forEach((btn) => {
      btn.addEventListener('click', () => floppySave(btn));
    });

    // ---- P1: Page-context sprite mounts (auto-inject by route) ----
    mountPageSprites();
  }

  /* Page-context sprite mounts: picks the right sprite for the current
     page and injects it into the best matching container. Driven by
     data-sprite hooks when present, falls back to route inference. */
  function mountPageSprites() {
    const path = location.pathname;
    // Universal mount target: the first card on the page (always present).
    // Pages can override via [data-sprite="robot-head"] etc.
    const firstCard = document.querySelector('.card');

    // Robot head → inference pages (live model thinking)
    if (/^\/inference/.test(path) || /\/chat\b/.test(path)) {
      const target = document.querySelector('[data-sprite="robot-head"]') || firstCard;
      if (target && !target.querySelector('.robot-sprite')) {
        const cap = document.createElement('div');
        cap.className = 'sprite-mount-caption';
        cap.textContent = 'NO MODEL LOADED'; // start honest; updated after fetch
        const sub = document.createElement('div');
        sub.className = 'sprite-mount-sub';
        sub.style.cssText = 'font-size:11px;color:var(--text-dim);margin-top:2px;';
        sub.textContent = 'checking status…';
        const host = document.createElement('div');
        host.className = 'sprite-mount';
        host.style.flexDirection = 'row';
        host.style.alignItems = 'center';
        host.style.gap = '14px';
        host.style.padding = '10px 14px';
        host.style.marginBottom = '10px';
        const textWrap = document.createElement('div');
        textWrap.style.display = 'flex';
        textWrap.style.flexDirection = 'column';
        textWrap.appendChild(cap);
        textWrap.appendChild(sub);
        host.appendChild(spriteRobotHead(64));
        host.appendChild(textWrap);
        target.prepend(host);

        document.addEventListener('fts:token', () => {
          robotFeed(host.querySelector('.robot-sprite'));
        });

        // Poll real model state — don't lie about MODEL ONLINE
        async function refreshModelState() {
          try {
            const [d, ie] = await Promise.all([
              fetch('/api/providers').then(r => r.json()).catch(() => null),
              fetch('/api/inference/status').then(r => r.json()).catch(() => null),
            ]);
            const active = d && d.active && d.active.loaded ? d.active : null;
            const infLoaded = ie && ie.loaded ? ie : null;
            if (active || infLoaded) {
              cap.textContent = '● MODEL ONLINE';
              cap.style.color = 'var(--accent)';
              const name = active ? (active.model_id || active.name || active.id) : (infLoaded.model || infLoaded.id);
              sub.textContent = String(name || '').split(/[\\/]/).pop();
            } else {
              cap.textContent = '○ NO MODEL LOADED';
              cap.style.color = 'var(--text-dim)';
              sub.innerHTML = '<a href="/inference" data-link style="color:var(--accent);">load one in Inference →</a>';
            }
          } catch (e) { /* ignore */ }
        }
        refreshModelState();
        setInterval(refreshModelState, 5000);
      }
    }

    // CPU chip → training pages
    if (/\/training/.test(path)) {
      const target = document.querySelector('[data-sprite="cpu-chip"]') || firstCard;
      if (target && !target.querySelector('.chip-sprite')) {
        const cap = document.createElement('div');
        cap.className = 'sprite-mount-caption';
        cap.textContent = 'GPU ACCELERATING';
        const host = document.createElement('div');
        host.className = 'sprite-mount';
        host.style.flexDirection = 'row';
        host.style.alignItems = 'center';
        host.style.gap = '14px';
        host.style.padding = '10px 14px';
        host.style.marginBottom = '10px';
        host.appendChild(spriteCpuChip(64));
        host.appendChild(cap);
        target.prepend(host);
      }
    }

    // Data stream → data prep pages
    if (/\/data-prep/.test(path) || /\/data\b/.test(path)) {
      const target = document.querySelector('[data-sprite="data-stream"]') || firstCard;
      if (target && !target.querySelector('.stream-sprite')) {
        const host = document.createElement('div');
        host.className = 'sprite-mount';
        host.style.flexDirection = 'row';
        host.style.justifyContent = 'flex-start';
        host.style.padding = '6px 14px';
        host.style.marginBottom = '10px';
        host.appendChild(spriteDataStream(120));
        target.prepend(host);
      }
    }

    // Corpus graph → RAG pages
    if (/\/rag/.test(path)) {
      const target = document.querySelector('[data-sprite="corpus-node"]') || firstCard;
      if (target && !target.querySelector('.corpus-sprite')) {
        const cap = document.createElement('div');
        cap.className = 'sprite-mount-caption';
        cap.textContent = 'RAG RETRIEVAL INDEX';
        const host = document.createElement('div');
        host.className = 'sprite-mount';
        host.style.flexDirection = 'row';
        host.style.alignItems = 'center';
        host.style.gap = '14px';
        host.style.padding = '10px 14px';
        host.style.marginBottom = '10px';
        host.appendChild(spriteCorpusNode(120));
        host.appendChild(cap);
        target.prepend(host);
      }
    }

    // Bench bars → benchmarks / testing pages
    if (/\/benchmarks/.test(path) || /\/testing/.test(path)) {
      const target = document.querySelector('[data-sprite="bench-bars"]') || firstCard;
      if (target && !target.querySelector('.bench-sprite')) {
        const cap = document.createElement('div');
        cap.className = 'sprite-mount-caption';
        cap.textContent = 'COMPUTING SCORES';
        const host = document.createElement('div');
        host.className = 'sprite-mount';
        host.style.flexDirection = 'row';
        host.style.alignItems = 'center';
        host.style.gap = '14px';
        host.style.padding = '10px 14px';
        host.style.marginBottom = '10px';
        host.appendChild(spriteBenchBars(120));
        host.appendChild(cap);
        target.prepend(host);

        document.addEventListener('fts:bench-progress', (ev) => {
          if (ev.detail && typeof ev.detail.score === 'number') {
            benchUpdate(host.querySelector('.bench-sprite'), ev.detail.score);
            cap.textContent = `SCORE ${(ev.detail.score * 100).toFixed(0)}%`;
          }
        });
      }
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', auto);
  } else {
    auto();
  }

  // SPA re-run
  document.addEventListener('fts:page-loaded', auto);
  // also re-run after SPA innerHTML replace
  const _origDispatch = window.dispatchEvent;
  // expose a helper for spa.js to call
  window.spritesInit = auto;
})();
