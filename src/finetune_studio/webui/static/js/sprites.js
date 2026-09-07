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
