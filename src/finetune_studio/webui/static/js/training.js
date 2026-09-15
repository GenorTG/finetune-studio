/* ============================================================
   Training page — preset chooser, custom overrides, launch
   ============================================================ */
(() => {
  const $ = (id) => document.getElementById(id);
  const status = (msg) => { const el = $("training-status"); if (el) el.textContent = msg; };

  let presets = [];
  let projects = [];
  let models = [];
  let datasets = [];
  let selectedPreset = null;
  let pollTimer = null;

  async function loadPresets() {
    const r = await fetch("/api/training/presets");
    presets = await r.json();
    const list = $("preset-list");
    list.innerHTML = "";
    for (const p of presets) {
      const card = document.createElement("div");
      card.className = "card preset-card";
      card.dataset.id = p.id;
      card.innerHTML = `
        <div class="flex items-center gap-2 mb-1">
          <input type="radio" name="preset" value="${p.id}" />
          <span class="font-bold text-sm">${p.name}</span>
          <span class="badge">${p.min_vram_gb}GB VRAM</span>
        </div>
        <div class="text-xs dim">${p.description}</div>
        <div class="text-xs mt-2 grid grid-4 gap-2">
          <div><span class="dim">epochs</span><br><span class="mono">${p.num_epochs}</span></div>
          <div><span class="dim">rank</span><br><span class="mono">${p.lora_rank}</span></div>
          <div><span class="dim">batch</span><br><span class="mono">${p.batch_size}</span></div>
          <div><span class="dim">seq</span><br><span class="mono">${p.max_seq_length}</span></div>
        </div>
      `;
      card.addEventListener("click", () => selectPreset(p.id));
      list.appendChild(card);
    }
  }

  function selectPreset(id) {
    selectedPreset = presets.find(p => p.id === id);
    // Update radio
    const radios = document.querySelectorAll('input[name="preset"]');
    radios.forEach(r => { r.checked = (r.value === id); });
    // Highlight card
    document.querySelectorAll(".preset-card").forEach(c => {
      c.classList.toggle("selected", c.dataset.id === id);
    });
    // Apply to overrides
    if (selectedPreset) {
      $("ov-lora_rank").value = selectedPreset.lora_rank;
      $("ov-lora_alpha").value = selectedPreset.lora_alpha;
      $("ov-learning_rate").value = selectedPreset.learning_rate;
      $("ov-weight_decay").value = selectedPreset.weight_decay;
      $("ov-warmup_steps").value = selectedPreset.warmup_steps;
      $("ov-num_epochs").value = selectedPreset.num_epochs;
      $("ov-batch_size").value = selectedPreset.batch_size;
      $("ov-gradient_accumulation_steps").value = selectedPreset.gradient_accumulation_steps;
      $("ov-max_seq_length").value = selectedPreset.max_seq_length;
      $("ov-save_steps").value = selectedPreset.save_steps;
      $("ov-logging_steps").value = selectedPreset.logging_steps;
      $("ov-bf16").value = String(selectedPreset.bf16);
      $("ov-unsloth").value = String(selectedPreset.unsloth);
      $("ov-merge_on_save").checked = !!selectedPreset.merge_on_save;
    }
    updateSummary();
    updateLaunch();
  }

  async function loadProjects() {
    const r = await fetch("/api/projects");
    projects = await r.json();
    const sel = $("train-project");
    sel.innerHTML = "";
    for (const p of projects) {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.name || p.id;
      sel.appendChild(opt);
    }
    if (projects.length > 0) {
      loadDatasets(projects[0].id);
    }
  }

  async function loadModels() {
    const r = await fetch("/api/models");
    models = await r.json();
    const sel = $("train-model");
    sel.innerHTML = "";
    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m.path;
      opt.textContent = `${m.name} (${m.format}, ${m.size_gb.toFixed(1)}GB)`;
      sel.appendChild(opt);
    }
    updateModelInfo();
  }

  async function loadDatasets(pid) {
    const r = await fetch(`/api/projects/${pid}/datasets`);
    const d = await r.json();
    datasets = d.datasets || [];
    const sel = $("train-dataset");
    sel.innerHTML = "";
    if (datasets.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "No datasets in project — upload or generate one";
      sel.appendChild(opt);
    }
    for (const ds of datasets) {
      const opt = document.createElement("option");
      opt.value = ds.data_path;
      opt.textContent = ds.name;
      sel.appendChild(opt);
    }
    updateDatasetInfo();
  }

  function updateModelInfo() {
    const sel = $("train-model");
    const m = models.find(x => x.path === sel.value);
    $("train-model-info").textContent = m ? `${m.format} · ${m.size_gb.toFixed(1)}GB · ${m.arch || ""}` : "";
  }

  function updateDatasetInfo() {
    const sel = $("train-dataset");
    const ds = datasets.find(x => x.data_path === sel.value);
    $("train-dataset-info").textContent = ds ? `${ds.name} · ${ds.example_count || "?"} examples` : "";
  }

  function updateSummary() {
    const summary = `
      <span class="dim">Selected:</span> <b>${selectedPreset ? selectedPreset.name : "Custom"}</b>
      <span class="dim"> · epochs=</span>${$("ov-num_epochs").value}
      <span class="dim"> rank=</span>${$("ov-lora_rank").value}
      <span class="dim"> batch=</span>${$("ov-batch_size").value}
      <span class="dim"> seq=</span>${$("ov-max_seq_length").value}
    `;
    $("param-summary").innerHTML = summary;
  }

  function updateLaunch() {
    const model = $("train-model").value;
    const dataset = $("train-dataset").value;
    const hasAll = model && dataset && ($("train-project").value);
    $("btn-start-training").disabled = !hasAll;
    $("launch-summary").innerHTML = hasAll ?
      `Ready: <b>${selectedPreset ? selectedPreset.name : "Custom config"}</b> on <b>${model.split("/").pop()}</b>` :
      "Select a preset, project, model, and dataset first.";
  }

  async function startTraining() {
    const body = {
      project_id: $("train-project").value,
      model_path: $("train-model").value,
      data_path: $("train-dataset").value,
      output_dir: $("train-output").value || "output",
      preset_id: selectedPreset ? selectedPreset.id : null,
      overrides: {
        lora_rank: parseInt($("ov-lora_rank").value, 10),
        lora_alpha: parseInt($("ov-lora_alpha").value, 10),
        learning_rate: parseFloat($("ov-learning_rate").value),
        weight_decay: parseFloat($("ov-weight_decay").value),
        warmup_steps: parseInt($("ov-warmup_steps").value, 10),
        num_epochs: parseInt($("ov-num_epochs").value, 10),
        batch_size: parseInt($("ov-batch_size").value, 10),
        gradient_accumulation_steps: parseInt($("ov-gradient_accumulation_steps").value, 10),
        max_seq_length: parseInt($("ov-max_seq_length").value, 10),
        save_steps: parseInt($("ov-save_steps").value, 10),
        logging_steps: parseInt($("ov-logging_steps").value, 10),
        bf16: $("ov-bf16").value === "true",
        unsloth: $("ov-unsloth").value === "true",
        merge_on_save: $("ov-merge_on_save").checked,
      }
    };

    // If no preset selected, send all params directly (no preset_id)
    if (!selectedPreset) {
      delete body.preset_id;
    }

    try {
      const r = await fetch("/api/training/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const d = await r.json();
      if (d.error) {
        status(`Error: ${d.error}`);
        return;
      }
      status("Started! Tracking progress...");
      $("btn-start-training").disabled = true;
      $("btn-stop-training").disabled = false;
      $("progress-card").style.display = "block";
      startPolling();
    } catch (e) {
      status(`Failed: ${e.message}`);
    }
  }

  async function stopTraining() {
    try {
      await fetch("/api/training/stop", { method: "POST" });
      status("Stopping...");
    } catch (e) {
      status(`Stop failed: ${e.message}`);
    }
  }

  function startPolling() {
    if (pollTimer) {
      if (typeof pollTimer === "function") pollTimer();
      else clearInterval(pollTimer);
      pollTimer = null;
    }
    const sub = window.fts && window.fts.subscribe;
    if (sub) {
      pollTimer = sub("/api/training/progress", applyLiveStatus, {
        pollUrl: "/api/training/status",
        fallbackMs: 5000,
      });
    } else {
      pollTimer = setInterval(pollStatus, 5000);
      pollStatus();
    }
  }

  function applyLiveStatus(s) {
    if (!s || typeof s !== "object") return;
    const step = s.step ?? s.current_step ?? 0;
    const total = s.total_steps || 0;
    $("m-step").textContent = total ? `${step}/${total}` : "—";
    $("m-loss").textContent = s.loss != null ? Number(s.loss).toFixed(4) : "—";
    $("m-lr").textContent = s.learning_rate ? Number(s.learning_rate).toExponential(2) : "—";
    $("m-elapsed").textContent = s.elapsed != null ? `${Number(s.elapsed).toFixed(0)}s` : "—";

    const msg = s.message || "";
    if (total > 0) {
      const pct = Math.min(100, (step / total) * 100).toFixed(1);
      $("progress-fill").style.width = `${pct}%`;
      $("progress-sub").textContent = `Step ${step}/${total}` +
        (s.loss != null ? ` · loss ${Number(s.loss).toFixed(4)}` : "") +
        (msg ? ` · ${msg}` : "") +
        ` · ${pct}%`;
    } else if (msg) {
      $("progress-sub").textContent = msg;
    }

    const logEl = $("training-log");
    if (logEl) {
      const lines = Array.isArray(s.log_lines) ? s.log_lines : [];
      logEl.style.display = "block";
      logEl.textContent = lines.length
        ? lines.slice(-40).join("\n")
        : (msg || "(waiting for step logs…)");
    }

    if (s.status === "done" || s.status === "error") {
      if (typeof pollTimer === "function") {
        pollTimer();
        pollTimer = null;
      } else if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
      status(s.status === "done" ? "Training complete!" : `Error: ${s.error || "unknown"}`);
      $("btn-start-training").disabled = false;
      $("btn-stop-training").disabled = true;
      loadPastRuns();
    }
  }

  async function pollStatus() {
    try {
      const r = await fetch("/api/training/status");
      applyLiveStatus(await r.json());
    } catch (e) { /* polling failed, ignore */ }
  }

  async function loadPastRuns() {
    const pid = $("train-project").value;
    if (!pid) return;
    try {
      const r = await fetch(`/api/training/runs/${pid}`);
      const runs = await r.json();
      const el = $("past-runs");
      if (runs.length === 0) {
        el.innerHTML = `<div class="text-xs dim">No past runs.</div>`;
        return;
      }
      el.innerHTML = runs.map(r => `
        <div class="flex items-center gap-3 text-xs mb-2">
          <span class="status-badge status-${r.status}">${r.status}</span>
          <span class="mono">${(r.name || r.id).slice(0, 50)}</span>
          <span class="dim">${r.created_at ? new Date(r.created_at * 1000).toLocaleString() : ""}</span>
          ${r.final_loss ? `<span class="mono">loss=${r.final_loss.toFixed(4)}</span>` : ""}
        </div>
      `).join("");
    } catch (e) {
      $("past-runs").innerHTML = `<div class="text-xs dim">Failed to load runs.</div>`;
    }
  }

  function wireEvents() {
    $("train-project").addEventListener("change", (e) => loadDatasets(e.target.value));
    $("train-model").addEventListener("change", updateModelInfo);
    $("train-dataset").addEventListener("change", updateDatasetInfo);
    $("btn-start-training").addEventListener("click", startTraining);
    $("btn-stop-training").addEventListener("click", stopTraining);
    // Update summary on any override change
    document.querySelectorAll("[id^='ov-']").forEach(el => {
      el.addEventListener("change", () => { updateSummary(); updateLaunch(); });
    });
  }

  // Init
  (async function init() {
    await Promise.all([loadPresets(), loadProjects(), loadModels()]);
    wireEvents();
    // Auto-select first preset
    if (presets.length > 0) selectPreset("standard");
    updateSummary();
    updateLaunch();
    loadPastRuns();
    // Start polling if training is already running
    fetch("/api/training/status").then(r => r.json()).then(s => {
      if (s.status === "training" || s.status === "loading" || s.status === "saving") {
        $("progress-card").style.display = "block";
        startPolling();
      }
    });
  })();
})();
