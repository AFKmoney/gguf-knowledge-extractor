/* GGUF Knowledge Extractor — Frontend Logic (v2 redesign) */

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

let selectedFile = null;
let selectedFilePath = null;
let selectedPacks = new Set();
let pollTimer = null;
let downloadPollTimer = null;

// ---------------------------------------------------------------- //
// Navigation
// ---------------------------------------------------------------- //
const VIEW_META = {
  dashboard: { title: 'Dashboard', subtitle: 'Overview of your GGUF analysis platform' },
  extract:   { title: 'Extract Knowledge', subtitle: 'Full pipeline: metadata + weights + probes + attribution' },
  trace:     { title: 'Causal Tracing', subtitle: 'Logit-lens per layer — find where each fact lives' },
  mediate:   { title: 'Causal Mediation', subtitle: 'Activation patching — the gold standard from ROME' },
  edit:      { title: 'ROME Editing', subtitle: 'Rank-1 fact editing without retraining' },
  compare:   { title: 'Model Comparison', subtitle: 'Cross-model fingerprint lineage analysis' },
  surgery:   { title: 'GGUF Surgery', subtitle: 'Direct model modification without retraining' },
  merge:     { title: 'Model Merging', subtitle: 'SLERP, TIES, DARE, Linear — combine two models' },
  transplant:{ title: 'Knowledge Transplant', subtitle: 'Extract knowledge from one model, inject into another' },
  imatrix:   { title: 'Importance Matrix', subtitle: 'Compute which tensors matter more for smarter quantization' },
  quantize:  { title: 'Smart Quantizer', subtitle: 'Compress GGUF models with optional imatrix guidance' },
  diff:      { title: 'GGUF Diff', subtitle: 'Compare two GGUFs at byte, metadata, and tensor level' },
  models:    { title: 'Hugging Face Hub', subtitle: 'Browse and download GGUF models' },
  job:       { title: 'Job Details', subtitle: 'Live progress and results' },
  jobs:      { title: 'Jobs', subtitle: 'All extraction, trace, edit, and surgery operations' },
  packs:     { title: 'Probe Packs', subtitle: 'YAML-defined test suites' },
  backends:  { title: 'Backends', subtitle: 'Inference backend availability' },
};

function navigateTo(view) {
  $$('.view').forEach(v => v.classList.remove('active'));
  const el = $(`#view-${view}`);
  if (el) el.classList.add('active');
  $$('.nav-item').forEach(a => a.classList.remove('active'));
  const navEl = $(`.nav-item[data-view="${view}"]`);
  if (navEl) navEl.classList.add('active');
  const meta = VIEW_META[view] || { title: view, subtitle: '' };
  $('#topbar-title').textContent = meta.title;
  $('#topbar-subtitle').textContent = meta.subtitle;

  // Load data for specific views
  if (view === 'dashboard') loadDashboard();
  if (view === 'models') { loadLocalModels(); loadDownloads(); }
  if (view === 'jobs') loadJobs();
  if (view === 'packs') loadPacksDetail();
  if (view === 'backends') loadBackendsDetail();
  if (view === 'extract') refreshLocalModelDropdown();
  if (view === 'surgery') refreshSurgeryLocalModels();
}

$$('.nav-item').forEach(item => {
  item.onclick = (e) => {
    e.preventDefault();
    const view = item.dataset.view;
    if (view) navigateTo(view);
  };
});

// ---------------------------------------------------------------- //
// Toast Notifications
// ---------------------------------------------------------------- //
function toast(message, type = 'info', duration = 4000) {
  const container = $('#toast-container');
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  const icons = {
    success: '<svg class="toast-icon" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M9 12l2 2 4-4M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>',
    error: '<svg class="toast-icon" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M15 9l-6 6M9 9l6 6"/></svg>',
    info: '<svg class="toast-icon" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/></svg>',
  };
  t.innerHTML = `${icons[type] || icons.info}<span>${message}</span>`;
  container.appendChild(t);
  setTimeout(() => {
    t.style.opacity = '0';
    t.style.transform = 'translateX(100%)';
    setTimeout(() => t.remove(), 200);
  }, duration);
}

// ---------------------------------------------------------------- //
// Dashboard
// ---------------------------------------------------------------- //
async function loadDashboard() {
  // Load local models count
  try {
    const res = await fetch('/api/models/local');
    const data = await res.json();
    $('#stat-models').textContent = data.count || 0;
  } catch (e) {}

  // Load jobs count
  try {
    const res = await fetch('/api/jobs');
    const jobs = await res.json();
    $('#stat-jobs').textContent = jobs.length;
    const badge = $('#badge-jobs');
    if (badge) badge.textContent = jobs.length;

    // Recent jobs (last 5)
    const recent = jobs.slice(-5).reverse();
    const container = $('#dashboard-recent-jobs');
    if (!recent.length) {
      container.innerHTML = '<div class="empty-state"><div class="empty-state-icon"><svg fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" width="24" height="24"><path d="M12 8v4l3 3M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg></div>No jobs yet. Run an extraction to get started.</div>';
      return;
    }
    let html = '<table class="table"><thead><tr><th>Status</th><th>Model</th><th>Type</th><th>Started</th></tr></thead><tbody>';
    for (const j of recent) {
      const kind = j.kind || (j.options?.do_attribution ? 'extract' : 'extract');
      html += `<tr style="cursor:pointer;" onclick="navigateTo('job'); pollJob('${j.id}')">
        <td><span class="status-pill ${j.status}"><span class="dot"></span>${j.status}</span></td>
        <td class="mono">${j.gguf_filename || '—'}</td>
        <td><span class="badge badge-neutral">${kind}</span></td>
        <td class="text-tertiary text-sm">${j.started_at || '—'}</td>
      </tr>`;
    }
    html += '</tbody></table>';
    container.innerHTML = html;
  } catch (e) {}
}

// ---------------------------------------------------------------- //
// File Upload (Extract)
// ---------------------------------------------------------------- //
const dz = $('#dropzone');
const fileInput = $('#file-input');
if (dz) {
  dz.onclick = () => fileInput.click();
  dz.ondragover = (e) => { e.preventDefault(); dz.classList.add('drag-over'); };
  dz.ondragleave = () => dz.classList.remove('drag-over');
  dz.ondrop = (e) => { e.preventDefault(); dz.classList.remove('drag-over'); if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]); };
  fileInput.onchange = (e) => { if (e.target.files.length) handleFile(e.target.files[0]); };
}

function handleFile(file) {
  if (!file.name.toLowerCase().endsWith('.gguf')) { toast('File must be a .gguf file', 'error'); return; }
  selectedFile = file;
  selectedFilePath = null;
  const sel = $('#local-model-select'); if (sel) sel.value = '';
  $('#selected-file').textContent = `✓ ${file.name} (${formatBytes(file.size)})`;
  $('#btn-extract').disabled = false;
}

// Local model dropdown
async function refreshLocalModelDropdown() {
  try {
    const res = await fetch('/api/models/local');
    const data = await res.json();
    const select = $('#local-model-select');
    if (!select) return;
    select.innerHTML = '<option value="">— Select a local model —</option>';
    for (const m of data.models) {
      const opt = document.createElement('option');
      opt.value = m.path;
      opt.textContent = `${m.filename} (${m.size_human})`;
      select.appendChild(opt);
    }
  } catch (e) {}
}
$('#btn-refresh-local-models')?.addEventListener('click', refreshLocalModelDropdown);
$('#link-to-models')?.addEventListener('click', (e) => { e.preventDefault(); navigateTo('models'); });
$('#local-model-select')?.addEventListener('change', (e) => {
  if (e.target.value) {
    selectedFilePath = e.target.value;
    selectedFile = null;
    const filename = e.target.value.split('/').pop();
    $('#selected-file').textContent = `✓ Selected: ${filename}`;
    $('#btn-extract').disabled = false;
  }
});

// ---------------------------------------------------------------- //
// Probe Packs
// ---------------------------------------------------------------- //
async function loadPacks() {
  const res = await fetch('/api/packs');
  const packs = await res.json();
  const container = $('#packs-list');
  if (!container) return;
  container.innerHTML = '';
  packs.forEach(p => {
    const card = document.createElement('div');
    card.className = 'pack-card selected';
    card.dataset.name = p.name;
    card.innerHTML = `
      <div class="pack-name">${p.name}</div>
      <div class="pack-meta"><span class="badge badge-accent">${p.category}</span> ${p.domain || 'general'} · ${p.n_probes} probes</div>
      <div class="pack-desc">${p.description}</div>`;
    card.onclick = () => {
      if (card.classList.contains('selected')) { card.classList.remove('selected'); selectedPacks.add(p.name); }
      else { card.classList.add('selected'); selectedPacks.delete(p.name); }
    };
    container.appendChild(card);
  });
}

async function loadPacksDetail() {
  const res = await fetch('/api/packs');
  const packs = await res.json();
  const container = $('#packs-detail');
  container.innerHTML = '';
  packs.forEach(p => {
    const card = document.createElement('div');
    card.className = 'pack-card';
    card.style.cursor = 'default';
    card.innerHTML = `
      <div class="pack-name">${p.name}</div>
      <div class="pack-meta"><span class="badge badge-accent">${p.category}</span> ${p.domain || 'general'} · ${p.n_probes} probes</div>
      <div class="pack-desc">${p.description}</div>
      <div class="pack-meta" style="margin-top:8px;">📁 ${p.path}</div>`;
    container.appendChild(card);
  });
}

// ---------------------------------------------------------------- //
// Backends
// ---------------------------------------------------------------- //
async function checkBackends() {
  const url = $('#opt-server-url')?.value || 'http://127.0.0.1:8080';
  const container = $('#backends-status');
  if (container) container.innerHTML = '<div class="spinner"></div>';
  try {
    const res = await fetch(`/api/backends?server_url=${encodeURIComponent(url)}`);
    const data = await res.json();
    if (container) {
      container.innerHTML = '';
      for (const [name, ok] of Object.entries(data)) {
        const row = document.createElement('div');
        row.className = `backend-row ${ok ? 'ok' : ''}`;
        row.innerHTML = `<span class="dot"></span> <strong>${name}</strong> — ${ok ? 'available' : 'not available'}`;
        container.appendChild(row);
      }
      if (Object.values(data).every(v => !v)) {
        toast('No inference backend available — probes will be skipped', 'info');
      }
    }
  } catch (e) { toast('Failed to check backends', 'error'); }
}
$('#btn-check-backends')?.addEventListener('click', checkBackends);

async function loadBackendsDetail() {
  const container = $('#backends-detail');
  if (!container) return;
  container.innerHTML = '<div class="spinner"></div>';
  try {
    const res = await fetch('/api/backends');
    const data = await res.json();
    container.innerHTML = '';
    for (const [name, ok] of Object.entries(data)) {
      const row = document.createElement('div');
      row.className = `backend-row ${ok ? 'ok' : ''}`;
      row.innerHTML = `<span class="dot"></span> <strong>${name}</strong> — ${ok ? 'available' : 'not available'}`;
      container.appendChild(row);
    }
    const help = document.createElement('div');
    help.style.cssText = 'margin-top:16px;padding:16px;background:var(--bg-elevated);border-radius:8px;';
    help.innerHTML = `<div class="form-label" style="margin-bottom:8px;">How to enable backends</div>
      <div class="form-hint" style="line-height:1.6;">
        <strong>llama.cpp server:</strong> <code class="code">llama-server -m model.gguf --port 8080</code><br>
        <strong>llama-cpp-python:</strong> <code class="code">pip install llama-cpp-python</code>
      </div>`;
    container.appendChild(help);
  } catch (e) {}
}

// ---------------------------------------------------------------- //
// Extract
// ---------------------------------------------------------------- //
$('#btn-extract')?.addEventListener('click', async () => {
  if (!selectedFile && !selectedFilePath) return;
  const fd = new FormData();
  if (selectedFile) fd.append('file', selectedFile);
  else if (selectedFilePath) fd.append('local_path', selectedFilePath);
  fd.append('packs', selectedPacks.size === 0 ? 'all' : Array.from(selectedPacks).join(','));
  fd.append('do_metadata', $('#opt-metadata').checked);
  fd.append('do_weights', $('#opt-weights').checked);
  fd.append('do_probes', $('#opt-probes').checked);
  fd.append('do_attribution', $('#opt-attribution').checked);
  fd.append('attribution_top_k', $('#opt-attribution-top-k').value);
  fd.append('server_url', $('#opt-server-url').value);
  fd.append('prefer_backend', $('#opt-prefer').value);
  fd.append('n_ctx', $('#opt-nctx').value);
  fd.append('n_gpu_layers', $('#opt-ngpu').value);

  $('#btn-extract').disabled = true;
  $('#btn-extract').textContent = 'Starting...';
  try {
    const url = selectedFilePath ? '/api/extract-local' : '/api/extract';
    const res = await fetch(url, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'failed');
    toast('Extraction started', 'success');
    navigateTo('job');
    pollJob(data.job_id);
  } catch (e) {
    toast('Failed: ' + e.message, 'error');
    $('#btn-extract').disabled = false;
    $('#btn-extract').textContent = 'Extract Knowledge →';
  }
});

// ---------------------------------------------------------------- //
// Job Polling
// ---------------------------------------------------------------- //
function pollJob(jobId) {
  if (pollTimer) clearInterval(pollTimer);
  const fetchJob = async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      const job = await res.json();
      renderJob(job);
      if (job.status === 'completed' || job.status === 'failed') {
        clearInterval(pollTimer);
        pollTimer = null;
        if (job.status === 'completed') toast('Job completed', 'success');
        else toast('Job failed', 'error');
      }
    } catch (e) {}
  };
  fetchJob();
  pollTimer = setInterval(fetchJob, 1500);
}

function formatBytes(n) {
  if (!n) return '?';
  if (n >= 1e9) return (n / 1e9).toFixed(2) + ' GB';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + ' MB';
  if (n >= 1e3) return (n / 1e3).toFixed(2) + ' KB';
  return n + ' B';
}

function formatNumber(n) {
  if (!n) return '0';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

function renderJob(job) {
  const c = $('#job-content');
  if (!c) return;
  const status = job.status;
  const progress = job.progress || {};

  let html = `<div class="card">
    <div class="card-header">
      <div>
        <div class="card-title">Job ${job.id}</div>
        <div class="card-subtitle">${job.gguf_filename || ''} · ${job.kind || 'extract'}</div>
      </div>
      <span class="status-pill ${status}"><span class="dot"></span>${status}</span>
    </div>`;

  if (status === 'running' || status === 'queued') {
    const pct = progress.total > 0 ? (progress.current / progress.total * 100) : 0;
    html += `<p style="margin-bottom:8px;"><strong>${progress.message || 'starting...'}</strong></p>
      <div class="progress-bar"><div class="progress-bar-fill" style="width:${pct}%"></div></div>
      <p class="form-hint" style="margin-top:8px;">${progress.current} / ${progress.total}</p>`;
  }

  if (status === 'failed') {
    html += `<div class="error-box">${job.error || 'Unknown error'}${job.traceback ? '\n\n' + job.traceback : ''}</div>`;
  }

  if (status === 'completed') {
    const r = job.report_preview || {};
    const stats = r.stats || {};
    const md = r.metadata || {};

    html += `<div class="stats-grid">`;
    if (md.vocab_size) html += `<div class="stat"><div class="stat-value">${formatNumber(md.vocab_size)}</div><div class="stat-label">Vocab</div></div>`;
    if (md.block_count) html += `<div class="stat"><div class="stat-value">${md.block_count}</div><div class="stat-label">Layers</div></div>`;
    const tp = r.weight_inspection?.total_parameters;
    if (tp) html += `<div class="stat"><div class="stat-value">${formatNumber(tp)}</div><div class="stat-label">Parameters</div></div>`;
    if (stats.n_probes !== undefined) html += `<div class="stat"><div class="stat-value accent">${stats.n_probes}</div><div class="stat-label">Probes</div></div>`;
    if (stats.n_probes_passed !== undefined) html += `<div class="stat"><div class="stat-value success">${stats.n_probes_passed}</div><div class="stat-label">Passed</div></div>`;
    if (stats.total_elapsed_seconds) html += `<div class="stat"><div class="stat-value teal">${stats.total_elapsed_seconds.toFixed(1)}s</div><div class="stat-label">Elapsed</div></div>`;
    html += `</div>`;

    // Attribution fingerprint
    const a = r.attribution;
    if (a && Object.keys(a).length > 0 && a.knowledge_fingerprint) {
      html += `<div class="fingerprint-display">
        <div class="fingerprint-label">Knowledge Fingerprint</div>
        <div class="fingerprint-value">${a.knowledge_fingerprint}</div>
      </div>`;
    }

    // Pack summaries
    if (r.pack_summaries && r.pack_summaries.length) {
      html += `<div class="section-divider">Probe Pack Results</div>`;
      html += `<table class="table"><thead><tr><th>Pack</th><th>Category</th><th>Probes</th><th>Passed</th><th>Failed</th><th>Pass Rate</th></tr></thead><tbody>`;
      r.pack_summaries.forEach(s => {
        html += `<tr><td><strong>${s.pack_name}</strong></td><td><span class="badge badge-accent">${s.category}</span></td>
          <td>${s.n_probes}</td><td class="text-success">${s.n_passed}</td><td class="text-danger">${s.n_failed}</td>
          <td><strong>${(s.pass_rate * 100).toFixed(1)}%</strong></td></tr>`;
      });
      html += `</tbody></table>`;
    }

    // Downloads
    const paths = job.result_paths || {};
    const hasDownloads = Object.keys(paths).length > 0;
    if (hasDownloads) {
      html += `<div class="section-divider">Downloads</div><div class="download-grid">`;
      const fmtMap = { json: ['📄', 'JSON', '.json'], markdown: ['📝', 'Markdown', '.md'], graphml: ['🕸', 'GraphML', '.graphml'], turtle: ['🁢', 'RDF/Turtle', '.ttl'], sqlite: ['🗄', 'SQLite', '.db'] };
      for (const [fmt, [icon, label, ext]] of Object.entries(fmtMap)) {
        if (paths[fmt]) html += `<a class="download-btn" href="/api/jobs/${job.id}/download/${fmt}"><div class="icon">${icon}</div><div class="label">${label}</div><div class="ext">${ext}</div></a>`;
      }
      if (paths.surgery_gguf) html += `<a class="download-btn" href="${paths.surgery_gguf}" download><div class="icon">🔬</div><div class="label">Surgery GGUF</div><div class="ext">.gguf</div></a>`;
      if (paths.edited_gguf) html += `<a class="download-btn" href="${paths.edited_gguf}" download><div class="icon">✏️</div><div class="label">Edited GGUF</div><div class="ext">.gguf</div></a>`;
      html += `</div>`;
    }
  }

  html += `</div>`;
  c.innerHTML = html;
}

// ---------------------------------------------------------------- //
// Jobs List
// ---------------------------------------------------------------- //
async function loadJobs() {
  try {
    const res = await fetch('/api/jobs');
    const jobs = await res.json();
    const container = $('#jobs-list');
    if (!container) return;
    if (!jobs.length) {
      container.innerHTML = '<div class="empty-state"><div class="empty-state-icon"><svg fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" width="24" height="24"><path d="M12 8v4l3 3M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg></div>No jobs yet. Run an extraction to get started.</div>';
      return;
    }
    jobs.reverse();
    let html = '<table class="table"><thead><tr><th>Status</th><th>Model</th><th>Type</th><th>Started</th></tr></thead><tbody>';
    for (const j of jobs) {
      const kind = j.kind || 'extract';
      html += `<tr style="cursor:pointer;" onclick="navigateTo('job'); pollJob('${j.id}')">
        <td><span class="status-pill ${j.status}"><span class="dot"></span>${j.status}</span></td>
        <td class="mono">${j.gguf_filename || '—'}</td>
        <td><span class="badge badge-neutral">${kind}</span></td>
        <td class="text-tertiary text-sm">${j.started_at || '—'}</td>
      </tr>`;
    }
    html += '</tbody></table>';
    container.innerHTML = html;
  } catch (e) {}
}

// ---------------------------------------------------------------- //
// v3: Causal Trace
// ---------------------------------------------------------------- //
let traceFile = null;
const traceDz = $('#trace-dropzone');
const traceInput = $('#trace-file-input');
if (traceDz) {
  traceDz.onclick = () => traceInput.click();
  traceDz.ondragover = (e) => { e.preventDefault(); traceDz.classList.add('drag-over'); };
  traceDz.ondragleave = () => traceDz.classList.remove('drag-over');
  traceDz.ondrop = (e) => { e.preventDefault(); traceDz.classList.remove('drag-over'); if (e.dataTransfer.files.length) { traceFile = e.dataTransfer.files[0]; $('#trace-selected-file').textContent = `✓ ${traceFile.name}`; $('#btn-trace').disabled = false; } };
  traceInput.onchange = (e) => { if (e.target.files.length) { traceFile = e.target.files[0]; $('#trace-selected-file').textContent = `✓ ${traceFile.name}`; $('#btn-trace').disabled = false; } };
}
$('#btn-trace')?.addEventListener('click', async () => {
  if (!traceFile) return;
  const fd = new FormData();
  fd.append('file', traceFile);
  fd.append('top_k', $('#trace-top-k').value);
  $('#btn-trace').disabled = true; $('#btn-trace').textContent = 'Starting...';
  try {
    const res = await fetch('/api/trace', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    toast('Trace started', 'success');
    navigateTo('job');
    pollJob(data.job_id);
  } catch (e) { toast('Failed: ' + e.message, 'error'); $('#btn-trace').disabled = false; $('#btn-trace').textContent = 'Run Causal Trace →'; }
});

// ---------------------------------------------------------------- //
// v6: Causal Mediation
// ---------------------------------------------------------------- //
let mediateFile = null;
const medDz = $('#mediate-dropzone');
const medInput = $('#mediate-file-input');
if (medDz) {
  medDz.onclick = () => medInput.click();
  medDz.ondragover = (e) => { e.preventDefault(); medDz.classList.add('drag-over'); };
  medDz.ondragleave = () => medDz.classList.remove('drag-over');
  medDz.ondrop = (e) => { e.preventDefault(); medDz.classList.remove('drag-over'); if (e.dataTransfer.files.length) { mediateFile = e.dataTransfer.files[0]; $('#mediate-selected-file').textContent = `✓ ${mediateFile.name}`; updateMediateBtn(); } };
  medInput.onchange = (e) => { if (e.target.files.length) { mediateFile = e.dataTransfer.files[0]; $('#mediate-selected-file').textContent = `✓ ${mediateFile.name}`; updateMediateBtn(); } };
}
function updateMediateBtn() {
  const ready = mediateFile && $('#mediate-prompt').value.trim();
  $('#btn-mediate').disabled = !ready;
}
$('#mediate-prompt')?.addEventListener('input', updateMediateBtn);
$('#btn-mediate')?.addEventListener('click', async () => {
  if (!mediateFile) return;
  const fd = new FormData();
  fd.append('file', mediateFile);
  fd.append('prompt', $('#mediate-prompt').value);
  fd.append('expected', $('#mediate-expected').value);
  fd.append('noise', $('#mediate-noise').value);
  $('#btn-mediate').disabled = true; $('#btn-mediate').textContent = 'Starting...';
  try {
    const res = await fetch('/api/mediate', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    toast('Mediation analysis started', 'success');
    navigateTo('job');
    pollJob(data.job_id);
  } catch (e) { toast('Failed: ' + e.message, 'error'); $('#btn-mediate').disabled = false; $('#btn-mediate').textContent = 'Run Mediation Analysis →'; }
});

// ---------------------------------------------------------------- //
// v3: ROME Edit
// ---------------------------------------------------------------- //
let editFile = null;
let editRequests = [];
const editDz = $('#edit-dropzone');
const editInput = $('#edit-file-input');
if (editDz) {
  editDz.onclick = () => editInput.click();
  editDz.ondragover = (e) => { e.preventDefault(); editDz.classList.add('drag-over'); };
  editDz.ondragleave = () => editDz.classList.remove('drag-over');
  editDz.ondrop = (e) => { e.preventDefault(); editDz.classList.remove('drag-over'); if (e.dataTransfer.files.length) { editFile = e.dataTransfer.files[0]; $('#edit-selected-file').textContent = `✓ ${editFile.name}`; updateEditBtn(); } };
  editInput.onchange = (e) => { if (e.target.files.length) { editFile = e.dataTransfer.files[0]; $('#edit-selected-file').textContent = `✓ ${editFile.name}`; updateEditBtn(); } };
}
function initEditView() {
  if (editRequests.length === 0) { editRequests = [{ subject: '', prompt: '', target_object: '' }]; renderEdits(); }
}
$('#btn-add-edit')?.addEventListener('click', () => { editRequests.push({ subject: '', prompt: '', target_object: '' }); renderEdits(); });
function renderEdits() {
  const container = $('#edits-list');
  if (!container) return;
  container.innerHTML = '';
  editRequests.forEach((req, idx) => {
    const div = document.createElement('div');
    div.style.cssText = 'background:var(--bg-elevated);border-radius:8px;padding:16px;margin-bottom:12px;';
    div.innerHTML = `<div style="display:flex;justify-content:space-between;margin-bottom:12px;"><strong>Edit #${idx + 1}</strong>${editRequests.length > 1 ? `<button class="btn btn-danger btn-sm" data-rm="${idx}">Remove</button>` : ''}</div>
      <div class="form-row"><div class="form-group"><div class="form-label">Subject</div><input type="text" class="form-input" data-field="subject" data-idx="${idx}" value="${req.subject.replace(/"/g, '&quot;')}" placeholder="The Eiffel Tower" /></div>
      <div class="form-group"><div class="form-label">Target Object</div><input type="text" class="form-input" data-field="target_object" data-idx="${idx}" value="${req.target_object.replace(/"/g, '&quot;')}" placeholder="Berlin" /></div></div>
      <div class="form-group"><div class="form-label">Prompt</div><input type="text" class="form-input" data-field="prompt" data-idx="${idx}" value="${req.prompt.replace(/"/g, '&quot;')}" placeholder="The Eiffel Tower is located in the city of" /></div>`;
    container.appendChild(div);
  });
  container.querySelectorAll('input[data-field]').forEach(inp => {
    inp.oninput = (e) => { editRequests[parseInt(e.target.dataset.idx)][e.target.dataset.field] = e.target.value; updateEditBtn(); };
  });
  container.querySelectorAll('button[data-rm]').forEach(btn => {
    btn.onclick = () => { editRequests.splice(parseInt(btn.dataset.rm), 1); renderEdits(); updateEditBtn(); };
  });
}
function updateEditBtn() {
  const ready = editFile && editRequests.some(r => r.subject && r.prompt && r.target_object);
  $('#btn-edit').disabled = !ready;
}
$('#btn-edit')?.addEventListener('click', async () => {
  if (!editFile) return;
  const valid = editRequests.filter(r => r.subject && r.prompt && r.target_object);
  const fd = new FormData();
  fd.append('file', editFile);
  fd.append('edits_json', JSON.stringify(valid));
  $('#btn-edit').disabled = true; $('#btn-edit').textContent = 'Starting...';
  try {
    const res = await fetch('/api/edit', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    toast('ROME edit started', 'success');
    navigateTo('job');
    pollJob(data.job_id);
  } catch (e) { toast('Failed: ' + e.message, 'error'); $('#btn-edit').disabled = false; $('#btn-edit').textContent = 'Apply ROME Edits →'; }
});

// ---------------------------------------------------------------- //
// v3: Compare
// ---------------------------------------------------------------- //
let compareFiles = [];
const cmpDz = $('#compare-dropzone');
const cmpInput = $('#compare-file-input');
if (cmpDz) {
  cmpDz.onclick = () => cmpInput.click();
  cmpDz.ondragover = (e) => { e.preventDefault(); cmpDz.classList.add('drag-over'); };
  cmpDz.ondragleave = () => cmpDz.classList.remove('drag-over');
  cmpDz.ondrop = (e) => { e.preventDefault(); cmpDz.classList.remove('drag-over'); for (const f of e.dataTransfer.files) handleCompareFile(f); };
  cmpInput.onchange = (e) => { for (const f of e.target.files) handleCompareFile(f); };
}
function handleCompareFile(file) {
  if (!file.name.toLowerCase().endsWith('.json')) return;
  if (compareFiles.find(f => f.name === file.name)) return;
  compareFiles.push(file);
  renderCompareFiles();
}
function renderCompareFiles() {
  const container = $('#compare-files-list');
  container.innerHTML = compareFiles.map((f, i) =>
    `<div style="display:flex;justify-content:space-between;align-items:center;padding:8px 12px;background:var(--bg-elevated);border-radius:6px;margin-bottom:6px;">
      <span class="mono text-sm text-teal">📄 ${f.name}</span>
      <button class="btn btn-danger btn-sm" data-rm="${i}">Remove</button></div>`
  ).join('');
  container.querySelectorAll('button[data-rm]').forEach(btn => {
    btn.onclick = () => { compareFiles.splice(parseInt(btn.dataset.rm), 1); renderCompareFiles(); $('#btn-compare').disabled = compareFiles.length < 2; };
  });
  $('#btn-compare').disabled = compareFiles.length < 2;
}
$('#btn-compare')?.addEventListener('click', async () => {
  if (compareFiles.length < 2) return;
  const fd = new FormData();
  for (const f of compareFiles) fd.append('reports', f, f.name);
  $('#btn-compare').disabled = true; $('#btn-compare').textContent = 'Comparing...';
  try {
    const res = await fetch('/api/compare', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    navigateTo('job');
    renderCompareResult(data.report);
  } catch (e) { toast('Failed: ' + e.message, 'error'); $('#btn-compare').disabled = false; $('#btn-compare').textContent = 'Compare Models →'; }
});

function renderCompareResult(report) {
  const c = $('#job-content');
  let html = `<div class="card"><div class="card-header"><div><div class="card-title">Cross-Model Comparison</div><div class="card-subtitle">${report.n_models} models · ${report.pairwise.length} comparisons</div></div><span class="status-pill completed"><span class="dot"></span>completed</span></div>`;
  html += `<div class="stats-grid">
    <div class="stat"><div class="stat-value">${report.n_models}</div><div class="stat-label">Models</div></div>
    <div class="stat"><div class="stat-value">${report.pairwise.length}</div><div class="stat-label">Comparisons</div></div>
    <div class="stat"><div class="stat-value accent">${report.stats.n_identical_fingerprints}</div><div class="stat-label">Identical</div></div>
    <div class="stat"><div class="stat-value success">${report.stats.n_strong_lineage}</div><div class="stat-label">Strong Lineage</div></div></div>`;
  html += `<div class="section-divider">Pairwise Comparisons</div>`;
  for (const p of report.pairwise) {
    const color = p.lineage_score > 0.7 ? 'var(--success)' : p.lineage_score > 0.4 ? 'var(--warning)' : 'var(--danger)';
    html += `<div style="background:var(--bg-elevated);border-radius:8px;padding:16px;margin-bottom:12px;border-left:4px solid ${color};">
      <div style="display:flex;justify-content:space-between;margin-bottom:8px;"><strong>${p.model_a} ↔ ${p.model_b}</strong>
      <span class="mono" style="font-size:18px;color:${color};font-weight:700;">${p.lineage_score.toFixed(3)}</span></div>
      <div class="text-sm" style="margin-bottom:8px;">${p.lineage_hypothesis}</div>
      <table class="table"><tr><td>Fingerprint match</td><td>${p.fingerprint_match ? '✓ identical' : '✗ different'}</td></tr>
      <tr><td>Same arch</td><td>${p.same_arch ? '✓' : '✗'}</td></tr>
      <tr><td>Top neuron Jaccard</td><td>${p.top_neuron_jaccard.toFixed(3)}</td></tr>
      <tr><td>Top token Jaccard</td><td>${p.top_token_jaccard.toFixed(3)}</td></tr>
      <tr><td>Concept mastery corr</td><td>${p.concept_mastery_correlation.toFixed(3)}</td></tr></table></div>`;
  }
  html += `</div>`;
  c.innerHTML = html;
}

// ---------------------------------------------------------------- //
// v5: Surgery
// ---------------------------------------------------------------- //
let surgeryFile = null;
let surgeryLocalPath = null;
let surgeryOps = [];
const surgeryDz = $('#surgery-dropzone');
const surgeryInput = $('#surgery-file-input');
if (surgeryDz) {
  surgeryDz.onclick = () => surgeryInput.click();
  surgeryDz.ondragover = (e) => { e.preventDefault(); surgeryDz.classList.add('drag-over'); };
  surgeryDz.ondragleave = () => surgeryDz.classList.remove('drag-over');
  surgeryDz.ondrop = (e) => { e.preventDefault(); surgeryDz.classList.remove('drag-over'); if (e.dataTransfer.files.length) { surgeryFile = e.dataTransfer.files[0]; surgeryLocalPath = null; $('#surgery-selected-file').textContent = `✓ ${surgeryFile.name}`; updateSurgeryButton(); } };
  surgeryInput.onchange = (e) => { if (e.target.files.length) { surgeryFile = e.dataTransfer.files[0]; surgeryLocalPath = null; $('#surgery-selected-file').textContent = `✓ ${surgeryFile.name}`; updateSurgeryButton(); } };
}
async function refreshSurgeryLocalModels() {
  try {
    const res = await fetch('/api/models/local');
    const data = await res.json();
    const select = $('#surgery-local-select');
    if (!select) return;
    select.innerHTML = '<option value="">— Select a local model —</option>';
    for (const m of data.models) {
      const opt = document.createElement('option');
      opt.value = m.path; opt.textContent = `${m.filename} (${m.size_human})`;
      select.appendChild(opt);
    }
  } catch (e) {}
}
$('#surgery-local-select')?.addEventListener('change', (e) => {
  if (e.target.value) { surgeryLocalPath = e.target.value; surgeryFile = null; $('#surgery-selected-file').textContent = `✓ Selected: ${e.target.value.split('/').pop()}`; updateSurgeryButton(); }
});
const OP_TYPES = [
  { value: 'bake_system_prompt', label: 'Bake System Prompt' },
  { value: 'set_chat_template', label: 'Set Chat Template' },
  { value: 'inject_dataset', label: 'Inject Dataset' },
  { value: 'add_token', label: 'Add New Token' },
  { value: 'add_steering_vector', label: 'Add Steering Vector' },
  { value: 'set_metadata', label: 'Set Metadata' },
  { value: 'remove_metadata', label: 'Remove Metadata' },
];
$('#btn-surgery-add-op')?.addEventListener('click', () => { surgeryOps.push({ type: 'bake_system_prompt', fields: {} }); renderSurgeryOps(); });
function renderSurgeryOps() {
  const container = $('#surgery-ops-list');
  if (!container) return;
  container.innerHTML = '';
  surgeryOps.forEach((op, idx) => {
    const div = document.createElement('div');
    div.style.cssText = 'background:var(--bg-elevated);border-radius:8px;padding:16px;margin-bottom:12px;';
    let html = `<div style="display:flex;justify-content:space-between;margin-bottom:12px;"><select class="form-select surgery-op-type" data-op-idx="${idx}" style="max-width:240px;">`;
    for (const t of OP_TYPES) html += `<option value="${t.value}" ${op.type === t.value ? 'selected' : ''}>${t.label}</option>`;
    html += `</select><button class="btn btn-danger btn-sm" data-rm-op="${idx}">Remove</button></div>`;
    const template = $(`#surgery-op-templates [data-template="${op.type}"]`);
    if (template) html += `<div class="surgery-op-fields" data-op-idx="${idx}">${template.innerHTML}</div>`;
    div.innerHTML = html;
    container.appendChild(div);
  });
  container.querySelectorAll('.surgery-op-type').forEach(sel => {
    sel.onchange = (e) => { surgeryOps[parseInt(e.target.dataset.opIdx)].type = e.target.value; surgeryOps[parseInt(e.target.dataset.opIdx)].fields = {}; renderSurgeryOps(); };
  });
  container.querySelectorAll('button[data-rm-op]').forEach(btn => {
    btn.onclick = () => { surgeryOps.splice(parseInt(btn.dataset.rmOp), 1); renderSurgeryOps(); updateSurgeryButton(); };
  });
  container.querySelectorAll('.surgery-op-fields').forEach(fc => {
    const idx = parseInt(fc.dataset.opIdx);
    fc.querySelectorAll('[data-field]').forEach(inp => {
      const fn = inp.dataset.field;
      if (surgeryOps[idx].fields[fn] !== undefined) inp.value = surgeryOps[idx].fields[fn];
      inp.oninput = () => { surgeryOps[idx].fields[fn] = inp.value; updateSurgeryButton(); };
    });
  });
}
function updateSurgeryButton() {
  const hasModel = surgeryFile || surgeryLocalPath;
  $('#btn-surgery-run').disabled = !(hasModel && surgeryOps.length > 0);
}
$('#btn-surgery-run')?.addEventListener('click', async () => {
  const operations = surgeryOps.map(op => {
    const r = { op: op.type }, f = op.fields;
    if (op.type === 'bake_system_prompt') r.prompt = f.prompt || '';
    else if (op.type === 'set_chat_template') r.template = f.template || '';
    else if (op.type === 'inject_dataset') { r.name = f.name || 'dataset'; r.description = f.description || ''; try { r.data = JSON.parse(f.data || '[]'); } catch { r.data = []; } }
    else if (op.type === 'add_token') { r.token = f.token || ''; if (f.embedding && f.embedding.trim()) r.embedding = f.embedding.split(',').map(parseFloat); }
    else if (op.type === 'add_steering_vector') { r.layer = parseInt(f.layer) || 0; r.name = f.name || 'default'; r.strength = parseFloat(f.strength) || 1.0; r.vector = f.vector.split(',').map(parseFloat); }
    else if (op.type === 'set_metadata') { r.key = f.key || ''; r.value = f.value || ''; if (f.value_type === 'int') r.value = parseInt(r.value); else if (f.value_type === 'float') r.value = parseFloat(r.value); else if (f.value_type === 'bool') r.value = r.value.toLowerCase() === 'true'; }
    else if (op.type === 'remove_metadata') r.key = f.key || '';
    return r;
  });
  const fd = new FormData();
  if (surgeryFile) fd.append('file', surgeryFile);
  else if (surgeryLocalPath) fd.append('local_path', surgeryLocalPath);
  fd.append('operations_json', JSON.stringify(operations));
  $('#btn-surgery-run').disabled = true; $('#btn-surgery-run').textContent = 'Running...';
  try {
    const url = surgeryLocalPath ? '/api/surgery-local' : '/api/surgery';
    const res = await fetch(url, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    toast('Surgery started', 'success');
    navigateTo('job');
    pollJob(data.job_id);
  } catch (e) { toast('Failed: ' + e.message, 'error'); $('#btn-surgery-run').disabled = false; $('#btn-surgery-run').textContent = 'Run Surgery →'; }
});

// ---------------------------------------------------------------- //
// v6: Merge
// ---------------------------------------------------------------- //
$('#btn-merge')?.addEventListener('click', async () => {
  const a = $('#merge-model-a').value.trim();
  const b = $('#merge-model-b').value.trim();
  if (!a || !b) { toast('Provide both model paths', 'error'); return; }
  const fd = new FormData();
  fd.append('model_a', a); fd.append('model_b', b);
  fd.append('algorithm', $('#merge-algorithm').value);
  fd.append('alpha', $('#merge-alpha').value);
  fd.append('filter', $('#merge-filter').value);
  $('#btn-merge').disabled = true; $('#btn-merge').textContent = 'Merging...';
  try {
    const res = await fetch('/api/merge', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    toast('Merge started', 'success');
    navigateTo('job');
    pollJob(data.job_id);
  } catch (e) { toast('Failed: ' + e.message, 'error'); $('#btn-merge').disabled = false; $('#btn-merge').textContent = 'Merge Models →'; }
});

// ---------------------------------------------------------------- //
// v6: Diff
// ---------------------------------------------------------------- //
$('#btn-diff')?.addEventListener('click', async () => {
  const a = $('#diff-model-a').value.trim();
  const b = $('#diff-model-b').value.trim();
  if (!a || !b) { toast('Provide both model paths', 'error'); return; }
  const fd = new FormData();
  fd.append('model_a', a); fd.append('model_b', b);
  $('#btn-diff').disabled = true; $('#btn-diff').textContent = 'Diffing...';
  try {
    const res = await fetch('/api/diff', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    renderDiffResult(data.report || data);
  } catch (e) { toast('Failed: ' + e.message, 'error'); $('#btn-diff').disabled = false; $('#btn-diff').textContent = 'Compute Diff →'; }
});

function renderDiffResult(report) {
  navigateTo('job');
  const c = $('#job-content');
  let html = `<div class="card"><div class="card-header"><div><div class="card-title">GGUF Diff</div><div class="card-subtitle">${report.source_a} vs ${report.source_b}</div></div><span class="status-pill completed"><span class="dot"></span>completed</span></div>`;
  html += `<div class="stats-grid">
    <div class="stat"><div class="stat-value">${formatBytes(report.file_size_a)}</div><div class="stat-label">File A</div></div>
    <div class="stat"><div class="stat-value">${formatBytes(report.file_size_b)}</div><div class="stat-label">File B</div></div>
    <div class="stat"><div class="stat-value">${report.summary.n_tensors_modified}</div><div class="stat-label">Modified</div></div>
    <div class="stat"><div class="stat-value teal">${report.summary.overall_similarity_percent.toFixed(1)}%</div><div class="stat-label">Similarity</div></div></div>`;
  const modified = (report.tensor_diffs || []).filter(t => t.status === 'modified').sort((a, b) => (a.f32_cosine_sim || 1) - (b.f32_cosine_sim || 1));
  if (modified.length) {
    html += `<div class="section-divider">Top 15 Most Different Tensors</div><table class="table"><thead><tr><th>Tensor</th><th>Cosine Sim</th><th>Mean Diff</th><th>Byte Diff</th></tr></thead><tbody>`;
    for (const t of modified.slice(0, 15)) {
      html += `<tr><td class="mono">${t.name}</td><td>${(t.f32_cosine_sim || 0).toFixed(4)}</td><td>${(t.f32_mean_abs_diff || 0).toFixed(4)}</td><td>${(t.byte_diff_percent || 0).toFixed(1)}%</td></tr>`;
    }
    html += `</tbody></table>`;
  }
  html += `</div>`;
  c.innerHTML = html;
}

// ---------------------------------------------------------------- //
// v4: Hugging Face Hub
// ---------------------------------------------------------------- //
$('#btn-models-search')?.addEventListener('click', async () => {
  const q = $('#models-search-query').value.trim();
  if (!q) { toast('Enter a search query', 'info'); return; }
  $('#btn-models-search').textContent = 'Searching...'; $('#btn-models-search').disabled = true;
  try {
    const res = await fetch(`/api/models/search?q=${encodeURIComponent(q)}&limit=30`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    renderSearchResults(data.results);
  } catch (e) { toast('Search failed: ' + e.message, 'error'); }
  finally { $('#btn-models-search').textContent = 'Search'; $('#btn-models-search').disabled = false; }
});
$('#models-search-query')?.addEventListener('keypress', (e) => { if (e.key === 'Enter') $('#btn-models-search').click(); });

function renderSearchResults(results) {
  const container = $('#models-search-results');
  if (!results.length) { container.innerHTML = '<div class="empty-state">No models found.</div>'; return; }
  let html = `<p class="form-hint">${results.length} models found — click a row to see files</p>`;
  html += '<table class="table"><thead><tr><th>Repo</th><th>Downloads</th><th>Likes</th><th>Tags</th><th>Modified</th></tr></thead><tbody>';
  for (const m of results) {
    const tags = (m.tags || []).slice(0, 3).join(', ');
    html += `<tr style="cursor:pointer;" data-repo="${m.repo_id}" class="search-row">
      <td><strong>${m.repo_id}</strong>${m.gated ? ' <span class="badge badge-warning">GATED</span>' : ''}</td>
      <td>${formatNumber(m.downloads)}</td><td>${m.likes}</td>
      <td class="text-sm text-tertiary">${tags}</td><td class="text-sm">${(m.last_modified || '').slice(0, 10)}</td></tr>`;
  }
  html += '</tbody></table>';
  container.innerHTML = html;
  container.querySelectorAll('.search-row').forEach(row => {
    row.onclick = () => { $('#models-info-repo').value = row.dataset.repo; $('#btn-models-info').click(); };
  });
}

$('#btn-models-info')?.addEventListener('click', async () => {
  const repo = $('#models-info-repo').value.trim();
  if (!repo) return;
  $('#btn-models-info').textContent = 'Loading...'; $('#btn-models-info').disabled = true;
  try {
    const res = await fetch(`/api/models/info/${repo}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    renderModelInfo(data);
  } catch (e) { toast('Info failed: ' + e.message, 'error'); }
  finally { $('#btn-models-info').textContent = 'Get info'; $('#btn-models-info').disabled = false; }
});

function renderModelInfo(info) {
  const container = $('#models-info-result');
  let html = `<div style="background:var(--bg-elevated);border-radius:12px;padding:16px;">
    <div style="display:flex;justify-content:space-between;margin-bottom:12px;">
      <div><strong style="font-size:14px;">${info.repo_id}</strong>${info.gated ? '<span class="badge badge-warning" style="margin-left:6px;">GATED</span>' : ''}
      <p class="form-hint" style="margin-top:4px;">by <strong>${info.author}</strong> · ${formatNumber(info.downloads)} downloads · ${info.likes} likes</p></div>
      <div class="text-sm text-tertiary" style="text-align:right;">${info.pipeline_tag || 'n/a'}<br>${(info.last_modified || '').slice(0, 10)}</div></div>`;
  if (info.tags && info.tags.length) {
    html += '<div style="margin-bottom:12px;">' + info.tags.map(t => `<span class="badge badge-neutral" style="margin:2px;">${t}</span>`).join('') + '</div>';
  }
  html += '<div class="section-divider" style="margin:12px 0;">Files</div>';
  html += '<table class="table"><thead><tr><th>Filename</th><th>Size</th><th></th></tr></thead><tbody>';
  for (const f of info.files) {
    if (f.is_gguf) {
      html += `<tr><td><span class="badge badge-accent">GGUF</span> ${f.filename}</td><td>${f.size_human}</td>
        <td><button class="btn btn-primary btn-sm" data-dl-repo="${info.repo_id}" data-dl-file="${f.filename}">Download</button></td></tr>`;
    } else {
      html += `<tr style="opacity:0.6;"><td>${f.filename}</td><td>${f.size_human}</td><td></td></tr>`;
    }
  }
  html += '</tbody></table></div>';
  container.innerHTML = html;
  container.querySelectorAll('button[data-dl-repo]').forEach(btn => {
    btn.onclick = async () => {
      const fd = new FormData();
      fd.append('repo_id', btn.dataset.dlRepo); fd.append('filename', btn.dataset.dlFile);
      btn.disabled = true; btn.textContent = 'Starting...';
      try {
        const res = await fetch('/api/models/download', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail);
        toast('Download started', 'success'); btn.textContent = 'Downloading...';
        startDownloadPolling();
      } catch (e) { toast('Download failed: ' + e.message, 'error'); btn.disabled = false; btn.textContent = 'Download'; }
    };
  });
}

async function loadLocalModels() {
  try {
    const res = await fetch('/api/models/local');
    const data = await res.json();
    $('#models-local-dir').textContent = `Directory: ${data.models_dir}`;
    const container = $('#models-local-list');
    if (!data.models.length) { container.innerHTML = '<div class="empty-state">No local models yet. Use the search above to find and download GGUF models.</div>'; return; }
    let html = '<table class="table"><thead><tr><th>Filename</th><th>Size</th><th>Source</th><th>Downloaded</th><th></th></tr></thead><tbody>';
    for (const m of data.models) {
      html += `<tr><td><strong>${m.filename}</strong></td><td>${m.size_human}</td><td class="text-sm">${m.repo_id || 'unknown'}</td><td class="text-sm">${(m.downloaded_at || '').slice(0, 16)}</td>
        <td><button class="btn btn-danger btn-sm" data-del="${m.filename}">Delete</button></td></tr>`;
    }
    html += '</tbody></table>';
    container.innerHTML = html;
    container.querySelectorAll('button[data-del]').forEach(btn => {
      btn.onclick = async () => {
        if (!confirm(`Delete ${btn.dataset.del}?`)) return;
        try {
          const res = await fetch(`/api/models/local/${encodeURIComponent(btn.dataset.del)}`, { method: 'DELETE' });
          if (!res.ok) throw new Error('failed');
          toast('Deleted', 'success'); loadLocalModels();
        } catch (e) { toast('Delete failed', 'error'); }
      };
    });
  } catch (e) {}
}

async function loadDownloads() {
  try {
    const res = await fetch('/api/models/downloads');
    const downloads = await res.json();
    renderDownloads(downloads);
    if (downloads.some(d => d.status === 'downloading' || d.status === 'queued')) startDownloadPolling();
  } catch (e) {}
}

function renderDownloads(downloads) {
  const container = $('#models-downloads-list');
  if (!container) return;
  if (!downloads.length) { container.innerHTML = '<div class="empty-state" style="padding:24px;">No downloads yet.</div>'; return; }
  const sorted = [...downloads].reverse();
  let html = '<table class="table"><thead><tr><th>Status</th><th>File</th><th>Progress</th><th>Speed</th><th>ETA</th></tr></thead><tbody>';
  for (const d of sorted.slice(0, 10)) {
    const pct = d.percent.toFixed(1);
    const sc = d.status === 'completed' ? 'var(--success)' : d.status === 'failed' ? 'var(--danger)' : 'var(--accent)';
    html += `<tr><td><span class="status-pill ${d.status}"><span class="dot"></span>${d.status}</span></td>
      <td class="mono text-sm">${d.filename}</td>
      <td><div style="display:flex;align-items:center;gap:8px;"><div class="progress-bar" style="flex:1;min-width:100px;"><div class="progress-bar-fill" style="width:${pct}%;background:${sc};"></div></div><span class="text-sm mono">${formatBytes(d.bytes_downloaded)} / ${formatBytes(d.total_bytes)}</span></div></td>
      <td class="text-sm">${d.speed_mbps > 0 ? d.speed_mbps.toFixed(1) + ' MB/s' : '—'}</td>
      <td class="text-sm">${d.eta_seconds > 0 ? Math.round(d.eta_seconds) + 's' : '—'}</td></tr>`;
  }
  html += '</tbody></table>';
  container.innerHTML = html;
}

function startDownloadPolling() {
  if (downloadPollTimer) return;
  downloadPollTimer = setInterval(async () => {
    try {
      const res = await fetch('/api/models/downloads');
      const downloads = await res.json();
      renderDownloads(downloads);
      if (!downloads.some(d => d.status === 'downloading' || d.status === 'queued')) {
        clearInterval(downloadPollTimer); downloadPollTimer = null; loadLocalModels();
      }
    } catch (e) {}
  }, 1500);
}

// ---------------------------------------------------------------- //
// Init
// ---------------------------------------------------------------- //
loadPacks();
loadDashboard();

// ---------------------------------------------------------------- //
// v7: Imatrix
// ---------------------------------------------------------------- //
let imatrixFile = null;
const imatrixDz = $('#imatrix-dropzone');
const imatrixInput = $('#imatrix-file-input');
if (imatrixDz) {
  imatrixDz.onclick = () => imatrixInput.click();
  imatrixDz.ondragover = (e) => { e.preventDefault(); imatrixDz.classList.add('drag-over'); };
  imatrixDz.ondragleave = () => imatrixDz.classList.remove('drag-over');
  imatrixDz.ondrop = (e) => { e.preventDefault(); imatrixDz.classList.remove('drag-over'); if (e.dataTransfer.files.length) { imatrixFile = e.dataTransfer.files[0]; $('#imatrix-selected-file').textContent = `✓ ${imatrixFile.name}`; $('#btn-imatrix').disabled = false; } };
  imatrixInput.onchange = (e) => { if (e.target.files.length) { imatrixFile = e.target.files[0]; $('#imatrix-selected-file').textContent = `✓ ${imatrixFile.name}`; $('#btn-imatrix').disabled = false; } };
}
$('#btn-imatrix')?.addEventListener('click', async () => {
  if (!imatrixFile) return;
  const fd = new FormData();
  fd.append('file', imatrixFile);
  $('#btn-imatrix').disabled = true; $('#btn-imatrix').textContent = 'Computing...';
  try {
    const res = await fetch('/api/imatrix', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    toast('Imatrix computation started', 'success');
    navigateTo('job');
    pollJob(data.job_id);
  } catch (e) { toast('Failed: ' + e.message, 'error'); $('#btn-imatrix').disabled = false; $('#btn-imatrix').textContent = 'Compute Imatrix →'; }
});

// ---------------------------------------------------------------- //
// v7: Quantize
// ---------------------------------------------------------------- //
let quantizeFile = null;
const quantDz = $('#quantize-dropzone');
const quantInput = $('#quantize-file-input');
if (quantDz) {
  quantDz.onclick = () => quantInput.click();
  quantDz.ondragover = (e) => { e.preventDefault(); quantDz.classList.add('drag-over'); };
  quantDz.ondragleave = () => quantDz.classList.remove('drag-over');
  quantDz.ondrop = (e) => { e.preventDefault(); quantDz.classList.remove('drag-over'); if (e.dataTransfer.files.length) { quantizeFile = e.dataTransfer.files[0]; $('#quantize-selected-file').textContent = `✓ ${quantizeFile.name}`; $('#btn-quantize').disabled = false; } };
  quantInput.onchange = (e) => { if (e.target.files.length) { quantizeFile = e.dataTransfer.files[0]; $('#quantize-selected-file').textContent = `✓ ${quantizeFile.name}`; $('#btn-quantize').disabled = false; } };
}
$('#btn-quantize')?.addEventListener('click', async () => {
  if (!quantizeFile) return;
  const fd = new FormData();
  fd.append('file', quantizeFile);
  fd.append('qtype', $('#quantize-qtype').value);
  fd.append('use_imatrix', $('#quantize-use-imatrix').value === 'true');
  $('#btn-quantize').disabled = true; $('#btn-quantize').textContent = 'Quantizing...';
  try {
    const res = await fetch('/api/quantize', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    toast('Quantization started', 'success');
    navigateTo('job');
    pollJob(data.job_id);
  } catch (e) { toast('Failed: ' + e.message, 'error'); $('#btn-quantize').disabled = false; $('#btn-quantize').textContent = 'Quantize Model →'; }
});

// ---------------------------------------------------------------- //
// v7: Transplant
// ---------------------------------------------------------------- //
$('#btn-transplant')?.addEventListener('click', async () => {
  const source = $('#transplant-source').value.trim();
  const target = $('#transplant-target').value.trim();
  if (!source || !target) { toast('Provide both source and target paths', 'error'); return; }
  const fd = new FormData();
  fd.append('source', source);
  fd.append('target', target);
  fd.append('strategy', $('#transplant-strategy').value);
  fd.append('strength', $('#transplant-strength').value);
  fd.append('facts_file', $('#transplant-facts-file').value);
  $('#btn-transplant').disabled = true; $('#btn-transplant').textContent = 'Transplanting...';
  try {
    const res = await fetch('/api/transplant', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    toast('Transplant started', 'success');
    navigateTo('job');
    pollJob(data.job_id);
  } catch (e) { toast('Failed: ' + e.message, 'error'); $('#btn-transplant').disabled = false; $('#btn-transplant').textContent = 'Transplant Knowledge →'; }
});
