/* GGUF Knowledge Extractor — frontend logic */

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

let selectedFile = null;
let selectedPacks = new Set(); // empty = "all"
let pollTimer = null;

// ---------------------------------------------------------------- //
// Navigation
// ---------------------------------------------------------------- //
function showView(name) {
  $$('.view').forEach(v => v.classList.remove('active'));
  $(`#view-${name}`).classList.add('active');
  $$('nav a').forEach(a => a.classList.remove('active'));
  $(`#nav-${name}`).classList.add('active');
}

$('#nav-extract').onclick = (e) => { e.preventDefault(); showView('extract'); };
$('#nav-trace').onclick = (e) => { e.preventDefault(); showView('trace'); };
$('#nav-edit').onclick = (e) => { e.preventDefault(); showView('edit'); initEditView(); };
$('#nav-compare').onclick = (e) => { e.preventDefault(); showView('compare'); };
$('#nav-models').onclick = (e) => { e.preventDefault(); showView('models'); loadLocalModels(); loadDownloads(); };
$('#nav-surgery').onclick = (e) => { e.preventDefault(); showView('surgery'); refreshSurgeryLocalModels(); };
$('#nav-jobs').onclick = (e) => { e.preventDefault(); showView('jobs'); loadJobs(); };
$('#nav-packs').onclick = (e) => { e.preventDefault(); showView('packs'); loadPacksDetail(); };
$('#nav-backends').onclick = (e) => { e.preventDefault(); showView('backends'); loadBackendsDetail(); };

// ---------------------------------------------------------------- //
// File upload
// ---------------------------------------------------------------- //
const dz = $('#dropzone');
const fileInput = $('#file-input');

dz.onclick = () => fileInput.click();
dz.ondragover = (e) => { e.preventDefault(); dz.classList.add('drag-over'); };
dz.ondragleave = () => dz.classList.remove('drag-over');
dz.ondrop = (e) => {
  e.preventDefault();
  dz.classList.remove('drag-over');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
};
fileInput.onchange = (e) => { if (e.target.files.length) handleFile(e.target.files[0]); };

function handleFile(file) {
  if (!file.name.toLowerCase().endsWith('.gguf')) {
    alert('File must be a .gguf file');
    return;
  }
  selectedFile = file;
  selectedFilePath = null;  // clear local path selection
  $('#local-model-select').value = '';  // reset dropdown
  $('#selected-file').textContent = `✓ ${file.name} (${(file.size / 1e9).toFixed(2)} GB)`;
  $('#btn-extract').disabled = false;
}

// ---------------------------------------------------------------- //
// Probe packs loading
// ---------------------------------------------------------------- //
async function loadPacks() {
  const res = await fetch('/api/packs');
  const packs = await res.json();
  const container = $('#packs-list');
  container.innerHTML = '';
  packs.forEach(p => {
    const card = document.createElement('div');
    card.className = 'pack-card selected';
    card.dataset.name = p.name;
    card.innerHTML = `
      <div class="pack-name">${p.name}</div>
      <div class="pack-meta">
        <span class="pack-category ${p.category}">${p.category}</span>
        ${p.domain || 'general'} · ${p.n_probes} probes
      </div>
      <div class="pack-desc">${p.description}</div>
    `;
    card.onclick = () => {
      if (card.classList.contains('selected')) {
        card.classList.remove('selected');
        selectedPacks.add(p.name);
      } else {
        card.classList.add('selected');
        selectedPacks.delete(p.name);
      }
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
    const item = document.createElement('div');
    item.className = 'pack-card';
    item.style.cursor = 'default';
    item.innerHTML = `
      <div class="pack-name">${p.name}</div>
      <div class="pack-meta">
        <span class="pack-category ${p.category}">${p.category}</span>
        ${p.domain || 'general'} · ${p.n_probes} probes
      </div>
      <div class="pack-desc">${p.description}</div>
      <div class="pack-meta" style="margin-top:6px;">📁 ${p.path}</div>
    `;
    container.appendChild(item);
  });
}

// ---------------------------------------------------------------- //
// Backends
// ---------------------------------------------------------------- //
async function checkBackends() {
  const url = $('#opt-server-url').value;
  $('#backends-status').innerHTML = '<p class="hint">Checking...</p>';
  const res = await fetch(`/api/backends?server_url=${encodeURIComponent(url)}`);
  const data = await res.json();
  const container = $('#backends-status');
  container.innerHTML = '';
  for (const [name, ok] of Object.entries(data)) {
    const pill = document.createElement('div');
    pill.className = `backend-pill ${ok ? 'ok' : ''}`;
    pill.innerHTML = `<span class="dot"></span> <strong>${name}</strong>: ${ok ? 'available' : 'not available'}`;
    container.appendChild(pill);
  }
  if (Object.values(data).every(v => !v)) {
    const warn = document.createElement('p');
    warn.className = 'hint';
    warn.style.color = 'var(--warn)';
    warn.textContent = '⚠ No inference backend available. Probes will be skipped — only metadata and weights will be extracted.';
    container.appendChild(warn);
  }
}

async function loadBackendsDetail() {
  const url = $('#opt-server-url').value;
  $('#backends-detail').innerHTML = '<p class="hint">Checking...</p>';
  const res = await fetch(`/api/backends?server_url=${encodeURIComponent(url)}`);
  const data = await res.json();
  const container = $('#backends-detail');
  container.innerHTML = '';
  for (const [name, ok] of Object.entries(data)) {
    const pill = document.createElement('div');
    pill.className = `backend-pill ${ok ? 'ok' : ''}`;
    pill.innerHTML = `<span class="dot"></span> <strong>${name}</strong>: ${ok ? 'available' : 'not available'}`;
    container.appendChild(pill);
  }
  const help = document.createElement('div');
  help.style.marginTop = '16px';
  help.innerHTML = `
    <p class="section-h3">How to enable each backend</p>
    <p><strong>llama.cpp server:</strong> run <code>llama-server -m your_model.gguf --port 8080</code></p>
    <p><strong>llama-cpp-python:</strong> <code>pip install llama-cpp-python</code> (requires build tools)</p>
  `;
  container.appendChild(help);
}

$('#btn-check-backends').onclick = checkBackends;

// ---------------------------------------------------------------- //
// Local model dropdown in Extract view
// ---------------------------------------------------------------- //
async function refreshLocalModelDropdown() {
  try {
    const res = await fetch('/api/models/local');
    const data = await res.json();
    const select = $('#local-model-select');
    if (!select) return;
    const current = select.value;
    select.innerHTML = '<option value="">— Select a downloaded model —</option>';
    for (const m of data.models) {
      const opt = document.createElement('option');
      opt.value = m.path;
      opt.textContent = `${m.filename} (${m.size_human})`;
      select.appendChild(opt);
    }
    if (current) select.value = current;
  } catch (e) { /* ignore — dropdown is optional */ }
}

$('#btn-refresh-local-models')?.addEventListener('click', refreshLocalModelDropdown);
$('#link-to-models')?.addEventListener('click', (e) => {
  e.preventDefault();
  showView('models');
  loadLocalModels();
  loadDownloads();
});

// When a local model is selected, fetch it as a File object
$('#local-model-select')?.addEventListener('change', async (e) => {
  const path = e.target.value;
  if (!path) return;
  // The server path is the absolute filesystem path. We need to fetch the file
  // from a new endpoint. For now, we'll use a fetch endpoint that streams the file.
  // Actually, simpler: extract requires a File upload. We can't easily set a File
  // from a path in the browser. Instead, we'll add a separate endpoint that
  // accepts a local path and runs extraction directly.
  // For now, we just show the selected path.
  const filename = path.split('/').pop();
  $('#selected-file').textContent = `✓ Selected local model: ${filename}`;
  selectedFile = null;  // clear any previously-dropped file
  selectedFilePath = path;  // store path for submission
  $('#btn-extract').disabled = false;
});
let selectedFilePath = null;

// Initial load of dropdown
refreshLocalModelDropdown();

// ---------------------------------------------------------------- //
// Extract
// ---------------------------------------------------------------- //
$('#btn-extract').onclick = async () => {
  if (!selectedFile && !selectedFilePath) return;

  const fd = new FormData();
  if (selectedFile) {
    fd.append('file', selectedFile);
  } else if (selectedFilePath) {
    // Use the local path endpoint
    fd.append('local_path', selectedFilePath);
  }
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
    if (!res.ok) throw new Error(data.detail || 'extraction failed to start');
    showView('job');
    pollJob(data.job_id);
  } catch (e) {
    alert('Failed to start extraction: ' + e.message);
    $('#btn-extract').disabled = false;
    $('#btn-extract').textContent = 'Extract knowledge →';
  }
};

// ---------------------------------------------------------------- //
// Job polling
// ---------------------------------------------------------------- //
function pollJob(jobId) {
  if (pollTimer) clearInterval(pollTimer);
  const fetchJob = async () => {
    const res = await fetch(`/api/jobs/${jobId}`);
    const job = await res.json();
    renderJob(job);
    if (job.status === 'completed' || job.status === 'failed') {
      clearInterval(pollTimer);
      pollTimer = null;
      $('#btn-extract').disabled = false;
      $('#btn-extract').textContent = 'Extract knowledge →';
    }
  };
  fetchJob();
  pollTimer = setInterval(fetchJob, 1500);
}

function renderJob(job) {
  const c = $('#job-content');
  const status = job.status;
  const progress = job.progress || {};

  let html = `
    <div class="card">
      <div class="job-header">
        <div>
          <h2 style="margin-bottom:4px;">Job ${job.id}</h2>
          <p class="hint">Model: ${job.gguf_filename}</p>
        </div>
        <span class="job-status ${status}">${status}</span>
      </div>
  `;

  if (status === 'running' || status === 'queued') {
    const pct = progress.total > 0 ? (progress.current / progress.total * 100) : 0;
    html += `
      <p><strong>${progress.message || 'starting...'}</strong></p>
      <div class="progress-bar"><div class="progress-bar-fill" style="width:${pct}%"></div></div>
      <p class="hint">${progress.current} / ${progress.total}</p>
    `;
  }

  if (status === 'failed') {
    html += `
      <div class="error-box">${job.error || 'Unknown error'}

${job.traceback || ''}</div>
    `;
  }

  if (status === 'completed') {
    const r = job.report_preview || {};
    const stats = r.stats || {};
    const md = r.metadata || {};

    // Stats grid
    html += `
      <div class="stats-grid">
        <div class="stat-block"><div class="stat-value">${formatNumber(md.vocab_size || 0)}</div><div class="stat-label">Vocab size</div></div>
        <div class="stat-block"><div class="stat-value">${md.block_count || 0}</div><div class="stat-label">Layers</div></div>
        <div class="stat-block"><div class="stat-value">${formatNumber(r.weight_inspection?.total_parameters || 0)}</div><div class="stat-label">Parameters</div></div>
        <div class="stat-block"><div class="stat-value">${stats.n_probes || 0}</div><div class="stat-label">Probes run</div></div>
        <div class="stat-block"><div class="stat-value">${stats.n_probes_passed || 0}</div><div class="stat-label">Probes passed</div></div>
        <div class="stat-block"><div class="stat-value">${stats.total_elapsed_seconds?.toFixed(1) || 0}s</div><div class="stat-label">Elapsed</div></div>
      </div>
    `;

    // Pack summaries
    if (r.pack_summaries && r.pack_summaries.length) {
      html += `<div class="section-h3">Probe Pack Results</div>`;
      html += `<table><thead><tr>
        <th>Pack</th><th>Category</th><th>Probes</th><th>Passed</th><th>Failed</th><th>Errors</th><th>Pass Rate</th>
      </tr></thead><tbody>`;
      r.pack_summaries.forEach(s => {
        html += `<tr>
          <td><strong>${s.pack_name}</strong></td>
          <td><span class="pack-category ${s.category}" style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:10px;color:white;">${s.category}</span></td>
          <td>${s.n_probes}</td>
          <td style="color:var(--success)">${s.n_passed}</td>
          <td style="color:var(--danger)">${s.n_failed}</td>
          <td>${s.n_errors}</td>
          <td><strong>${(s.pass_rate * 100).toFixed(1)}%</strong></td>
        </tr>`;
      });
      html += `</tbody></table>`;
    }

    // Concept mastery
    if (r.concepts && r.concepts.length) {
      html += `<div class="section-h3">Concept Mastery</div>`;
      html += `<table><thead><tr><th>Domain</th><th>Probes</th><th>Passed</th><th>Pass Rate</th><th>Mastery</th></tr></thead><tbody>`;
      r.concepts.forEach(c => {
        html += `<tr>
          <td><strong>${c.domain}</strong></td>
          <td>${c.n_probes}</td>
          <td>${c.n_passed}</td>
          <td>${(c.pass_rate * 100).toFixed(1)}%</td>
          <td><span class="badge ${c.mastery_level}">${c.mastery_level}</span></td>
        </tr>`;
      });
      html += `</tbody></table>`;
    }

    // Behavioral
    if (r.behavioral_profile) {
      const bp = r.behavioral_profile;
      html += `<div class="section-h3">Behavioral Profile</div>`;
      html += `<div class="stats-grid">
        <div class="stat-block"><div class="stat-value">${(bp.refusal_rate * 100).toFixed(1)}%</div><div class="stat-label">Refusal rate</div></div>
        <div class="stat-block"><div class="stat-value">${bp.n_refused}/${bp.n_refusal_probes}</div><div class="stat-label">Refused probes</div></div>
      </div>`;
      if (bp.detected_persona_excerpt) {
        html += `<p class="hint">Detected persona: "${bp.detected_persona_excerpt}"</p>`;
      }
    }

    // Calibration
    if (r.calibration) {
      const cal = r.calibration;
      html += `<div class="section-h3">Calibration & Hallucination</div>`;
      html += `<div class="stats-grid">`;
      if (cal.hallucination_ack_rate !== null && cal.hallucination_ack_rate !== undefined) {
        html += `<div class="stat-block"><div class="stat-value">${(cal.hallucination_ack_rate * 100).toFixed(1)}%</div><div class="stat-label">Hallucination ack rate</div></div>`;
      }
      if (cal.math_accuracy !== null && cal.math_accuracy !== undefined) {
        html += `<div class="stat-block"><div class="stat-value">${(cal.math_accuracy * 100).toFixed(1)}%</div><div class="stat-label">Math accuracy</div></div>`;
      }
      html += `</div>`;
    }

    // v2: Knowledge Attribution
    if (r.attribution && Object.keys(r.attribution).length > 0) {
      const a = r.attribution;
      const fp = a.knowledge_fingerprint || '';
      const brief = a.fingerprint_brief || {};
      const aStats = a.stats || {};
      html += `<div class="section-h3">v2: Knowledge Attribution (ROME/MEMIT-style)</div>`;

      // Fingerprint banner
      html += `<div style="background:linear-gradient(90deg,rgba(124,92,255,0.15),rgba(45,212,191,0.15));border:1px solid var(--accent);border-radius:8px;padding:14px;margin:12px 0;">
        <div style="font-size:11px;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.05em;">Knowledge Fingerprint</div>
        <div style="font-family:var(--mono);font-size:13px;color:var(--accent-2);word-break:break-all;margin-top:4px;">${fp}</div>
      </div>`;

      // Attribution stats
      html += `<div class="stats-grid">
        <div class="stat-block"><div class="stat-value">${aStats.n_layers_analyzed_mlp || 0}</div><div class="stat-label">MLP layers analyzed</div></div>
        <div class="stat-block"><div class="stat-value">${aStats.n_layers_analyzed_attn || 0}</div><div class="stat-label">Attn layers analyzed</div></div>
        <div class="stat-block"><div class="stat-value">${aStats.n_heads_total || 0}</div><div class="stat-label">Attention heads</div></div>
        <div class="stat-block"><div class="stat-value">${aStats.n_global_top_neurons || 0}</div><div class="stat-label">Top neurons extracted</div></div>
        <div class="stat-block"><div class="stat-value">${brief.strongest_layer !== null && brief.strongest_layer !== undefined ? 'L'+brief.strongest_layer : '?'}</div><div class="stat-label">Strongest layer</div></div>
        <div class="stat-block"><div class="stat-value">${brief.most_concentrated_layer !== null && brief.most_concentrated_layer !== undefined ? 'L'+brief.most_concentrated_layer : '?'}</div><div class="stat-label">Most concentrated</div></div>
      </div>`;

      // Top 5 neurons
      const top5 = brief.top_5_neurons || [];
      if (top5.length) {
        html += `<p style="margin-top:14px;font-size:13px;font-weight:600;color:var(--accent);">Top 5 Memory Neurons (across all layers)</p>`;
        html += `<table><thead><tr><th>Layer</th><th>Neuron</th><th>Strength</th><th>Top Activating Token</th></tr></thead><tbody>`;
        top5.forEach(n => {
          html += `<tr><td>L${n.layer}</td><td>N${n.neuron}</td><td>${(n.strength||0).toFixed(4)}</td><td><code>${(n.top_token||'').replace(/</g,'&lt;')}</code></td></tr>`;
        });
        html += `</tbody></table>`;
      }

      // Per-layer MLP summary
      const layers = (a.mlp_analysis && a.mlp_analysis.layers) || [];
      if (layers.length) {
        html += `<p style="margin-top:14px;font-size:13px;font-weight:600;color:var(--accent);">Per-Layer MLP Memory Map</p>`;
        html += `<table><thead><tr><th>Layer</th><th>Hidden</th><th>Gated</th><th>Neurons</th><th>Strength</th><th>Concentration</th><th>Top Token</th></tr></thead><tbody>`;
        layers.slice(0, 20).forEach(L => {
          const topNeurons = L.top_neurons || [];
          let topToken = '';
          if (topNeurons.length && topNeurons[0].top_activating_tokens && topNeurons[0].top_activating_tokens.length) {
            const t = topNeurons[0].top_activating_tokens[0];
            topToken = Array.isArray(t) ? t[0] : String(t);
          }
          html += `<tr><td>L${L.layer}</td><td>${L.hidden_dim}</td><td>${L.is_gated ? 'yes' : 'no'}</td><td>${L.n_neurons}</td><td>${(L.layer_strength||0).toFixed(2)}</td><td>${((L.concentration||0)*100).toFixed(1)}%</td><td><code>${String(topToken).replace(/</g,'&lt;').substring(0,30)}</code></td></tr>`;
        });
        if (layers.length > 20) {
          html += `<tr><td colspan="7" style="text-align:center;color:var(--text-dim);font-style:italic;">... and ${layers.length - 20} more layers (see JSON export)</td></tr>`;
        }
        html += `</tbody></table>`;
      }

      // Attention head specialization
      const attn = a.attention_analysis || {};
      const topCopy = attn.top_copy_heads || [];
      const topInd = attn.top_induction_heads || [];
      if (topCopy.length || topInd.length) {
        html += `<p style="margin-top:14px;font-size:13px;font-weight:600;color:var(--accent);">Attention Head Specialization</p>`;
        if (topCopy.length) {
          html += `<p style="font-size:12px;color:var(--text-dim);">Top Copy Heads (V·O ≈ identity — token-copying):</p>`;
          html += `<table><thead><tr><th>Layer</th><th>Head</th><th>Copy Score</th><th>Specialization</th></tr></thead><tbody>`;
          topCopy.slice(0, 8).forEach(h => {
            html += `<tr><td>L${h.layer}</td><td>H${h.head_index}</td><td>${(h.copy_score||0).toFixed(3)}</td><td>${(h.specialization||0).toFixed(3)}</td></tr>`;
          });
          html += `</tbody></table>`;
        }
        if (topInd.length) {
          html += `<p style="font-size:12px;color:var(--text-dim);margin-top:10px;">Top Induction Heads (low-rank Q·Q^T):</p>`;
          html += `<table><thead><tr><th>Layer</th><th>Head</th><th>Induction Score</th><th>Top SV</th></tr></thead><tbody>`;
          topInd.slice(0, 8).forEach(h => {
            html += `<tr><td>L${h.layer}</td><td>H${h.head_index}</td><td>${(h.induction_score||0).toFixed(3)}</td><td>${(h.top_singular_value||0).toFixed(2)}</td></tr>`;
          });
          html += `</tbody></table>`;
        }
      }

      // Per-fact attribution
      const factAttr = a.fact_attributions || [];
      if (factAttr.length) {
        html += `<p style="margin-top:14px;font-size:13px;font-weight:600;color:var(--accent);">Per-Fact Attribution</p>`;
        html += `<p class="hint">Each fact is attributed to the layer+neuron whose key vector is most strongly activated by tokens in the prompt.</p>`;
        html += `<table><thead><tr><th>Probe</th><th>Layer</th><th>Neuron</th><th>Confidence</th><th>Method</th></tr></thead><tbody>`;
        factAttr.slice(0, 25).forEach(f => {
          html += `<tr><td>${f.probe_id || ''}</td><td>${f.attributed_layer !== null ? 'L'+f.attributed_layer : '?'}</td><td>${f.attributed_neuron !== null ? 'N'+f.attributed_neuron : '?'}</td><td>${(f.attribution_confidence||0).toFixed(3)}</td><td><code>${f.method || ''}</code></td></tr>`;
        });
        if (factAttr.length > 25) {
          html += `<tr><td colspan="5" style="text-align:center;color:var(--text-dim);font-style:italic;">... and ${factAttr.length - 25} more (see JSON export)</td></tr>`;
        }
        html += `</tbody></table>`;
      }
    }

    // Downloads
    html += `<div class="section-h3">Download extracted knowledge</div>`;
    html += `<div class="download-grid">
      <a class="download-btn" href="/api/jobs/${job.id}/download/json"><span class="icon">📄</span><span class="label">JSON</span><span class="ext">.json</span></a>
      <a class="download-btn" href="/api/jobs/${job.id}/download/markdown"><span class="icon">📝</span><span class="label">Markdown</span><span class="ext">.md</span></a>
      <a class="download-btn" href="/api/jobs/${job.id}/download/graphml"><span class="icon">🕸</span><span class="label">GraphML</span><span class="ext">.graphml</span></a>
      <a class="download-btn" href="/api/jobs/${job.id}/download/turtle"><span class="icon">🁢</span><span class="label">RDF/Turtle</span><span class="ext">.ttl</span></a>
      <a class="download-btn" href="/api/jobs/${job.id}/download/sqlite"><span class="icon">🗄</span><span class="label">SQLite</span><span class="ext">.db</span></a>
    </div>`;
  }

  html += `</div>`;
  c.innerHTML = html;
}

function formatNumber(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(2) + 'K';
  return String(n);
}

// ---------------------------------------------------------------- //
// Jobs list
// ---------------------------------------------------------------- //
async function loadJobs() {
  const res = await fetch('/api/jobs');
  const jobs = await res.json();
  const container = $('#jobs-list');
  container.innerHTML = '';
  if (!jobs.length) {
    container.innerHTML = '<p class="hint">No jobs yet. Run an extraction first.</p>';
    return;
  }
  jobs.reverse().forEach(job => {
    const item = document.createElement('div');
    item.className = 'job-list-item';
    item.innerHTML = `
      <div>
        <strong>${job.gguf_filename}</strong>
        <p class="hint">Job ${job.id} · ${job.started_at}</p>
      </div>
      <span class="job-status ${job.status}">${job.status}</span>
    `;
    item.onclick = () => { showView('job'); pollJob(job.id); };
    container.appendChild(item);
  });
}

// ---------------------------------------------------------------- //
// Init
// ---------------------------------------------------------------- //
loadPacks();

// ---------------------------------------------------------------- //
// v3: Causal Trace
// ---------------------------------------------------------------- //
let traceFile = null;
const traceDz = $('#trace-dropzone');
const traceInput = $('#trace-file-input');

traceDz.onclick = () => traceInput.click();
traceDz.ondragover = (e) => { e.preventDefault(); traceDz.classList.add('drag-over'); };
traceDz.ondragleave = () => traceDz.classList.remove('drag-over');
traceDz.ondrop = (e) => {
  e.preventDefault();
  traceDz.classList.remove('drag-over');
  if (e.dataTransfer.files.length) handleTraceFile(e.dataTransfer.files[0]);
};
traceInput.onchange = (e) => { if (e.target.files.length) handleTraceFile(e.target.files[0]); };

function handleTraceFile(file) {
  if (!file.name.toLowerCase().endsWith('.gguf')) { alert('File must be .gguf'); return; }
  traceFile = file;
  $('#trace-selected-file').textContent = `✓ ${file.name} (${(file.size / 1e6).toFixed(2)} MB)`;
  $('#btn-trace').disabled = false;
}

$('#btn-trace').onclick = async () => {
  if (!traceFile) return;
  const fd = new FormData();
  fd.append('file', traceFile);
  fd.append('top_k', $('#trace-top-k').value);
  $('#btn-trace').disabled = true;
  $('#btn-trace').textContent = 'Starting...';
  try {
    const res = await fetch('/api/trace', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'trace failed to start');
    showView('job');
    pollJob(data.job_id);
  } catch (e) {
    alert('Failed: ' + e.message);
    $('#btn-trace').disabled = false;
    $('#btn-trace').textContent = 'Run causal trace →';
  }
};

// ---------------------------------------------------------------- //
// v3: ROME Edit
// ---------------------------------------------------------------- //
let editFile = null;
let editRequests = [];

const editDz = $('#edit-dropzone');
const editInput = $('#edit-file-input');
editDz.onclick = () => editInput.click();
editDz.ondragover = (e) => { e.preventDefault(); editDz.classList.add('drag-over'); };
editDz.ondragleave = () => editDz.classList.remove('drag-over');
editDz.ondrop = (e) => {
  e.preventDefault();
  editDz.classList.remove('drag-over');
  if (e.dataTransfer.files.length) handleEditFile(e.dataTransfer.files[0]);
};
editInput.onchange = (e) => { if (e.target.files.length) handleEditFile(e.target.files[0]); };

function handleEditFile(file) {
  if (!file.name.toLowerCase().endsWith('.gguf')) { alert('File must be .gguf'); return; }
  editFile = file;
  $('#edit-selected-file').textContent = `✓ ${file.name} (${(file.size / 1e6).toFixed(2)} MB)`;
  updateEditButton();
}

function initEditView() {
  if (editRequests.length === 0) {
    editRequests = [{ subject: '', prompt: '', target_object: '' }];
    renderEdits();
  }
}

$('#btn-add-edit').onclick = () => {
  editRequests.push({ subject: '', prompt: '', target_object: '' });
  renderEdits();
};

function renderEdits() {
  const container = $('#edits-list');
  container.innerHTML = '';
  editRequests.forEach((req, idx) => {
    const div = document.createElement('div');
    div.style.cssText = 'background:var(--bg-elev);border-radius:8px;padding:14px;margin-bottom:10px;';
    div.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <strong>Edit #${idx + 1}</strong>
        ${editRequests.length > 1 ? `<button class="btn-secondary" style="padding:4px 10px;font-size:11px;" data-rm="${idx}">Remove</button>` : ''}
      </div>
      <div class="form-row">
        <label>Subject<input type="text" data-field="subject" data-idx="${idx}" value="${req.subject.replace(/"/g, '&quot;')}" placeholder="e.g. The Eiffel Tower" /></label>
        <label>Target object<input type="text" data-field="target_object" data-idx="${idx}" value="${req.target_object.replace(/"/g, '&quot;')}" placeholder="e.g. Berlin" /></label>
      </div>
      <label style="display:flex;flex-direction:column;gap:4px;font-size:12px;color:var(--text-dim);margin-top:8px;">
        Prompt
        <input type="text" data-field="prompt" data-idx="${idx}" value="${req.prompt.replace(/"/g, '&quot;')}" placeholder="e.g. The Eiffel Tower is located in the city of" style="font-family:var(--mono);" />
      </label>
    `;
    container.appendChild(div);
  });
  // Wire up inputs
  container.querySelectorAll('input[data-field]').forEach(inp => {
    inp.oninput = (e) => {
      const idx = parseInt(e.target.dataset.idx);
      const field = e.target.dataset.field;
      editRequests[idx][field] = e.target.value;
      updateEditButton();
    };
  });
  container.querySelectorAll('button[data-rm]').forEach(btn => {
    btn.onclick = (e) => {
      const idx = parseInt(e.target.dataset.rm);
      editRequests.splice(idx, 1);
      renderEdits();
      updateEditButton();
    };
  });
}

function updateEditButton() {
  const hasFile = editFile !== null;
  const hasValidEdit = editRequests.some(r => r.subject && r.prompt && r.target_object);
  $('#btn-edit').disabled = !(hasFile && hasValidEdit);
}

$('#btn-edit').onclick = async () => {
  if (!editFile) return;
  const validEdits = editRequests.filter(r => r.subject && r.prompt && r.target_object);
  const fd = new FormData();
  fd.append('file', editFile);
  fd.append('edits_json', JSON.stringify(validEdits));
  $('#btn-edit').disabled = true;
  $('#btn-edit').textContent = 'Starting...';
  try {
    const res = await fetch('/api/edit', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'edit failed to start');
    showView('job');
    pollJob(data.job_id);
  } catch (e) {
    alert('Failed: ' + e.message);
    $('#btn-edit').disabled = false;
    $('#btn-edit').textContent = 'Apply ROME edits →';
  }
};

// ---------------------------------------------------------------- //
// v3: Compare
// ---------------------------------------------------------------- //
let compareFiles = [];
const cmpDz = $('#compare-dropzone');
const cmpInput = $('#compare-file-input');
cmpDz.onclick = () => cmpInput.click();
cmpDz.ondragover = (e) => { e.preventDefault(); cmpDz.classList.add('drag-over'); };
cmpDz.ondragleave = () => cmpDz.classList.remove('drag-over');
cmpDz.ondrop = (e) => {
  e.preventDefault();
  cmpDz.classList.remove('drag-over');
  for (const f of e.dataTransfer.files) handleCompareFile(f);
};
cmpInput.onchange = (e) => { for (const f of e.target.files) handleCompareFile(f); };

function handleCompareFile(file) {
  if (!file.name.toLowerCase().endsWith('.json')) return;
  if (compareFiles.find(f => f.name === file.name)) return;
  compareFiles.push(file);
  renderCompareFiles();
}

function renderCompareFiles() {
  const container = $('#compare-files-list');
  container.innerHTML = compareFiles.map((f, i) =>
    `<div style="display:flex;justify-content:space-between;align-items:center;padding:8px;background:var(--bg-elev);border-radius:6px;margin-bottom:6px;">
      <span style="font-family:var(--mono);font-size:12px;color:var(--accent-2);">📄 ${f.name}</span>
      <button class="btn-secondary" style="padding:4px 10px;font-size:11px;" data-rm="${i}">Remove</button>
    </div>`
  ).join('');
  container.querySelectorAll('button[data-rm]').forEach(btn => {
    btn.onclick = (e) => {
      compareFiles.splice(parseInt(e.target.dataset.rm), 1);
      renderCompareFiles();
      $('#btn-compare').disabled = compareFiles.length < 2;
    };
  });
  $('#btn-compare').disabled = compareFiles.length < 2;
}

$('#btn-compare').onclick = async () => {
  if (compareFiles.length < 2) return;
  const fd = new FormData();
  for (const f of compareFiles) fd.append('reports', f, f.name);
  $('#btn-compare').disabled = true;
  $('#btn-compare').textContent = 'Comparing...';
  try {
    const res = await fetch('/api/compare', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'compare failed');
    // Render immediately (compare is synchronous)
    showView('job');
    renderCompareResult(data.report);
  } catch (e) {
    alert('Failed: ' + e.message);
    $('#btn-compare').disabled = false;
    $('#btn-compare').textContent = 'Compare models →';
  }
};

function renderCompareResult(report) {
  const c = $('#job-content');
  let html = `<div class="card">
    <div class="job-header">
      <div>
        <h2 style="margin-bottom:4px;">Cross-Model Comparison</h2>
        <p class="hint">${report.n_models} models compared · ${report.pairwise.length} pairwise comparisons</p>
      </div>
      <span class="job-status completed">completed</span>
    </div>`;

  // Stats
  html += `<div class="stats-grid">
    <div class="stat-block"><div class="stat-value">${report.n_models}</div><div class="stat-label">Models</div></div>
    <div class="stat-block"><div class="stat-value">${report.pairwise.length}</div><div class="stat-label">Comparisons</div></div>
    <div class="stat-block"><div class="stat-value">${report.stats.n_identical_fingerprints}</div><div class="stat-label">Identical fingerprints</div></div>
    <div class="stat-block"><div class="stat-value">${report.stats.n_strong_lineage}</div><div class="stat-label">Strong lineage</div></div>
  </div>`;

  // Pairwise
  html += `<div class="section-h3">Pairwise Comparisons</div>`;
  for (const p of report.pairwise) {
    const scoreColor = p.lineage_score > 0.7 ? 'var(--success)' : p.lineage_score > 0.4 ? 'var(--warn)' : 'var(--danger)';
    html += `<div style="background:var(--bg-elev);border-radius:8px;padding:14px;margin-bottom:12px;border-left:4px solid ${scoreColor};">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <strong>${p.model_a}  ↔  ${p.model_b}</strong>
        <span style="font-family:var(--mono);font-size:18px;color:${scoreColor};font-weight:700;">${p.lineage_score.toFixed(3)}</span>
      </div>
      <div style="font-size:12px;color:var(--text);margin-bottom:8px;">${p.lineage_hypothesis}</div>
      <table style="font-size:11px;">
        <tr><td>Fingerprint match</td><td>${p.fingerprint_match ? '✓ IDENTICAL' : '✗ different'}</td></tr>
        <tr><td>Same architecture</td><td>${p.same_arch ? '✓' : '✗'}</td></tr>
        <tr><td>Top neuron Jaccard</td><td>${p.top_neuron_jaccard.toFixed(3)}</td></tr>
        <tr><td>Top token Jaccard</td><td>${p.top_token_jaccard.toFixed(3)}</td></tr>
        <tr><td>Concept mastery correlation</td><td>${p.concept_mastery_correlation.toFixed(3)}</td></tr>
        <tr><td>Behavioral similarity</td><td>${p.behavioral_similarity.toFixed(3)}</td></tr>
      </table>
    </div>`;
  }

  // Download
  html += `<div class="section-h3">Download</div>`;
  html += `<div class="download-grid">
    <a class="download-btn" href="/api/jobs/${report.job_id || ''}/download/json"><span class="icon">📄</span><span class="label">JSON</span><span class="ext">.json</span></a>
  </div>`;

  html += `</div>`;
  c.innerHTML = html;
}

// ---------------------------------------------------------------- //
// v4: Hugging Face Hub model browser
// ---------------------------------------------------------------- //
let downloadPollTimer = null;

function formatBytes(n) {
  if (!n) return '?';
  if (n >= 1e9) return (n / 1e9).toFixed(2) + ' GB';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + ' MB';
  if (n >= 1e3) return (n / 1e3).toFixed(2) + ' KB';
  return n + ' B';
}

function formatNumber(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

$('#btn-models-search').onclick = async () => {
  const q = $('#models-search-query').value.trim();
  if (!q) { alert('Enter a search query'); return; }
  $('#btn-models-search').textContent = 'Searching...';
  $('#btn-models-search').disabled = true;
  try {
    const res = await fetch(`/api/models/search?q=${encodeURIComponent(q)}&limit=30`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'search failed');
    renderSearchResults(data.results);
  } catch (e) {
    alert('Search failed: ' + e.message);
  } finally {
    $('#btn-models-search').textContent = 'Search';
    $('#btn-models-search').disabled = false;
  }
};

// Search on Enter key
$('#models-search-query').addEventListener('keypress', (e) => {
  if (e.key === 'Enter') $('#btn-models-search').click();
});

function renderSearchResults(results) {
  const container = $('#models-search-results');
  if (!results.length) {
    container.innerHTML = '<p class="hint">No models found.</p>';
    return;
  }
  container.innerHTML = `<p class="hint">${results.length} models found. Click a row to see files.</p>`;
  container.innerHTML += `<table><thead><tr>
    <th>Repo</th><th>Downloads</th><th>Likes</th><th>Tag</th><th>Last Modified</th>
  </tr></thead><tbody>`;
  for (const m of results) {
    const tags = (m.tags || []).slice(0, 3).join(', ');
    container.innerHTML += `<tr style="cursor:pointer;" data-repo="${m.repo_id}" class="search-row">
      <td><strong>${m.repo_id}</strong>${m.gated ? ' <span style="color:var(--warn);font-size:10px;">[GATED]</span>' : ''}</td>
      <td>${formatNumber(m.downloads)}</td>
      <td>${m.likes}</td>
      <td style="font-size:11px;color:var(--text-dim);">${tags}</td>
      <td style="font-size:11px;">${(m.last_modified || '').slice(0,10)}</td>
    </tr>`;
  }
  container.innerHTML += '</tbody></table>';
  // Click to load info
  container.querySelectorAll('.search-row').forEach(row => {
    row.onclick = () => {
      $('#models-info-repo').value = row.dataset.repo;
      $('#btn-models-info').click();
    };
  });
}

$('#btn-models-info').onclick = async () => {
  const repo = $('#models-info-repo').value.trim();
  if (!repo) { alert('Enter a repo ID'); return; }
  $('#btn-models-info').textContent = 'Loading...';
  $('#btn-models-info').disabled = true;
  try {
    const res = await fetch(`/api/models/info/${repo}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'info failed');
    renderModelInfo(data);
  } catch (e) {
    alert('Info failed: ' + e.message);
    $('#models-info-result').innerHTML = '';
  } finally {
    $('#btn-models-info').textContent = 'Get info';
    $('#btn-models-info').disabled = false;
  }
};

function renderModelInfo(info) {
  const container = $('#models-info-result');
  let html = `<div style="background:var(--bg-elev);border-radius:8px;padding:14px;">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
      <div>
        <strong style="font-size:14px;">${info.repo_id}</strong>
        ${info.gated ? '<span class="badge" style="background:var(--warn);color:black;margin-left:6px;">GATED</span>' : ''}
        <p class="hint" style="margin-top:4px;">by <strong>${info.author}</strong> · ${formatNumber(info.downloads)} downloads · ${info.likes} likes</p>
      </div>
      <div style="text-align:right;font-size:11px;color:var(--text-dim);">
        <div>${info.pipeline_tag || 'n/a'}</div>
        <div>Last modified: ${(info.last_modified || '').slice(0,10)}</div>
      </div>
    </div>`;
  if (info.tags && info.tags.length) {
    html += `<div style="margin-bottom:10px;">${info.tags.map(t => `<span style="display:inline-block;background:var(--bg-card);border:1px solid var(--border);border-radius:4px;padding:2px 8px;font-size:11px;margin:2px;">${t}</span>`).join('')}</div>`;
  }
  html += `<p style="margin:14px 0 6px;font-size:13px;font-weight:600;color:var(--accent);">Files</p>`;
  html += `<table><thead><tr><th>Filename</th><th>Size</th><th></th></tr></thead><tbody>`;
  for (const f of info.files) {
    if (f.is_gguf) {
      html += `<tr>
        <td><strong>[GGUF]</strong> ${f.filename}</td>
        <td>${f.size_human}</td>
        <td><button class="btn-primary" style="padding:4px 10px;font-size:11px;" data-dl-repo="${info.repo_id}" data-dl-file="${f.filename}">Download</button></td>
      </tr>`;
    } else {
      html += `<tr style="opacity:0.6;">
        <td>${f.filename}</td>
        <td>${f.size_human}</td>
        <td></td>
      </tr>`;
    }
  }
  html += `</tbody></table></div>`;
  container.innerHTML = html;
  // Wire download buttons
  container.querySelectorAll('button[data-dl-repo]').forEach(btn => {
    btn.onclick = async () => {
      const repo = btn.dataset.dlRepo;
      const file = btn.dataset.dlFile;
      btn.disabled = true;
      btn.textContent = 'Starting...';
      try {
        const fd = new FormData();
        fd.append('repo_id', repo);
        fd.append('filename', file);
        const res = await fetch('/api/models/download', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'download start failed');
        btn.textContent = 'Downloading...';
        startDownloadPolling();
      } catch (e) {
        alert('Download failed: ' + e.message);
        btn.disabled = false;
        btn.textContent = 'Download';
      }
    };
  });
}

async function loadLocalModels() {
  try {
    const res = await fetch('/api/models/local');
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'list failed');
    $('#models-local-dir').textContent = `Directory: ${data.models_dir}`;
    const container = $('#models-local-list');
    if (!data.models.length) {
      container.innerHTML = '<p class="hint">No local models yet. Use the search above to find and download GGUF models.</p>';
      return;
    }
    let html = `<table><thead><tr><th>Filename</th><th>Size</th><th>Source</th><th>Downloaded</th><th></th></tr></thead><tbody>`;
    for (const m of data.models) {
      html += `<tr>
        <td><strong>${m.filename}</strong></td>
        <td>${m.size_human}</td>
        <td style="font-size:11px;">${m.repo_id || 'unknown'}</td>
        <td style="font-size:11px;">${(m.downloaded_at || '').slice(0,16)}</td>
        <td><button class="btn-secondary" style="padding:4px 10px;font-size:11px;color:var(--danger);" data-del="${m.filename}">Delete</button></td>
      </tr>`;
    }
    html += '</tbody></table>';
    container.innerHTML = html;
    container.querySelectorAll('button[data-del]').forEach(btn => {
      btn.onclick = async () => {
        if (!confirm(`Delete ${btn.dataset.del}?`)) return;
        try {
          const res = await fetch(`/api/models/local/${encodeURIComponent(btn.dataset.del)}`, { method: 'DELETE' });
          if (!res.ok) throw new Error('delete failed');
          loadLocalModels();
        } catch (e) { alert('Delete failed: ' + e.message); }
      };
    });
  } catch (e) {
    $('#models-local-list').innerHTML = `<p style="color:var(--danger);">${e.message}</p>`;
  }
}

async function loadDownloads() {
  try {
    const res = await fetch('/api/models/downloads');
    const downloads = await res.json();
    renderDownloads(downloads);
    // If any are still downloading, keep polling
    if (downloads.some(d => d.status === 'downloading' || d.status === 'queued')) {
      startDownloadPolling();
    }
  } catch (e) {
    console.error('load downloads failed', e);
  }
}

function renderDownloads(downloads) {
  const container = $('#models-downloads-list');
  if (!downloads.length) {
    container.innerHTML = '<p class="hint">No downloads yet.</p>';
    return;
  }
  // Show most recent first
  const sorted = [...downloads].reverse();
  let html = `<table><thead><tr><th>Status</th><th>File</th><th>Progress</th><th>Speed</th><th>ETA</th></tr></thead><tbody>`;
  for (const d of sorted.slice(0, 10)) {
    const pct = d.percent.toFixed(1);
    const statusColor = d.status === 'completed' ? 'var(--success)' : d.status === 'failed' ? 'var(--danger)' : 'var(--accent)';
    html += `<tr>
      <td><span style="color:${statusColor};font-weight:600;">${d.status.toUpperCase()}</span></td>
      <td style="font-family:var(--mono);font-size:11px;">${d.filename}</td>
      <td>
        <div style="display:flex;align-items:center;gap:8px;">
          <div style="flex:1;height:6px;background:var(--bg-elev);border-radius:3px;overflow:hidden;min-width:100px;">
            <div style="height:100%;width:${pct}%;background:${statusColor};transition:width 0.3s;"></div>
          </div>
          <span style="font-size:11px;font-family:var(--mono);min-width:80px;">${formatBytes(d.bytes_downloaded)} / ${formatBytes(d.total_bytes)}</span>
        </div>
      </td>
      <td style="font-size:11px;">${d.speed_mbps > 0 ? d.speed_mbps.toFixed(1) + ' MB/s' : '-'}</td>
      <td style="font-size:11px;">${d.eta_seconds > 0 ? Math.round(d.eta_seconds) + 's' : '-'}</td>
    </tr>`;
  }
  html += '</tbody></table>';
  if (downloads.some(d => d.status === 'completed')) {
    html += '<p class="hint" style="margin-top:8px;">Completed downloads appear in the "Local Models" list below.</p>';
  }
  container.innerHTML = html;
}

function startDownloadPolling() {
  if (downloadPollTimer) return;
  downloadPollTimer = setInterval(async () => {
    try {
      const res = await fetch('/api/models/downloads');
      const downloads = await res.json();
      renderDownloads(downloads);
      // Also refresh local models when a download completes
      if (downloads.some(d => d.status === 'completed' && !d._seen)) {
        downloads.forEach(d => { if (d.status === 'completed') d._seen = true; });
        loadLocalModels();
      }
      // Stop polling when nothing is active
      if (!downloads.some(d => d.status === 'downloading' || d.status === 'queued')) {
        clearInterval(downloadPollTimer);
        downloadPollTimer = null;
        loadLocalModels();  // final refresh
      }
    } catch (e) { /* ignore */ }
  }, 1500);
}

// ---------------------------------------------------------------- //
// v5: GGUF Surgery
// ---------------------------------------------------------------- //
let surgeryFile = null;
let surgeryLocalPath = null;
let surgeryOps = [];

const surgeryDz = $('#surgery-dropzone');
const surgeryInput = $('#surgery-file-input');
surgeryDz.onclick = () => surgeryInput.click();
surgeryDz.ondragover = (e) => { e.preventDefault(); surgeryDz.classList.add('drag-over'); };
surgeryDz.ondragleave = () => surgeryDz.classList.remove('drag-over');
surgeryDz.ondrop = (e) => {
  e.preventDefault();
  surgeryDz.classList.remove('drag-over');
  if (e.dataTransfer.files.length) handleSurgeryFile(e.dataTransfer.files[0]);
};
surgeryInput.onchange = (e) => { if (e.target.files.length) handleSurgeryFile(e.target.files[0]); };

function handleSurgeryFile(file) {
  if (!file.name.toLowerCase().endsWith('.gguf')) { alert('File must be .gguf'); return; }
  surgeryFile = file;
  surgeryLocalPath = null;
  $('#surgery-local-select').value = '';
  $('#surgery-selected-file').textContent = `✓ ${file.name} (${formatBytes(file.size)})`;
  updateSurgeryButton();
}

async function refreshSurgeryLocalModels() {
  try {
    const res = await fetch('/api/models/local');
    const data = await res.json();
    const select = $('#surgery-local-select');
    select.innerHTML = '<option value="">— Select a local model —</option>';
    for (const m of data.models) {
      const opt = document.createElement('option');
      opt.value = m.path;
      opt.textContent = `${m.filename} (${m.size_human})`;
      select.appendChild(opt);
    }
  } catch (e) { /* ignore */ }
}

$('#surgery-local-select').onchange = (e) => {
  if (e.target.value) {
    surgeryLocalPath = e.target.value;
    surgeryFile = null;
    const filename = e.target.value.split('/').pop();
    $('#surgery-selected-file').textContent = `✓ Selected: ${filename}`;
    updateSurgeryButton();
  } else {
    surgeryLocalPath = null;
    $('#surgery-selected-file').textContent = '';
    updateSurgeryButton();
  }
};

const OP_TYPES = [
  {value: 'bake_system_prompt', label: 'Bake System Prompt'},
  {value: 'set_chat_template', label: 'Set Chat Template'},
  {value: 'inject_dataset', label: 'Inject Dataset (RAG-style)'},
  {value: 'add_token', label: 'Add New Token'},
  {value: 'add_steering_vector', label: 'Add Steering Vector'},
  {value: 'set_metadata', label: 'Set Metadata Field'},
  {value: 'remove_metadata', label: 'Remove Metadata Field'},
];

$('#btn-surgery-add-op').onclick = () => {
  surgeryOps.push({type: 'bake_system_prompt', fields: {}});
  renderSurgeryOps();
};

function renderSurgeryOps() {
  const container = $('#surgery-ops-list');
  container.innerHTML = '';
  surgeryOps.forEach((op, idx) => {
    const div = document.createElement('div');
    div.style.cssText = 'background:var(--bg-elev);border-radius:8px;padding:14px;margin-bottom:10px;';
    
    // Operation type selector
    let html = `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
      <select data-op-idx="${idx}" class="surgery-op-type" style="background:var(--bg-card);border:1px solid var(--border);border-radius:4px;padding:4px 8px;color:var(--text);font-size:12px;">`;
    for (const t of OP_TYPES) {
      html += `<option value="${t.value}" ${op.type === t.value ? 'selected' : ''}>${t.label}</option>`;
    }
    html += `</select>
      <button class="btn-secondary" style="padding:4px 10px;font-size:11px;color:var(--danger);" data-rm-op="${idx}">Remove</button>
    </div>`;
    
    // Template fields
    const template = $(`#surgery-op-templates [data-template="${op.type}"]`);
    if (template) {
      html += `<div class="surgery-op-fields" data-op-idx="${idx}">${template.innerHTML}</div>`;
    }
    
    div.innerHTML = html;
    container.appendChild(div);
  });
  
  // Wire up type selectors
  container.querySelectorAll('.surgery-op-type').forEach(sel => {
    sel.onchange = (e) => {
      const idx = parseInt(e.target.dataset.opIdx);
      surgeryOps[idx].type = e.target.value;
      surgeryOps[idx].fields = {};
      renderSurgeryOps();
    };
  });
  
  // Wire up remove buttons
  container.querySelectorAll('button[data-rm-op]').forEach(btn => {
    btn.onclick = (e) => {
      const idx = parseInt(e.target.dataset.rmOp);
      surgeryOps.splice(idx, 1);
      renderSurgeryOps();
      updateSurgeryButton();
    };
  });
  
  // Wire up field inputs
  container.querySelectorAll('.surgery-op-fields').forEach(fieldContainer => {
    const idx = parseInt(fieldContainer.dataset.opIdx);
    fieldContainer.querySelectorAll('[data-field]').forEach(inp => {
      const fieldName = inp.dataset.field;
      if (surgeryOps[idx].fields[fieldName] !== undefined) {
        inp.value = surgeryOps[idx].fields[fieldName];
      }
      inp.oninput = (e) => {
        surgeryOps[idx].fields[fieldName] = e.target.value;
        updateSurgeryButton();
      };
    });
  });
}

function updateSurgeryButton() {
  const hasModel = surgeryFile || surgeryLocalPath;
  const hasOps = surgeryOps.length > 0;
  $('#btn-surgery-run').disabled = !(hasModel && hasOps);
}

$('#btn-surgery-run').onclick = async () => {
  // Build operations JSON from the form
  const operations = surgeryOps.map(op => {
    const result = {op: op.type};
    const fields = op.fields;
    if (op.type === 'bake_system_prompt') {
      result.prompt = fields.prompt || '';
    } else if (op.type === 'set_chat_template') {
      result.template = fields.template || '';
    } else if (op.type === 'inject_dataset') {
      result.name = fields.name || 'dataset';
      result.description = fields.description || '';
      try { result.data = JSON.parse(fields.data || '[]'); }
      catch (e) { result.data = []; }
    } else if (op.type === 'add_token') {
      result.token = fields.token || '';
      if (fields.embedding && fields.embedding.trim()) {
        result.embedding = fields.embedding.split(',').map(parseFloat);
      }
    } else if (op.type === 'add_steering_vector') {
      result.layer = parseInt(fields.layer) || 0;
      result.name = fields.name || 'default';
      result.strength = parseFloat(fields.strength) || 1.0;
      result.vector = fields.vector.split(',').map(parseFloat);
    } else if (op.type === 'set_metadata') {
      result.key = fields.key || '';
      result.value = fields.value || '';
      if (fields.value_type === 'int') result.value = parseInt(result.value);
      else if (fields.value_type === 'float') result.value = parseFloat(result.value);
      else if (fields.value_type === 'bool') result.value = result.value.toLowerCase() === 'true';
    } else if (op.type === 'remove_metadata') {
      result.key = fields.key || '';
    }
    return result;
  });

  const fd = new FormData();
  if (surgeryFile) {
    fd.append('file', surgeryFile);
  } else if (surgeryLocalPath) {
    fd.append('local_path', surgeryLocalPath);
  }
  fd.append('operations_json', JSON.stringify(operations));

  $('#btn-surgery-run').disabled = true;
  $('#btn-surgery-run').textContent = 'Running...';

  try {
    const url = surgeryLocalPath ? '/api/surgery-local' : '/api/surgery';
    const res = await fetch(url, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'surgery failed to start');
    showView('job');
    pollJob(data.job_id);
  } catch (e) {
    alert('Failed: ' + e.message);
    $('#btn-surgery-run').disabled = false;
    $('#btn-surgery-run').textContent = 'Run surgery →';
  }
};
