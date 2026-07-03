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
// Extract
// ---------------------------------------------------------------- //
$('#btn-extract').onclick = async () => {
  if (!selectedFile) return;

  const fd = new FormData();
  fd.append('file', selectedFile);
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
    const res = await fetch('/api/extract', { method: 'POST', body: fd });
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
