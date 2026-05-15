import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Three.js scene setup ────────────────────────────────────────────────────
const viewerEl = document.getElementById('viewer');
const placeholder = document.getElementById('viewer-placeholder');

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x0f1117);
viewerEl.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 5000);
camera.position.set(0, -80, 40);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

function resize() {
  const w = viewerEl.clientWidth, h = viewerEl.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewerEl);
resize();

(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
})();

// ── PLY parser (ASCII only — matches our exporter) ──────────────────────────
function parsePly(text) {
  const lines = text.split('\n');
  let headerEnd = 0;
  let vertexCount = 0;
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i].trim();
    if (l.startsWith('element vertex')) vertexCount = parseInt(l.split(' ')[2]);
    if (l === 'end_header') { headerEnd = i + 1; break; }
  }

  const positions = new Float32Array(vertexCount * 3);
  const colors    = new Float32Array(vertexCount * 3);

  for (let i = 0; i < vertexCount; i++) {
    const parts = lines[headerEnd + i].trim().split(' ');
    positions[i * 3]     = parseFloat(parts[0]);
    positions[i * 3 + 1] = parseFloat(parts[1]);
    positions[i * 3 + 2] = parseFloat(parts[2]);
    colors[i * 3]        = parseInt(parts[3]) / 255;
    colors[i * 3 + 1]    = parseInt(parts[4]) / 255;
    colors[i * 3 + 2]    = parseInt(parts[5]) / 255;
  }
  return { positions, colors };
}

function displayPointCloud(text) {
  // Remove previous cloud
  scene.children.filter(c => c.isPoints).forEach(c => scene.remove(c));
  placeholder.style.display = 'none';

  const { positions, colors } = parsePly(text);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('color',    new THREE.BufferAttribute(colors, 3));
  geo.computeBoundingSphere();

  const mat = new THREE.PointsMaterial({ size: 0.05, vertexColors: true, sizeAttenuation: true });
  const cloud = new THREE.Points(geo, mat);
  scene.add(cloud);

  // Re-centre camera
  const centre = geo.boundingSphere.center;
  const r = geo.boundingSphere.radius;
  controls.target.copy(centre);
  camera.position.set(centre.x, centre.y - r * 1.5, centre.z + r * 0.8);
  controls.update();
}

// ── Upload / job polling ─────────────────────────────────────────────────────
let selectedFile = null;
let currentJobId = null;
let pollTimer = null;

const dropZone   = document.getElementById('drop-zone');
const fileInput  = document.getElementById('file-input');
const uploadBtn  = document.getElementById('upload-btn');
const downloadBtn = document.getElementById('download-btn');
const statusRows = document.getElementById('status-rows');

function setStatus(rows) {
  statusRows.innerHTML = rows.map(([k, v, cls]) =>
    `<div class="status-row"><span>${k}</span><span class="${cls || ''}">${v}</span></div>`
  ).join('');
}

function selectFile(file) {
  if (!file || !file.name.toLowerCase().endsWith('.e57')) {
    alert('Please select an .e57 file');
    return;
  }
  selectedFile = file;
  uploadBtn.disabled = false;
  setStatus([['File', file.name], ['Size', (file.size / 1e6).toFixed(1) + ' MB']]);
}

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => selectFile(e.target.files[0]));
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  selectFile(e.dataTransfer.files[0]);
});

uploadBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  uploadBtn.disabled = true;
  downloadBtn.style.display = 'none';

  const fd = new FormData();
  fd.append('file', selectedFile);

  try {
    const res = await fetch('/api/upload/e57', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    currentJobId = data.job_id;
    setStatus([['Job', currentJobId.slice(0, 8) + '…'], ['Status', '<span class="badge badge-queued">queued</span>']]);
    startPolling();
  } catch (err) {
    setStatus([['Error', err.message]]);
    uploadBtn.disabled = false;
  }
});

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(pollJob, 2000);
}

async function pollJob() {
  try {
    const res = await fetch(`/api/jobs/${currentJobId}`);
    const job = await res.json();

    const badgeClass = `badge badge-${job.status}`;
    const rows = [
      ['File', job.filename || '—'],
      ['Status', `<span class="${badgeClass}">${job.status}</span>`],
    ];

    if (job.status === 'done' && job.summary) {
      const s = job.summary;
      if (s.cylinder_radius_m) rows.push(['Radius', s.cylinder_radius_m.toFixed(2) + ' m']);
      if (s.cylinder_height_m) rows.push(['Height', s.cylinder_height_m.toFixed(2) + ' m']);
      Object.entries(s.point_counts || {}).forEach(([k, v]) => rows.push([k, v.toLocaleString()]));
    }
    if (job.status === 'error') rows.push(['Detail', job.error || 'unknown error']);

    setStatus(rows);

    if (job.status === 'done') {
      clearInterval(pollTimer);
      uploadBtn.disabled = false;
      downloadBtn.style.display = 'block';
      loadPointCloud();
    } else if (job.status === 'error') {
      clearInterval(pollTimer);
      uploadBtn.disabled = false;
    }
  } catch (err) {
    console.error('Poll error', err);
  }
}

async function loadPointCloud() {
  const res = await fetch(`/api/jobs/${currentJobId}/download`);
  const text = await res.text();
  displayPointCloud(text);
}

downloadBtn.addEventListener('click', () => {
  if (!currentJobId) return;
  window.location.href = `/api/jobs/${currentJobId}/download`;
});
