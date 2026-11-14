/* Camera windows on the Debug tab.
 *
 * Talks to ui/camera_api.py:
 *   GET  /api/cameras                 status of every configured camera (config/robot.yaml `cameras:`)
 *   GET  /api/cameras/<name>/stream   MJPEG, used directly as <img src>
 *   POST /api/cameras/<name>/capture  save frame + robot pose to data/captures/
 *   POST /api/cameras/<name>/device   { source, index } — remembered in data/memory/cameras.json
 *
 * Streams run only while the Debug tab is visible and the window isn't paused,
 * so the server releases the cameras when nobody is watching.
 * Uses notify() / eventLog() from the inline script in index.html.
 */
(() => {
  'use strict';

  const STATUS_POLL_MS = 1500;
  const MAX_INDEX = 5;               // OpenCV device numbers offered in the dropdown
  const BLANK = 'data:image/gif;base64,R0lGODlhAQABAAAAACw=';

  const container = document.getElementById('skCams');
  const badge = document.getElementById('skCamBadge');
  const debugPage = document.getElementById('page-debug');
  if (!container || !debugPage) return;

  /** name -> { root, img, overlay, stat, select, pauseBtn, paused, streaming } */
  const windows = new Map();

  const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const visible = () => debugPage.classList.contains('active') && !document.hidden;

  async function api(path, body) {
    const res = await fetch(path, body === undefined ? {} : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  }

  // ---------------------------------------------------------------- build windows
  function deviceOptions(cam) {
    const opts = [];
    for (let i = 0; i <= MAX_INDEX; i += 1) {
      opts.push(`<option value="opencv:${i}">Device ${i}</option>`);
    }
    opts.push('<option value="realsense:0">RealSense (colour + depth)</option>');
    return opts.join('');
  }

  function buildWindow(cam) {
    const root = document.createElement('section');
    root.className = 'sk-cam';
    root.innerHTML = `
      <header><strong>${esc(cam.label)}</strong><span class="chip">—</span></header>
      <div class="sk-view"><img alt="${esc(cam.label)} live view"><div class="sk-paused">PAUSED</div></div>
      <footer>
        <select class="field" title="Which camera feeds this window">${deviceOptions(cam)}</select>
        <button class="btn sm" data-act="pause">Pause</button>
        <button class="btn sm" data-act="capture">Capture</button>
        <span class="sk-camstat">—</span>
      </footer>`;
    const w = {
      root,
      img: root.querySelector('img'),
      overlay: root.querySelector('.sk-paused'),
      chip: root.querySelector('header .chip'),
      stat: root.querySelector('.sk-camstat'),
      select: root.querySelector('select'),
      pauseBtn: root.querySelector('[data-act="pause"]'),
      paused: false,
      streaming: false,
    };
    w.select.value = `${cam.source}:${cam.source === 'realsense' ? 0 : cam.index}`;
    w.select.addEventListener('change', () => selectDevice(cam.name, w));
    root.querySelector('footer').addEventListener('click', (e) => {
      const act = e.target.closest('button')?.dataset.act;
      if (act === 'pause') togglePause(cam.name, w);
      if (act === 'capture') capture(cam.name);
    });
    windows.set(cam.name, w);
    return root;
  }

  // ---------------------------------------------------------------- streaming on/off
  function startStream(name, w) {
    if (w.streaming) return;
    w.img.src = `/api/cameras/${encodeURIComponent(name)}/stream?t=${Date.now()}`;
    w.streaming = true;
    w.overlay.style.display = 'none';
  }

  function stopStream(w, label = 'PAUSED') {
    if (w.streaming) w.img.src = BLANK;     // aborts the MJPEG connection
    w.streaming = false;
    w.overlay.textContent = label;
    w.overlay.style.display = 'grid';
  }

  function syncStreams() {
    windows.forEach((w, name) => {
      if (visible() && !w.paused) startStream(name, w);
      else stopStream(w, w.paused ? 'PAUSED' : 'HIDDEN');
    });
  }

  function togglePause(name, w) {
    w.paused = !w.paused;
    w.pauseBtn.textContent = w.paused ? 'Resume' : 'Pause';
    syncStreams();
  }

  // ---------------------------------------------------------------- actions
  async function selectDevice(name, w) {
    const [source, index] = w.select.value.split(':');
    stopStream(w, 'SWITCHING…');
    try {
      await api(`/api/cameras/${encodeURIComponent(name)}/device`, { source, index: Number(index) });
      eventLog('INFO', `Camera ${name} → ${source} #${index}`);
    } catch (err) {
      notify(`Camera ${name}: ${err.message}`);
    } finally {
      syncStreams();
    }
  }

  async function capture(name) {
    try {
      const rec = await api(`/api/cameras/${encodeURIComponent(name)}/capture`, {});
      eventLog('INFO', `Captured ${rec.image}`);
      notify(`Saved ${rec.image}${rec.pose ? ' with robot pose' : ''}`);
    } catch (err) {
      notify(`Capture ${name}: ${err.message}`);
    }
  }

  // ---------------------------------------------------------------- status
  function renderStatus(cams) {
    let live = 0;
    cams.forEach((cam) => {
      const w = windows.get(cam.name);
      if (!w) return;
      const ok = cam.has_frame && !cam.error;
      if (ok) live += 1;
      w.chip.className = `chip ${ok ? 'green' : cam.error ? 'red' : 'amber'}`;
      w.chip.textContent = ok ? 'LIVE' : cam.error ? 'NO SIGNAL' : cam.running ? 'STARTING' : 'IDLE';
      w.stat.textContent = cam.error ? cam.error
        : cam.resolution ? `${cam.resolution[0]}×${cam.resolution[1]} · ${cam.fps}/${cam.target_fps} fps` : '—';
      w.stat.title = w.stat.textContent;
    });
    badge.className = `badge ${live === cams.length ? 'green' : live ? 'orange' : 'muted'}`;
    badge.textContent = `${live}/${cams.length} LIVE`;
  }

  async function pollStatus() {
    if (visible()) {
      try {
        renderStatus((await api('/api/cameras')).cameras);
      } catch (err) {
        badge.className = 'badge red';
        badge.textContent = 'SERVER OFFLINE';
      }
    }
    setTimeout(pollStatus, STATUS_POLL_MS);
  }

  // ---------------------------------------------------------------- boot
  async function init() {
    try {
      const { cameras } = await api('/api/cameras');
      if (!cameras.length) {
        container.innerHTML = '<div class="sk-line">No cameras configured in config/robot.yaml.</div>';
        return;
      }
      cameras.forEach((cam) => container.appendChild(buildWindow(cam)));
      renderStatus(cameras);
    } catch (err) {
      container.innerHTML = `<div class="sk-line">Cameras unavailable: ${esc(err.message)}</div>`;
      return;
    }
    // Start/stop streams when the Debug tab is shown/hidden or the window is hidden.
    new MutationObserver(syncStreams).observe(debugPage, { attributes: true, attributeFilter: ['class'] });
    document.addEventListener('visibilitychange', syncStreams);
    syncStreams();
    pollStatus();
  }

  init();
})();
