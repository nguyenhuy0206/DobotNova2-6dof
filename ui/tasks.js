/* Manual tasks: raw D455 capture, height geometry, and the live Debug executor. */
(() => {
  'use strict';

  const byId = (id) => document.getElementById(id);
  if (!byId('page-tasks')) return;
  const arm = byId('skArm');
  const canvas = byId('tasksCanvas');
  const context = canvas.getContext('2d');
  const state = { busy: false, image: null, location: null };
  const esc = (value) => String(value).replace(/[&<>"']/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[char]));
  const fmt = (value) => Array.isArray(value) ? value.map(fmt).join(', ')
    : typeof value === 'number' ? value.toFixed(2) : value ?? 'Unavailable';

  // Add another template here. A card with fields also gets an editable motion
  // form; its endpoint, slot title and numeric arguments are described together.
  const coordinates = [
    { key: 'x', label: 'X (mm)', required: true },
    { key: 'y', label: 'Y (mm)', required: true },
    { key: 'top_z', label: 'Top Z (mm)', required: true },
  ];
  const cards = [
    { id: 'locate', title: 'Locate (D455)', icon: 'cam', target: 'tasksCaptureCard',
      description: 'Capture raw colour, then click to measure a top face using D455 depth.' },
    { id: 'pick', title: 'Pick object', slotTitle: 'Object to pick', button: 'Pick', icon: 'hand',
      description: 'Open, approach, descend, grasp and lift the reviewed object.',
      fields: [...coordinates, { key: 'yaw_deg', label: 'Yaw (degrees)', required: true },
        { key: 'height_mm', label: 'Object height (mm)', min: 0.01, placeholder: 'Default: 26' }] },
    { id: 'place', title: 'Place object', slotTitle: 'Place target', button: 'Place', icon: 'target',
      description: 'Lower the held object onto the support top, release and lift.',
      fields: [...coordinates, { key: 'yaw_deg', label: 'Yaw (degrees)', placeholder: 'Keep current yaw' },
        { key: 'held_height_mm', label: 'Held object height (mm)', min: 0.01, placeholder: 'Default: 26' }] },
    { id: 'insert', title: 'Insertion (needle)', slotTitle: 'Needle insertion', button: 'Insert', icon: 'sliders',
      slotDesc: 'Straight along the gripper’s own +Z (straight down when Rx=180) at the slow insert speed. Aim the gripper first.',
      description: 'Push straight along the gripper axis by a depth, at the slow insert speed.',
      fields: [{ key: 'depth_mm', label: 'Depth (mm)', required: true, min: 0.01, max: 60 },
        { key: 'retract', label: 'Retract to the start after', type: 'checkbox' }] },
  ];
  const motionCards = cards.filter((card) => card.fields);

  function buildCards() {
    byId('tasksTemplates').innerHTML = cards.map((card) => `<div class="template-card">
      <div class="ticon"><svg class="icon lg"><use href="#ico-${esc(card.icon)}"/></svg></div>
      <h3>${esc(card.title)}</h3><p>${esc(card.description)}</p>
      <button class="btn sm" data-task-card="${esc(card.id)}" ${card.disabled ? 'disabled' : ''}>
        ${card.disabled ? 'Coming soon' : 'Open controls'}</button></div>`).join('');
    byId('tasksSlots').innerHTML = motionCards.map((card) => `<form class="card tasks-slot" id="tasks-${card.id}">
      <div class="card-head"><div><div class="card-title">${esc(card.slotTitle)}</div>
      <div class="card-desc">${esc(card.slotDesc || (card.id === 'place' ? 'Top Z is the support surface, not the held object.' : 'Top Z is the object’s top face.'))}</div></div></div>
      <div class="card-pad" style="padding-top:0"><p class="small-caption" data-slot-source>Use a locate result or enter coordinates by hand.</p>
      <fieldset><div class="formgrid">${card.fields.map((field) => field.type === 'checkbox'
        ? `<label class="form-label"><input type="checkbox" name="${esc(field.key)}"> ${esc(field.label)}</label>`
        : `<label class="form-label">${esc(field.label)}${field.required ? ' *' : ''}
        <input class="field" type="number" step="any" name="${esc(field.key)}"
          ${field.required ? 'required' : ''} ${field.min === undefined ? '' : `min="${field.min}"`}
          ${field.max === undefined ? '' : `max="${field.max}"`}
          placeholder="${esc(field.placeholder || 'Required')}"></label>`).join('')}</div></fieldset>
      <div class="tasks-actions"><button class="btn primary" type="submit" data-motion disabled>${esc(card.button)}</button></div></div></form>`).join('');
    motionCards.forEach((card) => {
      const form = byId(`tasks-${card.id}`);
      form.addEventListener('input', updateControls);
      form.addEventListener('submit', (event) => { event.preventDefault(); runMotion(card); });
    });
    byId('tasksTemplates').addEventListener('click', (event) => {
      const button = event.target.closest('[data-task-card]');
      const card = cards.find((item) => item.id === button?.dataset.taskCard);
      if (!card || card.disabled) return;
      const target = byId(card.target || `tasks-${card.id}`);
      target.scrollIntoView({ behavior: 'smooth', block: 'center' });
      target.querySelector('input, button')?.focus({ preventScroll: true });
    });
  }

  function updateControls() {
    const armed = arm?.checked === true;
    byId('tasksSafety').textContent = armed ? 'Workspace clear is ON.' : 'Motion disabled.';
    byId('tasksCapture').disabled = state.busy;
    byId('tasksUseObject').disabled = state.busy || !state.location?.ok;
    byId('tasksUseTarget').disabled = state.busy || !state.location?.ok;
    motionCards.forEach((card) => {
      const form = byId(`tasks-${card.id}`);
      // Check validity before disabling the fieldset (disabled inputs are exempt).
      const valid = card.fields.every((field) => {
        const input = form.elements.namedItem(field.key);
        if (field.type === 'checkbox') return true;
        return (!field.required && input.value === '') ||
          (input.value !== '' && Number.isFinite(input.valueAsNumber) &&
            (field.min === undefined || input.valueAsNumber >= field.min) &&
            (field.max === undefined || input.valueAsNumber <= field.max));
      });
      form.querySelector('[data-motion]').disabled = !armed || state.busy || !valid;
      form.querySelector('fieldset').disabled = state.busy;
    });
  }

  function setBusy(value) { state.busy = value; updateControls(); }
  function status(message, error = false) {
    byId('tasksStatus').textContent = message;
    byId('tasksStatus').classList.toggle('tasks-error', error);
  }

  async function post(path, body) {
    const response = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
  }

  function resetLocation(message) {
    state.location = null;
    byId('tasksLocateResult').textContent = message;
    updateControls();
  }

  async function capture() {
    if (state.busy) return;
    setBusy(true);
    state.image = null;
    canvas.hidden = true;
    byId('tasksFrameEmpty').hidden = false;
    resetLocation('Click the new frame to locate an object or surface.');
    status('Capturing env…');
    let url;
    try {
      const response = await fetch('/api/tasks/env/frame.jpg', { cache: 'no-store' });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || `HTTP ${response.status}`);
      }
      url = URL.createObjectURL(await response.blob());
      const image = new Image();
      image.src = url;
      await image.decode();
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      context.drawImage(image, 0, 0);
      state.image = image;
      canvas.hidden = false;
      byId('tasksFrameEmpty').hidden = true;
      byId('tasksFrameInfo').textContent = `${canvas.width} × ${canvas.height} full-image pixels · captured ${new Date().toLocaleTimeString()} · click a top face. Existing slots stay until replaced.`;
      status('Env captured. Click to locate.');
    } catch (error) {
      byId('tasksFrameInfo').textContent = 'No frame available. Capture env again when the depth helper is ready.';
      resetLocation('Capture env successfully before locating.');
      status(error.message, true);
    } finally {
      if (url) URL.revokeObjectURL(url);
      setBusy(false);
    }
  }

  function showLocation(result) {
    state.location = result.ok ? result : null;
    if (!result.ok) {
      byId('tasksLocateResult').textContent = result.error || 'Locate failed.';
      return;
    }
    const rows = [['top_center_mm', result.top_center_mm], ['height_mm', result.height_mm],
      ['footprint_mm', result.footprint_mm], ['yaw_deg', result.yaw_deg], ['kind', result.kind]];
    const warnings = result.quality || [];
    byId('tasksLocateResult').innerHTML = `<dl class="tasks-readout">${rows.map(([key, value]) =>
      `<div><dt>${esc(key)}</dt><dd>${esc(fmt(value))}</dd></div>`).join('')}</dl>`
      + (warnings.length ? `<div class="tasks-warning"><strong>Quality warnings — review before motion</strong><ul>${warnings.map((warning) => `<li>${esc(warning)}</li>`).join('')}</ul></div>`
        : '<p class="small-caption">No quality warnings reported.</p>');
  }

  async function locate(event) {
    if (state.busy || !state.image) return;
    const bounds = canvas.getBoundingClientRect();
    // The canvas has no border or padding; scale CSS coordinates to native pixels.
    const pixel = [Math.min(canvas.width - 1, Math.max(0, Math.floor((event.clientX - bounds.left) * canvas.width / bounds.width))),
      Math.min(canvas.height - 1, Math.max(0, Math.floor((event.clientY - bounds.top) * canvas.height / bounds.height)))];
    setBusy(true);
    resetLocation('Locating…');
    context.drawImage(state.image, 0, 0);
    context.strokeStyle = '#ffcc38';
    context.lineWidth = 2;
    context.beginPath();
    context.arc(pixel[0], pixel[1], 8, 0, Math.PI * 2);
    context.moveTo(pixel[0] - 12, pixel[1]); context.lineTo(pixel[0] + 12, pixel[1]);
    context.moveTo(pixel[0], pixel[1] - 12); context.lineTo(pixel[0], pixel[1] + 12);
    context.stroke();
    status(`Locating full-image pixel [${pixel.join(', ')}]…`);
    try {
      const result = await post('/api/tasks/locate', { pixel });
      showLocation(result);
      status(result.ok ? `Located at [${pixel.join(', ')}]. Review values and warnings.` : result.error || 'Locate failed.', !result.ok);
    } catch (error) {
      resetLocation(error.message);
      status(error.message, true);
    } finally { setBusy(false); }
  }

  function useLocation(id) {
    const result = state.location;
    if (state.busy || !result?.ok) return;
    const form = byId(`tasks-${id}`);
    const values = { x: result.top_center_mm[0], y: result.top_center_mm[1], top_z: result.top_center_mm[2],
      yaw_deg: result.yaw_deg };
    if (id === 'pick') values.height_mm = result.height_mm;
    Object.entries(values).forEach(([key, value]) => {
      form.elements.namedItem(key).value = typeof value === 'number' && Number.isFinite(value) ? value : '';
    });
    // Target height describes its support, never the height of the held object.
    form.querySelector('[data-slot-source]').textContent = `${result.kind} · footprint ${fmt(result.footprint_mm)} mm`
      + (result.quality?.length ? ` · Warnings: ${result.quality.join('; ')}` : ' · No quality warnings');
    form.querySelector('[data-slot-source]').classList.toggle('tasks-warning', !!result.quality?.length);
    updateControls();
  }

  function showExecution(envelope) {
    const result = envelope.result && typeof envelope.result === 'object' ? envelope.result : envelope;
    const ok = envelope.ok && result.ok !== false;
    byId('tasksExecution').innerHTML = `<p><span class="chip ${ok ? 'green' : 'red'}">${ok ? 'OK' : 'FAIL'}</span> ${esc(envelope.action || 'Task')}</p>`
      + (result.failed_step ? `<p class="tasks-error">Failed step: ${esc(result.failed_step)}</p>` : '')
      + `<p class="tasks-detail ${ok ? '' : 'tasks-error'}">${esc(result.detail || envelope.error || '')}</p>`
      + `<ol class="tasks-steps">${(result.steps || []).map((step) => `<li><span class="chip ${step.ok ? 'green' : 'red'}">${step.ok ? 'OK' : 'FAIL'}</span>${esc(step.phase || step.action)} · ${esc(step.tool || step.action)}${step.error ? `<div class="tasks-error">${esc(step.error)}</div>` : ''}</li>`).join('')}</ol>`;
  }

  async function runMotion(card) {
    if (state.busy || arm?.checked !== true) return;
    const form = byId(`tasks-${card.id}`);
    if (!form.reportValidity()) return;
    const args = {};
    card.fields.forEach((field) => {
      const input = form.elements.namedItem(field.key);
      if (field.type === 'checkbox') args[field.key] = input.checked;
      else if (input.value !== '') args[field.key] = input.valueAsNumber;
    });
    setBusy(true);
    byId('tasksExecution').textContent = `${card.button} running…`;
    status(`${card.button} running…`);
    try {
      const result = await post(`/api/tasks/${card.id}`, { ...args, confirm: arm.checked === true });
      showExecution(result);
      status(result.ok ? `${card.button} completed. Capture again after motion to refresh the scene.`
        : result.result?.detail || result.error || `${card.button} failed.`, !result.ok);
      if (card.id === 'pick' && result.ok) {
        byId('tasks-place').elements.namedItem('held_height_mm').value = args.height_mm ?? 26;
      }
    } catch (error) {
      showExecution({ ok: false, action: card.id, error: error.message });
      status(error.message, true);
    } finally { setBusy(false); }
  }

  byId('tasksStop').addEventListener('click', async () => {
    try {
      const result = await post('/api/skills/run', { name: 'stop', args: {} });
      status(result.ok ? 'STOP sent to the robot.' : result.error || 'STOP failed.', !result.ok);
    } catch (error) { status(`STOP failed: ${error.message}`, true); }
  });
  buildCards();
  byId('tasksCapture').addEventListener('click', capture);
  canvas.addEventListener('click', locate);
  byId('tasksUseObject').addEventListener('click', () => useLocation('pick'));
  byId('tasksUseTarget').addEventListener('click', () => useLocation('place'));
  arm?.addEventListener('change', updateControls);
  updateControls();
})();
