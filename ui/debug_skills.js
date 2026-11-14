/* Live Robot · Primitive Skills panel (Debug tab).
 *
 * Talks to ui/skills_api.py:
 *   GET  /api/skills        catalogue (rendered as forms below)
 *   GET  /api/skills/state  live robot + gripper state (polled)
 *   POST /api/skills/run    { name, args, confirm }
 *
 * Uses notify() and eventLog() defined by the inline script in index.html.
 * Everything else is scoped to this file.
 */
(() => {
  'use strict';

  const POLL_ACTIVE_MS = 600;     // Debug tab visible
  const POLL_IDLE_MS = 3000;      // other tab / window hidden
  const AXES = ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz'];
  // Dobot V3 RobotMode values
  const MODES = { 1: 'INIT', 2: 'BRAKE OPEN', 3: 'POWER OFF', 4: 'DISABLED', 5: 'ENABLED', 6: 'DRAG',
    7: 'RUNNING', 8: 'RECORDING', 9: 'ALARM', 10: 'PAUSED', 11: 'JOG' };

  const byId = (id) => document.getElementById(id);
  const ui = {
    panel: document.querySelector('.sk-panel'),
    conn: byId('skConn'), stop: byId('skStop'), arm: byId('skArm'),
    speed: byId('skSpeed'), speedVal: byId('skSpeedVal'),
    force: byId('skForce'), forceVal: byId('skForceVal'),
    gripPos: byId('skGripPos'), gripPosVal: byId('skGripPosVal'), gripSet: byId('skGripSet'),
    frame: byId('skFrame'), stepMm: byId('skStepMm'), stepDeg: byId('skStepDeg'),
    cartJog: byId('skCartJog'), jointJog: byId('skJointJog'),
    flags: byId('skFlags'), pose: byId('skPose'), joints: byId('skJoints'), gripper: byId('skGripper'),
    skills: byId('skSkills'), result: byId('skResult'), reload: byId('skReload'), save: byId('skSaveBtn'),
  };
  if (!ui.panel) return;

  const state = { catalogue: new Map(), live: null, busy: false, polling: false };

  // ---------------------------------------------------------------- helpers
  const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (n, d = 1) => (typeof n === 'number' ? n.toFixed(d) : '—');
  const isArmed = () => ui.arm.checked;
  const debugVisible = () => byId('page-debug')?.classList.contains('active') && !document.hidden;

  async function api(path, body) {
    const res = await fetch(path, body === undefined ? {} : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  }

  function setBusy(busy) {
    state.busy = busy;
    ui.panel.classList.toggle('busy', busy);
  }

  function showResult(obj, isError) {
    ui.result.textContent = typeof obj === 'string' ? obj : JSON.stringify(obj, null, 2);
    ui.result.classList.toggle('error', !!isError);
  }

  /** Fill in panel-wide defaults (speed slider, force slider) the skill accepts but the caller omitted. */
  function withDefaults(spec, args) {
    const props = spec?.parameters?.properties || {};
    const out = { ...args };
    const isGripperSkill = 'force' in props;
    if (isGripperSkill) {
      if (out.force === undefined) out.force = Number(ui.force.value);
    } else if ('speed' in props && out.speed === undefined) {
      out.speed = Number(ui.speed.value);
    }
    return out;
  }

  // ---------------------------------------------------------------- run a skill
  async function runSkill(name, args = {}) {
    const spec = state.catalogue.get(name);
    if (!spec) { notify(`Unknown skill ${name} — reload the skill list`); return; }
    if (spec.moves_robot && !isArmed()) { notify('Turn on "Workspace clear" first'); return; }
    if (state.busy) { notify('A skill is still running — wait, or press STOP'); return; }

    const finalArgs = withDefaults(spec, args);
    setBusy(true);
    eventLog('SKILL', `${name}(${JSON.stringify(finalArgs)})`);
    try {
      const result = await api('/api/skills/run', { name, args: finalArgs, confirm: isArmed() });
      showResult(result, !result.ok);
      if (result.ok) {
        eventLog('SKILL', `${name} ✓`);
      } else {
        eventLog('WARN', `${name}: ${result.error}`);
        notify(`${name}: ${result.error}`);
      }
    } catch (err) {
      showResult(err.message, true);
      eventLog('WARN', `${name}: ${err.message}`);
      notify(`${name}: ${err.message}`);
    } finally {
      setBusy(false);
      refreshState();
    }
  }

  async function emergencyStop() {
    // Bypasses busy/armed checks on purpose: STOP must always go through.
    eventLog('WARN', 'STOP pressed');
    try {
      const result = await api('/api/skills/run', { name: 'stop', args: {} });
      showResult(result, !result.ok);
      notify(result.ok ? 'Robot stopped' : `STOP failed: ${result.error}`);
    } catch (err) {
      showResult(err.message, true);
      notify(`STOP failed: ${err.message}`);
    }
  }

  async function systemCommand(action) {
    if (action === 'enable' && !isArmed()) { notify('Turn on "Workspace clear" first'); return; }
    eventLog(action === 'enable' ? 'WARN' : 'INFO', `${action} robot`);
    try {
      const result = await api('/api/skills/system', { action, confirm: isArmed() });
      showResult(result);
      notify(`${action}: ${result.result}`);
    } catch (err) {
      showResult(err.message, true);
      eventLog('WARN', `${action}: ${err.message}`);
      notify(`${action}: ${err.message}`);
    } finally {
      refreshState();
    }
  }

  // ---------------------------------------------------------------- live state
  function readout(labels, values, digits) {
    return labels.map((label, i) => `<div><span>${esc(label)}</span><strong>${fmt(values?.[i], digits)}</strong></div>`).join('');
  }

  function renderState(s, error) {
    state.live = s;
    if (error) {
      ui.conn.className = 'badge red';
      ui.conn.textContent = 'ROBOT OFFLINE';
      ui.conn.title = error;
      ui.flags.innerHTML = `<span class="chip red">${esc(error)}</span>`;
      ui.pose.innerHTML = readout(AXES, null);
      ui.joints.innerHTML = readout(AXES.map((_, i) => `J${i + 1}`), null);
      return;
    }
    ui.conn.className = s.error ? 'badge red' : s.moving ? 'badge blue' : 'badge green';
    ui.conn.textContent = s.error ? 'ALARM' : s.moving ? 'MOVING' : 'ONLINE';
    ui.conn.title = '';
    const chip = (ok, yes, no, warn = 'amber') => `<span class="chip ${ok ? 'green' : warn}">${ok ? yes : no}</span>`;
    ui.flags.innerHTML = [
      `<span class="chip ${s.mode === 5 ? 'green' : 'amber'}">MODE ${esc(s.mode)} · ${esc(MODES[s.mode] || '?')}</span>`,
      chip(s.enabled, 'ENABLED', 'DISABLED'),
      chip(!s.error, 'NO ALARM', 'ALARM', 'red'),
      chip(!s.moving, 'IDLE', 'MOVING', 'blue'),
      s.motion_locked ? '<span class="chip red">MOTION LOCK</span>' : '',
    ].join('');
    ui.pose.innerHTML = readout(['X mm', 'Y mm', 'Z mm', 'Rx °', 'Ry °', 'Rz °'], s.pose, 1);
    ui.joints.innerHTML = readout(AXES.map((_, i) => `J${i + 1} °`), s.joints_deg, 2);
    if (s.gripper) renderGripper(s.gripper);
  }

  function renderGripper(g) {
    ui.gripper.innerHTML = `Position <strong>${esc(g.position)}</strong> · ${esc(g.grip_status || g.grip_state)}`
      + (g.grip_state === 2 ? ' · <strong>holding object</strong>' : '');
  }

  async function refreshState() {
    try {
      renderState(await api('/api/skills/state'));
    } catch (err) {
      renderState(null, err.message);
    }
  }

  async function pollLoop() {
    if (debugVisible() && !state.polling) {
      state.polling = true;
      await refreshState();
      state.polling = false;
    }
    setTimeout(pollLoop, debugVisible() ? POLL_ACTIVE_MS : POLL_IDLE_MS);
  }

  // ---------------------------------------------------------------- quick controls
  function buildJogButtons() {
    const cart = [['X', 'dx'], ['Y', 'dy'], ['Z', 'dz']];
    ui.cartJog.innerHTML = cart.map(([label, key]) =>
      `<button class="btn sk-motion" data-cart="${key}" data-sign="-1">−${label}</button>`
      + `<button class="btn sk-motion" data-cart="${key}" data-sign="1">+${label}</button>`).join('');
    ui.jointJog.innerHTML = [1, 2, 3, 4, 5, 6].map((j) =>
      `<button class="btn sk-motion" data-joint="${j}" data-sign="-1">J${j} −</button>`
      + `<button class="btn sk-motion" data-joint="${j}" data-sign="1">J${j} +</button>`).join('');
  }

  function bindQuickControls() {
    ui.arm.addEventListener('change', () => {
      ui.panel.classList.toggle('armed', isArmed());
      eventLog(isArmed() ? 'WARN' : 'INFO', isArmed() ? 'Motion ENABLED from Debug panel' : 'Motion disabled');
    });
    ui.speed.addEventListener('input', () => { ui.speedVal.textContent = `${ui.speed.value}%`; });
    ui.force.addEventListener('input', () => { ui.forceVal.textContent = `${ui.force.value}%`; });
    ui.gripPos.addEventListener('input', () => { ui.gripPosVal.textContent = ui.gripPos.value; });

    ui.stop.addEventListener('click', emergencyStop);
    ui.gripSet.addEventListener('click', () => runSkill('set_gripper', { position: Number(ui.gripPos.value) }));
    ui.save.addEventListener('click', () => {
      const name = prompt('Save current position as:');
      if (name && name.trim()) runSkill('save_position', { name: name.trim() });
    });

    // One delegated handler for every data-* button in the quick-control area.
    ui.panel.querySelector('.sk-grid').addEventListener('click', (e) => {
      const btn = e.target.closest('button');
      if (!btn) return;
      const { skill, args, cart, joint, sign, system } = btn.dataset;
      if (system) {
        systemCommand(system);
      } else if (skill) {
        runSkill(skill, args ? JSON.parse(args) : {});
      } else if (cart) {
        const step = Number(ui.stepMm.value) * Number(sign);
        runSkill('move_relative', { [cart]: step, frame: ui.frame.value });
      } else if (joint) {
        runSkill('jog_joint', { joint: Number(joint), degrees: Number(ui.stepDeg.value) * Number(sign) });
      }
    });

    // Esc = STOP, from anywhere on the Debug tab.
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && byId('page-debug')?.classList.contains('active')) emergencyStop();
    });
  }

  // ---------------------------------------------------------------- generated skill forms
  function fieldHtml(skill, key, schema, required) {
    const id = `sk-${skill}-${key}`;
    const label = `${esc(key)}${required ? ' *' : ''}`;
    const range = [schema.minimum, schema.maximum].filter((v) => v !== undefined).join('…');
    if (schema.type === 'array') {
      const n = schema.maxItems || 6;
      const inputs = Array.from({ length: n }, (_, i) =>
        `<input class="field" type="number" step="any" data-array="${key}" data-index="${i}" placeholder="${i + 1}">`).join('');
      return `<label class="wide">${label}<div class="sk-array">${inputs}</div></label>`;
    }
    if (schema.type === 'boolean') {
      return `<label>${label}<input type="checkbox" class="switch" id="${id}" data-key="${key}" data-type="boolean"></label>`;
    }
    if (schema.enum) {
      const opts = schema.enum.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join('');
      return `<label>${label}<select class="field" id="${id}" data-key="${key}" data-type="string">${opts}</select></label>`;
    }
    const type = schema.type === 'string' ? 'text' : 'number';
    const step = schema.type === 'integer' ? '1' : 'any';
    const wide = schema.type === 'string' ? ' class="wide"' : '';
    return `<label${wide}>${label}<input class="field" id="${id}" type="${type}" step="${step}" `
      + `data-key="${key}" data-type="${schema.type}" placeholder="${esc(range || (required ? 'required' : 'optional'))}"></label>`;
  }

  function skillCardHtml(spec) {
    const props = spec.parameters?.properties || {};
    const required = new Set(spec.parameters?.required || []);
    const fields = Object.entries(props).map(([k, s]) => fieldHtml(spec.name, k, s, required.has(k))).join('');
    const canFillPose = ['x', 'y', 'z', 'rx', 'ry', 'rz'].every((k) => k in props);
    const canFillJoints = 'joints' in props;
    const tag = spec.read_only ? '<span class="chip green">READ</span>'
      : spec.moves_robot ? '<span class="chip amber">MOTION</span>' : '<span class="chip blue">ACTION</span>';
    return `<article class="sk-skill" data-name="${esc(spec.name)}">
      <header><code>${esc(spec.name)}</code>${tag}</header>
      <p>${esc(spec.description)}</p>
      ${fields ? `<div class="sk-fields">${fields}</div>` : ''}
      <footer>
        <button class="btn sm primary ${spec.moves_robot ? 'sk-motion' : ''}" data-run>Run</button>
        ${canFillPose ? '<button class="btn sm" data-fill="pose">Use current pose</button>' : ''}
        ${canFillJoints ? '<button class="btn sm" data-fill="joints">Use current joints</button>' : ''}
      </footer>
    </article>`;
  }

  /** Read a generated form back into an args object; blank optional fields are omitted. */
  function collectArgs(card) {
    const args = {};
    card.querySelectorAll('[data-key]').forEach((input) => {
      const { key, type } = input.dataset;
      if (type === 'boolean') { if (input.checked) args[key] = true; return; }
      if (input.value === '') return;
      args[key] = type === 'integer' ? parseInt(input.value, 10)
        : type === 'number' ? parseFloat(input.value) : input.value;
    });
    const arrays = {};
    card.querySelectorAll('[data-array]').forEach((input) => {
      (arrays[input.dataset.array] ||= [])[Number(input.dataset.index)] = input.value === '' ? null : parseFloat(input.value);
    });
    Object.entries(arrays).forEach(([key, values]) => {
      if (values.some((v) => v !== null)) args[key] = values;
    });
    return args;
  }

  function fillFromLive(card, what) {
    const live = state.live;
    if (!live) { notify('No live robot state yet'); return; }
    if (what === 'pose') {
      ['x', 'y', 'z', 'rx', 'ry', 'rz'].forEach((k, i) => {
        card.querySelector(`[data-key="${k}"]`).value = live.pose[i].toFixed(2);
      });
    } else {
      card.querySelectorAll('[data-array="joints"]').forEach((input, i) => {
        input.value = live.joints_deg[i].toFixed(2);
      });
    }
  }

  async function loadSkills() {
    try {
      const data = await api('/api/skills');
      state.catalogue = new Map(data.skills.map((s) => [s.name, s]));
      ui.skills.innerHTML = data.skills.map(skillCardHtml).join('');
      const d = data.defaults || {};
      if (d.speed) { ui.speed.value = d.speed; ui.speedVal.textContent = `${d.speed}%`; }
      if (d.gripper_force) { ui.force.value = d.gripper_force; ui.forceVal.textContent = `${d.gripper_force}%`; }
    } catch (err) {
      ui.skills.innerHTML = `<div class="sk-line">Could not load skills: ${esc(err.message)}. Is ui/app.py running?</div>`;
    }
  }

  function bindSkillForms() {
    ui.skills.addEventListener('click', (e) => {
      const btn = e.target.closest('button');
      const card = e.target.closest('.sk-skill');
      if (!btn || !card) return;
      if (btn.dataset.fill) fillFromLive(card, btn.dataset.fill);
      else if ('run' in btn.dataset) runSkill(card.dataset.name, collectArgs(card));
    });
    ui.reload.addEventListener('click', loadSkills);
  }

  // ---------------------------------------------------------------- boot
  // Deep link: http://127.0.0.1:8765/#debug opens this tab directly.
  if (location.hash.length > 1 && typeof setPage === 'function') setPage(location.hash.slice(1));

  buildJogButtons();
  bindQuickControls();
  bindSkillForms();
  renderState(null, 'waiting for first update');
  loadSkills();
  pollLoop();
})();
