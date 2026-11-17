/*
 * Side panels: robot dashboard, missions & score, setup, help.
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};
  const geom = FLL.geom;
  const field = FLL.field;

  const esc = (s) => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', '\'': '&#39;' }[c]));
  const $ = (id) => document.getElementById(id);
  const compassOf = (theta) => (Math.round(90 - geom.deg(theta)) % 360 + 360) % 360;
  const thetaOf = (compass) => geom.rad(90 - compass);

  class Panels {
    constructor(app) {
      this.app = app;
      this.active = 'robot';
      this.openMissions = new Set();
    }

    init() {
      document.querySelectorAll('#panelTabs .ptab[data-panel]').forEach(b => b.addEventListener('click', () => {
        // clicking the open tab again hides the panel so the map gets all the space
        if (b.dataset.panel === this.active && !this.app.panelsCollapsed) { this.app.setPanelsCollapsed(true); return; }
        if (this.app.panelsCollapsed) this.app.setPanelsCollapsed(false);
        this.show(b.dataset.panel);
      }));
      this.buildRobot();
      this.buildMissions();
      this.buildSetup();
      this.buildHelp();
      this.show('robot');
    }

    show(name) {
      this.active = name;
      document.querySelectorAll('#panelTabs .ptab[data-panel]').forEach(b => b.classList.toggle('active', b.dataset.panel === name));
      document.querySelectorAll('#panelArea .panel').forEach(p => p.classList.toggle('active', p.id === 'panel-' + name));
      if (name === 'missions') this.updateMissions(true);
      if (name === 'setup') this.syncSetup();
    }

    // ------------------------------------------------------------ robot dashboard
    buildRobot() {
      const el = $('panel-robot');
      el.innerHTML = `
        <div class="cards">
          <div class="card">
            <h3>Hub</h3>
            <div class="hub">
              <div class="hub-face">
                <div class="matrix" id="hubMatrix">${'<div></div>'.repeat(25)}</div>
                <div class="hub-buttons">
                  <button class="hub-btn" id="hubLeft" title="Left button (hold)">◀</button>
                  <div class="center-btn" id="hubCenter" title="Center button light"></div>
                  <button class="hub-btn" id="hubRight" title="Right button (hold)">▶</button>
                </div>
              </div>
              <div>
                <div class="kv">
                  <span class="k">Yaw (blocks)</span><span class="v" id="rYawBlocks">0°</span>
                  <span class="k">tilt_angles()[0]</span><span class="v" id="rYawPy">0</span>
                  <span class="k">Pybricks heading</span><span class="v" id="rHeadingPb">0°</span>
                  <span class="k">Program time</span><span class="v" id="rTime">0.0 s</span>
                </div>
                <div class="hub-text" id="hubText"></div>
              </div>
            </div>
          </div>
          <div class="card">
            <h3>Position on the mat</h3>
            <div class="kv">
              <span class="k">x (from left)</span><span class="v" id="rX">0 mm</span>
              <span class="k">y (from bottom)</span><span class="v" id="rY">0 mm</span>
              <span class="k">Facing</span><span class="v" id="rHead">0°</span>
              <span class="k">Grid cell</span><span class="v" id="rCell">-</span>
              <span class="k">Speed</span><span class="v" id="rSpeed">0 mm/s</span>
              <span class="k">Area</span><span class="v" id="rArea">-</span>
              <span class="k">Carrying</span><span class="v" id="rCargo">nothing</span>
            </div>
            <div class="small-note">Facing: 0° = towards the back wall (north), 90° = east. Drag the robot on the field to move it, drag the ⟳ handle to turn it.</div>
          </div>
          <div class="card">
            <h3>Sensors</h3>
            <div class="kv" id="rSensors"></div>
          </div>
          <div class="card">
            <h3>Motors</h3>
            <table class="motors-table">
              <thead><tr><th>Port</th><th>Use</th><th>Rel. pos</th><th>Abs</th><th>Speed</th></tr></thead>
              <tbody id="rMotors"></tbody>
            </table>
          </div>
        </div>`;
      const hold = (btnId, key) => {
        const b = $(btnId);
        const down = (e) => { e.preventDefault(); this.app.world.robot.hub.buttons[key] = performance.now(); b.classList.add('down'); };
        const up = () => { this.app.world.robot.hub.buttons[key] = 0; b.classList.remove('down'); };
        b.addEventListener('pointerdown', down);
        b.addEventListener('pointerup', up);
        b.addEventListener('pointerleave', up);
      };
      hold('hubLeft', 'left');
      hold('hubRight', 'right');
      this.matrixCells = Array.from($('hubMatrix').children);
    }

    updateRobot() {
      if (this.active !== 'robot') return;
      const app = this.app;
      const world = app.world;
      const rb = world.robot;
      const now = world.time;
      const m = rb.hub.matrix;
      for (let i = 0; i < 25; i++) {
        const v = m[i] || 0;
        this.matrixCells[i].style.background = v > 0 ? 'rgba(255,255,255,' + (0.25 + 0.75 * v / 100) + ')' : '#3a3a3a';
        this.matrixCells[i].style.boxShadow = v > 0 ? '0 0 6px rgba(255,255,255,' + (v / 140) + ')' : 'none';
      }
      $('hubCenter').style.background = FLL.COLOR_CSS[rb.hub.centerColor] || '#fff';
      $('hubText').textContent = rb.hub.text ? '“' + rb.hub.text + '”' : '';
      const yawCCW = geom.wrapDeg(rb.yawCCW(now));
      $('rYawBlocks').textContent = Math.trunc(-yawCCW) + '°';
      $('rYawPy').textContent = Math.round(yawCCW * (rb.config.hubYawCW ? -1 : 1) * 10) || 0;
      $('rHeadingPb').textContent = (-rb.yawCCW(now)).toFixed(1) + '°';
      const runner = app.runner;
      $('rTime').textContent = runner && runner.state !== 'idle' ? ((world.time - runner.startTime) / 1000).toFixed(1) + ' s' : '–';
      const p = rb.pose;
      $('rX').textContent = Math.round(p.x) + ' mm';
      $('rY').textContent = Math.round(p.y) + ' mm';
      $('rHead').textContent = compassOf(p.theta) + '°';
      $('rCell').textContent = field.cellName(p.x, p.y) || 'off the mat';
      $('rSpeed').textContent = Math.round(rb.linearSpeed || 0) + ' mm/s';
      const body = rb.bodyPoly();
      let area = 'on the field';
      if (field.polyInLaunchArea('left', body)) area = 'in the LEFT launch area ✔';
      else if (field.polyInLaunchArea('right', body)) area = 'in the RIGHT launch area ✔';
      else if (field.polyInHome('left', body)) area = 'in the left home';
      else if (field.polyInHome('right', body)) area = 'in the right home';
      $('rArea').textContent = area;
      const cargo = world.cargo();
      $('rCargo').textContent = cargo.length ? cargo.map(c => c.label || c.kind).join(', ') : 'nothing';

      // sensors
      const rows = [];
      for (const port of FLL.robot.PORTS) {
        const d = rb.device(port);
        if (d.type === 'color') {
          const w = rb.sensorWorld(port);
          const rgb = world.colorUnder(w[0], w[1]);
          const c = FLL.api.spike.classifyColor(rgb);
          const name = c < 0 ? 'no color (-1)' : FLL.api.spike.COLOR_NAMES[c] + ' (' + c + ')';
          rows.push(`<span class="k">${port} color</span><span class="v"><span class="swatch" style="background:rgb(${rgb.map(Math.round).join(',')})"></span>${esc(name)}</span>`);
          rows.push(`<span class="k">${port} reflection</span><span class="v">${FLL.api.spike.reflectionOf(rgb)} %</span>`);
        } else if (d.type === 'distance') {
          const dist = world.rayDistance(rb.sensorWorld(port), rb.pose.theta + geom.rad(d.angle || 0), 2100);
          rows.push(`<span class="k">${port} distance</span><span class="v">${dist < 40 || dist > 2000 ? '-1 (nothing)' : Math.round(dist) + ' mm'}</span>`);
        } else if (d.type === 'force') {
          const pr = world.bumperPressed(port);
          rows.push(`<span class="k">${port} force</span><span class="v">${pr ? 'pressed (6.0 N)' : 'released'}</span>`);
        }
      }
      rows.push(`<span class="k">Touching</span><span class="v">${rb.contacts && rb.contacts.length ? esc(rb.contacts.join(', ')) : '–'}</span>`);
      $('rSensors').innerHTML = rows.join('');

      const mrows = [];
      for (const port of FLL.robot.PORTS) {
        const mo = rb.motors[port];
        if (!mo) continue;
        const use = port === rb.config.driveLeft ? 'left wheel' : (port === rb.config.driveRight ? 'right wheel' : 'attachment');
        mrows.push(`<tr><td>${port}</td><td class="lbl">${use}</td><td>${Math.round(mo.pos)}°</td><td>${Math.round(mo.absolutePosition)}°</td><td>${Math.round(mo.speed)}°/s</td></tr>`);
      }
      $('rMotors').innerHTML = mrows.join('') || '<tr><td colspan="5" class="lbl muted">no motors</td></tr>';
    }

    // ------------------------------------------------------------ missions & score
    buildMissions() {
      const el = $('panel-missions');
      el.innerHTML = `
        <div class="score-head">
          <div><div class="muted">Total score</div><div class="total" id="mTotal">0</div></div>
          <div class="hint">Points update live while the robot works. Missions marked with 🚫 have the <b>No Equipment Constraint</b>: the robot must not touch that model at the end.
          Click a mission for details and simulator tips.</div>
        </div>
        <div class="mission-list" id="mList"></div>
        <h3 style="margin:14px 0 6px;font-size:13px;color:#6a737d;text-transform:uppercase">Mission log</h3>
        <div class="log" id="mLog"></div>`;
      $('mList').addEventListener('click', (e) => {
        const head = e.target.closest('.mhead');
        if (!head) return;
        const id = Number(head.dataset.id);
        if (this.openMissions.has(id)) this.openMissions.delete(id); else this.openMissions.add(id);
        this.updateMissions(true);
      });
    }

    updateMissions(force) {
      const world = this.app.world;
      const score = world.score();
      const key = JSON.stringify(score.rows.map(r => [r.id, r.points, r.note])) + '|' + Array.from(this.openMissions).join(',') + '|' + world.events.length;
      if (!force && key === this.lastMissionKey) return score;
      this.lastMissionKey = key;
      if (this.active !== 'missions' && !force) return score;
      $('mTotal').textContent = score.total;
      const html = [];
      for (const row of score.rows) {
        const m = world.missions.find(x => x.id === row.id);
        const def = m ? m.def : null;
        const open = this.openMissions.has(row.id);
        const cls = 'mission' + (row.points > 0 ? ' scored' : '') + (open ? ' open' : '');
        const num = row.id === 0 ? '✓' : (row.id === 99 ? '●' : String(row.id).padStart(2, '0'));
        const noEq = FLL.missions.NO_EQUIPMENT[row.id] ? ' 🚫' : '';
        const dock = def && def.dockModel ? (' · ' + (FLL.missions.dockOf(row.id, world.setup) || '?') + ' dock') : '';
        html.push(`<div class="${cls}">
          <div class="mhead" data-id="${row.id}"><span class="num">${num}</span><span class="mname">${esc(row.name)}${noEq}<span class="muted" style="font-weight:400">${esc(dock)}</span></span><span class="pts">${row.points}</span></div>
          <div class="mbody">
            ${def ? `<div class="desc">${esc(def.description)}</div>` : ''}
            <ul class="items">${row.items.map(it => `<li class="${it.points > 0 ? 'got' : ''}"><span>${esc(it.label)}</span><span class="p">${it.points}</span></li>`).join('')}</ul>
            ${row.note ? `<div class="note">${esc(row.note)}</div>` : ''}
            ${def && def.simHint ? `<div class="simhint">🤖 In the simulator: ${esc(def.simHint)}</div>` : ''}
          </div></div>`);
      }
      $('mList').innerHTML = html.join('');
      const log = world.events.slice(-60).reverse().map(e => `<div><span class="t">${(e.time / 1000).toFixed(1)}s</span>${e.mission ? 'M' + String(e.mission).padStart(2, '0') + ': ' : ''}${esc(e.text)}</div>`);
      $('mLog').innerHTML = log.join('') || '<div class="muted">Nothing happened yet. Run a program!</div>';
      return score;
    }

    // ------------------------------------------------------------ setup
    buildSetup() {
      const el = $('panel-setup');
      const portRows = FLL.robot.PORTS.map(p => `
        <tr>
          <td><b>${p}</b></td>
          <td><select data-port="${p}" data-k="type">
            <option value="none">– empty –</option>
            <option value="motor:medium">Medium motor</option>
            <option value="motor:large">Large motor</option>
            <option value="motor:small">Small motor</option>
            <option value="color">Color sensor</option>
            <option value="distance">Distance sensor</option>
            <option value="force">Force sensor</option>
          </select></td>
          <td><input type="number" data-port="${p}" data-k="x" step="1" title="mm forward of the wheel axle"></td>
          <td><input type="number" data-port="${p}" data-k="y" step="1" title="mm to the left of the centre"></td>
          <td><select data-port="${p}" data-k="mount" title="Where an attachment motor reaches">
            <option value="front">front</option><option value="back">back</option><option value="left">left</option><option value="right">right</option>
          </select></td>
          <td><input type="number" data-port="${p}" data-k="reach" step="5" title="How far the attachment reaches (mm)"></td>
        </tr>`).join('');
      el.innerHTML = `
        <div class="cards">
          <div class="card">
            <h3>Start position</h3>
            <div class="inline" style="margin-bottom:8px">
              <button class="btn small" id="sLeft">Left launch area</button>
              <button class="btn small" id="sRight">Right launch area</button>
            </div>
            <div class="form-grid">
              <span>x</span><span class="inline"><input type="number" id="sX" step="1"> <span class="unit">mm from the left mat edge</span></span>
              <span>y</span><span class="inline"><input type="number" id="sY" step="1"> <span class="unit">mm from the bottom wall</span></span>
              <span>Facing</span><span class="inline"><input type="number" id="sH" step="1"> <span class="unit">° (0 = north, 90 = east)</span></span>
            </div>
            <div class="inline" style="margin-top:8px">
              <button class="btn small primary" id="sApply">Move robot here</button>
              <button class="btn small" id="sUseCurrent" title="Remember where the robot is now as its start position">Use current position</button>
            </div>
            <div class="small-note" id="sCheck"></div>
            <hr style="border:0;border-top:1px solid #e5e7eb;margin:10px 0">
            <label class="inline"><input type="checkbox" id="sKeystone"> Load the keystone species (Mission 13) on the robot at launch</label>
          </div>
          <div class="card">
            <h3>Field setup</h3>
            <div class="form-grid">
              <span>Mine dock</span><select id="dMine"></select>
              <span>City dock</span><select id="dCity"></select>
              <span>Farm dock</span><select id="dFarm"></select>
              <span>M07 bonus</span><select id="dConn"><option value="0">no opposing team</option><option value="1">1 connection</option><option value="2">2 connections</option></select>
              <span>Inspection</span><label class="inline"><input type="checkbox" id="dInspect"> robot fits in one launch area (+20)</label>
            </div>
            <div class="small-note">Missions 13–15 sit on interchangeable docks – choosing the docks is part of your strategy. The katydid (Mission 4) hides in a random pod every time you reset the field.</div>
          </div>
          <div class="card" style="grid-column: 1 / -1">
            <h3>Robot design</h3>
            <div class="inline" style="margin-bottom:8px">
              <span class="muted">Preset:</span>
              <button class="btn small" data-preset="default">SPIKE driving base (guided mission)</button>
              <button class="btn small" data-preset="twocolor">2 color sensors (line following)</button>
              <button class="btn small" data-preset="distance">Distance + force sensor</button>
            </div>
            <div class="cards" style="grid-template-columns: repeat(auto-fill, minmax(230px, 1fr))">
              <div class="form-grid">
                <span>Left wheel motor</span><select id="cLeft">${FLL.robot.PORTS.map(p => `<option>${p}</option>`).join('')}</select>
                <span>Right wheel motor</span><select id="cRight">${FLL.robot.PORTS.map(p => `<option>${p}</option>`).join('')}</select>
                <span>Wheel diameter</span><span class="inline"><input type="number" id="cWheel" step="0.5"> <span class="unit">mm</span></span>
                <span>Axle track</span><span class="inline"><input type="number" id="cTrack" step="1"> <span class="unit">mm</span></span>
                <span>Left motor mirrored</span><label class="inline"><input type="checkbox" id="cMirror"> (normal SPIKE build)</label>
              </div>
              <div class="form-grid">
                <span>Front (from axle)</span><span class="inline"><input type="number" id="cFront" step="1"> <span class="unit">mm</span></span>
                <span>Rear (from axle)</span><span class="inline"><input type="number" id="cRear" step="1"> <span class="unit">mm</span></span>
                <span>Width</span><span class="inline"><input type="number" id="cWidth" step="1"> <span class="unit">mm</span></span>
                <span>Realism</span><select id="cRealism">
                  <option value="ideal">Ideal (perfect motors)</option>
                  <option value="realistic">Realistic (small errors)</option>
                  <option value="challenging">Challenging (bigger errors)</option>
                </select>
              </div>
            </div>
            <table class="ports-table" style="margin-top:8px">
              <thead><tr><th>Port</th><th>Device</th><th>x fwd</th><th>y left</th><th>Tool side</th><th>Reach</th></tr></thead>
              <tbody>${portRows}</tbody>
            </table>
            <div class="inline" style="margin-top:8px">
              <button class="btn small primary" id="cApply">Apply robot design</button>
              <span class="small-note" id="cMsg"></span>
            </div>
            <div class="small-note">Sensor positions are measured from the middle of the wheel axle: x forward, y to the left. Attachment motors interact with mission models in their tool zone (orange outline on the robot).</div>
          </div>
        </div>`;

      const dockOpts = [13, 14, 15].map(n => `<option value="${n}">M${n} ${esc(FLL.missions.list.find(m => m.id === n).name)}</option>`).join('');
      ['dMine', 'dCity', 'dFarm'].forEach(id => { $(id).innerHTML = dockOpts; });
      const dockIds = { dMine: 'mine', dCity: 'city', dFarm: 'farm' };
      for (const id of Object.keys(dockIds)) {
        $(id).addEventListener('change', (e) => {
          const val = Number(e.target.value);
          const docks = Object.assign({}, this.app.world.setup.docks);
          const mine = dockIds[id];
          const other = Object.keys(docks).find(k => docks[k] === val && k !== mine);
          if (other) docks[other] = docks[mine];
          docks[mine] = val;
          this.app.setDocks(docks);
          this.syncSetup();
        });
      }
      $('dConn').addEventListener('change', (e) => { this.app.world.setup.opponentConnections = Number(e.target.value); this.app.saveSettings(); });
      $('dInspect').addEventListener('change', (e) => { this.app.world.setup.inspectionBonus = e.target.checked; this.app.saveSettings(); });
      $('sKeystone').addEventListener('change', (e) => { this.app.settings.keystone = e.target.checked; this.app.saveSettings(); });

      $('sLeft').addEventListener('click', () => this.app.placeRobotPreset('left'));
      $('sRight').addEventListener('click', () => this.app.placeRobotPreset('right'));
      $('sApply').addEventListener('click', () => {
        const x = Number($('sX').value), y = Number($('sY').value), h = Number($('sH').value);
        this.app.setStartPose({ x: x, y: y, theta: thetaOf(h) }, true);
      });
      $('sUseCurrent').addEventListener('click', () => {
        const p = this.app.world.robot.pose;
        this.app.setStartPose({ x: p.x, y: p.y, theta: p.theta }, false);
        this.app.toast('Start position saved');
      });
      el.querySelectorAll('[data-preset]').forEach(b => b.addEventListener('click', () => this.app.applyRobotPreset(b.dataset.preset)));
      $('cApply').addEventListener('click', () => this.applyRobotForm());
    }

    syncSetup() {
      const app = this.app;
      const w = app.world;
      $('dMine').value = w.setup.docks.mine;
      $('dCity').value = w.setup.docks.city;
      $('dFarm').value = w.setup.docks.farm;
      $('dConn').value = w.setup.opponentConnections || 0;
      $('dInspect').checked = !!w.setup.inspectionBonus;
      $('sKeystone').checked = !!app.settings.keystone;
      const s = app.startPose;
      $('sX').value = Math.round(s.x);
      $('sY').value = Math.round(s.y);
      $('sH').value = compassOf(s.theta);
      const body = w.robot.bodyPoly(s);
      const ok = field.polyInLaunchArea('left', body) || field.polyInLaunchArea('right', body);
      $('sCheck').innerHTML = ok ? '✔ The start position is completely inside a launch area.' : '⚠ The robot is not completely inside a launch area at this start position (needed in a match).';
      const cfg = w.robot.config;
      $('cLeft').value = cfg.driveLeft;
      $('cRight').value = cfg.driveRight;
      $('cWheel').value = cfg.wheelDiameter;
      $('cTrack').value = cfg.trackWidth;
      $('cMirror').checked = !!cfg.leftReversed;
      $('cFront').value = cfg.body.front;
      $('cRear').value = cfg.body.rear;
      $('cWidth').value = cfg.body.width;
      $('cRealism').value = cfg.realism || 'realistic';
      for (const p of FLL.robot.PORTS) {
        const d = cfg.devices[p] || { type: 'none' };
        const typeSel = document.querySelector(`[data-port="${p}"][data-k="type"]`);
        typeSel.value = d.type === 'motor' ? 'motor:' + (d.kind || 'medium') : d.type;
        document.querySelector(`[data-port="${p}"][data-k="x"]`).value = d.x !== undefined ? d.x : '';
        document.querySelector(`[data-port="${p}"][data-k="y"]`).value = d.y !== undefined ? d.y : '';
        document.querySelector(`[data-port="${p}"][data-k="mount"]`).value = d.mount || 'front';
        document.querySelector(`[data-port="${p}"][data-k="reach"]`).value = d.reach !== undefined ? d.reach : '';
      }
    }

    applyRobotForm() {
      const cfg = JSON.parse(JSON.stringify(this.app.world.robot.config));
      const num = (id, def) => { const v = parseFloat($(id).value); return Number.isFinite(v) ? v : def; };
      cfg.driveLeft = $('cLeft').value;
      cfg.driveRight = $('cRight').value;
      cfg.wheelDiameter = geom.clamp(num('cWheel', 56), 10, 200);
      cfg.trackWidth = geom.clamp(num('cTrack', 112), 30, 400);
      cfg.leftReversed = $('cMirror').checked;
      cfg.body.front = geom.clamp(num('cFront', 112), 20, 400);
      cfg.body.rear = geom.clamp(num('cRear', 68), 0, 400);
      cfg.body.width = geom.clamp(num('cWidth', 156), 40, 400);
      cfg.realism = $('cRealism').value;
      for (const p of FLL.robot.PORTS) {
        const t = document.querySelector(`[data-port="${p}"][data-k="type"]`).value;
        const old = cfg.devices[p] || {};
        const d = {};
        if (t.indexOf('motor') === 0) { d.type = 'motor'; d.kind = t.split(':')[1]; }
        else d.type = t;
        const x = parseFloat(document.querySelector(`[data-port="${p}"][data-k="x"]`).value);
        const y = parseFloat(document.querySelector(`[data-port="${p}"][data-k="y"]`).value);
        const reach = parseFloat(document.querySelector(`[data-port="${p}"][data-k="reach"]`).value);
        if (d.type !== 'motor' && d.type !== 'none') {
          d.x = Number.isFinite(x) ? x : (d.type === 'color' ? 92 : cfg.body.front);
          d.y = Number.isFinite(y) ? y : 0;
        }
        if (d.type === 'motor') {
          d.mount = document.querySelector(`[data-port="${p}"][data-k="mount"]`).value;
          d.reach = Number.isFinite(reach) ? reach : 80;
          d.armLength = old.armLength || 90;
        }
        if (d.type === 'distance') d.angle = old.angle || 0;
        cfg.devices[p] = d;
      }
      const msg = this.validateConfig(cfg);
      if (msg) { $('cMsg').textContent = '⚠ ' + msg; return; }
      $('cMsg').textContent = '✔ Robot updated';
      this.app.setRobotConfig(cfg);
    }

    validateConfig(cfg) {
      if (cfg.driveLeft === cfg.driveRight) return 'The left and right wheel need different ports.';
      for (const p of [cfg.driveLeft, cfg.driveRight]) {
        if (!cfg.devices[p] || cfg.devices[p].type !== 'motor') return 'Port ' + p + ' must have a motor to drive a wheel.';
      }
      const motors = FLL.robot.PORTS.filter(p => cfg.devices[p] && cfg.devices[p].type === 'motor').length;
      if (motors > 4) return 'FLL rules allow at most 4 motors.';
      return '';
    }

    // ------------------------------------------------------------ help
    buildHelp() {
      $('panel-help').innerHTML = FLL.examples.helpHtml();
    }
  }

  FLL.Panels = Panels;
  FLL.ui = { compassOf, thetaOf, esc };
})(typeof globalThis !== 'undefined' ? globalThis : window);
