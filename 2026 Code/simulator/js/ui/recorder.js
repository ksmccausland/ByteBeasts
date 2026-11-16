/*
 * Record: writes a program while you drive the robot with the keyboard.
 * Every time a drive or arm key changes, the movement since the previous change becomes one step:
 * a SPIKE word block, a line of SPIKE Python, or both (two programs), as chosen next to the Record button.
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};
  const $ = (id) => document.getElementById(id);
  const MIN_DEG = 3;          // movements smaller than this are key bounce
  const ARM_SPEED = 400;      // same speed as the arm keys in Drive mode
  const DEG_PER_PCT = 11.1;   // SPIKE speed: 100 % = 1110 degrees per second
  const PAIR = 'motor_pair.PAIR_1';

  // What dL / dR motor degrees did to the robot (turn > 0 = to the right).
  function measure(dL, dR, cfg) {
    const k = Math.PI * cfg.wheelDiameter / 360;
    const sL = dL * (cfg.leftReversed ? -1 : 1) * k, sR = dR * k;
    return { sL: sL, sR: sR, mm: (sL + sR) / 2, turn: -(sR - sL) / cfg.trackWidth * 180 / Math.PI };
  }

  // pv = [left, right] motor-pair velocities (degrees per second, + = forward).
  function driveStep(pv, dL, dR, cfg) {
    const D = Math.round(Math.max(Math.abs(dL), Math.abs(dR)));
    if (D < MIN_DEG || (!pv[0] && !pv[1])) return null;
    return Object.assign({ kind: 'drive', pv: pv, D: D }, measure(dL, dR, cfg));
  }

  // ---------------------------------------------------------------- Python
  function stepText(s) {
    const pad = '    ';
    if (s.kind === 'note') return pad + '# ' + s.text;
    if (s.kind === 'arm') return pad + 'await motor.run_for_degrees(port.' + s.port + ', ' + s.deg + ', ' + ARM_SPEED + ')   # move arm ' + s.port;
    const mm = Math.abs(Math.round(s.mm)), turn = Math.abs(Math.round(s.turn));
    const travel = Math.abs(s.sL) + Math.abs(s.sR);   // named from the measured wheel travel, not from the velocities
    const what = Math.abs(s.sR - s.sL) <= 0.1 * travel ? 'drive ' + (s.mm >= 0 ? 'forward ' : 'backward ') + mm + ' mm'
      : Math.abs(s.sL + s.sR) <= 0.1 * travel ? 'turn ' + (s.turn >= 0 ? 'right ' : 'left ') + turn + ' deg'
      : 'curve ' + (s.turn >= 0 ? 'right' : 'left') + ', ' + mm + ' mm, ' + turn + ' deg';
    return pad + 'await motor_pair.move_tank_for_degrees(' + PAIR + ', ' + s.D + ', ' + s.pv[0] + ', ' + s.pv[1] + ')   # ' + what;
  }

  function renderPython(cfg, pose, steps) {
    return 'from hub import port\nimport motor, motor_pair, runloop\n\nasync def main():\n' +
      '    motor_pair.pair(' + PAIR + ', port.' + cfg.driveLeft + ', port.' + cfg.driveRight + ')\n' +
      '    # ' + startNote(pose) + '\n' +
      steps.map(s => stepText(s) + '\n').join('') + '\nrunloop.run(main())\n';
  }

  // ---------------------------------------------------------------- word blocks
  const N = (v) => ({ shadow: { type: 'math_number', fields: { NUM: v } } });
  const pct = (v) => Math.round(v / DEG_PER_PCT);
  // "1 motor rotation" in cm for this robot's wheels (one decimal, like in the SPIKE app).
  const cmPerRotation = (cfg) => Math.round(Math.PI * cfg.wheelDiameter) / 10;
  const startNote = (pose) => 'Recorded with the robot at x ' + pose.x + ', y ' + pose.y + ', facing ' + pose.heading + ' deg (0 = north). Put it back there before you run this.';

  // The word blocks for one step. `st.speed` is the movement speed set by earlier blocks.
  function stepBlocks(s, cfg, st) {
    if (s.kind === 'note') return [{ type: 'python_comment', fields: { TEXT: s.text } }];
    if (s.kind === 'arm') {
      return [{ type: 'flippermotor_motorTurnForDirection', fields: { PORT: s.port, DIRECTION: s.deg >= 0 ? 'clockwise' : 'counterclockwise', UNIT: 'degrees' }, inputs: { VALUE: N(Math.abs(s.deg)) } }];
    }
    const l = s.pv[0], r = s.pv[1];
    if (l !== r && l !== -r) {
      return [{ type: 'flippermoremove_moveDualSpeed', fields: { UNIT: 'degrees' }, inputs: { VALUE: N(s.D), LEFT: N(pct(l)), RIGHT: N(pct(r)) } }];
    }
    const out = [];
    const speed = Math.abs(pct(l));
    if (speed !== st.speed) {
      out.push({ type: 'flippermove_movementSpeed', inputs: { SPEED: N(speed) } });
      st.speed = speed;
    }
    if (l === r) {
      const cm = Math.round(s.D * cmPerRotation(cfg) / 360 * 10) / 10;
      out.push({ type: 'flippermove_move', fields: { DIRECTION: l > 0 ? 'forward' : 'back', UNIT: 'cm' }, inputs: { VALUE: N(cm) } });
    } else {
      out.push({ type: 'flippermove_steer', fields: { UNIT: 'degrees' }, inputs: { STEERING: N(l > 0 ? 100 : -100), VALUE: N(s.D) } });
    }
    return out;
  }

  // The whole program as Blockly JSON: start block, set-up blocks, then the blocks of every step.
  function renderBlocks(cfg, pose, steps) {
    const st = { speed: null };
    const list = [
      { type: 'python_comment', fields: { TEXT: startNote(pose) } },
      { type: 'flippermove_setMovementPair', fields: { PAIR: cfg.driveLeft + cfg.driveRight } },
      { type: 'flippermove_setDistance', fields: { UNIT: 'cm' }, inputs: { DISTANCE: N(cmPerRotation(cfg)) } }
    ];
    for (const s of steps) list.push(...stepBlocks(s, cfg, st));
    list.forEach((b, i) => { b.id = 'rec' + i; });
    for (let i = list.length - 2; i >= 0; i--) list[i].next = { block: list[i + 1] };
    return { blocks: { languageVersion: 0, blocks: [{ type: 'flipperevents_whenProgramStarts', id: 'recStart', x: 30, y: 30, next: { block: list[0] } }] } };
  }

  class Recorder {
    constructor(app) {
      this.app = app;
      this.active = false;
      this.progs = {};        // { blocks: program, python: program } being written
      this.steps = [];        // closed steps
      this.pv = [0, 0];       // drive velocities of the open segment
      this.armV = {};         // arm directions of the open segment
      this.snap = null;       // motor positions where the open segment began
      this.cfg = null;
      this.pose = null;
      this.prevId = null;
      this.label = '';
    }

    // 'blocks', 'python' or 'both' (chosen next to the Record button)
    format() { return this.app.settings.recordFormat || 'blocks'; }
    owns(id) { return Object.values(this.progs).some(p => p.id === id); }

    toggle() {
      if (this.active) this.stop(); else this.start();
      if (this.app.driveMode) { $('fieldCanvas').focus(); this.app.fieldFocused = true; }
    }

    snapshot() {
      const ms = this.app.world.robot.motors;
      this.snap = {};
      for (const p in ms) this.snap[p] = ms[p].pos;
    }

    // Steps for the movement since the last snapshot.
    pending() {
      const snap = this.snap, cfg = this.cfg;
      if (!snap) return [];
      const ms = this.app.world.robot.motors, out = [];
      const L = ms[cfg.driveLeft], R = ms[cfg.driveRight];
      if (L && R && snap[cfg.driveLeft] != null && snap[cfg.driveRight] != null) {
        const s = driveStep(this.pv, L.pos - snap[cfg.driveLeft], R.pos - snap[cfg.driveRight], cfg);
        if (s) out.push(s);
      }
      for (const port in this.armV) {
        const m = ms[port];
        if (!this.armV[port] || !m || snap[port] == null) continue;
        const deg = Math.round(m.pos - snap[port]);
        if (Math.abs(deg) >= MIN_DEG) out.push({ kind: 'arm', port: port, deg: deg });
      }
      return out;
    }

    // Close the open segment and start the next one.
    cut() {
      if (!this.active) return;
      const add = this.pending();
      this.snapshot();
      if (!add.length) return;
      this.steps.push(...add);
      this.update();
    }

    // Called by the app when the drive velocities or an arm key change.
    drive(pv) { if (!this.active) return; this.cut(); this.pv = pv; }
    arm(port, v) { if (!this.active) return; this.cut(); this.armV[port] = v; }

    // The robot was moved by hand: that is not driving, so it becomes a comment.
    note(text) {
      if (!this.active) return;
      this.cut();
      this.pv = [0, 0];
      this.armV = {};
      if (text) {
        this.steps.push({ kind: 'note', text: text });
        this.update();
      }
      this.snap = null;   // the next cut() takes a fresh snapshot
    }

    // The Python program also shows the move that is still going on.
    tick() {
      if (!this.active) return;
      if (this.progs.python) this.showPython(true);
      this.setButton();
    }

    update() {
      if (this.progs.blocks) this.showBlocks();
      if (this.progs.python) this.showPython(false);
      this.app.editors.savePrograms();
    }

    showBlocks() {
      const p = this.progs.blocks, ed = this.app.editors;
      p.blocks = renderBlocks(this.cfg, this.pose, this.steps);
      if (ed.currentId !== p.id) return;
      ed.loadBlocks(p);
      if (ed.tab === 'python') ed.refreshGenerated();
      let n = 0, b = p.blocks.blocks.blocks[0].next;
      while (b) { n++; b = b.block.next; }
      const last = 'rec' + (n - 1);
      setTimeout(() => { if (ed.workspace.getBlockById(last)) ed.workspace.centerOnBlock(last); }, 60);
    }

    // Write the code into the program and into the editor (only the part that changed).
    showPython(withPending) {
      const p = this.progs.python, ed = this.app.editors, cm = ed.cm;
      const text = renderPython(this.cfg, this.pose, withPending ? this.steps.concat(this.pending()) : this.steps);
      p.python = text;
      if (ed.currentId !== p.id || ed.tab !== 'python') return;
      const old = cm.getValue();
      if (old === text) return;
      let i = 0;
      const n = Math.min(old.length, text.length);
      while (i < n && old.charCodeAt(i) === text.charCodeAt(i)) i++;
      cm.replaceRange(text.slice(i), cm.posFromIndex(i), cm.posFromIndex(old.length));
      cm.scrollIntoView({ line: cm.lastLine(), ch: 0 }, 40);
    }

    setButton() {
      const b = $('btnRecord');
      if (!b) return;
      const n = this.active ? this.steps.filter(s => s.kind !== 'note').length + this.pending().length : 0;
      const label = this.active ? '■ Stop · ' + n + (n === 1 ? ' step' : ' steps') : '● Record';
      if (label !== this.label) { this.label = label; b.textContent = label; }
      b.classList.toggle('on', this.active);
      const sel = $('recFormat');
      if (sel) sel.disabled = this.active;
    }

    start() {
      if (this.active) return;
      const app = this.app, ed = app.editors;
      if (!app.driveMode) app.setDrive(true);
      if (!app.driveMode) return;   // refused (match mode or a running program); setDrive says why
      const rb = app.world.robot, p = rb.pose, fmt = this.format();
      this.cfg = rb.config;
      this.pose = { x: Math.round(p.x), y: Math.round(p.y), heading: FLL.ui.compassOf(p.theta) };
      this.steps = [];
      this.pv = [0, 0];
      this.armV = {};
      this.snap = null;
      this.prevId = ed.currentId;
      this.progs = {};
      // the last program added is the one that opens
      if (fmt !== 'blocks') this.progs.python = ed.addProgram({ name: fmt === 'both' ? 'Recorded drive (Python)' : 'Recorded drive', mode: 'python', python: renderPython(this.cfg, this.pose, []) }, fmt === 'python');
      if (fmt !== 'python') this.progs.blocks = ed.addProgram({ name: 'Recorded drive', mode: 'blocks', blocks: renderBlocks(this.cfg, this.pose, []) });
      this.active = true;   // only now, so selecting the new program does not stop the recording
      if (this.progs.python && ed.currentId === this.progs.python.id) ed.cm.setOption('readOnly', 'nocursor');
      app.resetDriveState();
      this.snapshot();
      this.setButton();
      app.fieldFocused = true;
      const names = Object.values(this.progs).map(x => '“' + x.name + '”').join(' and ');
      app.toast('● Recording – drive with the arrow keys. Press R or Esc to finish.', 3500);
      app.log('● Recording started. Your driving is written into ' + names + '.', 'info');
    }

    // Delete a recorded program that stayed empty.
    remove(p) {
      const ed = this.app.editors;
      if (ed.currentId === p.id) { ed.deleteCurrent(); return; }
      ed.programs = ed.programs.filter(x => x.id !== p.id);
      ed.refreshSelect();
      ed.savePrograms();
    }

    // keep = true: do not delete an empty recording and do not change the selected program.
    stop(keep) {
      if (!this.active) return;
      this.cut();
      this.active = false;
      const app = this.app, ed = app.editors, progs = Object.values(this.progs);
      const n = this.steps.filter(s => s.kind !== 'note').length;
      if (this.progs.python) this.showPython(false);
      if (this.owns(ed.currentId)) { ed.showTab(ed.tab); ed.cm.clearHistory(); }   // unlocks the Python editor
      this.setButton();
      if (!n) {
        app.toast('Nothing was recorded – hold an arrow key to drive.', 3000);
        if (!keep && this.owns(ed.currentId)) {
          progs.forEach(p => this.remove(p));
          if (ed.programs.some(x => x.id === this.prevId)) ed.select(this.prevId);
        }
        return;
      }
      ed.savePrograms();
      const names = progs.map(x => '“' + x.name + '”').join(' and ');
      app.toast('Recorded ' + n + (n === 1 ? ' step' : ' steps') + ' in ' + names + '. To replay it, put the robot back where you started recording and press ▶ Run (F5).', 6000);
      app.log('■ Recording finished: ' + n + (n === 1 ? ' step' : ' steps') + ' in ' + names + '.', 'ok');
    }
  }

  Recorder.renderBlocks = renderBlocks;
  Recorder.renderPython = renderPython;
  Recorder.driveStep = driveStep;
  Recorder.measure = measure;
  Recorder.stepText = stepText;
  Recorder.stepBlocks = stepBlocks;
  FLL.Recorder = Recorder;
})(typeof globalThis !== 'undefined' ? globalThis : window);
