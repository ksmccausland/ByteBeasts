/*
 * Main application: wires the world, renderer, editors and panels together.
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};
  const geom = FLL.geom;
  const field = FLL.field;
  const $ = (id) => document.getElementById(id);
  const SETTINGS_KEY = 'fll-bioglow-settings-v1';
  const DT = FLL.World.DT;

  const START_PRESETS = {
    left: { x: 220, y: 140, theta: Math.PI / 2 },
    right: { x: field.MAT_W - 220, y: 140, theta: Math.PI / 2 }
  };

  const PRESETS = {
    default: () => JSON.parse(JSON.stringify(FLL.robot.DEFAULT_CONFIG)),
    twocolor: () => {
      const c = JSON.parse(JSON.stringify(FLL.robot.DEFAULT_CONFIG));
      c.name = 'Driving base with two color sensors';
      c.devices.A = { type: 'color', x: 92, y: 24 };
      c.devices.B = { type: 'color', x: 92, y: -24 };
      c.devices.F = { type: 'none' };
      return c;
    },
    distance: () => {
      const c = JSON.parse(JSON.stringify(FLL.robot.DEFAULT_CONFIG));
      c.name = 'Driving base with distance and force sensor';
      c.devices.A = { type: 'distance', x: 112, y: 30, angle: 0 };
      c.devices.F = { type: 'force', x: 112, y: -30 };
      return c;
    },
    // The team robots, with the ports of their mission programs. Body, track and wheel size are the default design (not measured).
    mh20: () => {
      const c = JSON.parse(JSON.stringify(FLL.robot.DEFAULT_CONFIG));
      c.name = 'M & H 2.0 robot (drive A/E, attachments B/D)';
      c.driveLeft = 'A';
      c.driveRight = 'E';
      c.devices = {
        A: { type: 'motor', kind: 'medium' },
        B: { type: 'motor', kind: 'medium', mount: 'front', reach: 80, armLength: 90 },
        C: { type: 'none' },
        D: { type: 'motor', kind: 'medium', mount: 'front', reach: 80, armLength: 90 },
        E: { type: 'motor', kind: 'medium' },
        F: { type: 'none' }
      };
      return c;
    },
    specificity: () => {
      const c = JSON.parse(JSON.stringify(FLL.robot.DEFAULT_CONFIG));
      c.name = 'Specificity robot (drive A/C, attachments B/F)';
      c.driveLeft = 'A';
      c.driveRight = 'C';
      c.devices = {
        A: { type: 'motor', kind: 'medium' },
        B: { type: 'motor', kind: 'medium', mount: 'front', reach: 80, armLength: 90 },
        C: { type: 'motor', kind: 'medium' },
        D: { type: 'none' },
        E: { type: 'none' },
        F: { type: 'motor', kind: 'medium', mount: 'front', reach: 80, armLength: 90 }
      };
      return c;
    }
  };

  class App {
    constructor() {
      this.settings = this.loadSettings();
      this.startPose = Object.assign({}, this.settings.startPose || START_PRESETS.left);
      this.world = new FLL.World({
        robotConfig: this.settings.robotConfig || FLL.robot.DEFAULT_CONFIG,
        setup: this.settings.setup,
        matSampler: null,
        onLog: (e) => this.onMissionLog(e)
      });
      this.world.robot.pose = Object.assign({}, this.startPose);
      this.view = new FLL.FieldView($('fieldCanvas'), this);
      Object.assign(this.view.options, this.settings.view || {});
      this.runner = new FLL.ProgramRunner(this.world, {
        print: (s) => this.print(s),
        warn: (s) => this.log(s, 'warn'),
        onError: (e) => this.onProgramError(e),
        onFinish: (why) => this.onProgramFinish(why),
        onBeep: (f, d, v) => this.beep(f, d, v)
      });
      this.editors = new FLL.Editors(this);
      this.panels = new FLL.Panels(this);
      this.recorder = new FLL.Recorder(this);
      this.view3d = new FLL.View3D(this);
      this.speed = Number(this.settings.speed || 1);
      this.mode = this.settings.mode === 'simulator' ? 'simulator' : 'robot';
      this.setupRobot = null;   // the robot of the Setup panel while a Robot-mode run has its program's robot in place
      this.match = { active: false, started: false, over: false, timeLeft: 150, lastLaunch: 0 };
      this.drag = null;
      this.keys = {};
      this.driveMode = false;
      this.measureMode = false;
      this.acc = 0;
      this.lastFrame = 0;
      this.lastPanelUpdate = 0;
    }

    // ------------------------------------------------------------ settings
    loadSettings() {
      try { return JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}') || {}; } catch (e) { return {}; }
    }
    saveSettings() {
      const s = this.settings;
      s.robotConfig = this.setupRobot || this.world.robot.config;
      s.startPose = this.startPose;
      s.setup = { docks: this.world.setup.docks, opponentConnections: this.world.setup.opponentConnections, inspectionBonus: this.world.setup.inspectionBonus };
      s.view = this.view.options;
      s.speed = this.speed;
      s.mode = this.mode;
      try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(s)); } catch (e) { /* storage may be unavailable */ }
    }

    // ------------------------------------------------------------ start-up
    init() {
      this.loadImages();
      this.editors.init();
      this.panels.init();
      this.bindUi();
      this.bindField();
      this.bindKeys();
      this.view.resize();
      new ResizeObserver(() => { this.view.resize(); this.view3d.resize(); this.editors.resize(); }).observe($('fieldWrap'));
      new ResizeObserver(() => this.editors.resize()).observe($('editorArea'));
      if (this.settings.codeWidth) $('codePane').style.width = this.settings.codeWidth;
      this.setLayout(this.settings.layout || 'split', true);
      this.setPanelsCollapsed(this.settings.panelsCollapsed !== undefined ? this.settings.panelsCollapsed : true, true);
      this.world.robot.resetYaw(0, this.world.time);
      $('speedSel').value = String(this.speed);
      $('modeSel').value = this.mode;
      this.log('Welcome! Pick a program, press ▶ Run and watch the robot. Drag the robot to move it, scroll to zoom.', 'info');
      requestAnimationFrame((t) => this.frame(t));
    }

    loadImages() {
      const mat = new Image();
      mat.onload = () => {
        const c = document.createElement('canvas');
        c.width = mat.naturalWidth;
        c.height = mat.naturalHeight;
        const ctx = c.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(mat, 0, 0);
        try {
          const data = ctx.getImageData(0, 0, c.width, c.height).data;
          const W = c.width, H = c.height;
          const sx = W / field.MAT_W, sy = H / field.MAT_H;
          this.world.matSampler = (x, y) => {
            const px = Math.min(W - 1, Math.max(0, Math.floor(x * sx)));
            const py = Math.min(H - 1, Math.max(0, Math.floor((field.MAT_H - y) * sy)));
            const i = (py * W + px) * 4;
            return [data[i], data[i + 1], data[i + 2]];
          };
        } catch (e) {
          this.log('The color sensor cannot read the mat in this browser: ' + e.message, 'err');
        }
      };
      mat.src = FLL.assets.mat;
      const wire = new Image();
      wire.src = FLL.assets.wireframe;
      this.view.setImages(mat, wire);
    }

    bindUi() {
      $('btnRun').addEventListener('click', () => this.run());
      $('btnStop').addEventListener('click', () => this.stop());
      $('btnRobotStart').addEventListener('click', () => this.robotToStart());
      $('btnResetField').addEventListener('click', () => this.resetField());
      $('speedSel').addEventListener('change', (e) => { this.speed = Number(e.target.value); this.saveSettings(); });
      $('modeSel').addEventListener('change', (e) => { this.mode = e.target.value === 'simulator' ? 'simulator' : 'robot'; this.saveSettings(); e.target.blur(); });
      $('btnMatch').addEventListener('click', () => (this.match.active ? this.endMatch(true) : this.startMatch()));
      $('btnClearConsole').addEventListener('click', () => { $('console').innerHTML = ''; });
      $('console').addEventListener('click', (e) => {
        const a = e.target.closest('.lineref');
        if (a) this.editors.goToLine(Number(a.dataset.line));
      });

      // view toggles
      const bindOpt = (id, key) => {
        const el = $(id);
        el.checked = !!this.view.options[key];
        el.addEventListener('change', () => { this.view.options[key] = el.checked; this.saveSettings(); });
      };
      bindOpt('optZones', 'zones');
      bindOpt('optGrid', 'grid');
      bindOpt('optTrail', 'trail');
      bindOpt('optLabels', 'labels');
      bindOpt('optSchematic', 'schematic');
      $('btnFit').addEventListener('click', () => { if (this.view3d.on) this.view3d.resetCamera(); else this.view.fit(); });
      $('btn3d').addEventListener('click', (e) => { this.view3d.toggle(); e.currentTarget.blur(); });
      document.querySelectorAll('#layoutSwitch [data-layout]').forEach(b => b.addEventListener('click', () => this.setLayout(b.dataset.layout)));
      $('btnPanelToggle').addEventListener('click', () => this.setPanelsCollapsed(!this.panelsCollapsed));
      $('btnClearTrail').addEventListener('click', () => { this.world.robot.trail = []; this.world.robot.lastPoseForTrail = null; });
      $('btnMeasure').addEventListener('click', () => this.setMeasure(!this.measureMode));
      $('btnDrive').addEventListener('click', () => this.setDrive(!this.driveMode));
      $('btnRecord').addEventListener('click', (e) => { this.recorder.toggle(); e.currentTarget.blur(); });
      $('recFormat').value = this.recorder.format();
      $('recFormat').addEventListener('change', (e) => { this.settings.recordFormat = e.target.value; this.saveSettings(); e.target.blur(); });

      // program bar
      $('btnNew').addEventListener('click', () => this.newProgramDialog());
      $('btnOpen').addEventListener('click', () => $('fileInput').click());
      $('fileInput').addEventListener('change', (e) => { const f = e.target.files[0]; e.target.value = ''; if (f) this.openFile(f); });
      this.setupMenu('btnSaveMenu', 'saveMenu');
      this.setupMenu('btnExamples', 'examplesMenu');
      $('saveMenu').addEventListener('click', (e) => {
        const b = e.target.closest('button');
        if (!b) return;
        this.closeMenus();
        this.saveAction(b.dataset.act);
      });
      this.buildExamplesMenu();

      // splitter
      const sp = $('splitter');
      sp.addEventListener('pointerdown', (e) => {
        sp.setPointerCapture(e.pointerId);
        sp.classList.add('drag');
        const move = (ev) => {
          const swapped = $('layout').classList.contains('swap');
          const x = swapped ? window.innerWidth - ev.clientX : ev.clientX;
          const w = Math.min(window.innerWidth - 420, Math.max(320, x));
          $('codePane').style.width = w + 'px';
        };
        const up = () => {
          sp.classList.remove('drag');
          sp.removeEventListener('pointermove', move);
          sp.removeEventListener('pointerup', up);
          this.settings.codeWidth = $('codePane').style.width;
          this.saveSettings();
        };
        sp.addEventListener('pointermove', move);
        sp.addEventListener('pointerup', up);
      });
      window.addEventListener('click', (e) => { if (!e.target.closest('.dropdown')) this.closeMenus(); });
    }

    setupMenu(btnId, menuId) {
      $(btnId).addEventListener('click', (e) => {
        e.stopPropagation();
        const m = $(menuId);
        const open = m.classList.contains('open');
        this.closeMenus();
        if (!open) m.classList.add('open');
      });
    }
    closeMenus() { document.querySelectorAll('.menu.open').forEach(m => m.classList.remove('open')); }

    buildExamplesMenu() {
      const menu = $('examplesMenu');
      const groups = {};
      for (const ex of FLL.examples.list) (groups[ex.group] = groups[ex.group] || []).push(ex);
      const html = [];
      for (const g of Object.keys(groups)) {
        html.push(`<div class="menu-title">${FLL.ui.esc(g)}</div>`);
        for (const ex of groups[g]) {
          html.push(`<button data-ex="${ex.id}">${ex.mode === 'blocks' ? '🧩' : '🐍'} ${FLL.ui.esc(ex.name)}<span class="ex-desc">${FLL.ui.esc(ex.description)}</span></button>`);
        }
      }
      menu.innerHTML = html.join('');
      menu.addEventListener('click', (e) => {
        const b = e.target.closest('button[data-ex]');
        if (!b) return;
        this.closeMenus();
        this.loadExample(b.dataset.ex);
      });
    }

    loadExample(id) {
      const ex = FLL.examples.list.find(x => x.id === id);
      if (!ex) return;
      const prog = FLL.examples.instantiate(ex, this.world.robot.config);
      this.editors.addProgram(prog);
      if (ex.start) {
        this.setStartPose({ x: ex.start.x, y: ex.start.y, theta: FLL.ui.thetaOf(ex.start.heading) }, true);
      }
      this.toast('Loaded example “' + ex.name + '”' + (ex.start ? ' – the robot moved to its start position' : ''));
      if (ex.note) this.log(ex.note, 'info');
    }

    // ------------------------------------------------------------ console
    print(s) {
      const con = $('console');
      con.appendChild(document.createTextNode(s));
      this.trimConsole();
      con.scrollTop = con.scrollHeight;
    }
    log(text, cls) {
      const con = $('console');
      const div = document.createElement('div');
      if (cls) div.className = cls;
      div.textContent = text;
      con.appendChild(div);
      this.trimConsole();
      con.scrollTop = con.scrollHeight;
    }
    trimConsole() {
      const con = $('console');
      while (con.childNodes.length > 1500) con.removeChild(con.firstChild);
    }
    logError(err) {
      const con = $('console');
      const div = document.createElement('div');
      div.className = 'err';
      const html = FLL.ui.esc(err.traceback).replace(/line (\d+)/g, '<span class="lineref" data-line="$1">line $1</span>');
      div.innerHTML = html;
      con.appendChild(div);
      con.scrollTop = con.scrollHeight;
    }

    toast(msg, ms) {
      const t = $('toast');
      t.textContent = msg;
      t.classList.add('show');
      clearTimeout(this.toastTimer);
      this.toastTimer = setTimeout(() => t.classList.remove('show'), ms || 2600);
    }

    // ------------------------------------------------------------ running programs
    run() {
      if (this.runner.running) return;
      if (this.match.over) { this.toast('The match is over. Press “End match” to start practising again.'); return; }
      this.recorder.stop(true);
      const src = this.editors.getRunCode();
      if (!src) return;
      if (this.drag) this.drag = null;
      const body = this.world.robot.bodyPoly();
      const inLaunch = field.polyInLaunchArea('left', body) || field.polyInLaunchArea('right', body);
      if (this.match.active && !inLaunch) {
        this.toast('In a match the robot must start completely inside a launch area.', 4000);
        this.log('Launch refused: move the robot completely inside the left or right launch area first (⟲ Robot to start).', 'warn');
        return;
      }
      if (this.world.robotOverlaps()) this.log('Note: the robot overlaps a wall or mission model at the start.', 'warn');
      this.setDrive(false);
      this.editors.clearError();
      this.editors.clearRunHighlight();
      if (this.settings.keystone && !this.world.items.some(i => i.kind === 'keystone') && field.homeOf(body)) {
        this.world.carry({ kind: 'keystone', from: 'home', label: 'Keystone species' });
      }
      this.log('▶ Running “' + src.name + '”', 'ok');
      this.prepareRobot(src);
      const ok = this.runner.start(src.code, { filename: src.mode === 'blocks' ? 'blocks.py' : 'program.py' });
      if (!ok) this.restoreSetupRobot();
      if (ok) {
        this.match.lastLaunch = this.world.time;
        this.match.robotOut = true;
        if (this.match.active && !this.match.started) {
          this.match.started = true;
          this.toast('Match started – 2:30 on the clock!');
        }
      }
      this.updateRunState();
    }

    // Robot mode: a Python program runs on the robot it was written for, read from its settings (ports, wheel size, MOTOR_INVERT).
    // MOTOR_INVERT = -1 means positive speeds drive that robot backwards, so both wheels are wired the other way round, and
    // the hub's yaw then counts clockwise (the only direction in which the team's turn code can steer that robot).
    // That robot is for this run only: restoreSetupRobot puts the robot of the Setup panel back when the run ends.
    // Simulator mode, and word blocks, use the robot of the Setup panel as it is. The program is never changed.
    prepareRobot(src) {
      const cur = this.world.robot.config;
      let cfg = JSON.parse(JSON.stringify(cur));
      if (this.mode === 'robot' && src.mode === 'python') {
        const fit = FLL.github.robotFor(src.code, cur);
        if (fit.cfg) cfg = fit.cfg;
        const inv = fit.settings.invert === -1;
        cfg.leftReversed = !inv;
        cfg.rightReversed = inv;
        cfg.hubYawCW = inv;
        cfg.programWiring = true;
        if (fit.changes.length) this.log('Robot mode: the robot is set up like the program: ' + fit.changes.join('; ') + '.', 'info');
        if (inv) this.log('Robot mode: MOTOR_INVERT = -1, so positive speeds drive this robot backwards, and its yaw counts clockwise.', 'info');
        if ((!fit.settings.left || !fit.settings.right) && /motor_pair/.test(src.code)) this.log('Robot mode: the program does not say which ports its wheels are on, so the robot of the Setup panel is used.', 'warn');
      } else if (cfg.programWiring) {
        cfg.leftReversed = true;
        delete cfg.rightReversed; delete cfg.hubYawCW; delete cfg.programWiring;
      }
      if (JSON.stringify(cfg) === JSON.stringify(cur)) return;
      const bad = this.panels.validateConfig(cfg);
      if (bad) { this.log('The robot for this program cannot be set up: ' + bad + ' The robot of the Setup panel is used.', 'err'); return; }
      if (cfg.programWiring && !this.setupRobot) this.setupRobot = cur;
      this.setRobotConfig(cfg);
      this.panels.syncSetup();
    }

    // Put the robot of the Setup panel back once a Robot-mode run has ended (finished, stopped, failed or never started).
    restoreSetupRobot() {
      if (!this.setupRobot) return;
      const setup = this.setupRobot;
      this.setupRobot = null;
      this.setRobotConfig(setup);
      this.panels.syncSetup();
    }

    stop() {
      if (this.runner.running) {
        this.runner.stop();
        this.log('■ Stopped', 'warn');
      }
      this.world.robot.stopAll('brake');
      this.resetDriveState();
      this.updateRunState();
    }

    onProgramError(e) {
      this.restoreSetupRobot();
      this.logError(e);
      this.editors.markError(e.line);
      this.updateRunState();
      this.toast(e.type + ': ' + e.message, 5000);
    }
    onProgramFinish(why) {
      this.restoreSetupRobot();
      if (why === 'done' || why === 'exit') this.log('✔ Program finished after ' + ((this.world.time - this.runner.startTime) / 1000).toFixed(1) + ' s', 'ok');
      this.editors.clearRunHighlight();
      this.updateRunState();
    }
    updateRunState() {
      const st = $('runState');
      const s = this.runner.state;
      const labels = { idle: 'ready', running: 'running…', finished: 'finished', error: 'error', stopped: 'stopped' };
      st.textContent = labels[s] || s;
      st.className = 'run-state ' + s;
      $('btnRun').disabled = this.runner.running;
    }

    canEditPose() { return !this.runner.running && !this.match.over; }

    // ------------------------------------------------------------ robot placement
    setStartPose(pose, move) {
      this.startPose = { x: pose.x, y: pose.y, theta: pose.theta };
      this.saveSettings();
      if (move) this.placeRobot(this.startPose, true);
      if (this.panels.active === 'setup') this.panels.syncSetup();
    }
    placeRobotPreset(side) {
      const p = START_PRESETS[side];
      this.setStartPose(p, true);
    }
    robotToStart() {
      if (this.runner.running) this.stop();
      this.placeRobot(this.startPose, true);
    }
    // Moving the robot by hand. In a match this is an interruption.
    placeRobot(pose, clearTrail) {
      if (this.match.over) return;
      this.interruptionCheck();
      const rb = this.world.robot;
      rb.stopAll('coast');
      for (const p in rb.motors) rb.motors[p].speed = 0;
      rb.pose = { x: pose.x, y: pose.y, theta: pose.theta };
      rb.resetYaw(0, this.world.time);
      if (this.recorder.active) this.recorder.note('robot placed by hand: x ' + Math.round(pose.x) + ', y ' + Math.round(pose.y) + ', facing ' + FLL.ui.compassOf(pose.theta) + ' deg');
      this.resetDriveState();
      if (clearTrail) { rb.trail = []; rb.lastPoseForTrail = null; }
    }
    // Precision token rule: touching the robot while it is not completely in home costs a token.
    interruptionCheck() {
      if (!this.match.active || !this.match.started || !this.match.robotOut) return;
      this.match.robotOut = false;
      const body = this.world.robot.bodyPoly();
      if (field.homeOf(body)) return;
      const w = this.world;
      w.setup.tokens = Math.max(0, w.setup.tokens - 1);
      // things collected outside home since the last launch go to the referee
      const lost = w.items.filter(it => it.carried && it.pickedAt >= this.match.lastLaunch && it.kind !== 'keystone');
      for (const it of lost) { it.carried = false; it.placed = false; it.lost = true; it.x = -9999; }
      w.items = w.items.filter(it => !it.lost);
      this.log('Interruption outside home: one precision token lost (' + w.setup.tokens + ' left)' + (lost.length ? ', ' + lost.length + ' collected item(s) given to the referee' : '') + '.', 'warn');
      this.toast('Precision token lost – the robot was picked up outside home', 3500);
    }

    resetField() {
      if (this.runner.running) this.stop();
      this.world.resetField();
      if (this.match.active) this.world.setup.tokens = 6;
      this.placeRobot(this.startPose, true);
      this.panels.updateMissions(true);
      this.log('♻ Field reset – the katydid moved to a new hiding place.', 'info');
    }

    setDocks(docks) {
      this.world.setDocks(docks);
      this.saveSettings();
    }

    setRobotConfig(cfg) {
      if (this.runner.running) this.stop();
      if (this.recorder) this.recorder.stop();
      const pose = Object.assign({}, this.world.robot.pose);
      const hub = this.world.robot.hub;
      this.world.robot = new FLL.robot.Robot(cfg, (Date.now() & 0xffff) + 1);
      this.world.robot.pose = pose;
      this.world.robot.hub = hub;
      this.resetDriveState();
      this.saveSettings();
      this.editors.updateToolbox();
    }
    applyRobotPreset(name) {
      const cfg = PRESETS[name]();
      cfg.realism = this.world.robot.config.realism;
      this.setRobotConfig(cfg);
      this.panels.syncSetup();
      this.toast('Robot design: ' + cfg.name);
    }

    // ------------------------------------------------------------ match mode
    startMatch() {
      if (this.runner.running) this.stop();
      this.match = { active: true, started: false, over: false, timeLeft: 150, lastLaunch: 0, robotOut: false };
      this.setDrive(false);
      this.world.resetField();
      this.world.setup.tokens = 6;
      this.placeRobot(this.startPose, true);
      $('btnMatch').textContent = '⏹ End match';
      $('btnMatch').classList.add('active');
      this.log('🏁 Match mode: the 2:30 timer starts at your first launch. Launch from a launch area; picking the robot up outside home costs a precision token.', 'info');
      this.toast('Match ready – launch your first run!');
    }
    endMatch(manual) {
      if (this.runner.running) this.stop();
      const score = this.world.score();
      if (manual && this.match.started && !this.match.over) this.log('Match ended early. Score: ' + score.total, 'info');
      this.match = { active: false, started: false, over: false, timeLeft: 150, lastLaunch: 0 };
      this.world.setup.tokens = 6;
      $('btnMatch').textContent = '🏁 Start match';
      $('btnMatch').classList.remove('active');
      $('matchOver').classList.remove('show');
    }
    matchTimeUp() {
      this.match.over = true;
      if (this.runner.running) this.runner.stop(true);
      this.world.robot.stopAll('brake');
      const score = this.world.score();
      this.updateRunState();
      $('matchOver').innerHTML = '<div class="card">⏱ Time is up!<div class="big">' + score.total + '</div>points<div style="font-size:14px;margin-top:8px;color:#ccc">See the Missions &amp; score tab for details. Press “End match” to practise again.</div></div>';
      $('matchOver').classList.add('show');
      this.log('⏱ Match over – final score ' + score.total, 'ok');
      this.setPanelsCollapsed(false);
      this.panels.show('missions');
    }

    // ------------------------------------------------------------ files
    async openFile(file) {
      try {
        await this.openData(file.name, await file.arrayBuffer());
      } catch (e) {
        this.log('Could not open ' + file.name + ': ' + e.message, 'err');
      }
    }

    // Opens a program from the bytes of a file (one from the computer, or one from GitHub) and returns the new program.
    // Some .llsp3 files of the team are the Python text itself and not a zip, so a file that does not start with "PK" is read as text.
    async openData(fileName, buffer) {
      const name = fileName.replace(/\.[^.]+$/, '');
      const text = () => new TextDecoder().decode(buffer);
      const head = new Uint8Array(buffer, 0, Math.min(2, buffer.byteLength));
      const isZip = head[0] === 0x50 && head[1] === 0x4B;
      let prog;
      if (/\.llsp3$/i.test(fileName) && isZip) {
        const res = await FLL.files.readLlsp3(buffer);
        prog = this.editors.addProgram({ name: res.name || name, mode: res.mode, python: res.python || '', blocks: res.blocks || null });
        this.log('Opened SPIKE project “' + (res.name || name) + '” (' + (res.mode === 'blocks' ? 'word blocks' : 'Python') + ').', 'ok');
        if (res.warnings && res.warnings.length) this.log('Some blocks are not supported by the simulator and were replaced by ⚠ blocks: ' + res.warnings.join(', '), 'warn');
      } else if (/\.json$/i.test(fileName)) {
        const obj = JSON.parse(text());
        if (!obj || (obj.format !== 'fll-sim-program' && !obj.python && !obj.blocks)) throw new Error('This is not a simulator program file.');
        prog = this.editors.addProgram({ name: obj.name || name, mode: obj.mode || (obj.blocks ? 'blocks' : 'python'), python: obj.python || '', blocks: obj.blocks || null });
        this.log('Opened “' + (obj.name || name) + '”.', 'ok');
      } else {
        prog = this.editors.addProgram({ name: name, mode: 'python', python: text() });
        this.log('Opened Python file “' + fileName + '”' + (/\.llsp3$/i.test(fileName) ? ' (a SPIKE project saved as plain text).' : '.'), 'ok');
      }
      return prog;
    }

    download(name, blob) {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = name;
      document.body.appendChild(a);
      a.click();
      setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
    }
    safeName() {
      const p = this.editors.current();
      return (p ? p.name : 'program').replace(/[^\w\- ]+/g, '').trim().replace(/\s+/g, '_') || 'program';
    }

    saveAction(act) {
      const ed = this.editors;
      const p = ed.current();
      if (!p) return;
      switch (act) {
        case 'download-py':
          this.download(this.safeName() + '.py', new Blob([ed.currentPython()], { type: 'text/x-python' }));
          break;
        case 'download-json': {
          const obj = { format: 'fll-sim-program', version: 1, name: p.name, mode: p.mode, python: p.mode === 'python' ? ed.currentPython() : undefined, blocks: p.mode === 'blocks' ? p.blocks : undefined };
          this.download(this.safeName() + '.json', new Blob([JSON.stringify(obj, null, 1)], { type: 'application/json' }));
          break;
        }
        case 'download-llsp3':
          this.download(this.safeName() + '.llsp3', FLL.files.writeLlsp3Python(p.name, ed.currentPython()));
          this.log('Downloaded a SPIKE app Python project. If the SPIKE app cannot open it, copy the code into a new Python project instead.', 'info');
          break;
        case 'copy-py':
          navigator.clipboard.writeText(ed.currentPython()).then(() => this.toast('Python code copied – paste it into the SPIKE app'), () => this.toast('Copy failed'));
          break;
        case 'rename':
          this.prompt('Rename program', p.name, (v) => ed.renameCurrent(v));
          break;
        case 'duplicate':
          ed.duplicateCurrent();
          break;
        case 'to-python':
          if (p.mode !== 'blocks') { this.toast('This program is already Python.'); break; }
          ed.convertToPython();
          this.toast('Created an editable Python copy of your blocks');
          break;
        case 'manage':
          this.manageDialog();
          break;
        case 'delete':
          this.confirm('Delete “' + p.name + '”? This cannot be undone.', () => ed.deleteCurrent());
          break;
      }
    }

    // ------------------------------------------------------------ dialogs
    modal(html, onReady) {
      $('modalBody').innerHTML = html;
      $('modal').classList.remove('hidden');
      const close = () => $('modal').classList.add('hidden');
      $('modal').onclick = (e) => { if (e.target === $('modal')) close(); };
      if (onReady) onReady(close);
      return close;
    }
    prompt(title, value, cb) {
      this.modal(`<h3>${FLL.ui.esc(title)}</h3><input type="text" id="mInput" value="${FLL.ui.esc(value)}"><div class="row"><button class="btn" id="mCancel">Cancel</button><button class="btn primary" id="mOk">OK</button></div>`, (close) => {
        const inp = $('mInput');
        inp.focus();
        inp.select();
        const ok = () => { const v = inp.value.trim(); close(); if (v) cb(v); };
        $('mOk').onclick = ok;
        $('mCancel').onclick = close;
        inp.onkeydown = (e) => { if (e.key === 'Enter') ok(); if (e.key === 'Escape') close(); };
      });
    }
    confirm(text, cb) {
      this.modal(`<p>${FLL.ui.esc(text)}</p><div class="row"><button class="btn" id="mCancel">Cancel</button><button class="btn primary" id="mOk">OK</button></div>`, (close) => {
        $('mOk').onclick = () => { close(); cb(); };
        $('mCancel').onclick = close;
      });
    }
    // Delete several saved programs at once. Programs, opened files and recorded drives are all kept in one list.
    manageDialog() {
      const ed = this.editors, all = ed.programs.slice();
      const isRecording = (p) => /^Recorded drive/.test(p.name);
      const count = (n) => n + (n === 1 ? ' program' : ' programs');
      const rows = all.map(p => `<label class="pick-row"><input type="checkbox" value="${FLL.ui.esc(p.id)}"><span class="pick-name">${p.mode === 'blocks' ? '🧩' : '🐍'} ${FLL.ui.esc(p.name)}${p.id === ed.currentId ? ' <span class="muted">(open)</span>' : ''}</span><span class="muted">${p.updated ? new Date(p.updated).toLocaleString() : ''}</span></label>`).join('');
      this.modal(`<h3>Delete saved programs</h3>
        <p class="small-note">Programs, opened files and recorded drives are all kept in this browser. Tick the ones to delete.</p>
        <div class="pick-tools"><button class="btn small" id="mAll">Select all</button><button class="btn small" id="mNone">Select none</button>${all.some(isRecording) ? '<button class="btn small" id="mRecs">Select recorded drives</button>' : ''}</div>
        <div class="pick-list">${rows}</div>
        <div class="row"><button class="btn" id="mCancel">Cancel</button><button class="btn danger" id="mDelete" disabled>Delete selected</button></div>`, (close) => {
        const boxes = Array.from($('modalBody').querySelectorAll('.pick-list input'));
        const del = $('mDelete');
        const sync = () => {
          const n = boxes.filter(b => b.checked).length;
          del.disabled = n === 0;
          del.textContent = n ? 'Delete ' + count(n) : 'Delete selected';
        };
        const tick = (test) => { boxes.forEach((b, i) => { b.checked = test(all[i]); }); sync(); };
        boxes.forEach(b => { b.onchange = sync; });
        $('mAll').onclick = () => tick(() => true);
        $('mNone').onclick = () => tick(() => false);
        if ($('mRecs')) $('mRecs').onclick = () => tick(isRecording);
        $('mCancel').onclick = close;
        del.onclick = () => {
          const ids = boxes.filter(b => b.checked).map(b => b.value);
          const note = ids.length === all.length ? ' One example program is added back, so the list is never empty.' : '';
          this.confirm('Delete ' + count(ids.length) + '? This cannot be undone.' + note, () => {
            ed.deletePrograms(ids);
            this.toast('Deleted ' + count(ids.length));
          });
        };
      });
    }
    newProgramDialog() {
      this.modal(`<h3>New program</h3>
        <div class="choice-list">
          <button class="btn" data-kind="blocks"><b>🧩 SPIKE word blocks</b><span>Drag and drop blocks, like in the LEGO Education SPIKE app.</span></button>
          <button class="btn" data-kind="spike"><b>🐍 SPIKE Python</b><span>Python for the SPIKE App 3 (motor_pair, runloop, async/await).</span></button>
          <button class="btn" data-kind="pybricks"><b>🐍 Pybricks Python</b><span>Pybricks MicroPython with DriveBase – popular with FLL teams.</span></button>
        </div>
        <div class="row"><button class="btn" id="mCancel">Cancel</button></div>`, (close) => {
        $('mCancel').onclick = close;
        document.querySelectorAll('.choice-list [data-kind]').forEach(b => {
          b.onclick = () => { close(); this.editors.newProgram(b.dataset.kind); };
        });
      });
    }

    // ------------------------------------------------------------ field interaction
    bindField() {
      const cv = $('fieldCanvas');
      const view = this.view;
      const pt = (e) => {
        const r = cv.getBoundingClientRect();
        return [e.clientX - r.left, e.clientY - r.top];
      };
      cv.addEventListener('pointerdown', (e) => {
        cv.focus();
        this.fieldFocused = true;
        const [sx, sy] = pt(e);
        const w = view.toWorld(sx, sy);
        cv.setPointerCapture(e.pointerId);
        if (e.button === 1 || e.button === 2) {
          this.drag = { kind: 'pan', sx: sx, sy: sy, ox: view.ox, oy: view.oy };
          return;
        }
        if (this.measureMode) {
          view.measure = { a: w, b: w };
          this.drag = { kind: 'measure' };
          return;
        }
        const rb = this.world.robot;
        if (this.canEditPose()) {
          const h = view.rotateHandlePos();
          const hs = view.toScreen(h[0], h[1]);
          if (Math.hypot(hs[0] - sx, hs[1] - sy) < 14) {
            this.beginRobotDrag();
            this.drag = { kind: 'rotate' };
            return;
          }
          if (geom.pointInPoly(w, rb.bodyPoly())) {
            this.beginRobotDrag();
            this.drag = { kind: 'move', dx: rb.pose.x - w[0], dy: rb.pose.y - w[1] };
            cv.style.cursor = 'grabbing';
            return;
          }
        }
        this.drag = { kind: 'pan', sx: sx, sy: sy, ox: view.ox, oy: view.oy };
        cv.style.cursor = 'grabbing';
      });
      cv.addEventListener('pointermove', (e) => {
        const [sx, sy] = pt(e);
        const w = view.toWorld(sx, sy);
        view.hover = w;
        const d = this.drag;
        if (!d) {
          if (this.measureMode) cv.style.cursor = 'crosshair';
          else if (this.canEditPose() && geom.pointInPoly(w, this.world.robot.bodyPoly())) cv.style.cursor = 'grab';
          else {
            const h = view.rotateHandlePos();
            const hs = view.toScreen(h[0], h[1]);
            cv.style.cursor = this.canEditPose() && Math.hypot(hs[0] - sx, hs[1] - sy) < 14 ? 'alias' : 'default';
          }
          return;
        }
        const rb = this.world.robot;
        if (d.kind === 'pan') {
          view.ox = d.ox + (sx - d.sx);
          view.oy = d.oy + (sy - d.sy);
          if (Math.abs(sx - d.sx) + Math.abs(sy - d.sy) > 3) view.autoFit = false;
        } else if (d.kind === 'measure') {
          view.measure.b = w;
        } else if (d.kind === 'move') {
          const T = field.TABLE;
          rb.pose.x = geom.clamp(w[0] + d.dx, T.x0, T.x1);
          rb.pose.y = geom.clamp(w[1] + d.dy, T.y0, T.y1);
          rb.trail = [];
          rb.lastPoseForTrail = null;
        } else if (d.kind === 'rotate') {
          let ang = Math.atan2(w[1] - rb.pose.y, w[0] - rb.pose.x);
          const step = e.shiftKey ? 1 : 5;
          const comp = Math.round(FLL.ui.compassOf(ang) / step) * step;
          rb.pose.theta = FLL.ui.thetaOf(comp);
          rb.trail = [];
          rb.lastPoseForTrail = null;
        }
      });
      const end = () => {
        const d = this.drag;
        this.drag = null;
        cv.style.cursor = 'default';
        if (d && (d.kind === 'move' || d.kind === 'rotate')) this.endRobotDrag();
      };
      cv.addEventListener('pointerup', end);
      cv.addEventListener('pointercancel', end);
      cv.addEventListener('pointerleave', () => { view.hover = null; });
      cv.addEventListener('wheel', (e) => {
        e.preventDefault();
        const [sx, sy] = pt(e);
        view.zoomAt(sx, sy, e.deltaY < 0 ? 1.15 : 1 / 1.15);
      }, { passive: false });
      cv.addEventListener('dblclick', () => view.fit());
      cv.addEventListener('contextmenu', (e) => e.preventDefault());
      cv.tabIndex = 0;
    }

    beginRobotDrag() {
      const rb = this.world.robot;
      this.recorder.note();
      this.resetDriveState();
      rb.stopAll('coast');
      this.interruptionCheck();
    }
    endRobotDrag() {
      const rb = this.world.robot;
      rb.resetYaw(0, this.world.time);
      if (this.recorder.active) this.recorder.note('robot moved by hand to x ' + Math.round(rb.pose.x) + ', y ' + Math.round(rb.pose.y) + ', facing ' + FLL.ui.compassOf(rb.pose.theta) + ' deg');
      const body = rb.bodyPoly();
      if (field.polyInLaunchArea('left', body) || field.polyInLaunchArea('right', body)) {
        this.startPose = { x: rb.pose.x, y: rb.pose.y, theta: rb.pose.theta };
        this.saveSettings();
        if (this.panels.active === 'setup') this.panels.syncSetup();
        this.toast('Start position: x ' + Math.round(rb.pose.x) + ', y ' + Math.round(rb.pose.y) + ', facing ' + FLL.ui.compassOf(rb.pose.theta) + '°');
      } else if (this.world.robotOverlaps()) {
        this.toast('The robot overlaps a wall or mission model', 2500);
      }
    }

    // ------------------------------------------------------------ screen layout
    // split: code | map, split-r: map | code, map: map only, info: only the info panels (Robot, Missions, Setup, Help).
    // The info panel below the map can be collapsed.
    setLayout(name, quiet) {
      if (['split', 'split-r', 'map', 'info'].indexOf(name) < 0) name = 'split';
      this.layout = name;
      const lay = $('layout');
      lay.classList.toggle('swap', name === 'split-r');
      lay.classList.toggle('maponly', name === 'map');
      lay.classList.toggle('infoonly', name === 'info');
      document.querySelectorAll('#layoutSwitch [data-layout]').forEach(b => b.classList.toggle('on', b.dataset.layout === name));
      this.settings.layout = name;
      if (!quiet) this.saveSettings();
      this.afterLayoutChange();
    }
    setPanelsCollapsed(collapsed, quiet) {
      this.panelsCollapsed = !!collapsed;
      $('fieldPane').classList.toggle('panels-collapsed', this.panelsCollapsed);
      $('btnPanelToggle').textContent = this.panelsCollapsed ? '▴ Show info' : '▾ Hide info';
      this.settings.panelsCollapsed = this.panelsCollapsed;
      if (!quiet) this.saveSettings();
      this.afterLayoutChange();
    }
    afterLayoutChange() {
      requestAnimationFrame(() => {
        this.view.resize();
        this.view.fit();
        this.editors.resize();
      });
    }

    setMeasure(on) {
      this.measureMode = on;
      $('btnMeasure').classList.toggle('on', on);
      if (!on) this.view.measure = null;
      else this.toast('Click and drag on the field to measure. Distances are in mm; angles like a compass (0° = north).', 4000);
    }
    setDrive(on) {
      if (on && this.runner.running) { this.toast('Stop the program first'); return; }
      if (on && this.match.active) { this.toast('Remote control is not allowed in a match (FLL rule 5).'); return; }
      this.driveMode = on;
      $('btnDrive').classList.toggle('on', on);
      $('driveHelp').classList.toggle('show', on);
      if (on) {
        $('fieldCanvas').focus();
        this.fieldFocused = true;
        this.interruptionCheck();
      } else {
        this.recorder.stop();
        this.world.robot.stopAll('brake');
        this.resetDriveState();
      }
    }

    bindKeys() {
      document.addEventListener('keydown', (e) => {
        const inEditor = e.target.closest && (e.target.closest('.CodeMirror') || e.target.closest('#blocklyDiv') || /INPUT|SELECT|TEXTAREA/.test(e.target.tagName));
        if (e.key === 'F5') { e.preventDefault(); this.run(); return; }
        if (e.key === 'Escape' && this.runner.running) { this.stop(); return; }
        if (inEditor) return;
        const k = e.key;
        const rec = (k === 'r' || k === 'R') && !e.repeat && !e.ctrlKey && !e.metaKey && !e.altKey && $('modal').classList.contains('hidden');
        if (this.driveMode) {
          if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' ', 'q', 'a', 'w', 's', 'Q', 'A', 'W', 'S', 'Shift'].indexOf(k) >= 0) {
            this.keys[k.length === 1 ? k.toLowerCase() : k] = true;
            e.preventDefault();
          }
          if (rec) { this.recorder.toggle(); return; }
          if (k === 'Escape') this.setDrive(false);
          return;
        }
        if (k === 'Home') { e.preventDefault(); this.robotToStart(); return; }
        if (k === 'm' || k === 'M') { this.setMeasure(!this.measureMode); return; }
        if (k === 'd' || k === 'D') { this.setDrive(true); return; }
        if (rec) { this.recorder.toggle(); return; }
        if (!this.fieldFocused || !this.canEditPose()) return;
        const rb = this.world.robot;
        const step = e.shiftKey ? 10 : 1;
        let moved = true;
        if (k === 'ArrowUp') rb.pose.y += step;
        else if (k === 'ArrowDown') rb.pose.y -= step;
        else if (k === 'ArrowLeft') rb.pose.x -= step;
        else if (k === 'ArrowRight') rb.pose.x += step;
        else if (k === '[') rb.pose.theta += geom.rad(e.shiftKey ? 15 : 1);
        else if (k === ']') rb.pose.theta -= geom.rad(e.shiftKey ? 15 : 1);
        else moved = false;
        if (moved) { e.preventDefault(); this.interruptionCheck(); rb.trail = []; this.endRobotDragQuiet(); }
      });
      document.addEventListener('keyup', (e) => {
        const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
        delete this.keys[k];
      });
      window.addEventListener('blur', () => { this.keys = {}; });
      document.addEventListener('pointerdown', (e) => { if (!e.target.closest('#fieldCanvas')) this.fieldFocused = false; });
    }
    endRobotDragQuiet() {
      const rb = this.world.robot;
      const body = rb.bodyPoly();
      if (field.polyInLaunchArea('left', body) || field.polyInLaunchArea('right', body)) {
        this.startPose = { x: rb.pose.x, y: rb.pose.y, theta: rb.pose.theta };
        clearTimeout(this.saveTimer);
        this.saveTimer = setTimeout(() => this.saveSettings(), 500);
      }
    }

    // After the motors were stopped from outside, the next key press must issue its command again.
    resetDriveState() { this.lastDrive = null; this.armE = 0; this.armF = 0; }

    // Keyboard drive: tank speeds from arrow keys.
    applyDrive() {
      const rb = this.world.robot;
      const cfg = rb.config;
      const L = rb.motors[cfg.driveLeft], R = rb.motors[cfg.driveRight];
      if (!L || !R) return;
      const k = this.keys;
      const fast = k.Shift ? 220 : 520;
      let f = 0, t = 0;
      if (k.ArrowUp) f += 1;
      if (k.ArrowDown) f -= 1;
      if (k.ArrowLeft) t += 1;
      if (k.ArrowRight) t -= 1;
      if (k[' ']) { f = 0; t = 0; }
      const vl = (f - t * 0.6) * fast, vr = (f + t * 0.6) * fast;
      const sl = cfg.leftReversed ? -1 : 1;
      const sr = cfg.rightReversed ? -1 : 1;
      const want = [Math.round(vl * sl), Math.round(vr * sr)];
      if (!this.lastDrive || this.lastDrive[0] !== want[0] || this.lastDrive[1] !== want[1]) {
        if (want[0] === 0 && want[1] === 0) { L.stop('brake'); R.stop('brake'); }
        else { L.runAt(want[0], 3000); R.runAt(want[1], 3000); }
        this.lastDrive = want;
        this.recorder.drive([Math.round(vl) || 0, Math.round(vr) || 0]);   // forward = positive, whatever the wiring
      }
      const armDrive = (port, up, down) => {
        const m = rb.motors[port];
        if (!m || rb.isDrivePort(port)) return;
        const v = (k[up] ? 1 : 0) - (k[down] ? 1 : 0);
        const key = 'arm' + port;
        if (this[key] !== v) {
          if (v) m.runAt(v * 400, 4000); else m.stop('hold');
          this[key] = v;
          this.recorder.arm(port, v);
        }
      };
      armDrive('E', 'q', 'a');
      armDrive('F', 'w', 's');
    }

    // ------------------------------------------------------------ sound
    beep(freq, dur, vol) {
      try {
        if (!this.audio) this.audio = new (window.AudioContext || window.webkitAudioContext)();
        const ctx = this.audio;
        if (this.osc) { try { this.osc.stop(); } catch (e) { /* already stopped */ } this.osc = null; }
        if (!freq || !dur) return;
        const o = ctx.createOscillator();
        const g = ctx.createGain();
        o.type = 'square';
        o.frequency.value = Math.max(40, Math.min(8000, freq));
        g.gain.value = 0.04 * (vol || 100) / 100;
        o.connect(g).connect(ctx.destination);
        const realDur = Math.min(dur / 1000 / this.speed, 10);
        o.start();
        o.stop(ctx.currentTime + realDur);
        this.osc = o;
      } catch (e) { /* audio not available */ }
    }

    // ------------------------------------------------------------ main loop
    frame(ts) {
      const dtReal = Math.min(0.1, this.lastFrame ? (ts - this.lastFrame) / 1000 : 0.016);
      this.lastFrame = ts;
      if (!this.drag || this.drag.kind === 'pan' || this.drag.kind === 'measure') {
        this.acc += dtReal * this.speed;
        let steps = 0;
        const maxSteps = 400;
        while (this.acc >= DT && steps < maxSteps) {
          this.tick(DT);
          this.acc -= DT;
          steps++;
        }
        if (steps >= maxSteps) this.acc = 0;
      }
      if (this.view3d.on) this.view3d.draw(); else this.view.draw();
      if (ts - this.lastPanelUpdate > 100) {
        this.lastPanelUpdate = ts;
        this.panels.updateRobot();
        const score = this.panels.updateMissions(false);
        $('scoreTotal').textContent = score.total;
        this.updateTimer();
        if (this.runner.running) this.editors.highlightRunning(this.runner.activeLines());
        this.recorder.tick();
      }
      requestAnimationFrame((t) => this.frame(t));
    }

    tick(dt) {
      if (this.match.over) return;
      if (this.runner.running) this.runner.tick();
      if (this.driveMode) this.applyDrive();
      this.world.active = this.runner.running || this.driveMode;
      this.world.step(dt);
      if (this.match.active && this.match.started) {
        this.match.timeLeft -= dt;
        if (this.match.timeLeft <= 0) {
          this.match.timeLeft = 0;
          this.matchTimeUp();
        }
      }
    }

    updateTimer() {
      const t = Math.max(0, this.match.active ? this.match.timeLeft : 150);
      const m = Math.floor(t / 60), s = Math.floor(t % 60);
      $('matchTimer').textContent = m + ':' + String(s).padStart(2, '0');
      const box = $('matchTimer').parentElement;
      box.classList.toggle('running', this.match.active && this.match.started && !this.match.over);
      box.classList.toggle('low', this.match.active && t < 30);
    }

    onMissionLog(e) {
      if (this.runner.running || this.driveMode) {
        if (e.mission) this.toast('M' + String(e.mission).padStart(2, '0') + ': ' + e.text, 2200);
      }
    }

    missionDone(m) {
      const items = m.def.score(m, this.world.setup);
      return items.reduce((a, b) => a + b.points, 0) > 0 && !(m.id === 10);
    }
  }

  FLL.App = App;
  window.addEventListener('DOMContentLoaded', () => {
    try {
      const app = new App();
      FLL.app = app;
      app.init();
    } catch (e) {
      document.body.innerHTML = '<pre style="padding:20px;color:#b00">The simulator could not start:\n' + (e && e.stack || e) + '</pre>';
      throw e;
    }
  });
})(typeof globalThis !== 'undefined' ? globalThis : window);
