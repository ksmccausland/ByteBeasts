/*
 * Program management and the two editors (SPIKE word blocks via Blockly, Python via CodeMirror).
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};

  const STORE_KEY = 'fll-bioglow-programs-v1';
  const CURRENT_KEY = 'fll-bioglow-current-v1';

  const uid = () => 'p' + Date.now().toString(36) + Math.random().toString(36).slice(2, 7);

  // API names for Python autocompletion.
  const API = {
    motor_pair: ['PAIR_1', 'PAIR_2', 'PAIR_3', 'pair(', 'unpair(', 'move(', 'move_for_degrees(', 'move_for_time(', 'move_tank(', 'move_tank_for_degrees(', 'move_tank_for_time(', 'stop('],
    motor: ['run(', 'run_for_degrees(', 'run_for_time(', 'run_to_absolute_position(', 'run_to_relative_position(', 'stop(', 'relative_position(', 'reset_relative_position(', 'absolute_position(', 'velocity(', 'set_duty_cycle(', 'get_duty_cycle(', 'BRAKE', 'COAST', 'HOLD', 'CONTINUE', 'SMART_BRAKE', 'SMART_COAST', 'CLOCKWISE', 'COUNTERCLOCKWISE', 'SHORTEST_PATH', 'LONGEST_PATH', 'READY', 'RUNNING', 'STALLED', 'CANCELLED'],
    port: ['A', 'B', 'C', 'D', 'E', 'F'],
    runloop: ['run(', 'sleep_ms(', 'until('],
    motion_sensor: ['tilt_angles()', 'reset_yaw(0)', 'angular_velocity()', 'acceleration()', 'up_face()', 'gesture()', 'stable()'],
    color_sensor: ['color(', 'reflection(', 'rgbi('],
    color: ['BLACK', 'MAGENTA', 'PURPLE', 'BLUE', 'AZURE', 'TURQUOISE', 'GREEN', 'YELLOW', 'ORANGE', 'RED', 'WHITE', 'UNKNOWN'],
    distance_sensor: ['distance(', 'show(', 'clear('],
    force_sensor: ['force(', 'pressed(', 'raw('],
    light_matrix: ['write(', 'show_image(', 'show(', 'set_pixel(', 'get_pixel(', 'clear()', 'IMAGE_HEART', 'IMAGE_HAPPY', 'IMAGE_SAD', 'IMAGE_YES', 'IMAGE_NO', 'IMAGE_ARROW_N', 'IMAGE_ARROW_E', 'IMAGE_ARROW_S', 'IMAGE_ARROW_W'],
    sound: ['beep(', 'stop()'],
    button: ['pressed(', 'LEFT', 'RIGHT'],
    light: ['color(', 'POWER', 'CONNECT'],
    time: ['sleep(', 'sleep_ms(', 'ticks_ms()', 'ticks_diff('],
    math: ['sqrt(', 'sin(', 'cos(', 'tan(', 'atan2(', 'radians(', 'degrees(', 'pi', 'floor(', 'ceil(', 'fabs(', 'copysign('],
    random: ['randint(', 'random()', 'choice(', 'uniform('],
    robot: ['straight(', 'turn(', 'curve(', 'drive(', 'stop()', 'brake()', 'distance()', 'angle()', 'reset()', 'settings(', 'use_gyro(True)', 'done()', 'state()'],
    hub: ['display', 'light', 'speaker', 'imu', 'buttons', 'battery', 'system'],
    'hub.imu': ['heading()', 'reset_heading(0)', 'tilt()', 'angular_velocity()', 'stationary()'],
    'hub.display': ['text(', 'number(', 'char(', 'icon(', 'pixel(', 'off()'],
    'hub.speaker': ['beep(', 'play_notes('],
    'hub.light': ['on(', 'off()', 'blink('],
    Port: ['A', 'B', 'C', 'D', 'E', 'F'],
    Direction: ['CLOCKWISE', 'COUNTERCLOCKWISE'],
    Stop: ['HOLD', 'BRAKE', 'COAST', 'NONE'],
    Color: ['RED', 'GREEN', 'BLUE', 'YELLOW', 'WHITE', 'BLACK', 'NONE', 'ORANGE', 'VIOLET', 'MAGENTA', 'CYAN', 'GRAY', 'BROWN']
  };
  const MOTOR_METHODS = ['run(', 'run_angle(', 'run_target(', 'run_time(', 'run_until_stalled(', 'stop()', 'brake()', 'hold()', 'angle()', 'reset_angle(0)', 'speed()', 'dc(', 'done()', 'stalled()'];
  const KEYWORDS = ['async def ', 'await ', 'def ', 'for ', 'while ', 'if ', 'elif ', 'else:', 'return ', 'import ', 'from ', 'True', 'False', 'None', 'break', 'continue', 'print(', 'range(', 'len(', 'abs(', 'round(', 'int(', 'float(', 'str(', 'min(', 'max(', 'global '];

  class Editors {
    constructor(app) {
      this.app = app;
      this.programs = [];
      this.currentId = null;
      this.tab = 'python';
      this.workspace = null;
      this.gen = null;
      this.cm = null;
      this.runMap = null;
      this.highlighted = [];
      this.errorMark = null;
      this.suppressBlocksSave = false;
      this.highlightedBlock = null;
    }

    // ------------------------------------------------------------ setup
    init() {
      this.initCodeMirror();
      this.initBlockly();
      this.loadPrograms();
      document.querySelectorAll('#codeTabs .tab').forEach(btn => {
        btn.addEventListener('click', () => { if (!btn.disabled) this.showTab(btn.dataset.tab); });
      });
      document.getElementById('programSel').addEventListener('change', (e) => this.select(e.target.value));
    }

    initCodeMirror() {
      const ta = document.getElementById('pyText');
      this.cm = CodeMirror.fromTextArea(ta, {
        mode: { name: 'python', version: 3, singleLineStringErrors: false },
        lineNumbers: true,
        indentUnit: 4,
        tabSize: 4,
        indentWithTabs: false,
        matchBrackets: true,
        autoCloseBrackets: true,
        styleActiveLine: true,
        gutters: ['CodeMirror-linenumbers', 'errors'],
        extraKeys: {
          Tab: (cm) => {
            if (cm.somethingSelected()) cm.indentSelection('add');
            else cm.replaceSelection('    ', 'end');
          },
          'Shift-Tab': (cm) => cm.indentSelection('subtract'),
          'Ctrl-/': 'toggleComment',
          'Cmd-/': 'toggleComment',
          'Ctrl-Space': (cm) => this.showHint(cm),
          F5: () => this.app.run()
        }
      });
      this.cm.on('change', () => {
        const p = this.current();
        if (!p || p.mode !== 'python' || this.cm.getOption('readOnly')) return;
        p.python = this.cm.getValue();
        this.clearError();
        this.scheduleSave();
      });
      this.cm.on('inputRead', (cm, change) => {
        if (change.text && change.text[0] === '.') this.showHint(cm);
      });
    }

    showHint(cm) {
      cm.showHint({
        completeSingle: false,
        hint: () => {
          const cur = cm.getCursor();
          const line = cm.getLine(cur.line).slice(0, cur.ch);
          const m = /([A-Za-z_][\w.]*)?\.?([A-Za-z_]\w*)?$/.exec(line);
          let obj = null, word = '';
          const mm = /([A-Za-z_][\w]*(?:\.[A-Za-z_]\w*)*)\.(\w*)$/.exec(line);
          if (mm) { obj = mm[1]; word = mm[2]; }
          else {
            const w = /([A-Za-z_]\w*)$/.exec(line);
            word = w ? w[1] : '';
          }
          void m;
          let list;
          if (obj) {
            list = API[obj] || API[obj.split('.').slice(-2).join('.')] || API[obj.split('.').pop()];
            if (!list) {
              const code = cm.getValue();
              const re = new RegExp('\\b' + obj.replace('.', '\\.') + '\\s*=\\s*(\\w+)\\(');
              const t = re.exec(code);
              if (t && t[1] === 'DriveBase') list = API.robot;
              else if (t && t[1] === 'Motor') list = MOTOR_METHODS;
              else if (t && t[1] === 'PrimeHub') list = API.hub;
              else if (t && t[1] === 'ColorSensor') list = ['color()', 'reflection()', 'hsv()', 'ambient()'];
              else if (t && t[1] === 'UltrasonicSensor') list = ['distance()'];
              else if (t && t[1] === 'ForceSensor') list = ['force()', 'pressed()', 'touched()'];
            }
            list = list || [];
          } else {
            list = KEYWORDS.concat(Object.keys(API).filter(k => k.indexOf('.') < 0));
          }
          const filtered = list.filter(x => x.toLowerCase().startsWith(word.toLowerCase()));
          return { list: filtered, from: CodeMirror.Pos(cur.line, cur.ch - word.length), to: cur };
        }
      });
    }

    initBlockly() {
      const B = G.Blockly;
      FLL.blocks.register(B);
      this.gen = FLL.blocks.makeGenerator(B);
      const cfg = this.app.world.robot.config;
      this.workspace = B.inject('blocklyDiv', {
        toolbox: FLL.blocks.toolbox(this.blockDefaults(cfg)),
        renderer: 'zelos',
        theme: FLL.blocks.theme(B),
        media: 'vendor/blockly/media/',
        sounds: false,
        trashcan: true,
        zoom: { controls: true, wheel: false, startScale: 0.62, maxScale: 2, minScale: 0.3, scaleSpeed: 1.15, pinch: true },
        move: { scrollbars: true, drag: true, wheel: true },
        grid: { spacing: 40, length: 2, colour: '#e3e6ea', snap: false }
      });
      this.workspace.registerToolboxCategoryCallback('VARIABLE_SPIKE', FLL.blocks.variablesFlyout(B));
      this.workspace.addChangeListener((e) => {
        if (this.suppressBlocksSave) return;
        if (e.isUiEvent) return;
        const p = this.current();
        if (!p || p.mode !== 'blocks') return;
        p.blocks = B.serialization.workspaces.save(this.workspace);
        this.scheduleSave();
        if (this.tab === 'python') this.refreshGenerated();
      });
    }

    blockDefaults(cfg) {
      const dev = cfg.devices;
      const arm = ['E', 'F', 'A', 'B', 'C', 'D'].find(p => dev[p] && dev[p].type === 'motor' && p !== cfg.driveLeft && p !== cfg.driveRight) || 'E';
      const colorPort = ['B', 'A', 'C', 'D', 'E', 'F'].find(p => dev[p] && dev[p].type === 'color') || 'B';
      return { pair: cfg.driveLeft + cfg.driveRight, arm: arm, colorPort: colorPort };
    }
    updateToolbox() {
      if (this.workspace) this.workspace.updateToolbox(FLL.blocks.toolbox(this.blockDefaults(this.app.world.robot.config)));
    }

    // ------------------------------------------------------------ storage
    loadPrograms() {
      let list = [];
      try { list = JSON.parse(localStorage.getItem(STORE_KEY) || '[]'); } catch (e) { list = []; }
      if (!Array.isArray(list) || !list.length) {
        list = FLL.examples.defaultPrograms(this.app.world.robot.config).map(p => Object.assign({ id: uid(), updated: Date.now() }, p));
      }
      this.programs = list;
      let cur = null;
      try { cur = localStorage.getItem(CURRENT_KEY); } catch (e) { cur = null; }
      this.refreshSelect();
      this.select(this.programs.some(p => p.id === cur) ? cur : this.programs[0].id);
    }
    scheduleSave() {
      clearTimeout(this.saveTimer);
      this.saveTimer = setTimeout(() => this.savePrograms(), 400);
    }
    savePrograms() {
      const p = this.current();
      if (p) p.updated = Date.now();
      try {
        localStorage.setItem(STORE_KEY, JSON.stringify(this.programs));
        localStorage.setItem(CURRENT_KEY, this.currentId || '');
      } catch (e) {
        this.app.log('Could not save programs in this browser: ' + e.message, 'warn');
      }
    }
    refreshSelect() {
      const sel = document.getElementById('programSel');
      sel.innerHTML = '';
      for (const p of this.programs) {
        const o = document.createElement('option');
        o.value = p.id;
        o.textContent = (p.mode === 'blocks' ? '🧩 ' : '🐍 ') + p.name;
        sel.appendChild(o);
      }
      if (this.currentId) sel.value = this.currentId;
    }
    current() { return this.programs.find(p => p.id === this.currentId) || null; }

    select(id) {
      if (this.app.runner && this.app.runner.running) this.app.stop();
      const rec = this.app.recorder;
      if (rec && rec.active && !rec.owns(id)) rec.stop(true);
      this.currentId = id;
      const p = this.current();
      if (!p) return;
      document.getElementById('programSel').value = id;
      this.clearError();
      this.clearRunHighlight();
      const blocksTab = document.querySelector('#codeTabs [data-tab=blocks]');
      blocksTab.disabled = p.mode !== 'blocks';
      blocksTab.title = p.mode === 'blocks' ? '' : 'This program is written in Python. Create a new word-block program to use blocks.';
      if (p.mode === 'blocks') {
        this.loadBlocks(p);
        this.showTab('blocks');
      } else {
        this.showTab('python');
      }
      try { localStorage.setItem(CURRENT_KEY, id); } catch (e) { /* ignore */ }
    }

    loadBlocks(p) {
      const B = G.Blockly;
      this.suppressBlocksSave = true;
      try {
        this.workspace.clear();
        if (p.blocks) B.serialization.workspaces.load(p.blocks, this.workspace);
      } catch (e) {
        this.app.log('Could not load the blocks: ' + e.message, 'err');
      } finally {
        this.suppressBlocksSave = false;
      }
      setTimeout(() => {
        B.svgResize(this.workspace);
        const tops = this.workspace.getTopBlocks(true);
        if (tops.length) this.workspace.scrollCenter();
      }, 30);
    }

    showTab(tab) {
      const p = this.current();
      this.tab = tab;
      document.querySelectorAll('#codeTabs .tab').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
      document.getElementById('blocklyDiv').style.visibility = tab === 'blocks' ? 'visible' : 'hidden';
      document.getElementById('pythonDiv').style.visibility = tab === 'python' ? 'visible' : 'hidden';
      const note = document.getElementById('modeNote');
      if (tab === 'blocks') {
        G.Blockly.svgResize(this.workspace);
        note.textContent = 'SPIKE word blocks';
      } else {
        if (p && p.mode === 'blocks') {
          this.refreshGenerated();
          this.cm.setOption('readOnly', true);
          note.textContent = 'Python made from your blocks (read-only)';
        } else {
          this.cm.setOption('readOnly', this.app.recorder && this.app.recorder.active ? 'nocursor' : false);
          if (p && this.cm.getValue() !== p.python) this.cm.setValue(p.python || '');
          note.textContent = this.detectApi(p ? p.python : '');
        }
        setTimeout(() => this.cm.refresh(), 0);
      }
    }

    detectApi(code) {
      if (/\bpybricks\b/.test(code || '')) return 'Pybricks Python';
      if (/\b(motor_pair|runloop|from hub)\b/.test(code || '')) return 'SPIKE App 3 Python';
      return 'Python';
    }

    refreshGenerated() {
      const p = this.current();
      if (!p || p.mode !== 'blocks') return;
      const built = this.buildBlocksCode();
      this.runMap = built;
      if (this.cm.getValue() !== built.display) this.cm.setValue(built.display);
    }

    // Generate python from blocks: run version (with block markers) and display version.
    buildBlocksCode() {
      const B = G.Blockly;
      const defaults = this.blockDefaults(this.app.world.robot.config);
      this.gen.STATEMENT_PREFIX = '#blk:%1\n';
      let runCode;
      try {
        runCode = FLL.blocks.generateProgram(B, this.gen, this.workspace, defaults);
      } finally {
        this.gen.STATEMENT_PREFIX = null;
      }
      const lines = runCode.split('\n');
      const display = [];
      const lineToBlock = {};   // display line (1-based) -> block id
      let pending = null;
      for (const ln of lines) {
        const m = /^\s*#blk:'((?:[^'\\]|\\.)*)'\s*$/.exec(ln);
        if (m) { pending = m[1].replace(/\\(.)/g, '$1'); continue; }
        display.push(ln);
        if (pending && ln.trim()) { lineToBlock[display.length] = pending; pending = null; }
      }
      const code = display.join('\n');
      return { display: code, run: code, lineToBlock: lineToBlock };
    }

    // ------------------------------------------------------------ running
    getRunCode() {
      const p = this.current();
      if (!p) return null;
      if (p.mode === 'blocks') {
        const built = this.buildBlocksCode();
        this.runMap = built;
        if (this.tab === 'python' && this.cm.getValue() !== built.display) this.cm.setValue(built.display);
        return { code: built.run, mode: 'blocks', name: p.name };
      }
      p.python = this.tab === 'python' && !this.cm.getOption('readOnly') ? this.cm.getValue() : p.python;
      return { code: p.python || '', mode: 'python', name: p.name };
    }

    clearRunHighlight() {
      for (const h of this.highlighted) this.cm.removeLineClass(h, 'background', 'cm-running-line');
      this.highlighted = [];
      if (this.workspace && this.highlightedBlock) {
        this.workspace.highlightBlock(null);
        this.highlightedBlock = null;
      }
    }

    highlightRunning(lines) {
      const key = lines.join(',');
      if (key === this.lastHighlightKey) return;
      this.lastHighlightKey = key;
      for (const h of this.highlighted) this.cm.removeLineClass(h, 'background', 'cm-running-line');
      this.highlighted = [];
      for (const ln of lines) {
        if (ln - 1 < this.cm.lineCount()) {
          const h = this.cm.addLineClass(ln - 1, 'background', 'cm-running-line');
          this.highlighted.push(h);
        }
      }
      const p = this.current();
      if (p && p.mode === 'blocks' && this.runMap && this.workspace) {
        const id = this.blockForLine(lines[0]);
        if (id !== this.highlightedBlock) {
          this.workspace.highlightBlock(id || null);
          this.highlightedBlock = id;
        }
      }
    }

    blockForLine(line) {
      if (!this.runMap || !line) return null;
      for (let l = line; l > 0; l--) {
        if (this.runMap.lineToBlock[l]) return this.runMap.lineToBlock[l];
      }
      return null;
    }

    markError(line) {
      this.clearError();
      if (!line) return;
      const p = this.current();
      if (line - 1 < this.cm.lineCount()) {
        this.errorLine = this.cm.addLineClass(line - 1, 'background', 'cm-error-line');
        const marker = document.createElement('div');
        marker.className = 'cm-error-gutter';
        marker.textContent = '●';
        this.cm.setGutterMarker(line - 1, 'errors', marker);
        if (p && p.mode === 'python') this.cm.scrollIntoView({ line: line - 1, ch: 0 }, 80);
      }
      if (p && p.mode === 'blocks') {
        const id = this.blockForLine(line);
        if (id) {
          const blk = this.workspace.getBlockById(id);
          if (blk) { blk.select(); this.workspace.centerOnBlock(id); }
        }
      }
    }
    clearError() {
      if (this.errorLine) { this.cm.removeLineClass(this.errorLine, 'background', 'cm-error-line'); this.errorLine = null; }
      this.cm.clearGutter('errors');
    }
    goToLine(line) {
      const p = this.current();
      if (p && p.mode === 'blocks' && this.tab === 'blocks') {
        const id = this.blockForLine(line);
        if (id) { const blk = this.workspace.getBlockById(id); if (blk) { blk.select(); this.workspace.centerOnBlock(id); } }
        return;
      }
      this.showTab('python');
      this.cm.focus();
      this.cm.setCursor({ line: line - 1, ch: 0 });
      this.cm.scrollIntoView({ line: line - 1, ch: 0 }, 100);
    }

    // ------------------------------------------------------------ program actions
    addProgram(p, select) {
      const prog = Object.assign({ id: uid(), updated: Date.now() }, p);
      let name = prog.name || 'Program';
      let n = 2;
      while (this.programs.some(x => x.name === name)) name = (prog.name || 'Program') + ' ' + n++;
      prog.name = name;
      this.programs.push(prog);
      this.refreshSelect();
      if (select !== false) this.select(prog.id);
      this.savePrograms();
      return prog;
    }
    newProgram(kind) {
      const cfg = this.app.world.robot.config;
      const t = FLL.examples.template(kind, cfg);
      return this.addProgram(t);
    }
    renameCurrent(name) {
      const p = this.current();
      if (!p || !name) return;
      p.name = name;
      this.refreshSelect();
      this.savePrograms();
    }
    duplicateCurrent() {
      const p = this.current();
      if (!p) return;
      const copy = JSON.parse(JSON.stringify(p));
      delete copy.id;
      copy.name = p.name + ' copy';
      this.addProgram(copy);
    }
    // Delete one or more saved programs. A recording that is still writing one of them is cancelled first. When the
    // open program is one of them, the first program left is opened (an example program when none is left).
    deletePrograms(ids) {
      const rec = this.app.recorder;
      if (rec && rec.active && ids.some(id => rec.owns(id))) rec.cancel();
      const gone = new Set(ids);
      const openGone = gone.has(this.currentId);
      this.programs = this.programs.filter(x => !gone.has(x.id));
      if (!this.programs.length) this.programs = FLL.examples.defaultPrograms(this.app.world.robot.config).slice(0, 1).map(x => Object.assign({ id: uid() }, x));
      this.refreshSelect();
      if (openGone) this.select(this.programs[0].id);
      this.savePrograms();
    }
    deleteCurrent() {
      const p = this.current();
      if (!p) return;
      this.deletePrograms([p.id]);
    }
    convertToPython() {
      const p = this.current();
      if (!p || p.mode !== 'blocks') return null;
      const built = this.buildBlocksCode();
      return this.addProgram({ name: p.name + ' (Python)', mode: 'python', python: built.display });
    }
    currentPython() {
      const p = this.current();
      if (!p) return '';
      if (p.mode === 'blocks') return this.buildBlocksCode().display;
      return this.tab === 'python' && !this.cm.getOption('readOnly') ? this.cm.getValue() : (p.python || '');
    }
    resize() {
      if (this.workspace && this.tab === 'blocks') G.Blockly.svgResize(this.workspace);
      if (this.cm) this.cm.refresh();
    }
  }

  FLL.Editors = Editors;
})(typeof globalThis !== 'undefined' ? globalThis : window);
