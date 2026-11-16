/*
 * Generator-based Python interpreter for the FLL simulator.
 *
 * Every statement yields py.STEP so a scheduler can share time between tasks and
 * the robot simulation. Natives can return py.Suspend to block the running task
 * (e.g. time.sleep) and asynchronous APIs return py.Awaitable objects.
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};
  const py = FLL.py = FLL.py || {};
  const {
    None, PyFloat, PyList, PyTuple, PyDict, PySet, PySlice, PyIterator, DONE,
    PyFunction, PyBoundMethod, PyNative, PyClass, PyInstance, PyModule, PyCoroutine,
    PyStaticMethod, PyClassMethod, PyProperty, PySuper, PyException, Suspend, Awaitable, STEP,
    err, truthy
  } = py;

  const EMPTY_KW = Object.freeze(Object.create(null));
  const BREAK = { sig: 'break' };
  const CONTINUE = { sig: 'continue' };
  class Return { constructor(v) { this.value = v; } }
  const COMP_SCOPE = { kind: 'comp', locals: new Set(), globals: new Set(), nonlocals: new Set() };

  class Frame {
    constructor(scope, parent, module, fn, name) {
      this.vars = new Map();
      this.scope = scope;
      this.parent = parent;
      this.module = module || this;
      this.fn = fn || null;
      this.name = name || '<module>';
      this.line = 0;
    }
  }

  // Levenshtein distance for "did you mean" hints.
  function editDistance(a, b) {
    const m = a.length, n = b.length;
    if (Math.abs(m - n) > 3) return 99;
    const d = [];
    for (let i = 0; i <= m; i++) { d[i] = [i]; }
    for (let j = 1; j <= n; j++) d[0][j] = j;
    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        const c = a[i - 1].toLowerCase() === b[j - 1].toLowerCase() ? 0 : 1;
        d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c);
      }
    }
    return d[m][n];
  }
  function suggest(name, candidates) {
    let best = null, bestD = 99;
    for (const c of candidates) {
      if (typeof c !== 'string' || c.startsWith('__')) continue;
      const dd = editDistance(name, c);
      const limit = name.length <= 3 ? 1 : (name.length <= 6 ? 2 : 3);
      if (dd <= limit && dd < bestD) { best = c; bestD = dd; }
    }
    return best;
  }

  // Hints for names that usually come from an import.
  const IMPORT_HINTS = {
    port: 'from hub import port',
    motion_sensor: 'from hub import motion_sensor',
    light_matrix: 'from hub import light_matrix',
    button: 'from hub import button',
    sound: 'from hub import sound',
    light: 'from hub import light',
    motor: 'import motor',
    motor_pair: 'import motor_pair',
    runloop: 'import runloop',
    color_sensor: 'import color_sensor',
    color: 'import color',
    distance_sensor: 'import distance_sensor',
    force_sensor: 'import force_sensor',
    time: 'import time',
    math: 'import math',
    random: 'import random',
    PrimeHub: 'from pybricks.hubs import PrimeHub',
    Motor: 'from pybricks.pupdevices import Motor',
    ColorSensor: 'from pybricks.pupdevices import ColorSensor',
    UltrasonicSensor: 'from pybricks.pupdevices import UltrasonicSensor',
    ForceSensor: 'from pybricks.pupdevices import ForceSensor',
    Port: 'from pybricks.parameters import Port',
    Direction: 'from pybricks.parameters import Direction',
    Stop: 'from pybricks.parameters import Stop',
    Color: 'from pybricks.parameters import Color',
    Button: 'from pybricks.parameters import Button',
    DriveBase: 'from pybricks.robotics import DriveBase',
    wait: 'from pybricks.tools import wait',
    StopWatch: 'from pybricks.tools import StopWatch',
    multitask: 'from pybricks.tools import multitask',
    run_task: 'from pybricks.tools import run_task'
  };

  class Task {
    constructor(gen, name) {
      this.gen = gen;
      this.name = name;
      this.done = false;
      this.waiting = null;
      this.resumeValue = undefined;
      this.depth = 0;
      this.frame = null;
      this.excStack = [];
      this.error = null;
      this.result = None;
    }
    get line() { return this.frame ? this.frame.line : 0; }
  }

  class Interpreter {
    constructor(opts) {
      opts = opts || {};
      this.print = opts.print || (s => { if (typeof console !== 'undefined') console.log(s); });
      this.warn = opts.warn || this.print;
      this.moduleFactories = Object.assign({}, py.stdlibModules || {}, opts.modules || {});
      this.moduleCache = new Map();
      this.maxDepth = opts.maxDepth || 120;
      this.task = null;
      this._excStack = [];
      this.warnedLines = new Set();
      this.now = opts.now || (() => 0);
      this.scheduler = null;
      const self = this;
      this.ctx = {
        interp: this,
        host: opts.host || null,
        call: (f, a, k) => self.callSync(f, a || [], k || EMPTY_KW),
        now: () => self.now(),
        get task() { return self.task; },
        spawn: (gen, name) => self.scheduler.spawn(gen, name),
        runCoroutine: (c) => self.runCoroutine(c)
      };
      this.builtins = new Map();
      if (py.installBuiltins) py.installBuiltins(this);
      py.activeInterp = this;
      installHooks();
    }

    get excStack() { return this.task ? this.task.excStack : this._excStack; }

    parse(source) {
      try {
        return py.parse(source);
      } catch (e) {
        if (e instanceof py.SyntaxErr) {
          const pe = err(e.kind === 'IndentationError' ? 'IndentationError' : 'SyntaxError', e.message);
          pe.line = e.line;
          pe.col = e.col;
          pe.pyTraceback = [{ name: '<module>', line: e.line }];
          throw pe;
        }
        throw e;
      }
    }

    warnOnce(line, msg) {
      const key = line + ':' + msg;
      if (this.warnedLines.has(key)) return;
      this.warnedLines.add(key);
      this.warn('Warning (line ' + line + '): ' + msg);
    }

    // Convert any thrown value to a PyException (or null if it should not be caught).
    toPyException(e) {
      if (e instanceof PyException) return e;
      if (e && e.__stopTask) return null;
      if (e instanceof RangeError && /call stack/i.test(e.message)) {
        return err('RecursionError', 'maximum recursion depth exceeded');
      }
      if (typeof console !== 'undefined') console.error('Internal interpreter error', e);
      return err('RuntimeError', 'internal simulator error: ' + (e && e.message ? e.message : String(e)));
    }

    // ------------------------------------------------------------ names
    lookup(frame, name, node) {
      const sc = frame.scope;
      let v;
      if (sc.kind === 'module') {
        v = frame.vars.get(name);
        if (v !== undefined) return v;
        v = this.builtins.get(name);
        if (v !== undefined) return v;
        throw this.nameError(name, frame);
      }
      if (sc.kind === 'comp' || sc.kind === 'class') {
        v = frame.vars.get(name);
        if (v !== undefined) return v;
        return this.lookup(frame.parent, name, node);
      }
      if (sc.globals.has(name)) return this.lookupGlobal(frame.module, name, frame);
      if (sc.locals.has(name)) {
        v = frame.vars.get(name);
        if (v === undefined) {
          throw err('UnboundLocalError', "local variable '" + name + "' referenced before assignment" +
            (frame.module.vars.has(name) ? " (tip: add 'global " + name + "' at the start of the function to change the global variable)" : ''));
        }
        return v;
      }
      for (let f = frame.parent; f; f = f.parent) {
        if (f.scope.kind === 'module') break;
        if (f.scope.kind === 'class') continue;
        v = f.vars.get(name);
        if (v !== undefined) return v;
      }
      return this.lookupGlobal(frame.module, name, frame);
    }

    lookupGlobal(module, name, frame) {
      let v = module.vars.get(name);
      if (v !== undefined) return v;
      v = this.builtins.get(name);
      if (v !== undefined) return v;
      throw this.nameError(name, frame);
    }

    nameError(name, frame) {
      const cands = new Set(this.builtins.keys());
      for (let f = frame; f; f = f.parent) f.vars.forEach((_, k) => cands.add(k));
      frame.module.vars.forEach((_, k) => cands.add(k));
      let msg = "name '" + name + "' is not defined";
      if (IMPORT_HINTS[name]) msg += ". Did you forget '" + IMPORT_HINTS[name] + "'?";
      else {
        const s = suggest(name, cands);
        if (s) msg += ". Did you mean: '" + s + "'?";
      }
      return err('NameError', msg);
    }

    assignName(frame, name, value) {
      const sc = frame.scope;
      if (sc.kind === 'function') {
        if (sc.globals.has(name)) { frame.module.vars.set(name, value); return; }
        if (sc.nonlocals.has(name)) {
          for (let f = frame.parent; f; f = f.parent) {
            if (f.scope.kind === 'function' && f.vars.has(name)) { f.vars.set(name, value); return; }
          }
          throw err('SyntaxError', "no binding for nonlocal '" + name + "' found");
        }
      }
      frame.vars.set(name, value);
    }

    deleteName(frame, name) {
      const sc = frame.scope;
      const target = (sc.kind === 'function' && sc.globals.has(name)) ? frame.module : frame;
      if (!target.vars.has(name)) throw err('NameError', "name '" + name + "' is not defined");
      target.vars.delete(name);
    }

    // ------------------------------------------------------------ attributes
    findClassAttr(cls, name) {
      const mro = cls.mro;
      for (let i = 0; i < mro.length; i++) {
        const v = mro[i].dict.get(name);
        if (v !== undefined) return v;
      }
      return undefined;
    }

    bindAttr(v, inst, cls) {
      if (v instanceof PyFunction || v instanceof PyNative) return new PyBoundMethod(inst, v);
      if (v instanceof PyStaticMethod) return v.f;
      if (v instanceof PyClassMethod) return new PyBoundMethod(cls, v.f);
      if (v instanceof PyProperty) return this.callSync(v.fget, [inst]);
      return v;
    }

    getAttr(obj, name) {
      if (obj instanceof PyInstance) {
        const cls = obj.cls;
        const cv = this.findClassAttr(cls, name);
        if (cv instanceof PyProperty) return this.callSync(cv.fget, [obj]);
        const v = obj.dict.get(name);
        if (v !== undefined) return v;
        if (cv !== undefined) return this.bindAttr(cv, obj, cls);
        if (name === '__class__') return cls;
        if (name === '__dict__') {
          const d = new PyDict();
          obj.dict.forEach((val, k) => py.dictSet(d, k, val));
          return d;
        }
        const ga = this.findClassAttr(cls, '__getattr__');
        if (ga instanceof PyFunction) return this.callSync(ga, [obj, name]);
        const cands = Array.from(obj.dict.keys());
        cls.mro.forEach(c => c.dict.forEach((_, k) => cands.push(k)));
        throw this.attrError("'" + cls.name + "' object has no attribute '" + name + "'", name, cands);
      }
      if (obj instanceof PyModule) {
        const v = obj.dict.get(name);
        if (v !== undefined) return v;
        if (name === '__name__') return obj.name;
        throw this.attrError("module '" + obj.name + "' has no attribute '" + name + "'", name, Array.from(obj.dict.keys()));
      }
      if (obj instanceof PyClass) {
        if (name === '__name__') return obj.name;
        if (name === '__mro__') return new PyTuple(obj.mro.slice());
        if (name === '__bases__') return new PyTuple(obj.bases.slice());
        const v = this.findClassAttr(obj, name);
        if (v !== undefined) {
          if (v instanceof PyStaticMethod) return v.f;
          if (v instanceof PyClassMethod) return new PyBoundMethod(obj, v.f);
          return v;
        }
        const cands = [];
        obj.mro.forEach(c => c.dict.forEach((_, k) => cands.push(k)));
        throw this.attrError("type object '" + obj.name + "' has no attribute '" + name + "'", name, cands);
      }
      if (obj instanceof PySuper) {
        const isCls = obj.self instanceof PyClass;
        const mro = isCls ? obj.self.mro : obj.self.cls.mro;
        const start = mro.indexOf(obj.cls) + 1;
        for (let i = start; i < mro.length; i++) {
          const v = mro[i].dict.get(name);
          if (v !== undefined) {
            if (isCls) {
              if (v instanceof PyClassMethod) return new PyBoundMethod(obj.self, v.f);
              if (v instanceof PyStaticMethod) return v.f;
              return v;
            }
            return this.bindAttr(v, obj.self, obj.self.cls);
          }
        }
        if (name === '__init__') return new PyNative('__init__', () => None);
        throw err('AttributeError', "'super' object has no attribute '" + name + "'");
      }
      const b = py.builtinAttr(obj, name);
      if (b !== undefined) return b;
      throw err('AttributeError', "'" + py.typeName(obj) + "' object has no attribute '" + name + "'");
    }

    attrError(msg, name, cands) {
      const s = suggest(name, cands);
      if (s) msg += ". Did you mean: '" + s + "'?";
      return err('AttributeError', msg);
    }

    setAttr(obj, name, value) {
      if (obj instanceof PyInstance) {
        const cv = this.findClassAttr(obj.cls, name);
        if (cv instanceof PyProperty) {
          if (!cv.fset) throw err('AttributeError', "can't set attribute '" + name + "'");
          this.callSync(cv.fset, [obj, value]);
          return;
        }
        if (obj.native && obj.native.readonly && obj.native.readonly.indexOf(name) >= 0) {
          throw err('AttributeError', "attribute '" + name + "' is read-only");
        }
        obj.dict.set(name, value);
        return;
      }
      if (obj instanceof PyClass) { obj.dict.set(name, value); return; }
      if (obj instanceof PyModule) { obj.dict.set(name, value); return; }
      if (obj instanceof PyFunction) {
        if (!obj.attrs) obj.attrs = new Map();
        obj.attrs.set(name, value);
        return;
      }
      throw err('AttributeError', "'" + py.typeName(obj) + "' object has no attribute '" + name + "'");
    }

    delAttr(obj, name) {
      if ((obj instanceof PyInstance || obj instanceof PyClass || obj instanceof PyModule) && obj.dict.has(name)) {
        obj.dict.delete(name);
        return;
      }
      throw err('AttributeError', "'" + py.typeName(obj) + "' object has no attribute '" + name + "'");
    }

    // ------------------------------------------------------------ imports
    importModule(fullName) {
      const cached = this.moduleCache.get(fullName);
      if (cached) return cached;
      const parts = fullName.split('.');
      let parent = null;
      if (parts.length > 1) parent = this.importModule(parts.slice(0, -1).join('.'));
      const leaf = parts[parts.length - 1];
      let mod;
      const factory = this.moduleFactories[fullName];
      if (factory) mod = factory(this.ctx);
      else if (parent && parent.dict.get(leaf) instanceof PyModule) mod = parent.dict.get(leaf);
      else {
        let msg = "no module named '" + fullName + "'";
        const s = suggest(fullName, Object.keys(this.moduleFactories));
        if (s) msg += ". Did you mean: '" + s + "'?";
        throw err('ImportError', msg);
      }
      this.moduleCache.set(fullName, mod);
      if (parent) parent.dict.set(leaf, mod);
      return mod;
    }

    // ------------------------------------------------------------ execution
    *runModule(ast) {
      const frame = new Frame(ast.scope, null, null, null, '<module>');
      frame.module = frame;
      frame.vars.set('__name__', '__main__');
      this.mainFrame = frame;
      if (this.task) this.task.frame = frame;
      try {
        yield* this.execBlock(ast.body, frame);
      } catch (e) {
        const pe = this.toPyException(e);
        if (pe) {
          pe.pyTraceback.unshift({ name: '<module>', line: frame.line });
          if (!pe.line) pe.line = frame.line;
          throw pe;
        }
        throw e;
      }
    }

    *runCoroutine(coro) {
      if (coro.state !== 'created') throw err('RuntimeError', 'cannot reuse already awaited coroutine');
      coro.state = 'running';
      try {
        return yield* this.runFrame(coro.func, coro.frame);
      } finally {
        coro.state = 'done';
      }
    }

    *execBlock(stmts, frame) {
      for (let i = 0; i < stmts.length; i++) {
        const r = yield* this.exec(stmts[i], frame);
        if (r !== undefined) return r;
      }
      return undefined;
    }

    *exec(s, frame) {
      frame.line = s.line;
      yield STEP;
      switch (s.type) {
        case 'Expr': {
          const v = s.value.pure ? this.ev(s.value, frame) : yield* this.evalG(s.value, frame);
          if (v instanceof PyCoroutine && v.state === 'created') {
            this.warnOnce(s.line, "coroutine '" + v.func.name + "()' was called but never awaited. Did you forget 'await'?");
          } else if (v instanceof Awaitable && !v.awaited && v.hint) {
            this.warnOnce(s.line, v.hint);
          }
          return undefined;
        }
        case 'Assign': {
          const v = s.value.pure ? this.ev(s.value, frame) : yield* this.evalG(s.value, frame);
          for (let i = 0; i < s.targets.length; i++) yield* this.assign(s.targets[i], v, frame);
          return undefined;
        }
        case 'AugAssign':
          yield* this.augAssign(s, frame);
          return undefined;
        case 'If': {
          const t = s.test.pure ? this.ev(s.test, frame) : yield* this.evalG(s.test, frame);
          return yield* this.execBlock(truthy(t) ? s.body : s.orelse, frame);
        }
        case 'While': {
          for (;;) {
            const t = s.test.pure ? this.ev(s.test, frame) : yield* this.evalG(s.test, frame);
            if (!truthy(t)) return yield* this.execBlock(s.orelse, frame);
            const r = yield* this.execBlock(s.body, frame);
            if (r === BREAK) return undefined;
            if (r === CONTINUE || r === undefined) { frame.line = s.line; continue; }
            return r;
          }
        }
        case 'For': {
          if (s.isAsync) throw err('NotImplementedError', "'async for' is not supported");
          const iterable = s.iter.pure ? this.ev(s.iter, frame) : yield* this.evalG(s.iter, frame);
          const it = py.iterate(iterable);
          for (;;) {
            const n = it.next();
            if (n.done) break;
            yield* this.assign(s.target, n.value, frame);
            const r = yield* this.execBlock(s.body, frame);
            if (r === BREAK) return undefined;
            if (r === CONTINUE || r === undefined) { frame.line = s.line; continue; }
            return r;
          }
          return yield* this.execBlock(s.orelse, frame);
        }
        case 'Break': return BREAK;
        case 'Continue': return CONTINUE;
        case 'Pass': return undefined;
        case 'Return': {
          const v = s.value ? (s.value.pure ? this.ev(s.value, frame) : yield* this.evalG(s.value, frame)) : None;
          return new Return(v);
        }
        case 'FunctionDef': {
          const f = yield* this.makeFunction(s, frame);
          this.assignName(frame, s.name, f);
          return undefined;
        }
        case 'ClassDef': {
          const c = yield* this.makeClass(s, frame);
          this.assignName(frame, s.name, c);
          return undefined;
        }
        case 'Import':
          for (const a of s.names) {
            const mod = this.importModule(a.name);
            if (a.asname) this.assignName(frame, a.asname, mod);
            else this.assignName(frame, a.name.split('.')[0], this.importModule(a.name.split('.')[0]));
          }
          return undefined;
        case 'ImportFrom': {
          if (s.level > 0) throw err('ImportError', 'relative imports are not supported');
          const mod = this.importModule(s.module);
          if (s.names === '*') {
            mod.dict.forEach((v, k) => { if (!k.startsWith('_')) this.assignName(frame, k, v); });
            return undefined;
          }
          for (const a of s.names) {
            let v = mod.dict.get(a.name);
            if (v === undefined) {
              try { v = this.importModule(s.module + '.' + a.name); } catch (e) {
                let msg = "cannot import name '" + a.name + "' from '" + s.module + "'";
                const sg = suggest(a.name, Array.from(mod.dict.keys()));
                if (sg) msg += ". Did you mean: '" + sg + "'?";
                throw err('ImportError', msg);
              }
            }
            this.assignName(frame, a.asname || a.name, v);
          }
          return undefined;
        }
        case 'Global':
        case 'Nonlocal':
          return undefined;
        case 'Raise': {
          if (!s.exc) {
            const cur = this.excStack[this.excStack.length - 1];
            if (!cur) throw err('RuntimeError', 'No active exception to reraise');
            throw cur;
          }
          let v = s.exc.pure ? this.ev(s.exc, frame) : yield* this.evalG(s.exc, frame);
          if (v instanceof PyClass && py.isExceptionClass(v)) v = yield* this.instantiate(v, [], EMPTY_KW, s);
          if (!py.isExceptionInstance(v)) throw err('TypeError', 'exceptions must derive from BaseException');
          const pe = new PyException(v);
          pe.line = s.line;
          throw pe;
        }
        case 'Try':
          return yield* this.execTry(s, frame);
        case 'Assert': {
          const t = s.test.pure ? this.ev(s.test, frame) : yield* this.evalG(s.test, frame);
          if (!truthy(t)) {
            const msg = s.msg ? (s.msg.pure ? this.ev(s.msg, frame) : yield* this.evalG(s.msg, frame)) : undefined;
            throw err('AssertionError', msg);
          }
          return undefined;
        }
        case 'Delete':
          for (const t of s.targets) {
            if (t.type === 'Name') this.deleteName(frame, t.id);
            else if (t.type === 'Subscript') {
              const o = t.value.pure ? this.ev(t.value, frame) : yield* this.evalG(t.value, frame);
              const k = yield* this.evalSliceG(t.slice, frame);
              py.delItem(o, k);
            } else if (t.type === 'Attribute') {
              const o = t.value.pure ? this.ev(t.value, frame) : yield* this.evalG(t.value, frame);
              this.delAttr(o, t.attr);
            } else if (t.type === 'Tuple' || t.type === 'List') {
              for (const e of t.elts) if (e.type === 'Name') this.deleteName(frame, e.id);
            }
          }
          return undefined;
        case 'With':
          return yield* this.execWith(s, frame);
      }
      throw err('NotImplementedError', 'statement ' + s.type + ' is not supported');
    }

    *execTry(s, frame) {
      let result;
      try {
        let failed = false;
        try {
          result = yield* this.execBlock(s.body, frame);
        } catch (e) {
          const pe = this.toPyException(e);
          if (!pe) throw e;
          failed = true;
          let handler = null;
          for (const h of s.handlers) {
            if (!h.excType) { handler = h; break; }
            const t = h.excType.pure ? this.ev(h.excType, frame) : yield* this.evalG(h.excType, frame);
            if (this.excMatches(pe.exc, t)) { handler = h; break; }
          }
          if (!handler) throw pe;
          if (handler.name) this.assignName(frame, handler.name, pe.exc);
          const stack = this.excStack;
          stack.push(pe);
          try {
            frame.line = handler.line;
            result = yield* this.execBlock(handler.body, frame);
          } finally {
            stack.pop();
            if (handler.name) frame.vars.delete(handler.name);
          }
        }
        if (!failed && result === undefined && s.orelse.length) result = yield* this.execBlock(s.orelse, frame);
        return result;
      } finally {
        if (s.finalbody.length) {
          const fr = yield* this.execBlock(s.finalbody, frame);
          if (fr !== undefined) return fr; // eslint-disable-line no-unsafe-finally
        }
      }
    }

    excMatches(excInst, t) {
      if (t instanceof PyTuple) return t.items.some(x => this.excMatches(excInst, x));
      if (!(t instanceof PyClass) || !py.isExceptionClass(t)) {
        throw err('TypeError', 'catching classes that do not inherit from BaseException is not allowed');
      }
      return py.isSubclass(excInst.cls, t);
    }

    *execWith(s, frame) {
      // Minimal context manager support: __enter__/__exit__.
      const exits = [];
      for (const it of s.items) {
        const cm = it.expr.pure ? this.ev(it.expr, frame) : yield* this.evalG(it.expr, frame);
        const enter = this.getAttr(cm, '__enter__');
        const val = yield* this.callValue(enter, [], EMPTY_KW, s);
        exits.push(this.getAttr(cm, '__exit__'));
        if (it.target) yield* this.assign(it.target, val, frame);
      }
      let result;
      try {
        result = yield* this.execBlock(s.body, frame);
      } catch (e) {
        const pe = this.toPyException(e);
        if (!pe) throw e;
        for (let i = exits.length - 1; i >= 0; i--) {
          const r = yield* this.callValue(exits[i], [pe.exc.cls, pe.exc, None], EMPTY_KW, s);
          if (truthy(r)) return undefined;
        }
        throw pe;
      }
      for (let i = exits.length - 1; i >= 0; i--) yield* this.callValue(exits[i], [None, None, None], EMPTY_KW, s);
      return result;
    }

    *assign(target, value, frame) {
      switch (target.type) {
        case 'Name':
          this.assignName(frame, target.id, value);
          return;
        case 'Attribute': {
          const o = target.value.pure ? this.ev(target.value, frame) : yield* this.evalG(target.value, frame);
          this.setAttr(o, target.attr, value);
          return;
        }
        case 'Subscript': {
          const o = target.value.pure ? this.ev(target.value, frame) : yield* this.evalG(target.value, frame);
          const k = yield* this.evalSliceG(target.slice, frame);
          py.setItem(o, k, value);
          return;
        }
        case 'Tuple':
        case 'List': {
          const items = this.unpack(value, target.elts);
          for (let i = 0; i < target.elts.length; i++) {
            const t = target.elts[i];
            yield* this.assign(t.type === 'Starred' ? t.value : t, items[i], frame);
          }
          return;
        }
        case 'Starred':
          throw err('SyntaxError', 'starred assignment target must be in a list or tuple');
      }
      throw err('SyntaxError', 'cannot assign to ' + target.type);
    }

    assignSync(target, value, frame) {
      // Used by synchronous comprehension evaluation (targets are Names/Tuples).
      switch (target.type) {
        case 'Name': this.assignName(frame, target.id, value); return;
        case 'Tuple': case 'List': {
          const items = this.unpack(value, target.elts);
          target.elts.forEach((t, i) => this.assignSync(t.type === 'Starred' ? t.value : t, items[i], frame));
          return;
        }
        case 'Attribute': this.setAttr(this.ev(target.value, frame), target.attr, value); return;
        case 'Subscript': py.setItem(this.ev(target.value, frame), this.evalSlice(target.slice, frame), value); return;
      }
      throw err('SyntaxError', 'cannot assign to ' + target.type);
    }

    unpack(value, elts) {
      let items;
      try { items = py.toArray(value); } catch (e) {
        throw err('TypeError', 'cannot unpack non-iterable ' + py.typeName(value) + ' object');
      }
      const n = elts.length;
      const star = elts.findIndex(e => e.type === 'Starred');
      if (star < 0) {
        if (items.length > n) throw err('ValueError', 'too many values to unpack (expected ' + n + ')');
        if (items.length < n) throw err('ValueError', 'not enough values to unpack (expected ' + n + ', got ' + items.length + ')');
        return items;
      }
      const after = n - star - 1;
      if (items.length < n - 1) throw err('ValueError', 'not enough values to unpack (expected at least ' + (n - 1) + ', got ' + items.length + ')');
      const out = items.slice(0, star);
      out.push(new PyList(items.slice(star, items.length - after)));
      for (const x of items.slice(items.length - after)) out.push(x);
      return out;
    }

    inplaceOp(op, a, b) {
      if (a instanceof PyList) {
        if (op === '+') { for (const x of py.toArray(b)) a.items.push(x); return a; }
        if (op === '*') {
          const n = py.toInt(b); const copy = a.items.slice(); a.items.length = 0;
          for (let i = 0; i < n; i++) for (const x of copy) a.items.push(x);
          return a;
        }
      }
      if (a instanceof PySet && b instanceof PySet && '|&-^'.indexOf(op) >= 0) {
        const r = py.binop(op, a, b);
        a.map = r.map;
        return a;
      }
      if (a instanceof PyDict && op === '|' && b instanceof PyDict) {
        b.map.forEach((kv) => py.dictSet(a, kv[0], kv[1]));
        return a;
      }
      return py.binop(op, a, b);
    }

    *augAssign(s, frame) {
      const t = s.target;
      if (t.type === 'Name') {
        const cur = this.lookup(frame, t.id, t);
        const rhs = s.value.pure ? this.ev(s.value, frame) : yield* this.evalG(s.value, frame);
        this.assignName(frame, t.id, this.inplaceOp(s.op, cur, rhs));
      } else if (t.type === 'Attribute') {
        const o = t.value.pure ? this.ev(t.value, frame) : yield* this.evalG(t.value, frame);
        const cur = this.getAttr(o, t.attr);
        const rhs = s.value.pure ? this.ev(s.value, frame) : yield* this.evalG(s.value, frame);
        this.setAttr(o, t.attr, this.inplaceOp(s.op, cur, rhs));
      } else {
        const o = t.value.pure ? this.ev(t.value, frame) : yield* this.evalG(t.value, frame);
        const k = yield* this.evalSliceG(t.slice, frame);
        const cur = py.getItem(o, k);
        const rhs = s.value.pure ? this.ev(s.value, frame) : yield* this.evalG(s.value, frame);
        py.setItem(o, k, this.inplaceOp(s.op, cur, rhs));
      }
    }

    *makeFunction(s, frame) {
      const defaults = [];
      for (const p of s.args.params) {
        if (p.default) defaults.push(p.default.pure ? this.ev(p.default, frame) : yield* this.evalG(p.default, frame));
      }
      const kwdefaults = new Map();
      for (const p of s.args.kwonly) {
        if (p.default) kwdefaults.set(p.name, p.default.pure ? this.ev(p.default, frame) : yield* this.evalG(p.default, frame));
      }
      const closure = frame.scope.kind === 'class' ? frame.parent : frame;
      let f = new PyFunction(s.name, s, defaults, kwdefaults, closure, frame.module, s.isAsync);
      for (let i = s.decorators.length - 1; i >= 0; i--) {
        const d = s.decorators[i];
        const dv = d.pure ? this.ev(d, frame) : yield* this.evalG(d, frame);
        f = yield* this.callValue(dv, [f], EMPTY_KW, s);
      }
      return f;
    }

    makeLambda(n, frame) {
      const defaults = n.args.params.filter(p => p.default).map(p => this.ev(p.default, frame));
      const kwdefaults = new Map();
      n.args.kwonly.forEach(p => { if (p.default) kwdefaults.set(p.name, this.ev(p.default, frame)); });
      const closure = frame.scope.kind === 'class' ? frame.parent : frame;
      return new PyFunction('<lambda>', n, defaults, kwdefaults, closure, frame.module, false);
    }

    *makeClass(s, frame) {
      const bases = [];
      for (const b of s.bases) {
        const bv = b.pure ? this.ev(b, frame) : yield* this.evalG(b, frame);
        if (!(bv instanceof PyClass)) throw err('TypeError', 'class bases must be classes, not ' + py.typeName(bv));
        if (bv.isBuiltinType && bv !== py.types.object) {
          throw err('TypeError', "subclassing the built-in type '" + bv.name + "' is not supported in the simulator");
        }
        bases.push(bv);
      }
      const cframe = new Frame(s.scope, frame, frame.module, null, s.name);
      yield* this.execBlock(s.body, cframe);
      const dict = new Map(cframe.vars);
      let cls = new PyClass(s.name, bases.length ? bases : [py.types.object], dict);
      if (bases.some(b => b.isException)) cls.isException = true;
      dict.forEach((v) => {
        const f = v instanceof PyFunction ? v
          : (v instanceof PyStaticMethod || v instanceof PyClassMethod) ? v.f
            : (v instanceof PyProperty ? v.fget : null);
        if (f instanceof PyFunction) f.ownerClass = cls;
        if (v instanceof PyProperty && v.fset instanceof PyFunction) v.fset.ownerClass = cls;
      });
      for (let i = s.decorators.length - 1; i >= 0; i--) {
        const d = s.decorators[i];
        const dv = d.pure ? this.ev(d, frame) : yield* this.evalG(d, frame);
        cls = yield* this.callValue(dv, [cls], EMPTY_KW, s);
      }
      return cls;
    }

    // ------------------------------------------------------------ calls
    bindArgs(fn, args, kwargs) {
      const node = fn.node;
      const a = node.args;
      const frame = new Frame(node.scope, fn.closure, fn.module, fn, fn.name);
      const vars = frame.vars;
      const params = a.params;
      const np = params.length;
      const nargs = args.length;
      for (let i = 0; i < np && i < nargs; i++) vars.set(params[i].name, args[i]);
      if (nargs > np) {
        if (a.vararg) vars.set(a.vararg, new PyTuple(args.slice(np)));
        else {
          throw err('TypeError', fn.name + '() takes ' + np + ' positional argument' + (np === 1 ? '' : 's') +
            ' but ' + nargs + (nargs === 1 ? ' was' : ' were') + ' given');
        }
      } else if (a.vararg) vars.set(a.vararg, new PyTuple([]));
      let kwdict = null;
      if (a.kwarg) { kwdict = new PyDict(); vars.set(a.kwarg, kwdict); }
      if (kwargs !== EMPTY_KW) {
        for (const k in kwargs) {
          let pi = -1;
          for (let i = 0; i < np; i++) if (params[i].name === k) { pi = i; break; }
          if (pi >= 0) {
            if (pi < nargs) throw err('TypeError', fn.name + "() got multiple values for argument '" + k + "'");
            vars.set(k, kwargs[k]);
          } else if (a.kwonly.some(p => p.name === k)) vars.set(k, kwargs[k]);
          else if (kwdict) py.dictSet(kwdict, k, kwargs[k]);
          else {
            let msg = fn.name + "() got an unexpected keyword argument '" + k + "'";
            const sg = suggest(k, params.map(p => p.name).concat(a.kwonly.map(p => p.name)));
            if (sg) msg += ". Did you mean '" + sg + "'?";
            throw err('TypeError', msg);
          }
        }
      }
      if (nargs < np) {
        const firstDefault = np - fn.defaults.length;
        const missing = [];
        for (let i = nargs; i < np; i++) {
          const p = params[i];
          if (!vars.has(p.name)) {
            if (i >= firstDefault) vars.set(p.name, fn.defaults[i - firstDefault]);
            else missing.push("'" + p.name + "'");
          }
        }
        if (missing.length) {
          throw err('TypeError', fn.name + '() missing ' + missing.length + ' required positional argument' +
            (missing.length === 1 ? '' : 's') + ': ' + missing.join(' and '));
        }
      }
      for (const p of a.kwonly) {
        if (!vars.has(p.name)) {
          if (fn.kwdefaults.has(p.name)) vars.set(p.name, fn.kwdefaults.get(p.name));
          else throw err('TypeError', fn.name + "() missing 1 required keyword-only argument: '" + p.name + "'");
        }
      }
      return frame;
    }

    *callValue(fn, args, kwargs, node) {
      if (fn instanceof PyFunction) {
        const frame = this.bindArgs(fn, args, kwargs);
        if (fn.isAsync) {
          const c = new PyCoroutine(fn, frame);
          c.line = node ? node.line : 0;
          return c;
        }
        return yield* this.runFrame(fn, frame);
      }
      if (fn instanceof PyNative) {
        const r = fn.fn(args, kwargs, this.ctx);
        if (r instanceof Suspend) return yield r;
        if (r && r.__gen__) return yield* r.__gen__;
        return r === undefined ? None : r;
      }
      if (fn instanceof PyBoundMethod) {
        const all = new Array(args.length + 1);
        all[0] = fn.self;
        for (let i = 0; i < args.length; i++) all[i + 1] = args[i];
        return yield* this.callValue(fn.func, all, kwargs, node);
      }
      if (fn instanceof PyClass) return yield* this.instantiate(fn, args, kwargs, node);
      if (fn instanceof PyInstance) {
        const m = this.findClassAttr(fn.cls, '__call__');
        if (m !== undefined) return yield* this.callValue(this.bindAttr(m, fn, fn.cls), args, kwargs, node);
      }
      throw err('TypeError', "'" + py.typeName(fn) + "' object is not callable");
    }

    *runFrame(fn, frame) {
      const task = this.task;
      let prevFrame = null;
      if (task) {
        if (++task.depth > this.maxDepth) {
          task.depth--;
          throw err('RecursionError', 'maximum recursion depth exceeded');
        }
        prevFrame = task.frame;
        task.frame = frame;
      }
      try {
        if (fn.node.type === 'Lambda') {
          const b = fn.node.body;
          frame.line = b.line;
          return b.pure ? this.ev(b, frame) : yield* this.evalG(b, frame);
        }
        const r = yield* this.execBlock(fn.node.body, frame);
        if (r instanceof Return) return r.value;
        return None;
      } catch (e) {
        const pe = this.toPyException(e);
        if (pe) {
          pe.pyTraceback.unshift({ name: fn.name, line: frame.line });
          if (!pe.line) pe.line = frame.line;
          throw pe;
        }
        throw e;
      } finally {
        if (task) {
          task.depth--;
          task.frame = prevFrame;
        }
      }
    }

    *instantiate(cls, args, kwargs, node) {
      if (cls.callImpl) return cls.callImpl(args, kwargs, this.ctx);
      const inst = new PyInstance(cls);
      if (py.isSubclass(cls, py.exc.BaseException)) inst.dict.set('args', new PyTuple(args.slice()));
      const init = this.findClassAttr(cls, '__init__');
      if (init !== undefined) {
        const r = yield* this.callValue(init, [inst].concat(args), kwargs, node);
        if (r !== None && init instanceof PyFunction) {
          throw err('TypeError', "__init__() should return None, not '" + py.typeName(r) + "'");
        }
      } else if ((args.length || Object.keys(kwargs).length) && !inst.dict.has('args')) {
        throw err('TypeError', cls.name + '() takes no arguments');
      }
      return inst;
    }

    // Call a Python callable synchronously (used by natives like sorted(key=...)).
    callSync(fn, args, kwargs) {
      kwargs = kwargs || EMPTY_KW;
      if (fn instanceof PyNative) {
        const r = fn.fn(args, kwargs, this.ctx);
        if (r instanceof Suspend) {
          if (r.ready()) return r.getResult();
          throw err('RuntimeError', fn.name + '() needs to wait, which is not possible here');
        }
        return r === undefined ? None : r;
      }
      const gen = this.callValue(fn, args, kwargs, null);
      let r = gen.next();
      let steps = 0;
      while (!r.done) {
        const y = r.value;
        if (y instanceof Suspend) {
          if (y.ready()) { r = gen.next(y.getResult()); continue; }
          throw err('RuntimeError', "a function called here can't wait (no sleep or await allowed inside it)");
        }
        if (++steps > 2000000) throw err('RuntimeError', 'function call took too long (endless loop?)');
        r = gen.next();
      }
      return r.value;
    }

    makeSuper(frame) {
      let f = frame;
      while (f && !f.fn) f = f.parent;
      if (!f || !f.fn || !f.fn.ownerClass) throw err('RuntimeError', 'super(): no arguments (use it inside a method)');
      const a = f.fn.node.args;
      if (!a.params.length) throw err('RuntimeError', 'super(): no arguments');
      const self = f.vars.get(a.params[0].name);
      return new PySuper(f.fn.ownerClass, self);
    }

    // ------------------------------------------------------------ expressions
    evalSlice(n, frame) {
      if (n.type === 'Slice') {
        return new PySlice(
          n.lower ? this.ev(n.lower, frame) : None,
          n.upper ? this.ev(n.upper, frame) : None,
          n.step ? this.ev(n.step, frame) : None);
      }
      if (n.type === 'Tuple' && n.elts.some(e => e.type === 'Slice')) {
        return new PyTuple(n.elts.map(e => this.evalSlice(e, frame)));
      }
      return this.ev(n, frame);
    }

    *evalSliceG(n, frame) {
      if (n.pure) return this.evalSlice(n, frame);
      if (n.type === 'Slice') {
        const lo = n.lower ? (n.lower.pure ? this.ev(n.lower, frame) : yield* this.evalG(n.lower, frame)) : None;
        const up = n.upper ? (n.upper.pure ? this.ev(n.upper, frame) : yield* this.evalG(n.upper, frame)) : None;
        const st = n.step ? (n.step.pure ? this.ev(n.step, frame) : yield* this.evalG(n.step, frame)) : None;
        return new PySlice(lo, up, st);
      }
      if (n.type === 'Tuple') {
        const items = [];
        for (const e of n.elts) items.push(yield* this.evalSliceG(e, frame));
        return new PyTuple(items);
      }
      return yield* this.evalG(n, frame);
    }

    constValue(n) {
      if (n._v !== undefined) return n._v;
      let v;
      switch (n.kind) {
        case 'int': v = n.value; break;
        case 'float': v = new PyFloat(n.value); break;
        case 'str': case 'bytes': v = n.value; break;
        case 'bool': v = n.value; break;
        case 'None': v = None; break;
        case 'ellipsis': v = None; break;
      }
      n._v = v;
      return v;
    }

    formatPart(value, part, specStr) {
      if (part.conversion === 'r' || part.conversion === 'a') value = py.repr(value);
      else if (part.conversion === 's') value = py.str(value);
      return py.formatSpec(value, specStr || '');
    }

    // Synchronous evaluation of pure expressions (no calls / awaits inside).
    ev(n, frame) {
      switch (n.type) {
        case 'Constant': return n._v !== undefined ? n._v : this.constValue(n);
        case 'Name': return this.lookup(frame, n.id, n);
        case 'BinOp': return py.binop(n.op, this.ev(n.left, frame), this.ev(n.right, frame));
        case 'UnaryOp': return py.unaryop(n.op, this.ev(n.operand, frame));
        case 'BoolOp': {
          let v;
          if (n.op === 'and') {
            for (const x of n.values) { v = this.ev(x, frame); if (!truthy(v)) return v; }
          } else {
            for (const x of n.values) { v = this.ev(x, frame); if (truthy(v)) return v; }
          }
          return v;
        }
        case 'Compare': {
          let left = this.ev(n.left, frame);
          for (let i = 0; i < n.ops.length; i++) {
            const right = this.ev(n.comparators[i], frame);
            if (!py.compare(n.ops[i], left, right)) return false;
            left = right;
          }
          return true;
        }
        case 'IfExp': return truthy(this.ev(n.test, frame)) ? this.ev(n.body, frame) : this.ev(n.orelse, frame);
        case 'Attribute': return this.getAttr(this.ev(n.value, frame), n.attr);
        case 'Subscript': return py.getItem(this.ev(n.value, frame), this.evalSlice(n.slice, frame));
        case 'List': return new PyList(this.evalElts(n.elts, frame));
        case 'Tuple': return new PyTuple(this.evalElts(n.elts, frame));
        case 'Set': return py.makeSet(this.evalElts(n.elts, frame));
        case 'Dict': {
          const d = new PyDict();
          for (let i = 0; i < n.keys.length; i++) {
            if (n.keys[i] === null) {
              const src = this.ev(n.values[i], frame);
              if (!(src instanceof PyDict)) throw err('TypeError', "'" + py.typeName(src) + "' object is not a mapping");
              src.map.forEach(kv => py.dictSet(d, kv[0], kv[1]));
            } else py.dictSet(d, this.ev(n.keys[i], frame), this.ev(n.values[i], frame));
          }
          return d;
        }
        case 'JoinedStr': {
          let out = '';
          for (const part of n.parts) {
            if (typeof part === 'string') { out += part; continue; }
            const v = this.ev(part.expr, frame);
            let spec = '';
            if (part.spec) for (const sp of part.spec) spec += typeof sp === 'string' ? sp : this.formatPart(this.ev(sp.expr, frame), sp, '');
            out += this.formatPart(v, part, spec);
          }
          return out;
        }
        case 'Lambda': return this.makeLambda(n, frame);
        case 'ListComp': case 'SetComp': case 'GeneratorExp': case 'DictComp':
          return this.evalComp(n, frame);
        case 'Slice': return this.evalSlice(n, frame);
        case 'NamedExpr': {
          const v = this.ev(n.value, frame);
          let f = frame;
          while (f.scope.kind === 'comp') f = f.parent;
          this.assignName(f, n.target.id, v);
          return v;
        }
        case 'Starred': throw err('SyntaxError', "can't use starred expression here");
      }
      throw err('NotImplementedError', 'expression ' + n.type + ' is not supported');
    }

    evalElts(elts, frame) {
      const out = [];
      for (const e of elts) {
        if (e.type === 'Starred') { for (const x of py.toArray(this.ev(e.value, frame))) out.push(x); }
        else out.push(this.ev(e, frame));
      }
      return out;
    }

    evalComp(n, frame) {
      const cf = new Frame(COMP_SCOPE, frame, frame.module, frame.fn, frame.name);
      const out = [];
      const d = n.type === 'DictComp' ? new PyDict() : null;
      const gens = n.generators;
      const loop = (gi) => {
        const g = gens[gi];
        const it = py.iterate(this.ev(g.iter, gi === 0 ? frame : cf));
        for (let r = it.next(); !r.done; r = it.next()) {
          this.assignSync(g.target, r.value, cf);
          let ok = true;
          for (const c of g.ifs) if (!truthy(this.ev(c, cf))) { ok = false; break; }
          if (!ok) continue;
          if (gi + 1 < gens.length) loop(gi + 1);
          else if (d) py.dictSet(d, this.ev(n.key, cf), this.ev(n.value, cf));
          else out.push(this.ev(n.elt, cf));
        }
      };
      loop(0);
      if (d) return d;
      if (n.type === 'SetComp') return py.makeSet(out);
      if (n.type === 'GeneratorExp') return py.makeIterator(out, 'generator');
      return new PyList(out);
    }

    *evalCompG(n, frame) {
      const cf = new Frame(COMP_SCOPE, frame, frame.module, frame.fn, frame.name);
      const out = [];
      const d = n.type === 'DictComp' ? new PyDict() : null;
      const gens = n.generators;
      const self = this;
      function* loop(gi) {
        const g = gens[gi];
        const src = gi === 0 ? frame : cf;
        const iv = g.iter.pure ? self.ev(g.iter, src) : yield* self.evalG(g.iter, src);
        const it = py.iterate(iv);
        for (let r = it.next(); !r.done; r = it.next()) {
          yield* self.assign(g.target, r.value, cf);
          let ok = true;
          for (const c of g.ifs) {
            const cv = c.pure ? self.ev(c, cf) : yield* self.evalG(c, cf);
            if (!truthy(cv)) { ok = false; break; }
          }
          if (!ok) continue;
          if (gi + 1 < gens.length) yield* loop(gi + 1);
          else if (d) {
            const k = n.key.pure ? self.ev(n.key, cf) : yield* self.evalG(n.key, cf);
            const v = n.value.pure ? self.ev(n.value, cf) : yield* self.evalG(n.value, cf);
            py.dictSet(d, k, v);
          } else out.push(n.elt.pure ? self.ev(n.elt, cf) : yield* self.evalG(n.elt, cf));
          yield STEP;
        }
      }
      yield* loop(0);
      if (d) return d;
      if (n.type === 'SetComp') return py.makeSet(out);
      if (n.type === 'GeneratorExp') return py.makeIterator(out, 'generator');
      return new PyList(out);
    }

    // General evaluation (may call functions and suspend).
    *evalG(n, frame) {
      if (n.pure) return this.ev(n, frame);
      switch (n.type) {
        case 'Call': return yield* this.evalCall(n, frame);
        case 'Await': return yield* this.evalAwait(n, frame);
        case 'BinOp': {
          const l = n.left.pure ? this.ev(n.left, frame) : yield* this.evalG(n.left, frame);
          const r = n.right.pure ? this.ev(n.right, frame) : yield* this.evalG(n.right, frame);
          return py.binop(n.op, l, r);
        }
        case 'UnaryOp': {
          const v = n.operand.pure ? this.ev(n.operand, frame) : yield* this.evalG(n.operand, frame);
          return py.unaryop(n.op, v);
        }
        case 'BoolOp': {
          let v;
          for (const x of n.values) {
            v = x.pure ? this.ev(x, frame) : yield* this.evalG(x, frame);
            if (n.op === 'and' ? !truthy(v) : truthy(v)) return v;
          }
          return v;
        }
        case 'Compare': {
          let left = n.left.pure ? this.ev(n.left, frame) : yield* this.evalG(n.left, frame);
          for (let i = 0; i < n.ops.length; i++) {
            const c = n.comparators[i];
            const right = c.pure ? this.ev(c, frame) : yield* this.evalG(c, frame);
            if (!py.compare(n.ops[i], left, right)) return false;
            left = right;
          }
          return true;
        }
        case 'IfExp': {
          const t = n.test.pure ? this.ev(n.test, frame) : yield* this.evalG(n.test, frame);
          const b = truthy(t) ? n.body : n.orelse;
          return b.pure ? this.ev(b, frame) : yield* this.evalG(b, frame);
        }
        case 'Attribute': {
          const o = n.value.pure ? this.ev(n.value, frame) : yield* this.evalG(n.value, frame);
          return this.getAttr(o, n.attr);
        }
        case 'Subscript': {
          const o = n.value.pure ? this.ev(n.value, frame) : yield* this.evalG(n.value, frame);
          const k = yield* this.evalSliceG(n.slice, frame);
          return py.getItem(o, k);
        }
        case 'List': return new PyList(yield* this.evalEltsG(n.elts, frame));
        case 'Tuple': return new PyTuple(yield* this.evalEltsG(n.elts, frame));
        case 'Set': return py.makeSet(yield* this.evalEltsG(n.elts, frame));
        case 'Dict': {
          const d = new PyDict();
          for (let i = 0; i < n.keys.length; i++) {
            const vn = n.values[i];
            if (n.keys[i] === null) {
              const src = vn.pure ? this.ev(vn, frame) : yield* this.evalG(vn, frame);
              if (!(src instanceof PyDict)) throw err('TypeError', "'" + py.typeName(src) + "' object is not a mapping");
              src.map.forEach(kv => py.dictSet(d, kv[0], kv[1]));
            } else {
              const kn = n.keys[i];
              const k = kn.pure ? this.ev(kn, frame) : yield* this.evalG(kn, frame);
              const v = vn.pure ? this.ev(vn, frame) : yield* this.evalG(vn, frame);
              py.dictSet(d, k, v);
            }
          }
          return d;
        }
        case 'JoinedStr': {
          let out = '';
          for (const part of n.parts) {
            if (typeof part === 'string') { out += part; continue; }
            const v = part.expr.pure ? this.ev(part.expr, frame) : yield* this.evalG(part.expr, frame);
            let spec = '';
            if (part.spec) {
              for (const sp of part.spec) {
                if (typeof sp === 'string') spec += sp;
                else spec += this.formatPart(sp.expr.pure ? this.ev(sp.expr, frame) : yield* this.evalG(sp.expr, frame), sp, '');
              }
            }
            out += this.formatPart(v, part, spec);
          }
          return out;
        }
        case 'ListComp': case 'SetComp': case 'GeneratorExp': case 'DictComp':
          return yield* this.evalCompG(n, frame);
        case 'NamedExpr': {
          const v = n.value.pure ? this.ev(n.value, frame) : yield* this.evalG(n.value, frame);
          let f = frame;
          while (f.scope.kind === 'comp') f = f.parent;
          this.assignName(f, n.target.id, v);
          return v;
        }
        case 'Slice': return yield* this.evalSliceG(n, frame);
        case 'Lambda': return this.makeLambda(n, frame);
      }
      return this.ev(n, frame);
    }

    *evalEltsG(elts, frame) {
      const out = [];
      for (const e of elts) {
        if (e.type === 'Starred') {
          const v = e.value.pure ? this.ev(e.value, frame) : yield* this.evalG(e.value, frame);
          for (const x of py.toArray(v)) out.push(x);
        } else out.push(e.pure ? this.ev(e, frame) : yield* this.evalG(e, frame));
      }
      return out;
    }

    *evalCall(n, frame) {
      const fn = n.func.pure ? this.ev(n.func, frame) : yield* this.evalG(n.func, frame);
      const args = [];
      for (const a of n.args) {
        if (a.type === 'Starred') {
          const v = a.value.pure ? this.ev(a.value, frame) : yield* this.evalG(a.value, frame);
          for (const x of py.toArray(v)) args.push(x);
        } else args.push(a.pure ? this.ev(a, frame) : yield* this.evalG(a, frame));
      }
      let kwargs = EMPTY_KW;
      if (n.keywords.length) {
        kwargs = Object.create(null);
        for (const k of n.keywords) {
          const v = k.value.pure ? this.ev(k.value, frame) : yield* this.evalG(k.value, frame);
          if (k.name === null) {
            if (!(v instanceof PyDict)) throw err('TypeError', 'argument after ** must be a mapping, not ' + py.typeName(v));
            v.map.forEach(kv => {
              if (typeof kv[0] !== 'string') throw err('TypeError', 'keywords must be strings');
              kwargs[kv[0]] = kv[1];
            });
          } else {
            if (k.name in kwargs) throw err('SyntaxError', "keyword argument repeated: " + k.name);
            kwargs[k.name] = v;
          }
        }
      }
      if (fn === this.superFn && args.length === 0) return this.makeSuper(frame);
      return yield* this.callValue(fn, args, kwargs, n);
    }

    *evalAwait(n, frame) {
      const v = n.value.pure ? this.ev(n.value, frame) : yield* this.evalG(n.value, frame);
      if (v instanceof PyCoroutine) {
        return yield* this.runCoroutine(v);
      }
      if (v instanceof Awaitable) {
        v.awaited = true;
        if (!v.isDone()) {
          return yield new Suspend(() => v.isDone(), () => v.result(), v.name);
        }
        return v.result();
      }
      if (v instanceof PyInstance) {
        const m = this.findClassAttr(v.cls, '__await__');
        if (m) return yield* this.callValue(this.bindAttr(m, v, v.cls), [], EMPTY_KW, n);
      }
      let hint = '';
      if (v === None) hint = ' (this function does not need await)';
      throw err('TypeError', "object " + py.typeName(v) + " can't be used in 'await' expression" + hint);
    }
  }

  // Dunder hooks used by runtime.js for user-defined classes.
  function installHooks() {
    const H = py.hooks;
    const I = () => py.activeInterp;
    const userMethod = (inst, name) => {
      const m = I().findClassAttr(inst.cls, name);
      return (m instanceof PyFunction || (m instanceof PyNative && name !== '__init__')) ? m : undefined;
    };
    H.instanceRepr = (inst, which) => {
      let m = userMethod(inst, which);
      if (!m && which === '__str__') m = userMethod(inst, '__repr__');
      if (!m) return undefined;
      const r = I().callSync(m, [inst]);
      if (typeof r !== 'string') throw err('TypeError', which + ' returned non-string (type ' + py.typeName(r) + ')');
      return r;
    };
    H.instanceEq = (a, b) => {
      const m = userMethod(a, '__eq__');
      if (!m) return undefined;
      return truthy(I().callSync(m, [a, b]));
    };
    H.instanceCmp = (a, b, op) => {
      const name = { '<': '__lt__', '<=': '__le__', '>': '__gt__', '>=': '__ge__' }[op];
      const m = userMethod(a, name);
      if (!m) return undefined;
      const r = truthy(I().callSync(m, [a, b]));
      if (op === '<' || op === '<=') return r ? -1 : 1;
      return r ? 1 : -1;
    };
    H.instanceBinop = (op, a, b) => {
      const name = py.OP_NAMES[op];
      if (a instanceof PyInstance) {
        const m = userMethod(a, name);
        if (m) return I().callSync(m, [a, b]);
      }
      if (b instanceof PyInstance) {
        const m = userMethod(b, name.replace('__', '__r'));
        if (m) return I().callSync(m, [b, a]);
      }
      return undefined;
    };
    H.instanceUnary = (op, a) => {
      const m = userMethod(a, { '-': '__neg__', '+': '__pos__', '~': '__invert__' }[op]);
      return m ? I().callSync(m, [a]) : undefined;
    };
    H.instanceLen = (a) => {
      const m = userMethod(a, '__len__');
      return m ? py.toInt(I().callSync(m, [a])) : undefined;
    };
    H.instanceBool = (a) => {
      let m = userMethod(a, '__bool__');
      if (m) return truthy(I().callSync(m, [a]));
      m = userMethod(a, '__len__');
      if (m) return py.toInt(I().callSync(m, [a])) !== 0;
      return undefined;
    };
    H.instanceContains = (a, item) => {
      const m = userMethod(a, '__contains__');
      return m ? truthy(I().callSync(m, [a, item])) : undefined;
    };
    H.instanceGetItem = (a, key) => {
      const m = userMethod(a, '__getitem__');
      return m ? I().callSync(m, [a, key]) : undefined;
    };
    H.instanceSetItem = (a, key, value) => {
      const m = userMethod(a, '__setitem__');
      if (!m) return false;
      I().callSync(m, [a, key, value]);
      return true;
    };
    H.instanceIter = (a) => {
      const m = userMethod(a, '__iter__');
      if (!m) return undefined;
      const itObj = I().callSync(m, [a]);
      if (itObj instanceof PyIterator || !(itObj instanceof PyInstance)) return py.iterate(itObj);
      const nx = userMethod(itObj, '__next__');
      if (!nx) throw err('TypeError', "iter() returned non-iterator of type '" + py.typeName(itObj) + "'");
      return {
        next: () => {
          try {
            return { done: false, value: I().callSync(nx, [itObj]) };
          } catch (e) {
            if (e instanceof PyException && py.isSubclass(e.exc.cls, py.exc.StopIteration)) return DONE;
            throw e;
          }
        }
      };
    };
    H.instanceFormat = (a, spec) => {
      const m = userMethod(a, '__format__');
      return m ? py.str(I().callSync(m, [a, spec])) : undefined;
    };
  }

  // ------------------------------------------------------------ scheduler
  class Scheduler {
    constructor(interp) {
      this.interp = interp;
      interp.scheduler = this;
      this.tasks = [];
      this.stepsPerTick = 250;
      this.error = null;
    }
    spawn(gen, name) {
      const t = new Task(gen, name || 'task');
      this.tasks.push(t);
      return t;
    }
    get running() {
      for (const t of this.tasks) if (!t.done) return true;
      return false;
    }
    // Run every task for one time slice. Returns false when nothing is left to run.
    tick() {
      const n = this.tasks.length;
      for (let i = 0; i < n; i++) {
        const t = this.tasks[i];
        if (t.done) continue;
        if (t.waiting) {
          try {
            if (!t.waiting.ready()) continue;
            t.resumeValue = t.waiting.getResult();
          } catch (e) {
            t.done = true;
            t.error = this.interp.toPyException(e) || e;
            this.error = { task: t, error: t.error };
            return false;
          }
          t.waiting = null;
        }
        this.runTask(t, this.stepsPerTick);
        if (this.error) return false;
      }
      if (this.tasks.length > 16) this.tasks = this.tasks.filter(t => !t.done);
      return this.running;
    }
    runTask(t, budget) {
      const interp = this.interp;
      const prev = interp.task;
      interp.task = t;
      try {
        let send = t.resumeValue;
        t.resumeValue = undefined;
        while (budget > 0) {
          const r = t.gen.next(send);
          send = undefined;
          if (r.done) {
            t.done = true;
            t.result = r.value;
            return;
          }
          const y = r.value;
          if (y === STEP) { budget--; continue; }
          if (y instanceof Suspend) {
            if (y.ready()) { send = y.getResult(); budget--; continue; }
            t.waiting = y;
            return;
          }
          budget--;
        }
      } catch (e) {
        t.done = true;
        const pe = interp.toPyException(e);
        if (pe && py.isSubclass(pe.exc.cls, py.exc.SystemExit)) {
          // sys.exit(): end the whole program quietly.
          this.tasks.forEach(x => { x.done = true; });
          this.exited = true;
          return;
        }
        t.error = pe || e;
        this.error = { task: t, error: t.error };
      } finally {
        interp.task = prev;
      }
    }
    stop() {
      this.tasks.forEach(t => { t.done = true; t.waiting = null; });
      this.tasks = [];
    }
    activeLines() {
      return this.tasks.filter(t => !t.done).map(t => t.line).filter(l => l > 0);
    }
  }

  function formatTraceback(pe, filename) {
    filename = filename || 'program.py';
    const lines = ['Traceback (most recent call last):'];
    for (const fr of pe.pyTraceback) {
      lines.push('  File "' + filename + '", line ' + fr.line + ', in ' + fr.name);
    }
    const msg = pe.message;
    lines.push(pe.exc.cls.name + (msg ? ': ' + msg : ''));
    return lines.join('\n');
  }

  Object.assign(py, {
    Interpreter, Scheduler, Task, Frame, EMPTY_KW, formatTraceback, suggest, IMPORT_HINTS
  });
})(typeof globalThis !== 'undefined' ? globalThis : window);
