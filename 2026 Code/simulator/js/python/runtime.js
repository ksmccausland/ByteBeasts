/*
 * Python object model and core operations for the FLL simulator interpreter.
 *
 * Value representation:
 *   int   -> JS number (always integral)        bool -> JS boolean
 *   float -> PyFloat {v}                         str  -> JS string
 *   None  -> py.None                             list/tuple/dict/set -> PyList/PyTuple/PyDict/PySet
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};
  const py = FLL.py = FLL.py || {};

  // ------------------------------------------------------------------ classes
  class PyNoneType { }
  const None = new PyNoneType();

  class PyFloat { constructor(v) { this.v = v; } }
  class PyList { constructor(items) { this.items = items || []; } }
  class PyTuple { constructor(items) { this.items = items || []; } }
  class PyDict { constructor() { this.map = new Map(); } }
  class PySet { constructor(frozen) { this.map = new Map(); this.frozen = !!frozen; } }
  class PyRange {
    constructor(start, stop, step) {
      this.start = start; this.stop = stop; this.step = step;
    }
    get length() {
      const { start, stop, step } = this;
      if (step > 0) return Math.max(0, Math.ceil((stop - start) / step));
      return Math.max(0, Math.ceil((start - stop) / -step));
    }
    at(i) { return this.start + i * this.step; }
  }
  class PySlice { constructor(start, stop, step) { this.start = start; this.stop = stop; this.step = step; } }
  // A JS-backed iterator: next() returns a value or DONE.
  const DONE = { done: true };
  class PyIterator {
    constructor(nextFn, name) { this.nextFn = nextFn; this.name = name || 'iterator'; }
  }
  class PyFunction {
    constructor(name, node, defaults, kwdefaults, closure, module, isAsync) {
      this.name = name;
      this.node = node;
      this.defaults = defaults;
      this.kwdefaults = kwdefaults;
      this.closure = closure;
      this.module = module;
      this.isAsync = isAsync;
      this.ownerClass = null;
      this.attrs = null;
    }
  }
  class PyBoundMethod { constructor(self, func) { this.self = self; this.func = func; } }
  class PyNative {
    // fn(args: Array, kwargs: Object, ctx) -> value | Suspend
    constructor(name, fn, doc) { this.name = name; this.fn = fn; this.doc = doc || ''; }
  }
  class PyClass {
    constructor(name, bases, dict, opts) {
      this.name = name;
      this.bases = bases || [];
      this.dict = dict || new Map();
      opts = opts || {};
      this.callImpl = opts.callImpl || null;     // builtin type constructor
      this.nativeInit = opts.nativeInit || null; // native class __init__(self, args, kwargs, ctx)
      this.isBuiltinType = !!opts.isBuiltinType;
      this.module = opts.module || null;
      this.mro = computeMro(this);
    }
  }
  class PyInstance {
    constructor(cls) { this.cls = cls; this.dict = new Map(); this.native = null; }
  }
  class PyModule {
    constructor(name, attrs) {
      this.name = name;
      this.dict = new Map();
      if (attrs) for (const k of Object.keys(attrs)) this.dict.set(k, attrs[k]);
    }
  }
  class PyCoroutine {
    constructor(func, frame) { this.func = func; this.frame = frame; this.state = 'created'; this.line = 0; }
  }
  class PyStaticMethod { constructor(f) { this.f = f; } }
  class PyClassMethod { constructor(f) { this.f = f; } }
  class PyProperty { constructor(fget, fset) { this.fget = fget; this.fset = fset; } }
  class PySuper { constructor(cls, self) { this.cls = cls; this.self = self; } }

  // A native function may return a Suspend to block the calling task until ready() is true.
  class Suspend {
    constructor(ready, result, label) { this.ready = ready; this.result = result || null; this.label = label || ''; }
    getResult() { return this.result ? this.result() : None; }
  }
  // An awaitable returned by asynchronous robot API calls.
  class Awaitable {
    constructor(isDone, result, name) { this.isDone = isDone; this.result = result || (() => None); this.name = name || 'awaitable'; this.awaited = false; }
  }
  const STEP = { step: true };

  function computeMro(cls) {
    const out = [cls];
    for (const b of cls.bases) {
      for (const c of b.mro) if (out.indexOf(c) < 0) out.push(c);
    }
    return out;
  }

  // Python exception wrapper (thrown as JS error).
  class PyException extends Error {
    constructor(exc) {
      super();
      this.exc = exc;
      this.pyTraceback = [];  // [{name, line}] innermost last
      this.line = 0;
    }
    get message() { return excToString(this.exc); }
    set message(v) { }
  }

  // ---------------------------------------------------------------- type objects
  const types = {};
  function builtinType(name, base, callImpl) {
    const t = new PyClass(name, base ? [base] : [], new Map(), { isBuiltinType: true, callImpl: callImpl });
    types[name] = t;
    return t;
  }
  const tObject = builtinType('object', null, () => new PyInstance(tObject));
  builtinType('int', tObject);
  builtinType('bool', types.int);
  builtinType('float', tObject);
  builtinType('str', tObject);
  builtinType('list', tObject);
  builtinType('tuple', tObject);
  builtinType('dict', tObject);
  builtinType('set', tObject);
  builtinType('frozenset', tObject);
  builtinType('range', tObject);
  builtinType('slice', tObject);
  builtinType('NoneType', tObject);
  builtinType('function', tObject);
  builtinType('builtin_function_or_method', tObject);
  builtinType('method', tObject);
  builtinType('type', tObject);
  builtinType('module', tObject);
  builtinType('coroutine', tObject);
  builtinType('iterator', tObject);
  builtinType('awaitable', tObject);
  builtinType('staticmethod', tObject);
  builtinType('classmethod', tObject);
  builtinType('property', tObject);
  builtinType('super', tObject);

  // Exceptions
  const exc = {};
  function excClass(name, base) {
    const c = new PyClass(name, base ? [base] : [tObject], new Map());
    c.isException = true;
    exc[name] = c;
    return c;
  }
  excClass('BaseException', null);
  excClass('SystemExit', exc.BaseException);
  excClass('KeyboardInterrupt', exc.BaseException);
  excClass('GeneratorExit', exc.BaseException);
  excClass('Exception', exc.BaseException);
  excClass('StopIteration', exc.Exception);
  excClass('StopAsyncIteration', exc.Exception);
  excClass('ArithmeticError', exc.Exception);
  excClass('ZeroDivisionError', exc.ArithmeticError);
  excClass('OverflowError', exc.ArithmeticError);
  excClass('AssertionError', exc.Exception);
  excClass('AttributeError', exc.Exception);
  excClass('ImportError', exc.Exception);
  excClass('ModuleNotFoundError', exc.ImportError);
  excClass('LookupError', exc.Exception);
  excClass('IndexError', exc.LookupError);
  excClass('KeyError', exc.LookupError);
  excClass('NameError', exc.Exception);
  excClass('UnboundLocalError', exc.NameError);
  excClass('OSError', exc.Exception);
  excClass('RuntimeError', exc.Exception);
  excClass('NotImplementedError', exc.RuntimeError);
  excClass('RecursionError', exc.RuntimeError);
  excClass('SyntaxError', exc.Exception);
  excClass('IndentationError', exc.SyntaxError);
  excClass('TypeError', exc.Exception);
  excClass('ValueError', exc.Exception);
  excClass('UnicodeError', exc.ValueError);
  excClass('MemoryError', exc.Exception);

  function newExc(cls, args) {
    const inst = new PyInstance(cls);
    inst.dict.set('args', new PyTuple(args || []));
    return inst;
  }
  function err(name, msg) {
    const e = new PyException(newExc(exc[name] || exc.Exception, msg === undefined ? [] : [msg]));
    return e;
  }
  function isSubclass(c, base) {
    if (c === base) return true;
    if (!c || !c.mro) return false;
    return c.mro.indexOf(base) >= 0;
  }
  function isExceptionClass(c) { return c instanceof PyClass && isSubclass(c, exc.BaseException); }
  function isExceptionInstance(v) { return v instanceof PyInstance && isSubclass(v.cls, exc.BaseException); }

  function excToString(e) {
    if (!(e instanceof PyInstance)) return String(e);
    const args = e.dict.get('args');
    if (!args || !(args instanceof PyTuple) || args.items.length === 0) return '';
    if (args.items.length === 1) {
      if (e.cls === exc.KeyError) return repr(args.items[0]);
      return str(args.items[0]);
    }
    return repr(args);
  }

  // ------------------------------------------------------------ type helpers
  const F = (v) => new PyFloat(v);

  function typeOf(v) {
    if (v === None || v === null || v === undefined) return types.NoneType;
    switch (typeof v) {
      case 'number': return types.int;
      case 'boolean': return types.bool;
      case 'string': return types.str;
    }
    if (v instanceof PyFloat) return types.float;
    if (v instanceof PyInstance) return v.cls;
    if (v instanceof PyList) return types.list;
    if (v instanceof PyTuple) return types.tuple;
    if (v instanceof PyDict) return types.dict;
    if (v instanceof PySet) return v.frozen ? types.frozenset : types.set;
    if (v instanceof PyRange) return types.range;
    if (v instanceof PyFunction) return types.function;
    if (v instanceof PyNative) return types.builtin_function_or_method;
    if (v instanceof PyBoundMethod) return types.method;
    if (v instanceof PyClass) return types.type;
    if (v instanceof PyModule) return types.module;
    if (v instanceof PyCoroutine) return types.coroutine;
    if (v instanceof PyIterator) return types.iterator;
    if (v instanceof Awaitable) return types.awaitable;
    if (v instanceof PySlice) return types.slice;
    if (v instanceof PyStaticMethod) return types.staticmethod;
    if (v instanceof PyClassMethod) return types.classmethod;
    if (v instanceof PyProperty) return types.property;
    if (v instanceof PySuper) return types.super;
    return tObject;
  }
  function typeName(v) { return typeOf(v).name; }

  function isInt(v) { return typeof v === 'number' || typeof v === 'boolean'; }
  function isNumber(v) { return typeof v === 'number' || typeof v === 'boolean' || v instanceof PyFloat; }
  function numVal(v) {
    if (typeof v === 'number') return v;
    if (v instanceof PyFloat) return v.v;
    if (typeof v === 'boolean') return v ? 1 : 0;
    return undefined;
  }
  // Convert to a JS number for APIs (accepts int, float, bool).
  function toNumber(v, what) {
    const n = numVal(v);
    if (n === undefined) throw err('TypeError', (what ? what + ': ' : '') + "expected a number but got '" + typeName(v) + "'");
    return n;
  }
  // Convert to a JS integer for APIs that need int (like MicroPython does).
  function toInt(v, what) {
    if (typeof v === 'number') return v;
    if (typeof v === 'boolean') return v ? 1 : 0;
    if (v instanceof PyFloat) {
      throw err('TypeError', (what ? what + ': ' : '') + "can't convert float to int (tip: use int(...) or round(...))");
    }
    throw err('TypeError', (what ? what + ': ' : '') + "expected an int but got '" + typeName(v) + "'");
  }
  function toStr(v, what) {
    if (typeof v !== 'string') throw err('TypeError', (what ? what + ': ' : '') + "expected a str but got '" + typeName(v) + "'");
    return v;
  }
  // Make a Python number from a JS number: ints stay ints.
  function fromJsNumber(x, asFloat) {
    if (asFloat || !Number.isInteger(x)) return F(x);
    return x;
  }

  function truthy(v) {
    if (v === true) return true;
    if (v === false || v === None || v === null || v === undefined) return false;
    switch (typeof v) {
      case 'number': return v !== 0;
      case 'string': return v.length > 0;
    }
    if (v instanceof PyFloat) return v.v !== 0;
    if (v instanceof PyList || v instanceof PyTuple) return v.items.length > 0;
    if (v instanceof PyDict || v instanceof PySet) return v.map.size > 0;
    if (v instanceof PyRange) return v.length > 0;
    if (v instanceof PyInstance && py.hooks.instanceBool) {
      const r = py.hooks.instanceBool(v);
      if (r !== undefined) return r;
    }
    return true;
  }

  // ---------------------------------------------------------------- repr/str
  let objIds = new WeakMap();
  let nextId = 0x10000;
  function objectId(o) {
    if (o === null || typeof o !== 'object') return 0;
    let id = objIds.get(o);
    if (!id) { id = nextId; nextId += 16; objIds.set(o, id); }
    return id;
  }
  function hexAddr(o) { return '0x' + objectId(o).toString(16).padStart(8, '0'); }

  function floatRepr(x) {
    if (Number.isNaN(x)) return 'nan';
    if (x === Infinity) return 'inf';
    if (x === -Infinity) return '-inf';
    if (x === 0) return Object.is(x, -0) ? '-0.0' : '0.0';
    const a = Math.abs(x);
    let s;
    if (a >= 1e16 || a < 1e-4) {
      s = x.toExponential();
      s = s.replace(/e([+-])(\d)$/, 'e$10$2');
      return s;
    }
    s = String(x);
    if (s.indexOf('.') < 0 && s.indexOf('e') < 0) s += '.0';
    return s;
  }

  function strRepr(s) {
    const useDouble = s.indexOf('\'') >= 0 && s.indexOf('"') < 0;
    const q = useDouble ? '"' : '\'';
    let out = q;
    for (const ch of s) {
      const c = ch.codePointAt(0);
      if (ch === '\\') out += '\\\\';
      else if (ch === q) out += '\\' + q;
      else if (ch === '\n') out += '\\n';
      else if (ch === '\r') out += '\\r';
      else if (ch === '\t') out += '\\t';
      else if (c < 32 || c === 127) out += '\\x' + c.toString(16).padStart(2, '0');
      else out += ch;
    }
    return out + q;
  }

  // Sets of small non-negative ints iterate in ascending order in CPython/MicroPython.
  function setValues(set) {
    const vals = Array.from(set.map.values());
    if (vals.length > 1 && vals.every(x => typeof x === 'number' && x >= 0 && x < 1e6)) vals.sort((a, b) => a - b);
    return vals;
  }

  const reprStack = [];
  function repr(v) {
    if (v === None || v === null || v === undefined) return 'None';
    switch (typeof v) {
      case 'number': return Number.isFinite(v) ? String(v) : floatRepr(v);
      case 'boolean': return v ? 'True' : 'False';
      case 'string': return strRepr(v);
    }
    if (v instanceof PyFloat) return floatRepr(v.v);
    if (reprStack.indexOf(v) >= 0) {
      if (v instanceof PyList) return '[...]';
      if (v instanceof PyDict) return '{...}';
      return '...';
    }
    reprStack.push(v);
    try {
      if (v instanceof PyList) return '[' + v.items.map(repr).join(', ') + ']';
      if (v instanceof PyTuple) return '(' + v.items.map(repr).join(', ') + (v.items.length === 1 ? ',)' : ')');
      if (v instanceof PyDict) {
        const parts = [];
        for (const [, kv] of v.map) parts.push(repr(kv[0]) + ': ' + repr(kv[1]));
        return '{' + parts.join(', ') + '}';
      }
      if (v instanceof PySet) {
        if (!v.map.size) return v.frozen ? 'frozenset()' : 'set()';
        const s = '{' + setValues(v).map(repr).join(', ') + '}';
        return v.frozen ? 'frozenset(' + s + ')' : s;
      }
      if (v instanceof PyRange) return 'range(' + v.start + ', ' + v.stop + (v.step !== 1 ? ', ' + v.step : '') + ')';
      if (v instanceof PySlice) return 'slice(' + repr(v.start) + ', ' + repr(v.stop) + ', ' + repr(v.step) + ')';
      if (v instanceof PyFunction) return '<function ' + v.name + ' at ' + hexAddr(v) + '>';
      if (v instanceof PyNative) return '<built-in function ' + v.name + '>';
      if (v instanceof PyBoundMethod) return '<bound method ' + (v.func.name || '?') + ' of ' + repr(v.self) + '>';
      if (v instanceof PyClass) return "<class '" + v.name + "'>";
      if (v instanceof PyModule) return "<module '" + v.name + "'>";
      if (v instanceof PyCoroutine) return '<coroutine object ' + v.func.name + ' at ' + hexAddr(v) + '>';
      if (v instanceof Awaitable) return '<awaitable ' + v.name + '>';
      if (v instanceof PyIterator) return '<' + v.name + ' object at ' + hexAddr(v) + '>';
      if (v instanceof PyInstance) {
        if (py.hooks.instanceRepr) {
          const r = py.hooks.instanceRepr(v, '__repr__');
          if (r !== undefined) return r;
        }
        if (isExceptionInstance(v)) {
          const args = v.dict.get('args');
          const inner = args instanceof PyTuple ? args.items.map(repr).join(', ') : '';
          return v.cls.name + '(' + inner + ')';
        }
        if (v.native && v.native.repr) return v.native.repr();
        return '<' + v.cls.name + ' object at ' + hexAddr(v) + '>';
      }
      return '<object>';
    } finally {
      reprStack.pop();
    }
  }

  function str(v) {
    if (typeof v === 'string') return v;
    if (v instanceof PyInstance) {
      if (py.hooks.instanceRepr) {
        const r = py.hooks.instanceRepr(v, '__str__');
        if (r !== undefined) return r;
      }
      if (isExceptionInstance(v)) return excToString(v);
    }
    return repr(v);
  }

  // ------------------------------------------------------------ equality, order
  function eq(a, b) {
    if (a === b) return !(a instanceof PyFloat && Number.isNaN(a.v));
    const ta = typeof a, tb = typeof b;
    if ((ta === 'number' || ta === 'boolean' || a instanceof PyFloat) && (tb === 'number' || tb === 'boolean' || b instanceof PyFloat)) {
      return numVal(a) === numVal(b);
    }
    if (ta === 'string' || tb === 'string') return false;
    if (a instanceof PyList && b instanceof PyList) return seqEq(a.items, b.items);
    if (a instanceof PyTuple && b instanceof PyTuple) return seqEq(a.items, b.items);
    if (a instanceof PyDict && b instanceof PyDict) {
      if (a.map.size !== b.map.size) return false;
      for (const [k, kv] of a.map) {
        const o = b.map.get(k);
        if (!o || !eq(kv[1], o[1])) return false;
      }
      return true;
    }
    if (a instanceof PySet && b instanceof PySet) {
      if (a.map.size !== b.map.size) return false;
      for (const k of a.map.keys()) if (!b.map.has(k)) return false;
      return true;
    }
    if (a instanceof PyRange && b instanceof PyRange) {
      return a.length === b.length && (a.length === 0 || (a.start === b.start && (a.length === 1 || a.step === b.step)));
    }
    if (a instanceof PyInstance && py.hooks.instanceEq) {
      const r = py.hooks.instanceEq(a, b);
      if (r !== undefined) return r;
    }
    if (a instanceof PyBoundMethod && b instanceof PyBoundMethod) return a.self === b.self && a.func === b.func;
    return false;
  }
  function seqEq(x, y) {
    if (x.length !== y.length) return false;
    for (let i = 0; i < x.length; i++) if (!eq(x[i], y[i])) return false;
    return true;
  }

  // Returns -1, 0, 1 or throws TypeError for unorderable types.
  function cmp(a, b, op) {
    const na = numVal(a), nb = numVal(b);
    if (na !== undefined && nb !== undefined) return na < nb ? -1 : (na > nb ? 1 : (na === nb ? 0 : NaN));
    if (typeof a === 'string' && typeof b === 'string') return a < b ? -1 : (a > b ? 1 : 0);
    if ((a instanceof PyList && b instanceof PyList) || (a instanceof PyTuple && b instanceof PyTuple)) {
      const x = a.items, y = b.items;
      for (let i = 0; i < Math.min(x.length, y.length); i++) {
        if (!eq(x[i], y[i])) return cmp(x[i], y[i], op);
      }
      return x.length < y.length ? -1 : (x.length > y.length ? 1 : 0);
    }
    if (a instanceof PySet && b instanceof PySet) {
      const sub = Array.from(a.map.keys()).every(k => b.map.has(k));
      const sup = Array.from(b.map.keys()).every(k => a.map.has(k));
      if (sub && sup) return 0;
      if (sub) return -1;
      if (sup) return 1;
      return NaN;
    }
    if (a instanceof PyInstance && py.hooks.instanceCmp) {
      const r = py.hooks.instanceCmp(a, b, op);
      if (r !== undefined) return r;
    }
    throw err('TypeError', "'" + op + "' not supported between instances of '" + typeName(a) + "' and '" + typeName(b) + "'");
  }

  function compare(op, a, b) {
    switch (op) {
      case '==': return eq(a, b);
      case '!=': return !eq(a, b);
      case '<': return cmp(a, b, op) < 0;
      case '<=': return cmp(a, b, op) <= 0;
      case '>': return cmp(a, b, op) > 0;
      case '>=': return cmp(a, b, op) >= 0;
      case 'is': return a === b;
      case 'is not': return !(a === b);
      case 'in': return contains(b, a);
      case 'not in': return !contains(b, a);
    }
    throw err('RuntimeError', 'unknown comparison ' + op);
  }

  // ------------------------------------------------------------------ hashing
  function hashKey(v) {
    if (v === None) return 'N';
    switch (typeof v) {
      case 'number': return 'n' + v;
      case 'boolean': return v ? 'n1' : 'n0';
      case 'string': return 's' + v;
    }
    if (v instanceof PyFloat) return 'n' + v.v;
    if (v instanceof PyTuple) return 't(' + v.items.map(hashKey).join(',') + ')';
    if (v instanceof PySet && v.frozen) return 'f(' + Array.from(v.map.keys()).sort().join(',') + ')';
    if (v instanceof PyList || v instanceof PyDict || v instanceof PySet) {
      throw err('TypeError', "unhashable type: '" + typeName(v) + "'");
    }
    return 'o' + objectId(v);
  }

  // ----------------------------------------------------------- dict helpers
  function dictGet(d, k) { const e = d.map.get(hashKey(k)); return e ? e[1] : undefined; }
  function dictSet(d, k, v) {
    const h = hashKey(k);
    const e = d.map.get(h);
    if (e) e[1] = v; else d.map.set(h, [k, v]);
  }
  function dictDel(d, k) {
    const h = hashKey(k);
    if (!d.map.has(h)) throw err('KeyError', k);
    d.map.delete(h);
  }
  function makeDict(pairs) {
    const d = new PyDict();
    for (const [k, v] of pairs) dictSet(d, k, v);
    return d;
  }
  function makeSet(items, frozen) {
    const s = new PySet(frozen);
    for (const it of items) s.map.set(hashKey(it), it);
    return s;
  }

  // ------------------------------------------------------------------ iteration
  // Returns a JS iterator {next(): {done, value}} over Python values.
  function iterate(v) {
    if (v instanceof PyList || v instanceof PyTuple) {
      const items = v.items;
      let i = 0;
      return { next: () => (i < items.length ? { done: false, value: items[i++] } : DONE) };
    }
    if (typeof v === 'string') {
      const chars = Array.from(v);
      let i = 0;
      return { next: () => (i < chars.length ? { done: false, value: chars[i++] } : DONE) };
    }
    if (v instanceof PyRange) {
      const n = v.length;
      let i = 0;
      return { next: () => (i < n ? { done: false, value: v.at(i++) } : DONE) };
    }
    if (v instanceof PyDict) {
      const keys = Array.from(v.map.values()).map(kv => kv[0]);
      let i = 0;
      return { next: () => (i < keys.length ? { done: false, value: keys[i++] } : DONE) };
    }
    if (v instanceof PySet) {
      const keys = setValues(v);
      let i = 0;
      return { next: () => (i < keys.length ? { done: false, value: keys[i++] } : DONE) };
    }
    if (v instanceof PyIterator) {
      return {
        next: () => {
          const r = v.nextFn();
          return r === DONE ? DONE : { done: false, value: r };
        }
      };
    }
    if (v instanceof PyInstance && py.hooks.instanceIter) {
      const it = py.hooks.instanceIter(v);
      if (it) return it;
    }
    throw err('TypeError', "'" + typeName(v) + "' object is not iterable");
  }
  function toArray(v) {
    if (v instanceof PyList || v instanceof PyTuple) return v.items.slice();
    const out = [];
    const it = iterate(v);
    for (let r = it.next(); !r.done; r = it.next()) out.push(r.value);
    return out;
  }
  function makeIterator(jsIterable, name) {
    const it = jsIterable[Symbol.iterator] ? jsIterable[Symbol.iterator]() : jsIterable;
    return new PyIterator(() => {
      const r = it.next();
      return r.done ? DONE : r.value;
    }, name);
  }

  function len(v) {
    if (typeof v === 'string') return Array.from(v).length;
    if (v instanceof PyList || v instanceof PyTuple) return v.items.length;
    if (v instanceof PyDict || v instanceof PySet) return v.map.size;
    if (v instanceof PyRange) return v.length;
    if (v instanceof PyInstance && py.hooks.instanceLen) {
      const r = py.hooks.instanceLen(v);
      if (r !== undefined) return r;
    }
    throw err('TypeError', "object of type '" + typeName(v) + "' has no len()");
  }

  function contains(container, item) {
    if (typeof container === 'string') {
      if (typeof item !== 'string') throw err('TypeError', "'in <string>' requires string as left operand, not " + typeName(item));
      return container.indexOf(item) >= 0;
    }
    if (container instanceof PyList || container instanceof PyTuple) {
      for (const x of container.items) if (eq(x, item)) return true;
      return false;
    }
    if (container instanceof PyDict || container instanceof PySet) return container.map.has(hashKey(item));
    if (container instanceof PyRange) {
      const n = numVal(item);
      if (n === undefined || !Number.isInteger(n)) return false;
      const { start, step } = container;
      const len_ = container.length;
      if (len_ === 0) return false;
      const k = (n - start) / step;
      return Number.isInteger(k) && k >= 0 && k < len_;
    }
    if (container instanceof PyInstance && py.hooks.instanceContains) {
      const r = py.hooks.instanceContains(container, item);
      if (r !== undefined) return r;
    }
    const it = iterate(container);
    for (let r = it.next(); !r.done; r = it.next()) if (eq(r.value, item)) return true;
    return false;
  }

  // -------------------------------------------------------------- arithmetic
  const OP_NAMES = {
    '+': '__add__', '-': '__sub__', '*': '__mul__', '/': '__truediv__', '//': '__floordiv__',
    '%': '__mod__', '**': '__pow__', '<<': '__lshift__', '>>': '__rshift__', '&': '__and__',
    '|': '__or__', '^': '__xor__', '@': '__matmul__'
  };

  function intBitOp(op, a, b) {
    const small = Math.abs(a) < 0x40000000 && Math.abs(b) < 0x40000000;
    if (small && op !== '<<') {
      switch (op) {
        case '&': return a & b;
        case '|': return a | b;
        case '^': return a ^ b;
        case '>>': if (b < 0) throw err('ValueError', 'negative shift count'); return b > 31 ? (a < 0 ? -1 : 0) : a >> b;
      }
    }
    if ((op === '<<' || op === '>>') && b < 0) throw err('ValueError', 'negative shift count');
    const A = BigInt(a), B = BigInt(b);
    let r;
    switch (op) {
      case '&': r = A & B; break;
      case '|': r = A | B; break;
      case '^': r = A ^ B; break;
      case '<<': r = A << B; break;
      case '>>': r = A >> B; break;
    }
    return Number(r);
  }

  function binop(op, a, b) {
    if (typeof a === 'number' && typeof b === 'number') {
      switch (op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/':
          if (b === 0) throw err('ZeroDivisionError', 'division by zero');
          return F(a / b);
        case '//':
          if (b === 0) throw err('ZeroDivisionError', 'integer division or modulo by zero');
          return Math.floor(a / b);
        case '%':
          if (b === 0) throw err('ZeroDivisionError', 'integer modulo by zero');
          { const r = a % b; return (r !== 0 && (r < 0) !== (b < 0)) ? r + b : r; }
        case '**':
          if (b < 0) {
            if (a === 0) throw err('ZeroDivisionError', '0.0 cannot be raised to a negative power');
            return F(Math.pow(a, b));
          }
          return Math.pow(a, b);
        case '<<': case '>>': case '&': case '|': case '^':
          return intBitOp(op, a, b);
      }
    }
    const na = numVal(a), nb = numVal(b);
    if (na !== undefined && nb !== undefined) {
      if (!(a instanceof PyFloat) && !(b instanceof PyFloat)) return binop(op, na, nb);
      switch (op) {
        case '+': return F(na + nb);
        case '-': return F(na - nb);
        case '*': return F(na * nb);
        case '/':
          if (nb === 0) throw err('ZeroDivisionError', 'float division by zero');
          return F(na / nb);
        case '//':
          if (nb === 0) throw err('ZeroDivisionError', 'float floor division by zero');
          return F(Math.floor(na / nb));
        case '%':
          if (nb === 0) throw err('ZeroDivisionError', 'float modulo');
          return F(na - nb * Math.floor(na / nb));
        case '**': {
          if (na === 0 && nb < 0) throw err('ZeroDivisionError', '0.0 cannot be raised to a negative power');
          const r = Math.pow(na, nb);
          if (Number.isNaN(r) && !Number.isNaN(na) && !Number.isNaN(nb)) throw err('ValueError', 'math domain error (negative number to a fractional power)');
          return F(r);
        }
      }
    }
    if (op === '+') {
      if (typeof a === 'string' && typeof b === 'string') return a + b;
      if (a instanceof PyList && b instanceof PyList) return new PyList(a.items.concat(b.items));
      if (a instanceof PyTuple && b instanceof PyTuple) return new PyTuple(a.items.concat(b.items));
      if (typeof a === 'string' || typeof b === 'string') {
        const other = typeof a === 'string' ? b : a;
        if (isNumber(other)) {
          throw err('TypeError', typeof a === 'string'
            ? 'can only concatenate str (not "' + typeName(b) + '") to str (tip: use str(...) or an f-string)'
            : "unsupported operand type(s) for +: '" + typeName(a) + "' and 'str' (tip: use str(...))");
        }
      }
    }
    if (op === '*') {
      let seq = null, n = null;
      if (isInt(b) && (typeof a === 'string' || a instanceof PyList || a instanceof PyTuple)) { seq = a; n = numVal(b); }
      else if (isInt(a) && (typeof b === 'string' || b instanceof PyList || b instanceof PyTuple)) { seq = b; n = numVal(a); }
      if (seq !== null) {
        n = Math.max(0, n);
        if (typeof seq === 'string') return seq.repeat(n);
        const out = [];
        for (let i = 0; i < n; i++) for (const x of seq.items) out.push(x);
        return seq instanceof PyList ? new PyList(out) : new PyTuple(out);
      }
    }
    if (op === '%' && typeof a === 'string') return percentFormat(a, b);
    if (a instanceof PySet && b instanceof PySet) {
      const out = new PySet(a.frozen);
      switch (op) {
        case '|': a.map.forEach((v, k) => out.map.set(k, v)); b.map.forEach((v, k) => out.map.set(k, v)); return out;
        case '&': a.map.forEach((v, k) => { if (b.map.has(k)) out.map.set(k, v); }); return out;
        case '-': a.map.forEach((v, k) => { if (!b.map.has(k)) out.map.set(k, v); }); return out;
        case '^':
          a.map.forEach((v, k) => { if (!b.map.has(k)) out.map.set(k, v); });
          b.map.forEach((v, k) => { if (!a.map.has(k)) out.map.set(k, v); });
          return out;
      }
    }
    if (op === '|' && a instanceof PyDict && b instanceof PyDict) {
      const out = new PyDict();
      a.map.forEach((kv, k) => out.map.set(k, [kv[0], kv[1]]));
      b.map.forEach((kv, k) => out.map.set(k, [kv[0], kv[1]]));
      return out;
    }
    if ((a instanceof PyInstance || b instanceof PyInstance) && py.hooks.instanceBinop) {
      const r = py.hooks.instanceBinop(op, a, b);
      if (r !== undefined) return r;
    }
    throw err('TypeError', "unsupported operand type(s) for " + op + ": '" + typeName(a) + "' and '" + typeName(b) + "'");
  }

  function unaryop(op, a) {
    if (op === 'not') return !truthy(a);
    if (typeof a === 'number') {
      if (op === '-') return a === 0 ? 0 : -a;
      if (op === '+') return a;
      if (op === '~') return -(a + 1);
    }
    if (typeof a === 'boolean') return unaryop(op, a ? 1 : 0);
    if (a instanceof PyFloat) {
      if (op === '-') return F(-a.v);
      if (op === '+') return a;
    }
    if (a instanceof PyInstance && py.hooks.instanceUnary) {
      const r = py.hooks.instanceUnary(op, a);
      if (r !== undefined) return r;
    }
    throw err('TypeError', "bad operand type for unary " + op + ": '" + typeName(a) + "'");
  }

  // ------------------------------------------------------------- formatting
  function roundHalfEven(x, nd) {
    const m = Math.pow(10, nd || 0);
    const y = x * m;
    if (!Number.isFinite(y)) return x;
    const r = Math.round(y);
    const isHalf = Math.abs(y - Math.trunc(y)) === 0.5;
    let res = isHalf ? (2 * Math.round(y / 2)) : r;
    return res / m;
  }

  function addGrouping(intPart, sep) {
    return intPart.replace(/\B(?=(\d{3})+(?!\d))/g, sep);
  }

  function fmtExp(x, prec, upper) {
    let s = x.toExponential(prec);
    s = s.replace(/e([+-])(\d)$/, 'e$10$2');
    return upper ? s.toUpperCase() : s;
  }

  function fmtGeneral(x, prec, upper, alt) {
    if (prec === 0) prec = 1;
    if (x === 0) return alt ? (0).toFixed(prec - 1) : '0';
    if (!Number.isFinite(x)) return Number.isNaN(x) ? 'nan' : (x > 0 ? 'inf' : '-inf');
    const e = Math.floor(Math.log10(Math.abs(Number(x.toPrecision(prec)))));
    let s;
    if (e >= -4 && e < prec) {
      s = x.toFixed(Math.max(0, prec - 1 - e));
      if (!alt && s.indexOf('.') >= 0) s = s.replace(/\.?0+$/, '');
    } else {
      s = x.toExponential(prec - 1);
      let [m, ex] = s.split('e');
      if (!alt && m.indexOf('.') >= 0) m = m.replace(/\.?0+$/, '');
      ex = ex.replace(/^([+-])(\d)$/, '$10$2');
      s = m + 'e' + ex;
    }
    return upper ? s.toUpperCase() : s;
  }

  const SPEC_RE = /^(?:(.)?([<>=^]))?([+\- ])?(z)?(#)?(0)?(\d+)?([,_])?(?:\.(\d+))?([bcdeEfFgGnosxX%])?$/s;

  function formatSpec(v, spec) {
    if (spec === '' || spec === undefined) return str(v);
    if (v instanceof PyInstance && py.hooks.instanceFormat) {
      const r = py.hooks.instanceFormat(v, spec);
      if (r !== undefined) return r;
    }
    const m = SPEC_RE.exec(spec);
    if (!m) throw err('ValueError', "Invalid format specifier '" + spec + "'");
    let [, fill, align, sign, , alt, zero, width, grouping, precision, type] = m;
    width = width ? parseInt(width, 10) : 0;
    precision = precision !== undefined ? parseInt(precision, 10) : undefined;
    if (zero && !align) { fill = '0'; align = '='; }
    fill = fill || ' ';
    let body;
    let isNum = isNumber(v);
    if (typeof v === 'string') {
      if (type && type !== 's') throw err('ValueError', "Unknown format code '" + type + "' for object of type 'str'");
      body = precision !== undefined ? v.slice(0, precision) : v;
      align = align || '<';
      isNum = false;
    } else if (isNum) {
      let x = numVal(v);
      const isIntVal = !(v instanceof PyFloat);
      let neg = isIntVal ? x < 0 : (x < 0 || Object.is(x, -0));
      let ax = Math.abs(x);
      if (!type) type = isIntVal ? 'd' : (precision !== undefined ? 'g*' : 'r');
      switch (type) {
        case 'd': case 'n':
          if (!isIntVal) throw err('ValueError', "Unknown format code 'd' for object of type 'float'");
          body = String(ax);
          if (grouping) body = addGrouping(body, grouping);
          break;
        case 'b': case 'o': case 'x': case 'X':
          if (!isIntVal) throw err('ValueError', "Unknown format code '" + type + "' for object of type 'float'");
          body = ax.toString({ b: 2, o: 8, x: 16, X: 16 }[type]);
          if (type === 'X') body = body.toUpperCase();
          if (alt) body = '0' + type + body;
          break;
        case 'c':
          body = String.fromCodePoint(x); neg = false;
          break;
        case 'f': case 'F': {
          const p = precision === undefined ? 6 : precision;
          body = Number.isFinite(ax) ? ax.toFixed(p) : (Number.isNaN(ax) ? 'nan' : 'inf');
          if (grouping) { const parts = body.split('.'); parts[0] = addGrouping(parts[0], grouping); body = parts.join('.'); }
          if (type === 'F') body = body.toUpperCase();
          break;
        }
        case 'e': case 'E':
          body = fmtExp(ax, precision === undefined ? 6 : precision, type === 'E');
          break;
        case 'g': case 'G':
          body = fmtGeneral(ax, precision === undefined ? 6 : precision, type === 'G', !!alt);
          break;
        case 'g*': // float with precision and no type
          body = fmtGeneral(ax, precision, false, false);
          if (/^\d+$/.test(body)) body += '.0';
          break;
        case 'r':
          body = isIntVal ? String(ax) : floatRepr(ax);
          break;
        case '%': {
          const p = precision === undefined ? 6 : precision;
          body = (ax * 100).toFixed(p) + '%';
          break;
        }
        case 's':
          throw err('ValueError', "Unknown format code 's' for object of type '" + typeName(v) + "'");
      }
      const sgn = neg ? '-' : (sign === '+' ? '+' : (sign === ' ' ? ' ' : ''));
      align = align || '>';
      if (align === '=') {
        const padLen = Math.max(0, width - sgn.length - body.length);
        return sgn + fill.repeat(padLen) + body;
      }
      body = sgn + body;
    } else {
      body = str(v);
      align = align || '<';
    }
    const padLen = Math.max(0, width - Array.from(body).length);
    if (!padLen) return body;
    if (align === '<') return body + fill.repeat(padLen);
    if (align === '>' || align === '=') return fill.repeat(padLen) + body;
    if (align === '^') {
      const left = Math.floor(padLen / 2);
      return fill.repeat(left) + body + fill.repeat(padLen - left);
    }
    return body;
  }

  // "%d %s" % values
  function percentFormat(fmt, values) {
    let args, dictArg = null;
    if (values instanceof PyTuple) args = values.items.slice();
    else if (values instanceof PyDict) { dictArg = values; args = []; }
    else args = [values];
    let ai = 0;
    const out = fmt.replace(/%(\(([^)]*)\))?([-+ #0]*)(\*|\d+)?(?:\.(\*|\d+))?([diouxXeEfFgGcrsa%])/g,
      (all, _k, key, flags, width, prec, conv) => {
        if (conv === '%') return '%';
        let val;
        if (key !== undefined) {
          if (!dictArg) throw err('TypeError', 'format requires a mapping');
          val = dictGet(dictArg, key);
          if (val === undefined) throw err('KeyError', key);
        } else {
          if (width === '*') width = String(toInt(args[ai++]));
          if (prec === '*') prec = String(toInt(args[ai++]));
          if (ai >= args.length) throw err('TypeError', 'not enough arguments for format string');
          val = args[ai++];
        }
        let s;
        const p = prec !== undefined ? parseInt(prec, 10) : undefined;
        switch (conv) {
          case 's': s = str(val); if (p !== undefined) s = s.slice(0, p); break;
          case 'r': case 'a': s = repr(val); break;
          case 'c': s = typeof val === 'string' ? val : String.fromCodePoint(toInt(val)); break;
          case 'd': case 'i': case 'u': {
            const n = numVal(val);
            if (n === undefined) throw err('TypeError', '%d format: a number is required, not ' + typeName(val));
            s = String(Math.trunc(n));
            break;
          }
          default: {
            const n = numVal(val);
            if (n === undefined) throw err('TypeError', 'must be real number, not ' + typeName(val));
            const spec = (flags.indexOf('+') >= 0 ? '+' : (flags.indexOf(' ') >= 0 ? ' ' : '')) + (flags.indexOf('#') >= 0 ? '#' : '') + (p !== undefined ? '.' + p : '') + conv;
            s = formatSpec(['x', 'X', 'o'].indexOf(conv) >= 0 ? Math.trunc(n) : F(n), spec);
          }
        }
        if (flags.indexOf('+') >= 0 && (conv === 'd' || conv === 'i') && numVal(val) >= 0) s = '+' + s;
        const w = width ? parseInt(width, 10) : 0;
        if (s.length < w) {
          if (flags.indexOf('-') >= 0) s = s + ' '.repeat(w - s.length);
          else if (flags.indexOf('0') >= 0 && conv !== 's' && conv !== 'r') {
            const neg = s[0] === '-' || s[0] === '+';
            s = (neg ? s[0] : '') + '0'.repeat(w - s.length) + (neg ? s.slice(1) : s);
          } else s = ' '.repeat(w - s.length) + s;
        }
        return s;
      });
    if (!dictArg && ai < args.length) throw err('TypeError', 'not all arguments converted during string formatting');
    return out;
  }

  // "{} {name:>5}".format(...)
  function strFormat(fmt, args, kwargs, getAttrFn) {
    let auto = 0;
    let out = '';
    let i = 0;
    while (i < fmt.length) {
      const c = fmt[i];
      if (c === '{') {
        if (fmt[i + 1] === '{') { out += '{'; i += 2; continue; }
        let j = i + 1, depth = 0;
        for (; j < fmt.length; j++) {
          if (fmt[j] === '{') depth++;
          else if (fmt[j] === '}') { if (depth === 0) break; depth--; }
        }
        if (j >= fmt.length) throw err('ValueError', "Single '{' encountered in format string");
        let field = fmt.slice(i + 1, j);
        let spec = '', conv = null;
        const colon = field.search(/:(?![^\[]*\])/);
        if (colon >= 0) { spec = field.slice(colon + 1); field = field.slice(0, colon); }
        const bang = field.indexOf('!');
        if (bang >= 0) { conv = field.slice(bang + 1); field = field.slice(0, bang); }
        // nested replacement fields in spec
        if (spec.indexOf('{') >= 0) spec = strFormat(spec, args, kwargs, getAttrFn);
        const m = /^([^.\[]*)(.*)$/.exec(field);
        let key = m[1], rest = m[2];
        let val;
        if (key === '') {
          if (auto >= args.length) throw err('IndexError', 'Replacement index ' + auto + ' out of range for positional args tuple');
          val = args[auto++];
        } else if (/^\d+$/.test(key)) {
          const k = parseInt(key, 10);
          if (k >= args.length) throw err('IndexError', 'Replacement index ' + k + ' out of range for positional args tuple');
          val = args[k];
        } else {
          if (!(key in kwargs)) throw err('KeyError', key);
          val = kwargs[key];
        }
        const re = /\.([A-Za-z_]\w*)|\[([^\]]+)\]/g;
        let mm;
        while ((mm = re.exec(rest))) {
          if (mm[1]) val = getAttrFn(val, mm[1]);
          else {
            const idx = /^\d+$/.test(mm[2]) ? parseInt(mm[2], 10) : mm[2];
            val = getItem(val, idx);
          }
        }
        if (conv === 'r' || conv === 'a') val = repr(val);
        else if (conv === 's') val = str(val);
        out += formatSpec(val, spec);
        i = j + 1;
        continue;
      }
      if (c === '}') {
        if (fmt[i + 1] === '}') { out += '}'; i += 2; continue; }
        throw err('ValueError', "Single '}' encountered in format string");
      }
      out += c;
      i++;
    }
    return out;
  }

  // ------------------------------------------------------------------ indexing
  function normIndex(i, n, what) {
    const k = toIndex(i, what);
    const r = k < 0 ? k + n : k;
    if (r < 0 || r >= n) throw err('IndexError', what + ' index out of range');
    return r;
  }
  function toIndex(i, what) {
    if (typeof i === 'number') return i;
    if (typeof i === 'boolean') return i ? 1 : 0;
    throw err('TypeError', what + ' indices must be integers or slices, not ' + typeName(i));
  }
  // Python slice.indices(length)
  function sliceIndices(sl, n) {
    const step = sl.step === None || sl.step === undefined ? 1 : toIndex(sl.step, 'slice');
    if (step === 0) throw err('ValueError', 'slice step cannot be zero');
    let start, stop;
    const conv = (v, def, lo, hi) => {
      if (v === None || v === undefined) return def;
      let x = toIndex(v, 'slice');
      if (x < 0) { x += n; if (x < lo) x = lo; }
      else if (x > hi) x = hi;
      return x;
    };
    if (step > 0) { start = conv(sl.start, 0, 0, n); stop = conv(sl.stop, n, 0, n); }
    else { start = conv(sl.start, n - 1, -1, n - 1); stop = conv(sl.stop, -1, -1, n - 1); }
    return [start, stop, step];
  }
  function sliceArray(arr, sl) {
    const [start, stop, step] = sliceIndices(sl, arr.length);
    const out = [];
    if (step > 0) for (let i = start; i < stop; i += step) out.push(arr[i]);
    else for (let i = start; i > stop; i += step) out.push(arr[i]);
    return out;
  }

  function getItem(obj, key) {
    if (obj instanceof PyList || obj instanceof PyTuple) {
      if (key instanceof PySlice) {
        const items = sliceArray(obj.items, key);
        return obj instanceof PyList ? new PyList(items) : new PyTuple(items);
      }
      return obj.items[normIndex(key, obj.items.length, obj instanceof PyList ? 'list' : 'tuple')];
    }
    if (typeof obj === 'string') {
      const chars = Array.from(obj);
      if (key instanceof PySlice) return sliceArray(chars, key).join('');
      return chars[normIndex(key, chars.length, 'string')];
    }
    if (obj instanceof PyDict) {
      const v = dictGet(obj, key);
      if (v === undefined) throw err('KeyError', key);
      return v;
    }
    if (obj instanceof PyRange) {
      if (key instanceof PySlice) {
        const [start, stop, step] = sliceIndices(key, obj.length);
        const s = obj.at(start), st = obj.step * step;
        const cnt = step > 0 ? Math.max(0, Math.ceil((stop - start) / step)) : Math.max(0, Math.ceil((start - stop) / -step));
        return new PyRange(s, s + cnt * st, st);
      }
      return obj.at(normIndex(key, obj.length, 'range object'));
    }
    if (obj instanceof PyInstance && py.hooks.instanceGetItem) {
      const r = py.hooks.instanceGetItem(obj, key);
      if (r !== undefined) return r;
    }
    if (obj instanceof PyClass && obj.isBuiltinType) return obj; // list[int] style hints
    throw err('TypeError', "'" + typeName(obj) + "' object is not subscriptable");
  }

  function setItem(obj, key, value) {
    if (obj instanceof PyList) {
      if (key instanceof PySlice) {
        const vals = toArray(value);
        const [start, stop, step] = sliceIndices(key, obj.items.length);
        if (step === 1) {
          obj.items.splice(start, Math.max(0, stop - start), ...vals);
        } else {
          const idx = [];
          if (step > 0) for (let i = start; i < stop; i += step) idx.push(i);
          else for (let i = start; i > stop; i += step) idx.push(i);
          if (idx.length !== vals.length) throw err('ValueError', 'attempt to assign sequence of size ' + vals.length + ' to extended slice of size ' + idx.length);
          idx.forEach((ix, k) => { obj.items[ix] = vals[k]; });
        }
        return;
      }
      obj.items[normIndex(key, obj.items.length, 'list assignment')] = value;
      return;
    }
    if (obj instanceof PyDict) { dictSet(obj, key, value); return; }
    if (obj instanceof PyInstance && py.hooks.instanceSetItem) {
      if (py.hooks.instanceSetItem(obj, key, value)) return;
    }
    throw err('TypeError', "'" + typeName(obj) + "' object does not support item assignment");
  }

  function delItem(obj, key) {
    if (obj instanceof PyList) {
      if (key instanceof PySlice) {
        const [start, stop, step] = sliceIndices(key, obj.items.length);
        if (step === 1) { obj.items.splice(start, Math.max(0, stop - start)); return; }
        const idx = [];
        if (step > 0) for (let i = start; i < stop; i += step) idx.push(i);
        else for (let i = start; i > stop; i += step) idx.push(i);
        idx.sort((a, b) => b - a).forEach(i => obj.items.splice(i, 1));
        return;
      }
      obj.items.splice(normIndex(key, obj.items.length, 'list assignment'), 1);
      return;
    }
    if (obj instanceof PyDict) { dictDel(obj, key); return; }
    throw err('TypeError', "'" + typeName(obj) + "' object doesn't support item deletion");
  }

  // ------------------------------------------------------- builtin type methods
  // Each entry: (self, args, kwargs, ctx) -> value
  function argCount(name, args, min, max) {
    if (args.length < min || args.length > max) {
      const exp = min === max ? String(min) : (min + ' to ' + max);
      throw err('TypeError', name + '() takes ' + exp + ' argument' + (max === 1 ? '' : 's') + ' (' + args.length + ' given)');
    }
  }

  const strMethods = {
    upper: (s) => s.toUpperCase(),
    lower: (s) => s.toLowerCase(),
    capitalize: (s) => s ? s[0].toUpperCase() + s.slice(1).toLowerCase() : s,
    title: (s) => s.replace(/[A-Za-z]+/g, w => w[0].toUpperCase() + w.slice(1).toLowerCase()),
    swapcase: (s) => Array.from(s).map(c => c === c.toUpperCase() ? c.toLowerCase() : c.toUpperCase()).join(''),
    strip: (s, a) => stripChars(s, a[0], true, true),
    lstrip: (s, a) => stripChars(s, a[0], true, false),
    rstrip: (s, a) => stripChars(s, a[0], false, true),
    split: (s, a, kw) => {
      const sep = a.length > 0 ? a[0] : (kw.sep !== undefined ? kw.sep : None);
      const maxsplit = a.length > 1 ? toInt(a[1]) : (kw.maxsplit !== undefined ? toInt(kw.maxsplit) : -1);
      let parts;
      if (sep === None) {
        parts = s.trim().split(/\s+/).filter(x => x !== '');
        if (maxsplit >= 0 && parts.length > maxsplit + 1) {
          // re-split preserving remainder
          const re = /\S+/g; const out = []; let m; let count = 0;
          while ((m = re.exec(s)) && count < maxsplit) { out.push(m[0]); count++; }
          if (m) out.push(s.slice(m.index).replace(/\s+$/, ''));
          parts = out;
        }
      } else {
        toStr(sep);
        if (sep === '') throw err('ValueError', 'empty separator');
        parts = s.split(sep);
        if (maxsplit >= 0 && parts.length > maxsplit + 1) parts = parts.slice(0, maxsplit).concat([parts.slice(maxsplit).join(sep)]);
      }
      return new PyList(parts);
    },
    rsplit: (s, a, kw) => {
      const sep = a.length > 0 ? a[0] : (kw.sep !== undefined ? kw.sep : None);
      const maxsplit = a.length > 1 ? toInt(a[1]) : (kw.maxsplit !== undefined ? toInt(kw.maxsplit) : -1);
      if (sep === None) {
        let parts = s.trim().split(/\s+/).filter(x => x !== '');
        if (maxsplit >= 0 && parts.length > maxsplit + 1) parts = [parts.slice(0, parts.length - maxsplit).join(' ')].concat(parts.slice(parts.length - maxsplit));
        return new PyList(parts);
      }
      let parts = s.split(sep);
      if (maxsplit >= 0 && parts.length > maxsplit + 1) parts = [parts.slice(0, parts.length - maxsplit).join(sep)].concat(parts.slice(parts.length - maxsplit));
      return new PyList(parts);
    },
    splitlines: (s) => new PyList(s.split(/\r\n|\r|\n/).filter((x, i, arr) => !(i === arr.length - 1 && x === ''))),
    join: (s, a) => {
      argCount('join', a, 1, 1);
      const items = toArray(a[0]);
      items.forEach((x, i) => {
        if (typeof x !== 'string') throw err('TypeError', 'sequence item ' + i + ": expected str instance, " + typeName(x) + ' found');
      });
      return items.join(s);
    },
    replace: (s, a) => {
      argCount('replace', a, 2, 3);
      const old = toStr(a[0]), nw = toStr(a[1]);
      let count = a.length > 2 ? toInt(a[2]) : -1;
      if (count < 0) return s.split(old).join(nw);
      let out = '', idx = 0;
      while (count-- > 0) {
        const j = s.indexOf(old, idx);
        if (j < 0) break;
        out += s.slice(idx, j) + nw;
        idx = j + old.length;
        if (old === '') { out += s[idx] || ''; idx++; }
      }
      return out + s.slice(idx);
    },
    startswith: (s, a) => {
      const p = a[0];
      const start = a.length > 1 ? toInt(a[1]) : 0;
      if (p instanceof PyTuple) return p.items.some(x => s.startsWith(toStr(x), start));
      return s.startsWith(toStr(p), start);
    },
    endswith: (s, a) => {
      const p = a[0];
      if (p instanceof PyTuple) return p.items.some(x => s.endsWith(toStr(x)));
      return s.endsWith(toStr(p));
    },
    find: (s, a) => s.indexOf(toStr(a[0]), a.length > 1 ? toInt(a[1]) : 0),
    rfind: (s, a) => s.lastIndexOf(toStr(a[0])),
    index: (s, a) => {
      const r = s.indexOf(toStr(a[0]), a.length > 1 ? toInt(a[1]) : 0);
      if (r < 0) throw err('ValueError', 'substring not found');
      return r;
    },
    rindex: (s, a) => {
      const r = s.lastIndexOf(toStr(a[0]));
      if (r < 0) throw err('ValueError', 'substring not found');
      return r;
    },
    count: (s, a) => {
      const sub = toStr(a[0]);
      if (sub === '') return Array.from(s).length + 1;
      return s.split(sub).length - 1;
    },
    format: (s, a, kw, ctx) => strFormat(s, a, kw, (o, n) => ctx.interp.getAttr(o, n)),
    isdigit: (s) => s.length > 0 && /^[0-9]+$/.test(s),
    isnumeric: (s) => s.length > 0 && /^[0-9]+$/.test(s),
    isdecimal: (s) => s.length > 0 && /^[0-9]+$/.test(s),
    isalpha: (s) => s.length > 0 && /^[A-Za-z]+$/.test(s),
    isalnum: (s) => s.length > 0 && /^[A-Za-z0-9]+$/.test(s),
    isspace: (s) => s.length > 0 && /^\s+$/.test(s),
    isupper: (s) => /[A-Z]/.test(s) && s === s.toUpperCase(),
    islower: (s) => /[a-z]/.test(s) && s === s.toLowerCase(),
    center: (s, a) => {
      const w = toInt(a[0]); const f = a.length > 1 ? toStr(a[1]) : ' ';
      const n = Math.max(0, w - s.length); const l = Math.floor(n / 2) + (n % 2 && w % 2 ? 1 : 0);
      return f.repeat(l) + s + f.repeat(n - l);
    },
    ljust: (s, a) => { const w = toInt(a[0]); const f = a.length > 1 ? toStr(a[1]) : ' '; return s + f.repeat(Math.max(0, w - s.length)); },
    rjust: (s, a) => { const w = toInt(a[0]); const f = a.length > 1 ? toStr(a[1]) : ' '; return f.repeat(Math.max(0, w - s.length)) + s; },
    zfill: (s, a) => {
      const w = toInt(a[0]);
      const sign = (s[0] === '-' || s[0] === '+') ? s[0] : '';
      const body = sign ? s.slice(1) : s;
      return sign + '0'.repeat(Math.max(0, w - s.length)) + body;
    },
    encode: (s) => s,
    partition: (s, a) => {
      const sep = toStr(a[0]); const i = s.indexOf(sep);
      return i < 0 ? new PyTuple([s, '', '']) : new PyTuple([s.slice(0, i), sep, s.slice(i + sep.length)]);
    }
  };
  function stripChars(s, chars, left, right) {
    if (chars === undefined || chars === None) {
      if (left) s = s.replace(/^\s+/, '');
      if (right) s = s.replace(/\s+$/, '');
      return s;
    }
    chars = toStr(chars);
    let a = 0, b = s.length;
    if (left) while (a < b && chars.indexOf(s[a]) >= 0) a++;
    if (right) while (b > a && chars.indexOf(s[b - 1]) >= 0) b--;
    return s.slice(a, b);
  }

  function sortItems(items, keyFn, reverse, ctx) {
    const keyed = items.map((v, i) => ({ k: keyFn ? ctx.call(keyFn, [v]) : v, v: v, i: i }));
    keyed.sort((x, y) => {
      const c = reverse ? cmp(y.k, x.k, '<') : cmp(x.k, y.k, '<');
      if (Number.isNaN(c) || c === 0) return x.i - y.i;
      return c;
    });
    return keyed.map(o => o.v);
  }

  const listMethods = {
    append: (l, a) => { argCount('append', a, 1, 1); l.items.push(a[0]); return None; },
    extend: (l, a) => { argCount('extend', a, 1, 1); for (const x of toArray(a[0])) l.items.push(x); return None; },
    insert: (l, a) => {
      argCount('insert', a, 2, 2);
      let i = toInt(a[0]); const n = l.items.length;
      if (i < 0) i = Math.max(0, i + n); if (i > n) i = n;
      l.items.splice(i, 0, a[1]); return None;
    },
    pop: (l, a) => {
      if (!l.items.length) throw err('IndexError', 'pop from empty list');
      const i = a.length ? normIndex(a[0], l.items.length, 'pop') : l.items.length - 1;
      return l.items.splice(i, 1)[0];
    },
    remove: (l, a) => {
      const i = l.items.findIndex(x => eq(x, a[0]));
      if (i < 0) throw err('ValueError', 'list.remove(x): x not in list');
      l.items.splice(i, 1); return None;
    },
    index: (l, a) => {
      const i = l.items.findIndex(x => eq(x, a[0]));
      if (i < 0) throw err('ValueError', repr(a[0]) + ' is not in list');
      return i;
    },
    count: (l, a) => l.items.filter(x => eq(x, a[0])).length,
    sort: (l, a, kw, ctx) => {
      const keyFn = kw.key !== undefined && kw.key !== None ? kw.key : null;
      const sorted = sortItems(l.items, keyFn, kw.reverse !== undefined && truthy(kw.reverse), ctx);
      l.items.length = 0; for (const x of sorted) l.items.push(x);
      return None;
    },
    reverse: (l) => { l.items.reverse(); return None; },
    clear: (l) => { l.items.length = 0; return None; },
    copy: (l) => new PyList(l.items.slice())
  };

  const tupleMethods = {
    index: (t, a) => {
      const i = t.items.findIndex(x => eq(x, a[0]));
      if (i < 0) throw err('ValueError', 'tuple.index(x): x not in tuple');
      return i;
    },
    count: (t, a) => t.items.filter(x => eq(x, a[0])).length
  };

  const dictMethods = {
    keys: (d) => new PyList(Array.from(d.map.values()).map(kv => kv[0])),
    values: (d) => new PyList(Array.from(d.map.values()).map(kv => kv[1])),
    items: (d) => new PyList(Array.from(d.map.values()).map(kv => new PyTuple([kv[0], kv[1]]))),
    get: (d, a) => { const v = dictGet(d, a[0]); return v === undefined ? (a.length > 1 ? a[1] : None) : v; },
    pop: (d, a) => {
      const v = dictGet(d, a[0]);
      if (v === undefined) { if (a.length > 1) return a[1]; throw err('KeyError', a[0]); }
      dictDel(d, a[0]); return v;
    },
    popitem: (d) => {
      if (!d.map.size) throw err('KeyError', 'popitem(): dictionary is empty');
      const lastKey = Array.from(d.map.keys()).pop();
      const kv = d.map.get(lastKey); d.map.delete(lastKey);
      return new PyTuple([kv[0], kv[1]]);
    },
    setdefault: (d, a) => {
      const v = dictGet(d, a[0]);
      if (v !== undefined) return v;
      const def = a.length > 1 ? a[1] : None;
      dictSet(d, a[0], def); return def;
    },
    update: (d, a, kw) => {
      if (a.length) {
        const src = a[0];
        if (src instanceof PyDict) src.map.forEach(kv => dictSet(d, kv[0], kv[1]));
        else for (const pair of toArray(src)) { const p = toArray(pair); dictSet(d, p[0], p[1]); }
      }
      for (const k of Object.keys(kw)) dictSet(d, k, kw[k]);
      return None;
    },
    clear: (d) => { d.map.clear(); return None; },
    copy: (d) => { const o = new PyDict(); d.map.forEach((kv, h) => o.map.set(h, [kv[0], kv[1]])); return o; }
  };

  const setMethods = {
    add: (s, a) => { s.map.set(hashKey(a[0]), a[0]); return None; },
    remove: (s, a) => { const h = hashKey(a[0]); if (!s.map.has(h)) throw err('KeyError', a[0]); s.map.delete(h); return None; },
    discard: (s, a) => { s.map.delete(hashKey(a[0])); return None; },
    pop: (s) => {
      if (!s.map.size) throw err('KeyError', 'pop from an empty set');
      const [h, v] = s.map.entries().next().value; s.map.delete(h); return v;
    },
    clear: (s) => { s.map.clear(); return None; },
    copy: (s) => { const o = new PySet(s.frozen); s.map.forEach((v, h) => o.map.set(h, v)); return o; },
    union: (s, a) => { let r = s; for (const x of a) r = binop('|', r, makeSet(toArray(x))); return r; },
    intersection: (s, a) => { let r = s; for (const x of a) r = binop('&', r, makeSet(toArray(x))); return r; },
    difference: (s, a) => { let r = s; for (const x of a) r = binop('-', r, makeSet(toArray(x))); return r; },
    symmetric_difference: (s, a) => binop('^', s, makeSet(toArray(a[0]))),
    update: (s, a) => { for (const x of a) for (const it of toArray(x)) s.map.set(hashKey(it), it); return None; },
    issubset: (s, a) => { const o = makeSet(toArray(a[0])); return Array.from(s.map.keys()).every(k => o.map.has(k)); },
    issuperset: (s, a) => { const o = makeSet(toArray(a[0])); return Array.from(o.map.keys()).every(k => s.map.has(k)); },
    isdisjoint: (s, a) => { const o = makeSet(toArray(a[0])); return Array.from(o.map.keys()).every(k => !s.map.has(k)); }
  };

  const intMethods = {
    bit_length: (n) => (n === 0 ? 0 : Math.floor(Math.log2(Math.abs(n))) + 1),
    to_bytes: () => { throw err('NotImplementedError', 'to_bytes is not supported'); }
  };
  const floatMethods = {
    is_integer: (f) => Number.isInteger(f.v),
    as_integer_ratio: () => { throw err('NotImplementedError', 'as_integer_ratio is not supported'); }
  };

  function methodTableFor(obj) {
    if (typeof obj === 'string') return strMethods;
    if (obj instanceof PyList) return listMethods;
    if (obj instanceof PyDict) return dictMethods;
    if (obj instanceof PySet) return setMethods;
    if (obj instanceof PyTuple) return tupleMethods;
    if (typeof obj === 'number' || typeof obj === 'boolean') return intMethods;
    if (obj instanceof PyFloat) return floatMethods;
    return null;
  }

  // Attribute lookup for builtin values (used by the interpreter's getAttr).
  function builtinAttr(obj, name) {
    const table = methodTableFor(obj);
    if (table && Object.prototype.hasOwnProperty.call(table, name)) {
      const impl = table[name];
      return new PyNative(name, (args, kwargs, ctx) => impl(obj, args, kwargs || {}, ctx));
    }
    if (obj instanceof PyFloat) {
      if (name === 'real') return obj;
      if (name === 'imag') return F(0);
    }
    if (typeof obj === 'number' || typeof obj === 'boolean') {
      if (name === 'real') return numVal(obj);
      if (name === 'imag') return 0;
    }
    if (obj instanceof PyRange) {
      if (name === 'start') return obj.start;
      if (name === 'stop') return obj.stop;
      if (name === 'step') return obj.step;
    }
    if (obj instanceof PySlice) {
      if (name === 'start') return obj.start;
      if (name === 'stop') return obj.stop;
      if (name === 'step') return obj.step;
    }
    if (obj instanceof PyFunction || obj instanceof PyNative) {
      if (name === '__name__') return obj.name;
      if (name === '__doc__') return obj.doc || None;
      if (obj instanceof PyFunction && obj.attrs && obj.attrs.has(name)) return obj.attrs.get(name);
    }
    if (obj instanceof PyBoundMethod) {
      if (name === '__self__') return obj.self;
      if (name === '__func__') return obj.func;
      if (name === '__name__') return obj.func.name;
    }
    if (obj instanceof PyCoroutine && name === '__name__') return obj.func.name;
    return undefined;
  }

  // Hooks let the interpreter plug in user-defined dunder methods.
  py.hooks = {};

  Object.assign(py, {
    None, PyFloat, PyList, PyTuple, PyDict, PySet, PyRange, PySlice, PyIterator, DONE,
    PyFunction, PyBoundMethod, PyNative, PyClass, PyInstance, PyModule, PyCoroutine,
    PyStaticMethod, PyClassMethod, PyProperty, PySuper, PyException, Suspend, Awaitable, STEP,
    types, exc, newExc, err, isSubclass, isExceptionClass, isExceptionInstance, excToString,
    F, typeOf, typeName, isInt, isNumber, numVal, toNumber, toInt, toStr, fromJsNumber, truthy,
    objectId, floatRepr, repr, str, eq, cmp, compare, hashKey, dictGet, dictSet, dictDel, makeDict, makeSet,
    iterate, toArray, makeIterator, len, contains, binop, unaryop, roundHalfEven, formatSpec,
    percentFormat, strFormat, getItem, setItem, delItem, sliceIndices, sortItems, builtinAttr, argCount,
    OP_NAMES, computeMro
  });
})(typeof globalThis !== 'undefined' ? globalThis : window);
