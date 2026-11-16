/*
 * Builtin functions/types and small standard modules (math, random, time, sys)
 * for the FLL simulator's Python interpreter.
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};
  const py = FLL.py = FLL.py || {};
  const {
    None, PyFloat, PyList, PyTuple, PyDict, PySet, PyRange, PySlice, PyIterator, DONE,
    PyFunction, PyNative, PyClass, PyInstance, PyModule, PyStaticMethod, PyClassMethod, PyProperty,
    Suspend, err, F, types, truthy, numVal, toInt, toNumber
  } = py;

  function native(name, fn, doc) { return new PyNative(name, fn, doc); }
  function kw(kwargs, name, def) { return kwargs && kwargs[name] !== undefined ? kwargs[name] : def; }
  function noKw(name, kwargs) {
    for (const k in kwargs) throw err('TypeError', name + "() got an unexpected keyword argument '" + k + "'");
  }

  // ------------------------------------------------------------- conversions
  function parseIntStr(s, base) {
    const orig = s;
    s = s.trim().replace(/_/g, '');
    let sign = 1;
    if (s[0] === '-' || s[0] === '+') { if (s[0] === '-') sign = -1; s = s.slice(1); }
    if (base === 0) {
      if (/^0[xX]/.test(s)) { base = 16; s = s.slice(2); }
      else if (/^0[oO]/.test(s)) { base = 8; s = s.slice(2); }
      else if (/^0[bB]/.test(s)) { base = 2; s = s.slice(2); }
      else base = 10;
    } else if (base === 16 && /^0[xX]/.test(s)) s = s.slice(2);
    else if (base === 8 && /^0[oO]/.test(s)) s = s.slice(2);
    else if (base === 2 && /^0[bB]/.test(s)) s = s.slice(2);
    const digits = '0123456789abcdefghijklmnopqrstuvwxyz'.slice(0, base);
    if (!s.length || !Array.from(s.toLowerCase()).every(c => digits.indexOf(c) >= 0)) {
      throw err('ValueError', 'invalid literal for int() with base ' + base + ': ' + py.repr(orig));
    }
    return sign * parseInt(s, base);
  }

  types.int.callImpl = (args, kwargs) => {
    const base = args.length > 1 ? toInt(args[1]) : (kwargs.base !== undefined ? toInt(kwargs.base) : null);
    if (!args.length) return 0;
    const v = args[0];
    if (typeof v === 'string') return parseIntStr(v, base === null ? 10 : base);
    if (base !== null) throw err('TypeError', "int() can't convert non-string with explicit base");
    if (typeof v === 'number') return v;
    if (typeof v === 'boolean') return v ? 1 : 0;
    if (v instanceof PyFloat) {
      if (Number.isNaN(v.v)) throw err('ValueError', 'cannot convert float NaN to integer');
      if (!Number.isFinite(v.v)) throw err('OverflowError', 'cannot convert float infinity to integer');
      return Math.trunc(v.v);
    }
    if (v instanceof PyInstance && py.activeInterp) {
      const m = py.activeInterp.findClassAttr(v.cls, '__int__');
      if (m) return py.activeInterp.callSync(m, [v]);
    }
    throw err('TypeError', "int() argument must be a string or a number, not '" + py.typeName(v) + "'");
  };
  types.bool.callImpl = (args) => (args.length ? truthy(args[0]) : false);
  types.float.callImpl = (args) => {
    if (!args.length) return F(0);
    const v = args[0];
    if (typeof v === 'string') {
      const s = v.trim().toLowerCase().replace(/_/g, '');
      if (s === 'inf' || s === '+inf' || s === 'infinity' || s === '+infinity') return F(Infinity);
      if (s === '-inf' || s === '-infinity') return F(-Infinity);
      if (s === 'nan' || s === '+nan' || s === '-nan') return F(NaN);
      if (!/^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/.test(s)) throw err('ValueError', 'could not convert string to float: ' + py.repr(v));
      return F(parseFloat(s));
    }
    const n = numVal(v);
    if (n === undefined) throw err('TypeError', "float() argument must be a string or a number, not '" + py.typeName(v) + "'");
    return F(n);
  };
  types.str.callImpl = (args) => (args.length ? py.str(args[0]) : '');
  types.list.callImpl = (args) => new PyList(args.length ? py.toArray(args[0]) : []);
  types.tuple.callImpl = (args) => (args.length && args[0] instanceof PyTuple ? args[0] : new PyTuple(args.length ? py.toArray(args[0]) : []));
  types.set.callImpl = (args) => py.makeSet(args.length ? py.toArray(args[0]) : []);
  types.frozenset.callImpl = (args) => py.makeSet(args.length ? py.toArray(args[0]) : [], true);
  types.dict.callImpl = (args, kwargs) => {
    const d = new PyDict();
    if (args.length) {
      const src = args[0];
      if (src instanceof PyDict) src.map.forEach(kv => py.dictSet(d, kv[0], kv[1]));
      else {
        for (const pair of py.toArray(src)) {
          const p = py.toArray(pair);
          if (p.length !== 2) throw err('ValueError', 'dictionary update sequence element has length ' + p.length + '; 2 is required');
          py.dictSet(d, p[0], p[1]);
        }
      }
    }
    for (const k in kwargs) py.dictSet(d, k, kwargs[k]);
    return d;
  };
  types.range.callImpl = (args) => {
    if (!args.length || args.length > 3) throw err('TypeError', 'range expected 1 to 3 arguments, got ' + args.length);
    const ints = args.map(a => {
      if (a instanceof PyFloat) throw err('TypeError', "'float' object cannot be interpreted as an integer (tip: use int(...))");
      return toInt(a);
    });
    let start = 0, stop, step = 1;
    if (ints.length === 1) stop = ints[0];
    else { start = ints[0]; stop = ints[1]; if (ints.length === 3) step = ints[2]; }
    if (step === 0) throw err('ValueError', 'range() arg 3 must not be zero');
    return new PyRange(start, stop, step);
  };
  types.slice.callImpl = (args) => {
    if (args.length === 1) return new PySlice(None, args[0], None);
    return new PySlice(args[0], args[1], args.length > 2 ? args[2] : None);
  };
  types.type.callImpl = (args) => {
    if (args.length === 1) return py.typeOf(args[0]);
    throw err('TypeError', 'type() with 3 arguments is not supported');
  };
  types.staticmethod.callImpl = (args) => new PyStaticMethod(args[0]);
  types.classmethod.callImpl = (args) => new PyClassMethod(args[0]);
  types.property.callImpl = (args, kwargs) => {
    const p = new PyProperty(args[0] !== undefined ? args[0] : kw(kwargs, 'fget', None), args[1] !== undefined ? args[1] : kw(kwargs, 'fset', null));
    return p;
  };
  // property.setter decorator support: @x.setter
  const propertySetter = (p) => native('setter', (a) => new PyProperty(p.fget, a[0]));

  // ------------------------------------------------------------- min / max
  function minmax(name, args, kwargs, ctx, pickMax) {
    const keyFn = kw(kwargs, 'key', None);
    let items;
    if (args.length === 1) items = py.toArray(args[0]);
    else if (args.length > 1) items = args;
    else throw err('TypeError', name + ' expected at least 1 argument, got 0');
    if (!items.length) {
      if (kwargs.default !== undefined) return kwargs.default;
      throw err('ValueError', name + '() arg is an empty sequence');
    }
    let best = items[0];
    let bestK = keyFn !== None ? ctx.call(keyFn, [best]) : best;
    for (let i = 1; i < items.length; i++) {
      const k = keyFn !== None ? ctx.call(keyFn, [items[i]]) : items[i];
      const c = py.cmp(k, bestK, pickMax ? '>' : '<');
      if (pickMax ? c > 0 : c < 0) { best = items[i]; bestK = k; }
    }
    return best;
  }

  function roundBuiltin(args) {
    const x = args[0];
    const nd = args.length > 1 && args[1] !== None ? toInt(args[1]) : null;
    if (typeof x === 'number' || typeof x === 'boolean') {
      const n = numVal(x);
      if (nd === null || nd >= 0) return n;
      const m = Math.pow(10, -nd);
      return py.roundHalfEven(n / m, 0) * m;
    }
    if (x instanceof PyFloat) {
      if (nd === null) {
        if (!Number.isFinite(x.v)) throw err(Number.isNaN(x.v) ? 'ValueError' : 'OverflowError', 'cannot convert float ' + py.floatRepr(x.v) + ' to integer');
        return py.roundHalfEven(x.v, 0);
      }
      // Use decimal string rounding to avoid binary artefacts (round(2.675, 2) == 2.67 like CPython)
      const r = Number(x.v.toFixed(Math.min(Math.max(nd, 0), 20)));
      if (nd >= 0) {
        const exact = py.roundHalfEven(x.v, nd);
        return F(Math.abs(exact - r) < Math.pow(10, -nd) / 2 ? exact : r);
      }
      const m = Math.pow(10, -nd);
      return F(py.roundHalfEven(x.v / m, 0) * m);
    }
    if (x instanceof PyInstance && py.activeInterp) {
      const m = py.activeInterp.findClassAttr(x.cls, '__round__');
      if (m) return py.activeInterp.callSync(m, args);
    }
    throw err('TypeError', "type " + py.typeName(x) + " doesn't define __round__ method");
  }

  function makeIter(v) {
    if (v instanceof PyIterator) return v;
    const it = py.iterate(v);
    return new PyIterator(() => { const r = it.next(); return r.done ? DONE : r.value; }, py.typeName(v) + '_iterator');
  }

  function installBuiltins(interp) {
    const B = interp.builtins;
    const add = (name, fn, doc) => B.set(name, native(name, fn, doc));
    const out = (s) => interp.print(s);

    add('print', (args, kwargs) => {
      const sep = kwargs.sep !== undefined && kwargs.sep !== None ? py.str(kwargs.sep) : ' ';
      const end = kwargs.end !== undefined && kwargs.end !== None ? py.str(kwargs.end) : '\n';
      out(args.map(py.str).join(sep) + end);
      return None;
    });
    add('len', (a) => { py.argCount('len', a, 1, 1); return py.len(a[0]); });
    add('abs', (a) => {
      py.argCount('abs', a, 1, 1);
      const x = a[0];
      if (typeof x === 'number') return Math.abs(x);
      if (typeof x === 'boolean') return x ? 1 : 0;
      if (x instanceof PyFloat) return F(Math.abs(x.v));
      if (x instanceof PyInstance) {
        const m = interp.findClassAttr(x.cls, '__abs__');
        if (m) return interp.callSync(m, [x]);
      }
      throw err('TypeError', "bad operand type for abs(): '" + py.typeName(x) + "'");
    });
    add('min', (a, k, ctx) => minmax('min', a, k, ctx, false));
    add('max', (a, k, ctx) => minmax('max', a, k, ctx, true));
    add('sum', (a, k) => {
      let total = a.length > 1 ? a[1] : kw(k, 'start', 0);
      for (const x of py.toArray(a[0])) total = py.binop('+', total, x);
      return total;
    });
    add('round', (a) => { py.argCount('round', a, 1, 2); return roundBuiltin(a); });
    add('divmod', (a) => new PyTuple([py.binop('//', a[0], a[1]), py.binop('%', a[0], a[1])]));
    add('pow', (a) => {
      if (a.length === 3) {
        const b = toInt(a[0]), e = toInt(a[1]), m = toInt(a[2]);
        let r = 1n; let base = BigInt(b) % BigInt(m); let ex = BigInt(e);
        while (ex > 0n) { if (ex & 1n) r = (r * base) % BigInt(m); base = (base * base) % BigInt(m); ex >>= 1n; }
        return Number(r);
      }
      return py.binop('**', a[0], a[1]);
    });
    add('sorted', (a, k, ctx) => {
      py.argCount('sorted', a, 1, 1);
      const keyFn = kw(k, 'key', None);
      return new PyList(py.sortItems(py.toArray(a[0]), keyFn !== None ? keyFn : null, truthy(kw(k, 'reverse', false)), ctx));
    });
    add('reversed', (a) => {
      const v = a[0];
      let arr;
      if (v instanceof PyList || v instanceof PyTuple) arr = v.items.slice().reverse();
      else if (typeof v === 'string' || v instanceof PyRange) arr = py.toArray(v).reverse();
      else if (v instanceof PyDict) arr = py.toArray(v).reverse();
      else throw err('TypeError', "'" + py.typeName(v) + "' object is not reversible");
      return py.makeIterator(arr, 'reversed');
    });
    add('enumerate', (a, k) => {
      const it = py.iterate(a[0]);
      let i = a.length > 1 ? toInt(a[1]) : toInt(kw(k, 'start', 0));
      return new PyIterator(() => {
        const r = it.next();
        return r.done ? DONE : new PyTuple([i++, r.value]);
      }, 'enumerate');
    });
    add('zip', (a) => {
      const its = a.map(x => py.iterate(x));
      return new PyIterator(() => {
        if (!its.length) return DONE;
        const vals = [];
        for (const it of its) { const r = it.next(); if (r.done) return DONE; vals.push(r.value); }
        return new PyTuple(vals);
      }, 'zip');
    });
    add('map', (a, k, ctx) => {
      if (a.length < 2) throw err('TypeError', 'map() must have at least two arguments.');
      const fn = a[0];
      const its = a.slice(1).map(x => py.iterate(x));
      return new PyIterator(() => {
        const vals = [];
        for (const it of its) { const r = it.next(); if (r.done) return DONE; vals.push(r.value); }
        return ctx.call(fn, vals);
      }, 'map');
    });
    add('filter', (a, k, ctx) => {
      const fn = a[0];
      const it = py.iterate(a[1]);
      return new PyIterator(() => {
        for (;;) {
          const r = it.next();
          if (r.done) return DONE;
          const keep = fn === None ? truthy(r.value) : truthy(ctx.call(fn, [r.value]));
          if (keep) return r.value;
        }
      }, 'filter');
    });
    add('any', (a) => { for (const x of py.toArray(a[0])) if (truthy(x)) return true; return false; });
    add('all', (a) => { for (const x of py.toArray(a[0])) if (!truthy(x)) return false; return true; });
    add('iter', (a) => makeIter(a[0]));
    add('next', (a) => {
      const it = a[0];
      if (it instanceof PyIterator) {
        const r = it.nextFn();
        if (r === DONE) { if (a.length > 1) return a[1]; throw err('StopIteration'); }
        return r;
      }
      if (it instanceof PyInstance) {
        const m = interp.findClassAttr(it.cls, '__next__');
        if (m) return interp.callSync(m, [it]);
      }
      throw err('TypeError', "'" + py.typeName(it) + "' object is not an iterator");
    });
    add('isinstance', (a) => {
      py.argCount('isinstance', a, 2, 2);
      const t = py.typeOf(a[0]);
      const check = (c) => {
        if (c instanceof PyTuple) return c.items.some(check);
        if (!(c instanceof PyClass)) throw err('TypeError', 'isinstance() arg 2 must be a type or tuple of types');
        return py.isSubclass(t, c);
      };
      return check(a[1]);
    });
    add('issubclass', (a) => {
      const check = (c) => (c instanceof PyTuple ? c.items.some(check) : py.isSubclass(a[0], c));
      return check(a[1]);
    });
    add('hasattr', (a) => {
      try { interp.getAttr(a[0], py.toStr(a[1])); return true; } catch (e) {
        if (e instanceof py.PyException && py.isSubclass(e.exc.cls, py.exc.AttributeError)) return false;
        throw e;
      }
    });
    add('getattr', (a) => {
      try { return interp.getAttr(a[0], py.toStr(a[1])); } catch (e) {
        if (a.length > 2 && e instanceof py.PyException && py.isSubclass(e.exc.cls, py.exc.AttributeError)) return a[2];
        throw e;
      }
    });
    add('setattr', (a) => { interp.setAttr(a[0], py.toStr(a[1]), a[2]); return None; });
    add('delattr', (a) => { interp.delAttr(a[0], py.toStr(a[1])); return None; });
    add('callable', (a) => {
      const v = a[0];
      return v instanceof PyFunction || v instanceof PyNative || v instanceof py.PyBoundMethod || v instanceof PyClass ||
        (v instanceof PyInstance && interp.findClassAttr(v.cls, '__call__') !== undefined);
    });
    add('repr', (a) => py.repr(a[0]));
    add('ascii', (a) => py.repr(a[0]).replace(/[^\x00-\x7f]/g, c => '\\u' + c.charCodeAt(0).toString(16).padStart(4, '0')));
    add('format', (a) => py.formatSpec(a[0], a.length > 1 ? py.toStr(a[1]) : ''));
    add('chr', (a) => String.fromCodePoint(toInt(a[0])));
    add('ord', (a) => {
      const s = py.toStr(a[0]);
      if (Array.from(s).length !== 1) throw err('TypeError', 'ord() expected a character, but string of length ' + s.length + ' found');
      return s.codePointAt(0);
    });
    add('hex', (a) => { const n = toInt(a[0]); return (n < 0 ? '-0x' : '0x') + Math.abs(n).toString(16); });
    add('oct', (a) => { const n = toInt(a[0]); return (n < 0 ? '-0o' : '0o') + Math.abs(n).toString(8); });
    add('bin', (a) => { const n = toInt(a[0]); return (n < 0 ? '-0b' : '0b') + Math.abs(n).toString(2); });
    add('id', (a) => py.objectId(a[0]) || (typeof a[0] === 'number' ? a[0] * 2 + 1 : 7));
    add('hash', (a) => {
      const h = py.hashKey(a[0]);
      if (typeof a[0] === 'number') return a[0];
      let x = 0;
      for (let i = 0; i < h.length; i++) x = (Math.imul(x, 31) + h.charCodeAt(i)) | 0;
      return x;
    });
    add('input', () => { throw err('OSError', 'input() is not available: the robot has no keyboard'); });
    add('exit', (a) => { throw err('SystemExit', a.length ? a[0] : undefined); });
    add('quit', (a) => { throw err('SystemExit', a.length ? a[0] : undefined); });
    add('dir', (a) => {
      const v = a[0];
      let names = [];
      if (v instanceof PyModule) names = Array.from(v.dict.keys());
      else if (v instanceof PyInstance) { names = Array.from(v.dict.keys()); v.cls.mro.forEach(c => c.dict.forEach((_, k) => names.push(k))); }
      else if (v instanceof PyClass) v.mro.forEach(c => c.dict.forEach((_, k) => names.push(k)));
      else if (!a.length) names = Array.from(interp.mainFrame ? interp.mainFrame.vars.keys() : []);
      return new PyList(Array.from(new Set(names)).sort());
    });
    add('globals', () => {
      const d = new PyDict();
      if (interp.mainFrame) interp.mainFrame.vars.forEach((v, k) => py.dictSet(d, k, v));
      return d;
    });
    const superFn = native('super', () => { throw err('RuntimeError', 'super() with arguments is not supported; use super() inside a method'); });
    B.set('super', superFn);
    interp.superFn = superFn;

    // Types (callable)
    for (const t of ['int', 'float', 'str', 'bool', 'list', 'tuple', 'dict', 'set', 'frozenset', 'range', 'object', 'type', 'slice', 'staticmethod', 'classmethod', 'property']) {
      B.set(t, types[t]);
    }
    // Exceptions
    for (const name of Object.keys(py.exc)) B.set(name, py.exc[name]);
    B.set('EnvironmentError', py.exc.OSError);
    B.set('IOError', py.exc.OSError);
    B.set('NotImplemented', None);
    B.set('Ellipsis', None);
    B.set('__name__', '__main__');
  }

  // property.setter support via attribute access on PyProperty
  const origBuiltinAttr = py.builtinAttr;
  py.builtinAttr = function (obj, name) {
    if (obj instanceof PyProperty) {
      if (name === 'setter') return propertySetter(obj);
      if (name === 'getter') return native('getter', (a) => new PyProperty(a[0], obj.fset));
      if (name === 'fget') return obj.fget;
    }
    return origBuiltinAttr(obj, name);
  };

  // ------------------------------------------------------------------ modules
  // Deterministic PRNG so results can be reproduced with random.seed().
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function mathModule() {
    const f1 = (name, fn) => native(name, (a) => {
      py.argCount(name, a, 1, 1);
      const r = fn(toNumber(a[0], 'math.' + name));
      if (Number.isNaN(r) && !Number.isNaN(toNumber(a[0]))) throw err('ValueError', 'math domain error');
      return F(r);
    });
    return new PyModule('math', {
      pi: F(Math.PI), e: F(Math.E), tau: F(2 * Math.PI), inf: F(Infinity), nan: F(NaN),
      sqrt: native('sqrt', (a) => {
        const x = toNumber(a[0], 'math.sqrt');
        if (x < 0) throw err('ValueError', 'math domain error (square root of a negative number)');
        return F(Math.sqrt(x));
      }),
      exp: f1('exp', Math.exp),
      log: native('log', (a) => {
        const x = toNumber(a[0], 'math.log');
        if (x <= 0) throw err('ValueError', 'math domain error');
        if (a.length > 1) return F(Math.log(x) / Math.log(toNumber(a[1])));
        return F(Math.log(x));
      }),
      log10: native('log10', (a) => { const x = toNumber(a[0]); if (x <= 0) throw err('ValueError', 'math domain error'); return F(Math.log10(x)); }),
      log2: native('log2', (a) => { const x = toNumber(a[0]); if (x <= 0) throw err('ValueError', 'math domain error'); return F(Math.log2(x)); }),
      pow: native('pow', (a) => F(Math.pow(toNumber(a[0]), toNumber(a[1])))),
      sin: f1('sin', Math.sin), cos: f1('cos', Math.cos), tan: f1('tan', Math.tan),
      asin: f1('asin', Math.asin), acos: f1('acos', Math.acos), atan: f1('atan', Math.atan),
      sinh: f1('sinh', Math.sinh), cosh: f1('cosh', Math.cosh), tanh: f1('tanh', Math.tanh),
      atan2: native('atan2', (a) => F(Math.atan2(toNumber(a[0]), toNumber(a[1])))),
      hypot: native('hypot', (a) => F(Math.hypot.apply(null, a.map(x => toNumber(x))))),
      degrees: f1('degrees', x => x * 180 / Math.PI),
      radians: f1('radians', x => x * Math.PI / 180),
      floor: native('floor', (a) => { const x = toNumber(a[0], 'math.floor'); return Math.floor(x); }),
      ceil: native('ceil', (a) => { const x = toNumber(a[0], 'math.ceil'); return Math.ceil(x); }),
      trunc: native('trunc', (a) => Math.trunc(toNumber(a[0]))),
      fabs: f1('fabs', Math.abs),
      fmod: native('fmod', (a) => F(toNumber(a[0]) % toNumber(a[1]))),
      copysign: native('copysign', (a) => {
        const x = Math.abs(toNumber(a[0])); const y = toNumber(a[1]);
        return F(y < 0 || Object.is(y, -0) ? -x : x);
      }),
      isnan: native('isnan', (a) => Number.isNaN(toNumber(a[0]))),
      isinf: native('isinf', (a) => { const x = toNumber(a[0]); return x === Infinity || x === -Infinity; }),
      isfinite: native('isfinite', (a) => Number.isFinite(toNumber(a[0]))),
      isclose: native('isclose', (a, k) => {
        const x = toNumber(a[0]), y = toNumber(a[1]);
        const rel = toNumber(kw(k, 'rel_tol', F(1e-9))), abs = toNumber(kw(k, 'abs_tol', 0));
        return Math.abs(x - y) <= Math.max(rel * Math.max(Math.abs(x), Math.abs(y)), abs);
      }),
      factorial: native('factorial', (a) => {
        const n = toInt(a[0]);
        if (n < 0) throw err('ValueError', 'factorial() not defined for negative values');
        let r = 1; for (let i = 2; i <= n; i++) r *= i; return r;
      }),
      gcd: native('gcd', (a) => {
        let x = Math.abs(toInt(a[0])), y = Math.abs(toInt(a[1]));
        while (y) { [x, y] = [y, x % y]; } return x;
      }),
      modf: native('modf', (a) => { const x = toNumber(a[0]); const ip = Math.trunc(x); return new PyTuple([F(x - ip), F(ip)]); })
    });
  }

  function randomModule(ctx) {
    let rng = mulberry32((Date.now() ^ 0x5bd1e995) >>> 0);
    const rand = () => rng();
    const randint = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1));
    return new PyModule('random', {
      seed: native('seed', (a) => {
        const s = a.length && a[0] !== None ? py.hashKey(a[0]) : String(Date.now());
        let h = 2166136261;
        for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
        rng = mulberry32(h >>> 0);
        return None;
      }),
      random: native('random', () => F(rand())),
      uniform: native('uniform', (a) => { const lo = toNumber(a[0]), hi = toNumber(a[1]); return F(lo + (hi - lo) * rand()); }),
      randint: native('randint', (a) => {
        py.argCount('randint', a, 2, 2);
        const lo = toInt(a[0]), hi = toInt(a[1]);
        if (hi < lo) throw err('ValueError', 'empty range for randint(' + lo + ', ' + hi + ')');
        return randint(lo, hi);
      }),
      randrange: native('randrange', (a) => {
        let start = 0, stop, step = 1;
        if (a.length === 1) stop = toInt(a[0]);
        else { start = toInt(a[0]); stop = toInt(a[1]); if (a.length > 2) step = toInt(a[2]); }
        const n = step > 0 ? Math.ceil((stop - start) / step) : Math.ceil((start - stop) / -step);
        if (n <= 0) throw err('ValueError', 'empty range for randrange()');
        return start + step * Math.floor(rand() * n);
      }),
      choice: native('choice', (a) => {
        const items = py.toArray(a[0]);
        if (!items.length) throw err('IndexError', 'Cannot choose from an empty sequence');
        return items[Math.floor(rand() * items.length)];
      }),
      shuffle: native('shuffle', (a) => {
        const l = a[0];
        if (!(l instanceof PyList)) throw err('TypeError', 'shuffle() needs a list');
        for (let i = l.items.length - 1; i > 0; i--) {
          const j = Math.floor(rand() * (i + 1));
          [l.items[i], l.items[j]] = [l.items[j], l.items[i]];
        }
        return None;
      }),
      getrandbits: native('getrandbits', (a) => Math.floor(rand() * Math.pow(2, Math.min(32, toInt(a[0])))))
    });
  }

  function sleepFor(ctx, ms) {
    const until = ctx.now() + Math.max(0, ms);
    return new Suspend(() => ctx.now() >= until - 1e-9, null, 'sleep');
  }

  function timeModule(ctx) {
    const EPOCH = 1767225600; // 2026-01-01, purely cosmetic for time.time()
    return new PyModule('time', {
      sleep: native('sleep', (a) => sleepFor(ctx, toNumber(a[0], 'time.sleep') * 1000)),
      sleep_ms: native('sleep_ms', (a) => sleepFor(ctx, toNumber(a[0], 'time.sleep_ms'))),
      sleep_us: native('sleep_us', (a) => sleepFor(ctx, toNumber(a[0], 'time.sleep_us') / 1000)),
      ticks_ms: native('ticks_ms', () => Math.floor(ctx.now())),
      ticks_us: native('ticks_us', () => Math.floor(ctx.now() * 1000)),
      ticks_cpu: native('ticks_cpu', () => Math.floor(ctx.now() * 1000)),
      ticks_diff: native('ticks_diff', (a) => toInt(a[0]) - toInt(a[1])),
      ticks_add: native('ticks_add', (a) => toInt(a[0]) + toInt(a[1])),
      time: native('time', () => F(EPOCH + ctx.now() / 1000)),
      time_ns: native('time_ns', () => Math.floor((EPOCH * 1000 + ctx.now()) * 1e6)),
      monotonic: native('monotonic', () => F(ctx.now() / 1000)),
      perf_counter: native('perf_counter', () => F(ctx.now() / 1000))
    });
  }

  function sysModule(ctx) {
    return new PyModule('sys', {
      exit: native('exit', (a) => { throw err('SystemExit', a.length ? a[0] : undefined); }),
      version: '3.4.0; MicroPython (FLL simulator)',
      platform: 'SPIKE Prime (simulated)',
      implementation: new PyTuple(['micropython']),
      maxsize: 2147483647,
      argv: new PyList([]),
      modules: new PyDict(),
      print_exception: native('print_exception', (a) => {
        const e = a[0];
        ctx.interp.print(e.cls.name + ': ' + py.excToString(e) + '\n');
        return None;
      })
    });
  }

  function micropythonModule() {
    return new PyModule('micropython', {
      const: native('const', (a) => a[0]),
      opt_level: native('opt_level', () => 0),
      mem_info: native('mem_info', () => None)
    });
  }
  function gcModule() {
    return new PyModule('gc', {
      collect: native('collect', () => None),
      mem_free: native('mem_free', () => 200000),
      mem_alloc: native('mem_alloc', () => 50000),
      enable: native('enable', () => None),
      disable: native('disable', () => None)
    });
  }

  function asyncioModule(ctx) {
    const sleepAw = (ms) => {
      const until = ctx.now() + Math.max(0, ms);
      return new py.Awaitable(() => ctx.now() >= until - 1e-9, () => None, 'sleep');
    };
    const spawnAll = (coros) => coros.map((c, i) => {
      if (!(c instanceof py.PyCoroutine)) throw err('TypeError', 'a coroutine was expected, got ' + py.typeName(c));
      return ctx.spawn(ctx.runCoroutine(c), c.func.name + '#' + i);
    });
    const taskAwaitable = (t) => new py.Awaitable(() => t.done, () => t.result, 'Task');
    return new PyModule('asyncio', {
      run: native('run', (a) => {
        const tasks = spawnAll([a[0]]);
        return new Suspend(() => tasks[0].done, () => tasks[0].result, 'asyncio.run');
      }),
      sleep: native('sleep', (a) => sleepAw(toNumber(a[0], 'asyncio.sleep') * 1000)),
      sleep_ms: native('sleep_ms', (a) => sleepAw(toNumber(a[0], 'asyncio.sleep_ms'))),
      gather: native('gather', (a) => {
        const tasks = spawnAll(a);
        return new py.Awaitable(() => tasks.every(t => t.done), () => new PyList(tasks.map(t => t.result)), 'gather');
      }),
      create_task: native('create_task', (a) => taskAwaitable(spawnAll([a[0]])[0]))
    });
  }

  py.stdlibModules = {
    math: mathModule,
    random: randomModule,
    time: timeModule,
    utime: timeModule,
    sys: sysModule,
    micropython: micropythonModule,
    gc: gcModule,
    asyncio: asyncioModule,
    uasyncio: asyncioModule
  };
  py.installBuiltins = installBuiltins;
  py.native = native;
  py.kwArg = kw;
  py.noKw = noKw;
  py.sleepFor = sleepFor;
  py.mulberry32 = mulberry32;
})(typeof globalThis !== 'undefined' ? globalThis : window);
