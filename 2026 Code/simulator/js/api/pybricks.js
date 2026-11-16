/*
 * Pybricks API (v3.x) for the simulated SPIKE Prime robot.
 * Blocking calls (robot.straight(), wait(), ...) suspend the program; inside
 * run_task()/multitask() coroutines they return awaitables instead.
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};
  const py = FLL.py;
  const geom = FLL.geom;
  const { None, PyModule, PyTuple, PyList, PyInstance, PyClass, PyNative, Awaitable, Suspend, err, F, toInt, toNumber } = py;
  const PORTS = FLL.robot.PORTS;
  const spikeApi = () => FLL.api.spike;

  const native = (name, fn) => new PyNative(name, fn);
  const kw = (k, name, def) => (k[name] !== undefined ? k[name] : def);
  const arg = (a, k, i, name, def) => (a.length > i ? a[i] : kw(k, name, def));

  function nativeClass(name, methods, init, opts) {
    const dict = new Map();
    dict.set('__init__', native('__init__', (a, k, ctx) => { init(a[0], a.slice(1), k, ctx); return None; }));
    for (const m of Object.keys(methods)) {
      const fn = methods[m];
      dict.set(m, native(m, (a, k, ctx) => fn(a[0], a.slice(1), k, ctx)));
    }
    const cls = new PyClass(name, [py.types.object], dict, opts);
    return cls;
  }
  function enumClass(name, members) {
    const cls = new PyClass(name, [py.types.object], new Map());
    for (const m of Object.keys(members)) {
      const inst = new PyInstance(cls);
      inst.native = { name: m, value: members[m], repr: () => name + '.' + m };
      cls.dict.set(m, inst);
    }
    return cls;
  }

  function createModules(env) {
    const world = env.world;
    const robot = () => world.robot;
    const now = () => world.time;

    // Blocking or awaitable depending on whether we are inside run_task().
    function waitFor(ctx, isDone, result, name, wait) {
      if (wait === false) return None;
      const task = ctx.task;
      if (task && task.pbAsync) {
        return new Awaitable(isDone, result || (() => None), name);
      }
      return new Suspend(isDone, result || null, name);
    }

    // ------------------------------------------------------------ parameters
    const Port = enumClass('Port', { A: 0, B: 1, C: 2, D: 3, E: 4, F: 5 });
    const Direction = enumClass('Direction', { CLOCKWISE: 0, COUNTERCLOCKWISE: 1 });
    const Stop = enumClass('Stop', { COAST: 'coast', COAST_SMART: 'coast', BRAKE: 'brake', HOLD: 'hold', NONE: 'continue' });
    const Button = enumClass('Button', { LEFT: 'left', RIGHT: 'right', CENTER: 'center', BLUETOOTH: 'bluetooth', LEFT_UP: 'lu', LEFT_DOWN: 'ld', RIGHT_UP: 'ru', RIGHT_DOWN: 'rd', LEFT_PLUS: 'lp', LEFT_MINUS: 'lm', RIGHT_PLUS: 'rp', RIGHT_MINUS: 'rm', UP: 'up', DOWN: 'down', BEACON: 'beacon' });
    const Side = enumClass('Side', { TOP: 0, FRONT: 1, RIGHT: 2, BOTTOM: 3, BACK: 4, LEFT: 5 });
    const Axis = enumClass('Axis', { X: 0, Y: 1, Z: 2 });

    // Color with hue/saturation/value.
    const ColorCls = new PyClass('Color', [py.types.object], new Map());
    const makeColor = (h, s, v, nm) => {
      const c = new PyInstance(ColorCls);
      c.dict.set('h', h); c.dict.set('s', s); c.dict.set('v', v);
      c.native = { h: h, s: s, v: v, repr: () => (nm ? 'Color.' + nm : 'Color(h=' + h + ', s=' + s + ', v=' + v + ')') };
      return c;
    };
    ColorCls.dict.set('__init__', native('__init__', (a, k) => {
      const self = a[0];
      const h = toInt(arg(a, k, 1, 'h', 0)), s = toInt(arg(a, k, 2, 's', 100)), v = toInt(arg(a, k, 3, 'v', 100));
      self.dict.set('h', h); self.dict.set('s', s); self.dict.set('v', v);
      self.native = { h: h, s: s, v: v, repr: () => 'Color(h=' + h + ', s=' + s + ', v=' + v + ')' };
      return None;
    }));
    ColorCls.dict.set('__eq__', native('__eq__', (a) => {
      const x = a[0], y = a[1];
      if (!(y instanceof PyInstance) || y.cls !== ColorCls) return false;
      return x.native.h === y.native.h && x.native.s === y.native.s && x.native.v === y.native.v;
    }));
    const COLORS = {
      NONE: [0, 0, 0], BLACK: [0, 0, 10], GRAY: [0, 0, 50], WHITE: [0, 0, 100], RED: [0, 100, 100],
      ORANGE: [30, 100, 100], BROWN: [30, 100, 50], YELLOW: [60, 100, 100], GREEN: [120, 100, 100],
      CYAN: [180, 100, 100], BLUE: [240, 100, 100], VIOLET: [270, 100, 100], MAGENTA: [300, 100, 100]
    };
    for (const n of Object.keys(COLORS)) ColorCls.dict.set(n, makeColor(COLORS[n][0], COLORS[n][1], COLORS[n][2], n));
    const colorRgbName = (c) => {
      if (!(c instanceof PyInstance) || !c.native) return 'white';
      const { h, s, v } = c.native;
      if (v < 15) return 'off';
      if (s < 20) return 'white';
      if (h < 15 || h >= 345) return 'red';
      if (h < 45) return 'orange';
      if (h < 75) return 'yellow';
      if (h < 160) return 'green';
      if (h < 200) return 'azure';
      if (h < 255) return 'blue';
      if (h < 290) return 'violet';
      return 'magenta';
    };

    // Icons (5x5 brightness matrices)
    const iconRows = {
      UP: '0010001110101010010000100', DOWN: '0010000100101010111000100', LEFT: '0010001000111110100000100', RIGHT: '0010000010111110001000100',
      HEART: '0101011111111110111000100', HAPPY: '0000001010000001000101110', SAD: '0000001010000000111010001',
      CIRCLE: '0111010001100011000101110', SQUARE: '1111110001100011000111111', TRIANGLE_UP: '0000000100010101111100000',
      FULL: '1111111111111111111111111', EMPTY: '0000000000000000000000000', TRUE: '0000000001000101010001000', FALSE: '1000101010001000101010001',
      PAUSE: '0101001010010100101001010', ARROW_UP: '0010001110101010010000100'
    };
    const IconCls = new PyClass('Icon', [py.types.object], new Map());
    for (const n of Object.keys(iconRows)) {
      IconCls.dict.set(n, new PyList(Array.from(iconRows[n]).map(c => (c === '1' ? 100 : 0))));
    }

    const parametersMod = new PyModule('pybricks.parameters', {
      Port: Port, Direction: Direction, Stop: Stop, Color: ColorCls, Button: Button, Side: Side, Axis: Axis, Icon: IconCls
    });

    const enumVal = (v, what) => {
      if (v instanceof PyInstance && v.native && v.native.value !== undefined) return v.native.value;
      throw err('TypeError', what + ' must be a ' + what + ' value (for example Port.A)');
    };
    const portOf = (v) => PORTS[enumVal(v, 'Port')];
    const stopOf = (v, def) => (v === undefined || v === None ? def : enumVal(v, 'Stop'));

    // ------------------------------------------------------------ hub
    const hubState = () => robot().hub;
    const performanceNow = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
    const PrimeHub = nativeClass('PrimeHub', {}, (self) => {
      const display = new PyModule('display', {
        text: native('text', (a, k, ctx) => {
          const text = py.str(a[0]);
          const on = toNumber(arg(a, k, 1, 'on', 500)), off = toNumber(arg(a, k, 2, 'off', 50));
          const chars = Array.from(text.toUpperCase());
          const start = now();
          const font = env.font;
          const end = start + chars.length * (on + off);
          hubState().text = text;
          const show = () => {
            const t = now() - start;
            const i = Math.min(chars.length - 1, Math.floor(t / (on + off)));
            const inOff = (t % (on + off)) > on;
            const bits = inOff ? null : (font[chars[i]] || font['?']);
            for (let p = 0; p < 25; p++) hubState().matrix[p] = bits && bits[p] === '1' ? 100 : 0;
          };
          show();
          return waitFor(ctx, () => { show(); return now() >= end; }, null, 'display.text');
        }),
        number: native('number', (a) => {
          const n = Math.round(toNumber(a[0]));
          hubState().text = String(n);
          const s = String(Math.abs(n) % 100).padStart(2, '0');
          const digits = { 0: '111101101101111', 1: '010110010010111', 2: '111001111100111', 3: '111001111001111', 4: '101101111001001', 5: '111100111001111', 6: '111100111101111', 7: '111001001001001', 8: '111101111101111', 9: '111101111001111' };
          const m = new Array(25).fill(0);
          // two 2-column digits squeezed into the 5x5 matrix
          for (let r = 0; r < 5; r++) {
            const d1 = digits[s[0]], d2 = digits[s[1]];
            const row1 = d1.substr(Math.min(r, 4) * 3, 3), row2 = d2.substr(Math.min(r, 4) * 3, 3);
            m[r * 5 + 0] = row1[0] === '1' ? 100 : 0; m[r * 5 + 1] = row1[2] === '1' ? 100 : 0;
            m[r * 5 + 3] = row2[0] === '1' ? 100 : 0; m[r * 5 + 4] = row2[2] === '1' ? 100 : 0;
          }
          if (n < 0) m[12] = 100;
          for (let p = 0; p < 25; p++) hubState().matrix[p] = m[p];
          return None;
        }),
        char: native('char', (a) => {
          const bits = env.font[py.str(a[0]).toUpperCase()[0]] || env.font['?'];
          for (let p = 0; p < 25; p++) hubState().matrix[p] = bits[p] === '1' ? 100 : 0;
          return None;
        }),
        icon: native('icon', (a) => {
          const vals = py.toArray(a[0]).flatMap(x => (x instanceof PyList || x instanceof PyTuple ? py.toArray(x) : [x])).map(v => toNumber(v));
          for (let p = 0; p < 25; p++) hubState().matrix[p] = geom.clamp(vals[p] || 0, 0, 100);
          return None;
        }),
        pixel: native('pixel', (a) => {
          const r = toInt(a[0]), c = toInt(a[1]);
          hubState().matrix[r * 5 + c] = geom.clamp(toNumber(a.length > 2 ? a[2] : 100), 0, 100);
          return None;
        }),
        off: native('off', () => { hubState().matrix.fill(0); hubState().text = ''; return None; }),
        orientation: native('orientation', () => None),
        animate: native('animate', () => None)
      });
      const light = new PyModule('light', {
        on: native('on', (a) => { hubState().centerColor = colorRgbName(a[0]); return None; }),
        off: native('off', () => { hubState().centerColor = 'off'; return None; }),
        blink: native('blink', (a) => { hubState().centerColor = colorRgbName(a[0]); return None; }),
        animate: native('animate', () => None)
      });
      const speaker = new PyModule('speaker', {
        beep: native('beep', (a, k, ctx) => {
          const f = toNumber(arg(a, k, 0, 'frequency', 500)), d = toNumber(arg(a, k, 1, 'duration', 100));
          if (env.onBeep) env.onBeep(f, d, 80);
          const end = now() + d;
          return waitFor(ctx, () => now() >= end, null, 'speaker.beep');
        }),
        play_notes: native('play_notes', (a, k, ctx) => {
          const notes = py.toArray(a[0]);
          const tempo = toNumber(arg(a, k, 1, 'tempo', 120));
          const end = now() + notes.length * 60000 / tempo / 4;
          return waitFor(ctx, () => now() >= end, null, 'speaker.play_notes');
        }),
        volume: native('volume', () => 100)
      });
      const buttons = new PyModule('buttons', {
        pressed: native('pressed', () => {
          const b = hubState().buttons;
          const out = [];
          if (b.left) out.push(Button.dict.get('LEFT'));
          if (b.right) out.push(Button.dict.get('RIGHT'));
          if (b.center) out.push(Button.dict.get('CENTER'));
          return py.makeSet(out);
        })
      });
      const imu = new PyModule('imu', {
        heading: native('heading', () => F(-robot().yawCCW(now()))),
        reset_heading: native('reset_heading', (a) => { robot().resetYaw(-toNumber(a.length ? a[0] : 0), now()); return None; }),
        tilt: native('tilt', () => new PyTuple([0, 0])),
        acceleration: native('acceleration', (a) => (a.length ? F(0) : new PyTuple([F(0), F(0), F(9810)]))),
        angular_velocity: native('angular_velocity', (a) => (a.length ? F(-(robot().yawRate || 0)) : new PyTuple([F(0), F(0), F(-(robot().yawRate || 0))]))),
        ready: native('ready', () => true),
        stationary: native('stationary', () => Math.abs(robot().linearSpeed || 0) < 1),
        up: native('up', () => Side.dict.get('TOP')),
        orientation: native('orientation', () => None),
        rotation: native('rotation', () => F(-robot().yawCCW(now()))),
        settings: native('settings', () => None)
      });
      const battery = new PyModule('battery', {
        voltage: native('voltage', () => 8150), current: native('current', () => 180)
      });
      const system = new PyModule('system', {
        set_stop_button: native('set_stop_button', () => None),
        name: native('name', () => 'Pybricks Hub'),
        shutdown: native('shutdown', () => { throw err('SystemExit'); }),
        storage: native('storage', () => None),
        reset_reason: native('reset_reason', () => 0)
      });
      for (const [k, v] of [['display', display], ['light', light], ['speaker', speaker], ['buttons', buttons], ['imu', imu], ['battery', battery], ['system', system]]) self.dict.set(k, v);
    });
    const hubsMod = new PyModule('pybricks.hubs', { PrimeHub: PrimeHub, InventorHub: PrimeHub, EssentialHub: PrimeHub });

    // ------------------------------------------------------------ motors
    const getSimMotor = (P) => {
      const m = robot().motors[P];
      if (!m) {
        const d = robot().device(P);
        throw err('OSError', '[Errno 19] ENODEV: no motor on port ' + P + (d && d.type !== 'none' ? ' (a ' + d.type + ' sensor is there)' : '') + ' – check the Robot setup');
      }
      return m;
    };
    const motorState = (self) => {
      if (!self.native || !self.native.motor) throw err('RuntimeError', 'motor object is not initialised');
      return self.native;
    };
    const mangle = (s, deg) => deg * s.sign * s.gear;   // output-shaft degrees -> raw motor degrees
    const unmangle = (s, raw) => raw * s.sign / s.gear; // raw motor degrees -> output-shaft degrees
    const Motor = nativeClass('Motor', {
      angle: (self) => Math.round(unmangle(motorState(self), motorState(self).motor.pos) - motorState(self).offset),
      reset_angle: (self, a) => {
        const s = motorState(self);
        const v = a.length && a[0] !== None ? toNumber(a[0]) : unmangle(s, s.motor.absolutePosition);
        s.offset = unmangle(s, s.motor.pos) - v;
        return None;
      },
      speed: (self) => Math.round(unmangle(motorState(self), motorState(self).motor.speed)),
      load: (self) => 0,
      stalled: (self) => motorState(self).motor.stallTime > 0.2,
      done: (self) => { const m = motorState(self).motor; return !m.cmd || m.cmd.done; },
      run: (self, a) => {
        const s = motorState(self);
        s.motor.runAt(mangle(s, toNumber(a[0], 'speed')), 2000);
        return None;
      },
      stop: (self) => { motorState(self).motor.stop('coast'); return None; },
      brake: (self) => { motorState(self).motor.stop('brake'); return None; },
      hold: (self) => { motorState(self).motor.stop('hold'); return None; },
      dc: (self, a) => { motorState(self).motor.setDuty(toNumber(a[0]) * 100); return None; },
      run_time: (self, a, k, ctx) => {
        const s = motorState(self);
        const cmd = s.motor.runForTime(toNumber(a[1], 'time'), mangle(s, toNumber(a[0], 'speed')), { end: stopOf(arg(a, k, 2, 'then', undefined), 'hold'), accel: 2000, decel: 2000 }, now());
        return waitFor(ctx, () => cmd.done, null, 'run_time', py.truthy(arg(a, k, 3, 'wait', true)));
      },
      run_angle: (self, a, k, ctx) => {
        const s = motorState(self);
        const speed = toNumber(a[0], 'speed');
        const ang = toNumber(a[1], 'rotation_angle');
        const raw = mangle(s, Math.abs(ang) * Math.sign(ang) * (speed < 0 ? -1 : 1));
        const cmd = s.motor.runToPosition(s.motor.pos + raw, Math.abs(speed) * s.gear, { end: stopOf(arg(a, k, 2, 'then', undefined), 'hold'), accel: 2000, decel: 2000 });
        return waitFor(ctx, () => cmd.done, null, 'run_angle', py.truthy(arg(a, k, 3, 'wait', true)));
      },
      run_target: (self, a, k, ctx) => {
        const s = motorState(self);
        const target = toNumber(a[1], 'target_angle');
        const rawTarget = mangle(s, target + s.offset);
        const cmd = s.motor.runToPosition(rawTarget, Math.abs(toNumber(a[0], 'speed')) * s.gear, { end: stopOf(arg(a, k, 2, 'then', undefined), 'hold'), accel: 2000, decel: 2000 });
        return waitFor(ctx, () => cmd.done, null, 'run_target', py.truthy(arg(a, k, 3, 'wait', true)));
      },
      run_until_stalled: (self, a, k, ctx) => {
        const s = motorState(self);
        const m = s.motor;
        m.runAt(mangle(s, toNumber(a[0], 'speed')), 2000);
        const then = stopOf(arg(a, k, 1, 'then', undefined), 'coast');
        let finished = false;
        return waitFor(ctx, () => {
          if (finished) return true;
          if (m.stallTime > 0.25) { m.stop(then); finished = true; }
          return finished;
        }, () => Math.round(unmangle(s, m.pos) - s.offset), 'run_until_stalled');
      },
      track_target: (self, a) => {
        const s = motorState(self);
        s.motor.runToPosition(mangle(s, toNumber(a[0]) + s.offset), s.motor.maxSpeed, { end: 'hold', accel: 20000, decel: 20000 });
        return None;
      },
      settings: () => None,
      close: () => None
    }, (self, a, k) => {
      const P = portOf(arg(a, k, 0, 'port'));
      const motor = getSimMotor(P);
      const dir = arg(a, k, 1, 'positive_direction', None);
      const sign = dir !== None && enumVal(dir, 'Direction') === 1 ? -1 : 1;
      let gear = 1;
      const gears = arg(a, k, 2, 'gears', None);
      if (gears !== None) {
        const lists = py.toArray(gears);
        const chains = (lists.length && (lists[0] instanceof PyList || lists[0] instanceof PyTuple)) ? lists : [gears];
        for (const ch of chains) { const g = py.toArray(ch).map(x => toNumber(x)); gear *= g[g.length - 1] / g[0]; }
      }
      self.native = { motor: motor, port: P, sign: sign, gear: gear, offset: 0 };
      const resetAngle = py.truthy(arg(a, k, 3, 'reset_angle', true));
      self.native.offset = resetAngle ? unmangle(self.native, motor.pos) - unmangle(self.native, motor.absolutePosition) : 0;
    });

    // ------------------------------------------------------------ sensors
    const needDevice = (P, type) => {
      const d = robot().device(P);
      if (!d || d.type !== type) throw err('OSError', '[Errno 19] ENODEV: no ' + type + ' sensor on port ' + P + ' – check the Robot setup');
    };
    const DEFAULT_DETECT = ['RED', 'YELLOW', 'GREEN', 'BLUE', 'WHITE', 'NONE'];
    const ColorSensor = nativeClass('ColorSensor', {
      reflection: (self) => {
        const rgb = world.colorUnder.apply(world, robot().sensorWorld(self.native.port));
        return spikeApi().reflectionOf(rgb);
      },
      ambient: () => 20,
      hsv: (self) => {
        const rgb = world.colorUnder.apply(world, robot().sensorWorld(self.native.port));
        const [h, s, v] = spikeApi().rgbToHsv(rgb[0], rgb[1], rgb[2]);
        return makeColor(Math.round(h), Math.round(s * 100), Math.round(v * 100));
      },
      color: (self) => {
        const rgb = world.colorUnder.apply(world, robot().sensorWorld(self.native.port));
        let [h, s, v] = spikeApi().rgbToHsv(rgb[0], rgb[1], rgb[2]);
        s *= 100; v *= 100;
        const chromatic = self.native.detect.filter(c => c.native.s >= 50 && c.native.v >= 50);
        const plain = self.native.detect.filter(c => !(c.native.s >= 50 && c.native.v >= 50));
        const nearestV = (list) => list.reduce((b, c) => (!b || Math.abs(c.native.v - v) < Math.abs(b.native.v - v) ? c : b), null);
        if ((s < 25 || v < 18) && plain.length) return nearestV(plain);
        if (!chromatic.length) return nearestV(plain);
        let best = null, bestD = Infinity;
        for (const c of chromatic) {
          let dh = Math.abs(h - c.native.h); if (dh > 180) dh = 360 - dh;
          if (dh < bestD) { bestD = dh; best = c; }
        }
        return best;
      },
      detectable_colors: (self, a) => {
        if (a.length) self.native.detect = py.toArray(a[0]);
        return new PyTuple(self.native.detect.slice());
      },
      lights: (self) => None
    }, (self, a, k) => {
      const P = portOf(arg(a, k, 0, 'port'));
      needDevice(P, 'color');
      self.native = { port: P, detect: DEFAULT_DETECT.map(n => ColorCls.dict.get(n)) };
      self.dict.set('lights', new PyModule('lights', { on: native('on', () => None), off: native('off', () => None) }));
    });
    const UltrasonicSensor = nativeClass('UltrasonicSensor', {
      distance: (self) => {
        const rb = robot();
        const d = rb.device(self.native.port);
        const dist = world.rayDistance(rb.sensorWorld(self.native.port), rb.pose.theta + geom.rad(d.angle || 0), 2100);
        return dist > 2000 ? 2000 : Math.round(dist);
      },
      presence: () => false
    }, (self, a, k) => {
      const P = portOf(arg(a, k, 0, 'port'));
      needDevice(P, 'distance');
      self.native = { port: P };
      self.dict.set('lights', new PyModule('lights', { on: native('on', () => None), off: native('off', () => None) }));
    });
    const ForceSensor = nativeClass('ForceSensor', {
      force: (self) => F(world.bumperPressed(self.native.port) ? 6.0 : 0.0),
      distance: (self) => F(world.bumperPressed(self.native.port) ? 0.0 : 8.0),
      pressed: (self) => world.bumperPressed(self.native.port),
      touched: (self) => world.bumperPressed(self.native.port)
    }, (self, a, k) => {
      const P = portOf(arg(a, k, 0, 'port'));
      needDevice(P, 'force');
      self.native = { port: P };
    });
    const pupMod = new PyModule('pybricks.pupdevices', {
      Motor: Motor, ColorSensor: ColorSensor, UltrasonicSensor: UltrasonicSensor, ForceSensor: ForceSensor
    });

    // ------------------------------------------------------------ robotics
    const dbState = (self) => {
      if (!self.native || !self.native.left) throw err('RuntimeError', 'DriveBase is not initialised');
      return self.native;
    };
    // heading (clockwise degrees) seen by the drive base
    const dbAngle = (s) => {
      if (s.gyro) return -robot().yawCCW(now()) - s.gyroOffset;
      const l = unmangle(s.left, s.left.motor.pos), r = unmangle(s.right, s.right.motor.pos);
      return ((l - s.l0) - (r - s.r0)) * s.mmPerDeg / s.track * 180 / Math.PI;
    };
    const dbDistance = (s) => {
      const l = unmangle(s.left, s.left.motor.pos), r = unmangle(s.right, s.right.motor.pos);
      return ((l - s.l0) + (r - s.r0)) / 2 * s.mmPerDeg - s.dist0;
    };
    // Start a synchronized move. wheelL/wheelR = wheel travel in mm (forward positive).
    function dbMove(s, wheelL, wheelR, speedMm, accelMm, then, headingChange) {
      const degL = wheelL / s.mmPerDeg, degR = wheelR / s.mmPerDeg;
      const master = Math.max(Math.abs(degL), Math.abs(degR));
      const rates = master > 0 ? [mangle(s.left, degL / master), mangle(s.right, degR / master)] : [0, 0];
      const opts = { distance: master, velocity: speedMm / s.mmPerDeg, accel: accelMm / s.mmPerDeg, decel: accelMm / s.mmPerDeg, end: then };
      if (s.gyro && master > 0) {
        // Gyro mode: correct the wheel setpoints so the heading follows the plan.
        const startHeading = -robot().yawCCW(now());
        const kHead = s.mmPerDeg / s.track * 180 / Math.PI; // heading degrees per wheel-degree difference
        let c = 0;
        opts.corrector = (sync) => {
          const expected = startHeading + headingChange * (sync.p / master);
          const e = (-robot().yawCCW(now())) - expected;
          c = geom.clamp(c + 0.25 * e / kHead, -720, 720);
          return [mangle(s.left, -c), mangle(s.right, c)];
        };
      }
      const sync = new FLL.robot.SyncMove([s.left.motor, s.right.motor], rates, opts);
      robot().addSync(sync);
      s.last = sync;
      return sync;
    }
    const DriveBase = nativeClass('DriveBase', {
      straight: (self, a, k, ctx) => {
        const s = dbState(self);
        const d = toNumber(a[0], 'distance');
        const sync = dbMove(s, d, d, s.settings.straight_speed, s.settings.straight_acceleration, stopOf(arg(a, k, 1, 'then', undefined), 'hold'), 0);
        return waitFor(ctx, () => sync.cmd.done, null, 'straight', py.truthy(arg(a, k, 2, 'wait', true)));
      },
      turn: (self, a, k, ctx) => {
        const s = dbState(self);
        const ang = toNumber(a[0], 'angle');
        const arc = ang * Math.PI / 180 * s.track / 2;
        const speed = s.settings.turn_rate * Math.PI / 180 * s.track / 2;
        const acc = s.settings.turn_acceleration * Math.PI / 180 * s.track / 2;
        const sync = dbMove(s, arc, -arc, speed, acc, stopOf(arg(a, k, 1, 'then', undefined), 'hold'), ang);
        return waitFor(ctx, () => sync.cmd.done, null, 'turn', py.truthy(arg(a, k, 2, 'wait', true)));
      },
      curve: (self, a, k, ctx) => {
        const s = dbState(self);
        const r = toNumber(a[0], 'radius'), ang = toNumber(a[1], 'angle');
        const th = ang * Math.PI / 180;
        const wl = (r + s.track / 2) * th, wr = (r - s.track / 2) * th;
        const sync = dbMove(s, wl, wr, s.settings.straight_speed, s.settings.straight_acceleration, stopOf(arg(a, k, 2, 'then', undefined), 'hold'), ang);
        return waitFor(ctx, () => sync.cmd.done, null, 'curve', py.truthy(arg(a, k, 3, 'wait', true)));
      },
      arc: (self, a, k, ctx) => {
        const s = dbState(self);
        const r = toNumber(a[0], 'radius');
        let ang = arg(a, k, 1, 'angle', None);
        const dist = arg(a, k, 2, 'distance', None);
        let th;
        if (ang !== None) th = toNumber(ang) * Math.PI / 180;
        else th = toNumber(dist) / (r || 1);
        const wl = (r + s.track / 2) * th, wr = (r - s.track / 2) * th;
        const sync = dbMove(s, wl, wr, s.settings.straight_speed, s.settings.straight_acceleration, stopOf(kw(k, 'then', undefined), 'hold'), th * 180 / Math.PI);
        return waitFor(ctx, () => sync.cmd.done, null, 'arc', py.truthy(kw(k, 'wait', true)));
      },
      drive: (self, a) => {
        const s = dbState(self);
        const speed = toNumber(a[0], 'speed'), rate = toNumber(a[1], 'turn_rate');
        const diff = rate * Math.PI / 180 * s.track / 2;
        const vl = (speed + diff) / s.mmPerDeg, vr = (speed - diff) / s.mmPerDeg;
        s.left.motor.runAt(mangle(s.left, vl), 3000);
        s.right.motor.runAt(mangle(s.right, vr), 3000);
        return None;
      },
      stop: (self) => { const s = dbState(self); s.left.motor.stop('coast'); s.right.motor.stop('coast'); return None; },
      brake: (self) => { const s = dbState(self); s.left.motor.stop('brake'); s.right.motor.stop('brake'); return None; },
      distance: (self) => Math.round(dbDistance(dbState(self))),
      angle: (self) => Math.round(dbAngle(dbState(self))),
      state: (self) => {
        const s = dbState(self);
        const ls = unmangle(s.left, s.left.motor.speed) * s.mmPerDeg, rs = unmangle(s.right, s.right.motor.speed) * s.mmPerDeg;
        return new PyTuple([Math.round(dbDistance(s)), Math.round((ls + rs) / 2), Math.round(dbAngle(s)), Math.round((ls - rs) / s.track * 180 / Math.PI)]);
      },
      reset: (self, a) => {
        const s = dbState(self);
        s.l0 = unmangle(s.left, s.left.motor.pos);
        s.r0 = unmangle(s.right, s.right.motor.pos);
        s.dist0 = 0;
        if (a.length && a[0] !== None) s.dist0 = -toNumber(a[0]);
        s.gyroOffset = -robot().yawCCW(now()) - (a.length > 1 ? toNumber(a[1]) : 0);
        return None;
      },
      settings: (self, a, k) => {
        const s = dbState(self);
        const names = ['straight_speed', 'straight_acceleration', 'turn_rate', 'turn_acceleration'];
        if (!a.length && !Object.keys(k).length) return new PyTuple(names.map(n => Math.round(s.settings[n])));
        names.forEach((n, i) => {
          const v = a.length > i ? a[i] : k[n];
          if (v !== undefined && v !== None) s.settings[n] = Math.abs(toNumber(v, n));
        });
        return None;
      },
      done: (self) => { const s = dbState(self); return !s.last || s.last.cmd.done; },
      stalled: (self) => { const s = dbState(self); return s.left.motor.stallTime > 0.2 || s.right.motor.stallTime > 0.2; },
      use_gyro: (self, a) => { const s = dbState(self); s.gyro = py.truthy(a[0]); s.gyroOffset = -robot().yawCCW(now()) - dbAngle(Object.assign({}, s, { gyro: false })); return None; }
    }, (self, a, k) => {
      const L = arg(a, k, 0, 'left_motor'), R = arg(a, k, 1, 'right_motor');
      if (!(L instanceof PyInstance) || !L.native || !L.native.motor) throw err('TypeError', 'DriveBase needs two Motor objects');
      if (!(R instanceof PyInstance) || !R.native || !R.native.motor) throw err('TypeError', 'DriveBase needs two Motor objects');
      const wd = toNumber(arg(a, k, 2, 'wheel_diameter'), 'wheel_diameter');
      const track = toNumber(arg(a, k, 3, 'axle_track'), 'axle_track');
      const mmPerDeg = Math.PI * wd / 360;
      const maxMm = L.native.motor.maxSpeed * mmPerDeg;
      self.native = {
        left: L.native, right: R.native, wd: wd, track: track, mmPerDeg: mmPerDeg, gyro: false, gyroOffset: 0,
        l0: unmangle(L.native, L.native.motor.pos), r0: unmangle(R.native, R.native.motor.pos), dist0: 0, last: null,
        settings: {
          straight_speed: Math.round(maxMm * 0.4), straight_acceleration: Math.round(maxMm * 1.5),
          turn_rate: Math.round(maxMm * 0.4 / (track / 2) * 180 / Math.PI), turn_acceleration: Math.round(maxMm * 1.5 / (track / 2) * 180 / Math.PI)
        }
      };
      self.native.gyroOffset = -robot().yawCCW(now());
    });
    const roboticsMod = new PyModule('pybricks.robotics', { DriveBase: DriveBase, GyroDriveBase: DriveBase });

    // ------------------------------------------------------------ tools
    const StopWatch = nativeClass('StopWatch', {
      time: (self) => {
        const s = self.native;
        return Math.round(s.paused ? s.acc : s.acc + (now() - s.start));
      },
      pause: (self) => { const s = self.native; if (!s.paused) { s.acc += now() - s.start; s.paused = true; } return None; },
      resume: (self) => { const s = self.native; if (s.paused) { s.start = now(); s.paused = false; } return None; },
      reset: (self) => { const s = self.native; s.acc = 0; s.start = now(); return None; }
    }, (self) => { self.native = { start: now(), acc: 0, paused: false }; });
    const spawnAsync = (ctx, coros) => coros.map((c, i) => {
      if (!(c instanceof py.PyCoroutine)) throw err('TypeError', 'expected a coroutine (call your async function), got ' + py.typeName(c));
      const t = ctx.spawn(ctx.runCoroutine(c), c.func.name + '#' + i);
      t.pbAsync = true;
      return t;
    });
    const toolsMod = new PyModule('pybricks.tools', {
      wait: native('wait', (a, k, ctx) => {
        const end = now() + Math.max(0, toNumber(a[0], 'time'));
        return waitFor(ctx, () => now() >= end, null, 'wait');
      }),
      StopWatch: StopWatch,
      multitask: native('multitask', (a, k, ctx) => {
        const race = py.truthy(kw(k, 'race', false));
        const tasks = spawnAsync(ctx, a);
        const isDone = () => (race ? tasks.some(t => t.done) : tasks.every(t => t.done));
        return new Awaitable(() => {
          const d = isDone();
          if (d && race) tasks.forEach(t => { if (!t.done) t.done = true; });
          return d;
        }, () => new PyTuple(tasks.map(t => (t.done ? t.result : None))), 'multitask');
      }),
      run_task: native('run_task', (a, k, ctx) => {
        const tasks = spawnAsync(ctx, [a[0]]);
        return new Suspend(() => tasks[0].done, () => tasks[0].result, 'run_task');
      }),
      hub_menu: native('hub_menu', (a) => (a.length ? a[0] : None)),
      read_input_byte: native('read_input_byte', () => None),
      cross: native('cross', () => None)
    });
    const pybricksMod = new PyModule('pybricks', {
      hubs: hubsMod, pupdevices: pupMod, parameters: parametersMod, robotics: roboticsMod, tools: toolsMod, version: new PyTuple(['primehub', '3.5.0', 'simulated'])
    });

    return {
      pybricks: pybricksMod,
      'pybricks.hubs': hubsMod,
      'pybricks.pupdevices': pupMod,
      'pybricks.parameters': parametersMod,
      'pybricks.robotics': roboticsMod,
      'pybricks.tools': toolsMod
    };
  }

  FLL.api = FLL.api || {};
  FLL.api.pybricks = { createModules };
})(typeof globalThis !== 'undefined' ? globalThis : window);
