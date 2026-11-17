/*
 * SPIKE Prime (SPIKE App 3) Python API for the simulated robot.
 * Modules: hub (port, light_matrix, light, sound, button, motion_sensor), motor, motor_pair,
 * color_sensor, color, distance_sensor, force_sensor, runloop, device, app.
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};
  const py = FLL.py;
  const geom = FLL.geom;
  const { None, PyModule, PyTuple, PyList, PyInstance, PyClass, PyNative, Awaitable, Suspend, err, F, toInt, toNumber } = py;
  const STATUS = FLL.robot.STATUS;
  const PORTS = FLL.robot.PORTS;

  const native = (name, fn) => new PyNative(name, fn);
  const kw = (k, name, def) => (k[name] !== undefined ? k[name] : def);

  // ------------------------------------------------------------------ colours
  const COLOR = { BLACK: 0, MAGENTA: 1, PURPLE: 2, BLUE: 3, AZURE: 4, TURQUOISE: 5, GREEN: 6, YELLOW: 7, ORANGE: 8, RED: 9, WHITE: 10, UNKNOWN: -1 };
  const COLOR_NAMES = ['black', 'magenta', 'violet', 'blue', 'azure', 'turquoise', 'green', 'yellow', 'orange', 'red', 'white'];

  function rgbToHsv(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const d = max - min;
    let h = 0;
    if (d > 0) {
      if (max === r) h = 60 * (((g - b) / d) % 6);
      else if (max === g) h = 60 * ((b - r) / d + 2);
      else h = 60 * ((r - g) / d + 4);
    }
    if (h < 0) h += 360;
    return [h, max === 0 ? 0 : d / max, max];
  }

  // Classify an RGB reading like the SPIKE colour sensor does (approximation).
  function classifyColor(rgb) {
    const [h, s, v] = rgbToHsv(rgb[0], rgb[1], rgb[2]);
    if (v < 0.22) return COLOR.BLACK;
    if (s < 0.22) {
      if (v > 0.68) return COLOR.WHITE;
      if (v < 0.38) return COLOR.BLACK;
      return COLOR.UNKNOWN;
    }
    if (h < 12 || h >= 345) return COLOR.RED;
    if (h < 40) return COLOR.ORANGE;
    if (h < 70) return COLOR.YELLOW;
    if (h < 160) return COLOR.GREEN;
    if (h < 185) return COLOR.TURQUOISE;
    if (h < 212) return COLOR.AZURE;
    if (h < 250) return COLOR.BLUE;
    if (h < 290) return COLOR.PURPLE;
    return COLOR.MAGENTA;
  }
  function reflectionOf(rgb) {
    // Perceived brightness, mapped so that the black lines read ~10 and white ~95.
    const lum = 0.30 * rgb[0] + 0.55 * rgb[1] + 0.15 * rgb[2];
    return geom.clamp(Math.round(lum / 255 * 92 + 4), 0, 100);
  }

  // ------------------------------------------------------------------ images & font
  const IMG = {
    1: '0101011111111110111000100', 2: '0000001010011100010000000', 3: '0000001010000001000101110',
    4: '0000001010000001000101110', 5: '0000001010000000111010001', 6: '0000001010000000101010101',
    7: '1000101010000001111110101', 8: '0000011011000000111000000', 9: '0101000000001000101000100',
    10: '1000100000111110010100111', 11: '1111111111000000101001110', 12: '0101000000000000111000000',
    13: '0000000001000101010001000', 14: '1000101010001000101010001',
    27: '0010001110101010010000100', 28: '0011100011001010100010000', 29: '0010000010111110001000100',
    30: '1000001001001010001100111', 31: '0010000100101010111000100', 32: '0000100010101001100011100',
    33: '0010001000111110100000100', 34: '1110011000101000001000001',
    35: '0010000010111110001000100', 36: '0010001000111110100000100', 37: '0010001110101010010000100', 38: '0010000100101010111000100',
    39: '0000000100010101111100000', 40: '0000100011001110001100001', 41: '0101010101010101010101010',
    42: '0010001010100010101000100', 43: '0000000100010100010000000', 44: '1111110001100011000111111',
    45: '0000001110010100111000000', 54: '0010001010101010101000100', 58: '0010001110111110101001110',
    62: '1111110101111111111110101', 65: '0111010101111110111001110'
  };
  const FONT = {
    'A': '0110010010111101001010010', 'B': '1110010010111001001011100', 'C': '0111010000100001000001110',
    'D': '1110010010100101001011100', 'E': '1111010000111001000011110', 'F': '1111010000111001000010000',
    'G': '0111010000100111001001110', 'H': '1001010010111101001010010', 'I': '1110001000010000100011100',
    'J': '1111000010000101001001100', 'K': '1001010100110001010010010', 'L': '1000010000100001000011110',
    'M': '1000111011101011000110001', 'N': '1000111001101011001110001', 'O': '0110010010100101001001100',
    'P': '1110010010111001000010000', 'Q': '0110010010100101011001110', 'R': '1110010010111001010010010',
    'S': '0111010000011000001011100', 'T': '1111100100001000010000100', 'U': '1001010010100101001001100',
    'V': '1000110001100010101000100', 'W': '1000110001101011101110001', 'X': '1001010010011001001010010',
    'Y': '1000101010001000010000100', 'Z': '1111000010001000100011110',
    '0': '0110010010100101001001100', '1': '0010001100001000010001110', '2': '1110000010011001000011110',
    '3': '1110000010011000001011100', '4': '0011001010100101111100010', '5': '1111010000111000001011100',
    '6': '0111010000111001001001100', '7': '1111000010001000100001000', '8': '0110010010011001001001100',
    '9': '0110010010011100001001100', ' ': '0000000000000000000000000', '!': '0010000100001000000000100',
    '?': '0110000010001100000000100', '.': '0000000000000000000000100', '-': '0000000000011100000000000',
    '+': '0000000100011100010000000', ':': '0000000100000000010000000', '=': '0000001110000000111000000',
    ',': '0000000000000000010001000', '_': '0000000000000000000011111', '%': '1100111010001000101110011'
  };
  function bitsToMatrix(bits, intensity) {
    return Array.from(bits).map(b => (b === '1' ? intensity : 0));
  }

  // ------------------------------------------------------------------ module factory
  function createModules(env) {
    const world = env.world;
    const robot = () => world.robot;
    const now = () => world.time;
    const pairs = [null, null, null];
    let matrixOrientation = 0;
    const motorSpeedOf = {};

    const portName = (p) => {
      const i = toInt(p, 'port');
      if (i < 0 || i > 5) throw err('ValueError', 'invalid port ' + i + ' (use port.A ... port.F)');
      return PORTS[i];
    };
    const describeDevice = (P) => {
      const d = robot().device(P);
      if (!d || d.type === 'none') return 'nothing is connected there';
      if (d.type === 'motor') return 'a motor is connected there';
      return 'a ' + d.type + ' sensor is connected there';
    };
    const getMotor = (p) => {
      const P = portName(p);
      const m = robot().motors[P];
      if (!m) throw err('OSError', '[Errno 19] ENODEV: no motor on port ' + P + ' (' + describeDevice(P) + '; check the Robot setup)');
      return m;
    };
    const needSensor = (p, type) => {
      const P = portName(p);
      const d = robot().device(P);
      if (!d || d.type !== type) {
        throw err('OSError', '[Errno 19] ENODEV: no ' + type + ' sensor on port ' + P + ' (' + describeDevice(P) + '; check the Robot setup)');
      }
      return P;
    };
    const stopMode = (v) => {
      const s = v === undefined ? 1 : toInt(v, 'stop');
      return ({ 0: 'coast', 1: 'brake', 2: 'hold', 3: 'continue', 4: 'coast', 5: 'brake' })[s] || 'brake';
    };
    const cmdAwaitable = (cmd, name) => {
      const a = new Awaitable(() => cmd.done, () => cmd.status, name);
      a.hint = name + '() was not awaited, so the program did not wait for it to finish. Put \'await\' in front of it (inside an async function).';
      return a;
    };
    const velArg = (v, what) => toInt(v, what || 'velocity');

    // ---------------------------------------------------------------- hub.port
    const portMod = new PyModule('hub.port', { A: 0, B: 1, C: 2, D: 3, E: 4, F: 5 });

    // ---------------------------------------------------------------- light matrix
    const hub = () => robot().hub;
    const setMatrix = (arr) => { for (let i = 0; i < 25; i++) hub().matrix[i] = geom.clamp(arr[i] || 0, 0, 100); };
    const rotIndex = (x, y) => {
      // orientation: 0 = up (default)
      switch (matrixOrientation) {
        case 1: return (4 - x) * 5 + y;        // left
        case 2: return (4 - y) * 5 + (4 - x);  // down
        case 3: return x * 5 + (4 - y);        // right
        default: return y * 5 + x;
      }
    };
    const lightMatrix = new PyModule('hub.light_matrix', {
      show_image: native('show_image', (a) => {
        const id = toInt(a[0], 'image');
        setMatrix(bitsToMatrix(IMG[id] || IMG[1], 100));
        return None;
      }),
      show: native('show', (a) => {
        const vals = py.toArray(a[0]).map(v => toInt(v));
        if (vals.length !== 25) throw err('ValueError', 'show() needs a list of 25 brightness values');
        setMatrix(vals);
        return None;
      }),
      write: native('write', (a, k) => {
        const text = py.str(a[0]);
        const intensity = toInt(a.length > 1 ? a[1] : kw(k, 'intensity', 100));
        const per = toNumber(a.length > 2 ? a[2] : kw(k, 'time_per_character', 500));
        const chars = Array.from(text.toUpperCase());
        const start = now();
        hub().text = text;
        const show = () => {
          const i = Math.min(chars.length - 1, Math.floor((now() - start) / per));
          setMatrix(bitsToMatrix(FONT[chars[i]] || FONT['?'], intensity));
        };
        show();
        if (chars.length <= 1) return new Awaitable(() => true, () => None, 'write');
        const end = start + per * chars.length;
        return new Awaitable(() => { show(); return now() >= end; }, () => None, 'write');
      }),
      set_pixel: native('set_pixel', (a) => {
        const x = toInt(a[0]), y = toInt(a[1]);
        if (x < 0 || x > 4 || y < 0 || y > 4) throw err('ValueError', 'pixel coordinates must be 0..4');
        hub().matrix[rotIndex(x, y)] = geom.clamp(toInt(a[2]), 0, 100);
        return None;
      }),
      get_pixel: native('get_pixel', (a) => hub().matrix[rotIndex(toInt(a[0]), toInt(a[1]))]),
      clear: native('clear', () => { setMatrix([]); hub().text = ''; return None; }),
      set_orientation: native('set_orientation', (a) => { matrixOrientation = toInt(a[0]) % 4; return matrixOrientation; }),
      get_orientation: native('get_orientation', () => matrixOrientation)
    });
    const imageNames = ['HEART', 'HEART_SMALL', 'HAPPY', 'SMILE', 'SAD', 'CONFUSED', 'ANGRY', 'ASLEEP', 'SURPRISED', 'SILLY', 'FABULOUS', 'MEH', 'YES', 'NO',
      'CLOCK12', 'CLOCK1', 'CLOCK2', 'CLOCK3', 'CLOCK4', 'CLOCK5', 'CLOCK6', 'CLOCK7', 'CLOCK8', 'CLOCK9', 'CLOCK10', 'CLOCK11',
      'ARROW_N', 'ARROW_NE', 'ARROW_E', 'ARROW_SE', 'ARROW_S', 'ARROW_SW', 'ARROW_W', 'ARROW_NW', 'GO_RIGHT', 'GO_LEFT', 'GO_UP', 'GO_DOWN',
      'TRIANGLE', 'TRIANGLE_LEFT', 'CHESSBOARD', 'DIAMOND', 'DIAMOND_SMALL', 'SQUARE', 'SQUARE_SMALL', 'RABBIT', 'COW', 'MUSIC_CROTCHET',
      'MUSIC_QUAVER', 'MUSIC_QUAVERS', 'PITCHFORK', 'XMAS', 'PACMAN', 'TARGET', 'TSHIRT', 'ROLLERSKATE', 'DUCK', 'HOUSE', 'TORTOISE',
      'BUTTERFLY', 'STICKFIGURE', 'GHOST', 'SWORD', 'GIRAFFE', 'SKULL', 'UMBRELLA', 'SNAKE'];
    imageNames.forEach((n, i) => lightMatrix.dict.set('IMAGE_' + n, i + 1));

    // ---------------------------------------------------------------- light
    const lightMod = new PyModule('hub.light', {
      POWER: 0, CONNECT: 1,
      color: native('color', (a) => {
        const which = toInt(a[0]);
        const c = toInt(a[1]);
        if (which === 0) hub().centerColor = COLOR_NAMES[c] || 'off';
        else hub().connectColor = COLOR_NAMES[c] || 'off';
        return None;
      })
    });

    // ---------------------------------------------------------------- sound
    const soundMod = new PyModule('hub.sound', {
      ANY: -2, DEFAULT: -1, WAVEFORM_SINE: 1, WAVEFORM_SQUARE: 2, WAVEFORM_SAWTOOTH: 3, WAVEFORM_TRIANGLE: 4,
      beep: native('beep', (a, k) => {
        const freq = toInt(a.length > 0 ? a[0] : kw(k, 'freq', 440));
        const dur = toInt(a.length > 1 ? a[1] : kw(k, 'duration', 500));
        const vol = toInt(a.length > 2 ? a[2] : kw(k, 'volume', 100));
        const end = now() + dur;
        if (env.onBeep) env.onBeep(freq, dur, vol);
        return new Awaitable(() => now() >= end, () => None, 'beep');
      }),
      stop: native('stop', () => { if (env.onBeep) env.onBeep(0, 0, 0); return None; }),
      volume: native('volume', () => None)
    });

    // ---------------------------------------------------------------- button
    const buttonMod = new PyModule('hub.button', {
      LEFT: 1, RIGHT: 2,
      pressed: native('pressed', (a) => {
        const b = toInt(a[0]);
        const st = hub().buttons;
        const since = b === 1 ? st.left : (b === 2 ? st.right : 0);
        return since ? Math.max(1, Math.round(performanceNow() - since)) : 0;
      })
    });
    const performanceNow = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

    // ---------------------------------------------------------------- motion sensor
    const yawSign = () => (robot().config.hubYawCW ? -1 : 1);
    const motionMod = new PyModule('hub.motion_sensor', {
      TAPPED: 0, DOUBLE_TAPPED: 1, SHAKEN: 2, FALLING: 3, UNKNOWN: -1,
      TOP: 0, FRONT: 1, RIGHT: 2, BOTTOM: 3, BACK: 4, LEFT: 5,
      // Robot mode can make the hub count its yaw clockwise (config.hubYawCW); the simulator's own hub counts counter-clockwise.
      tilt_angles: native('tilt_angles', () => {
        let yaw = geom.wrapDeg(robot().yawCCW(now()));
        return new PyTuple([Math.round(yaw * yawSign() * 10) || 0, 0, 0]);
      }),
      reset_yaw: native('reset_yaw', (a) => {
        const v = a.length ? toNumber(a[0], 'reset_yaw') : 0;
        robot().resetYaw(v * yawSign() / 10, now());
        return None;
      }),
      angular_velocity: native('angular_velocity', () => new PyTuple([0, 0, Math.round((robot().yawRate || 0) * yawSign() * 10) || 0])),
      acceleration: native('acceleration', () => new PyTuple([0, 0, 981])),
      gesture: native('gesture', () => -1),
      up_face: native('up_face', () => 0),
      set_yaw_face: native('set_yaw_face', () => true),
      get_yaw_face: native('get_yaw_face', () => 0),
      stable: native('stable', () => Math.abs(robot().linearSpeed || 0) < 1),
      tap_count: native('tap_count', () => 0),
      reset_tap_count: native('reset_tap_count', () => None),
      quaternion: native('quaternion', () => {
        const t = geom.rad(robot().yawCCW(now())) * yawSign() / 2;
        return new PyTuple([F(Math.cos(t)), F(0), F(0), F(Math.sin(t))]);
      })
    });

    const hubMod = new PyModule('hub', {
      port: portMod,
      light_matrix: lightMatrix,
      light: lightMod,
      sound: soundMod,
      button: buttonMod,
      motion_sensor: motionMod,
      device_uuid: native('device_uuid', () => 'SIMULATED-SPIKE-PRIME-HUB'),
      hardware_id: native('hardware_id', () => 'SIM0000000000001'),
      power_off: native('power_off', () => { throw err('SystemExit'); }),
      temperature: native('temperature', () => 245)
    });

    // ---------------------------------------------------------------- motor
    const motorMod = new PyModule('motor', {
      READY: STATUS.READY, RUNNING: STATUS.RUNNING, STALLED: STATUS.STALLED, CANCELLED: STATUS.CANCELLED,
      ERROR: STATUS.ERROR, DISCONNECTED: STATUS.DISCONNECTED,
      COAST: 0, BRAKE: 1, HOLD: 2, CONTINUE: 3, SMART_COAST: 4, SMART_BRAKE: 5,
      CLOCKWISE: 0, COUNTERCLOCKWISE: 1, SHORTEST_PATH: 2, LONGEST_PATH: 3,
      absolute_position: native('absolute_position', (a) => Math.round(getMotor(a[0]).absolutePosition)),
      relative_position: native('relative_position', (a) => Math.round(getMotor(a[0]).pos)),
      reset_relative_position: native('reset_relative_position', (a) => {
        const m = getMotor(a[0]);
        const v = a.length > 1 ? toNumber(a[1]) : 0;
        const diff = v - m.pos;
        m.pos = v;
        if (m.ctrl.target !== undefined) m.ctrl.target += diff;
        if (m.owner) m.owner.start = m.owner.start.map((s, i) => (m.owner.motors[i] === m ? s + diff : s));
        return None;
      }),
      velocity: native('velocity', (a) => Math.round(getMotor(a[0]).speed)),
      get_duty_cycle: native('get_duty_cycle', (a) => {
        const m = getMotor(a[0]);
        return Math.round(m.speed / m.maxSpeed * 10000);
      }),
      set_duty_cycle: native('set_duty_cycle', (a) => { getMotor(a[0]).setDuty(toInt(a[1])); return None; }),
      run: native('run', (a, k) => {
        const m = getMotor(a[0]);
        m.runAt(velArg(a[1]), toNumber(kw(k, 'acceleration', 1000)));
        return None;
      }),
      stop: native('stop', (a, k) => { getMotor(a[0]).stop(stopMode(kw(k, 'stop', 1))); return None; }),
      run_for_degrees: native('run_for_degrees', (a, k) => {
        if (a.length < 3) throw err('TypeError', 'run_for_degrees(port, degrees, velocity) needs 3 arguments');
        const m = getMotor(a[0]);
        const deg = toInt(a[1], 'degrees');
        const vel = velArg(a[2]);
        const dir = Math.sign(deg) * (vel < 0 ? -1 : 1);
        const cmd = m.runToPosition(m.pos + Math.abs(deg) * dir, Math.abs(vel), {
          end: stopMode(kw(k, 'stop', 1)), accel: toNumber(kw(k, 'acceleration', 1000)), decel: toNumber(kw(k, 'deceleration', 1000))
        });
        return cmdAwaitable(cmd, 'motor.run_for_degrees');
      }),
      run_for_time: native('run_for_time', (a, k) => {
        if (a.length < 3) throw err('TypeError', 'run_for_time(port, duration, velocity) needs 3 arguments');
        const m = getMotor(a[0]);
        const cmd = m.runForTime(toInt(a[1], 'duration'), velArg(a[2]), {
          end: stopMode(kw(k, 'stop', 1)), accel: toNumber(kw(k, 'acceleration', 1000)), decel: toNumber(kw(k, 'deceleration', 1000))
        }, now());
        return cmdAwaitable(cmd, 'motor.run_for_time');
      }),
      run_to_relative_position: native('run_to_relative_position', (a, k) => {
        const m = getMotor(a[0]);
        const cmd = m.runToPosition(toInt(a[1], 'position'), Math.abs(velArg(a[2])), {
          end: stopMode(kw(k, 'stop', 1)), accel: toNumber(kw(k, 'acceleration', 1000)), decel: toNumber(kw(k, 'deceleration', 1000))
        });
        return cmdAwaitable(cmd, 'motor.run_to_relative_position');
      }),
      run_to_absolute_position: native('run_to_absolute_position', (a, k) => {
        const m = getMotor(a[0]);
        const target = toInt(a[1], 'position');
        const direction = toInt(kw(k, 'direction', 2));
        const cur = m.absolutePosition;
        let diff = geom.wrapDeg(target - cur);
        if (direction === 0) diff = ((target - cur) % 360 + 360) % 360;
        else if (direction === 1) diff = -(((cur - target) % 360 + 360) % 360);
        else if (direction === 3) diff = diff >= 0 ? diff - 360 : diff + 360;
        const cmd = m.runToPosition(m.pos + diff, Math.abs(velArg(a[2])), {
          end: stopMode(kw(k, 'stop', 1)), accel: toNumber(kw(k, 'acceleration', 1000)), decel: toNumber(kw(k, 'deceleration', 1000))
        });
        return cmdAwaitable(cmd, 'motor.run_to_absolute_position');
      })
    });

    // ---------------------------------------------------------------- motor_pair
    const getPair = (p) => {
      const i = toInt(p, 'pair');
      if (i < 0 || i > 2) throw err('ValueError', 'invalid motor pair ' + i);
      const pr = pairs[i];
      if (!pr) throw err('OSError', 'motor pair ' + (i + 1) + ' is not paired yet. Call motor_pair.pair(motor_pair.PAIR_' + (i + 1) + ', port.C, port.D) first.');
      return pr;
    };
    const steerSplit = (steering, v) => {
      const s = geom.clamp(steering, -100, 100);
      return s >= 0 ? [v, v * (1 - s / 50)] : [v * (1 + s / 50), v];
    };
    // Pairs drive the first (left) motor mirrored so positive velocity means forward.
    const pairRates = (vl, vr) => {
      const vmax = Math.max(Math.abs(vl), Math.abs(vr)) || 1;
      return [-vl / vmax, vr / vmax];
    };
    const syncMove = (pr, vl, vr, opts) => {
      const rates = pairRates(vl, vr);
      const s = new FLL.robot.SyncMove([pr.left, pr.right], rates, Object.assign({ velocity: Math.max(Math.abs(vl), Math.abs(vr)) }, opts));
      robot().addSync(s);
      return s;
    };
    const runPairContinuous = (pr, vl, vr, accel) => {
      const vmax = Math.max(Math.abs(vl), Math.abs(vr)) || 1;
      pr.left.runAt(-vl, accel * Math.abs(vl) / vmax || accel);
      pr.right.runAt(vr, accel * Math.abs(vr) / vmax || accel);
    };
    const motorPairMod = new PyModule('motor_pair', {
      PAIR_1: 0, PAIR_2: 1, PAIR_3: 2,
      pair: native('pair', (a) => {
        if (a.length !== 3) throw err('TypeError', 'pair(pair, left_motor, right_motor) needs 3 arguments');
        const i = toInt(a[0], 'pair');
        const left = getMotor(a[1]), right = getMotor(a[2]);
        if (left === right) throw err('ValueError', 'a motor pair needs two different motors');
        pairs[i] = { left: left, right: right };
        return None;
      }),
      unpair: native('unpair', (a) => { pairs[toInt(a[0])] = null; return None; }),
      move: native('move', (a, k) => {
        const pr = getPair(a[0]);
        const steering = toNumber(a.length > 1 ? a[1] : kw(k, 'steering', 0));
        const v = velArg(kw(k, 'velocity', 360));
        const [vl, vr] = steerSplit(steering, v);
        runPairContinuous(pr, vl, vr, toNumber(kw(k, 'acceleration', 1000)));
        return None;
      }),
      move_tank: native('move_tank', (a, k) => {
        const pr = getPair(a[0]);
        runPairContinuous(pr, velArg(a[1], 'left_velocity'), velArg(a[2], 'right_velocity'), toNumber(kw(k, 'acceleration', 1000)));
        return None;
      }),
      move_for_degrees: native('move_for_degrees', (a, k) => {
        if (a.length < 3) throw err('TypeError', 'move_for_degrees(pair, degrees, steering, velocity=...) needs at least 3 arguments');
        const pr = getPair(a[0]);
        const deg = toInt(a[1], 'degrees');
        const v = velArg(kw(k, 'velocity', 360)) * (deg < 0 ? -1 : 1);
        const [vl, vr] = steerSplit(toNumber(a[2], 'steering'), v);
        const s = syncMove(pr, vl, vr, { distance: Math.abs(deg), end: stopMode(kw(k, 'stop', 1)), accel: toNumber(kw(k, 'acceleration', 1000)), decel: toNumber(kw(k, 'deceleration', 1000)) });
        return cmdAwaitable(s.cmd, 'motor_pair.move_for_degrees');
      }),
      move_for_time: native('move_for_time', (a, k) => {
        if (a.length < 3) throw err('TypeError', 'move_for_time(pair, duration, steering, velocity=...) needs at least 3 arguments');
        const pr = getPair(a[0]);
        const dur = toInt(a[1], 'duration');
        const v = velArg(kw(k, 'velocity', 360));
        const [vl, vr] = steerSplit(toNumber(a[2], 'steering'), v);
        const s = syncMove(pr, vl, vr, { continuous: true, until: now() + dur, end: stopMode(kw(k, 'stop', 1)), accel: toNumber(kw(k, 'acceleration', 1000)), decel: toNumber(kw(k, 'deceleration', 1000)) });
        return cmdAwaitable(s.cmd, 'motor_pair.move_for_time');
      }),
      move_tank_for_degrees: native('move_tank_for_degrees', (a, k) => {
        if (a.length < 4) throw err('TypeError', 'move_tank_for_degrees(pair, degrees, left_velocity, right_velocity) needs 4 arguments');
        const pr = getPair(a[0]);
        const deg = toInt(a[1], 'degrees');
        const sg = deg < 0 ? -1 : 1;
        const s = syncMove(pr, velArg(a[2]) * sg, velArg(a[3]) * sg, { distance: Math.abs(deg), end: stopMode(kw(k, 'stop', 1)), accel: toNumber(kw(k, 'acceleration', 1000)), decel: toNumber(kw(k, 'deceleration', 1000)) });
        return cmdAwaitable(s.cmd, 'motor_pair.move_tank_for_degrees');
      }),
      move_tank_for_time: native('move_tank_for_time', (a, k) => {
        if (a.length < 4) throw err('TypeError', 'move_tank_for_time(pair, left_velocity, right_velocity, duration) needs 4 arguments');
        const pr = getPair(a[0]);
        const s = syncMove(pr, velArg(a[1]), velArg(a[2]), { continuous: true, until: now() + toInt(a[3], 'duration'), end: stopMode(kw(k, 'stop', 1)), accel: toNumber(kw(k, 'acceleration', 1000)), decel: toNumber(kw(k, 'deceleration', 1000)) });
        return cmdAwaitable(s.cmd, 'motor_pair.move_tank_for_time');
      }),
      stop: native('stop', (a, k) => {
        const pr = getPair(a[0]);
        const mode = stopMode(kw(k, 'stop', 1));
        pr.left.stop(mode);
        pr.right.stop(mode);
        return None;
      })
    });

    // ---------------------------------------------------------------- sensors
    const noise = (amp) => (amp ? (robot().rng() - 0.5) * 2 * amp : 0);
    const readRGB = (P) => {
      const p = robot().sensorWorld(P);
      return world.colorUnder(p[0], p[1]);
    };
    const colorSensorMod = new PyModule('color_sensor', {
      color: native('color', (a) => classifyColor(readRGB(needSensor(a[0], 'color')))),
      reflection: native('reflection', (a) => {
        const P = needSensor(a[0], 'color');
        return geom.clamp(Math.round(reflectionOf(readRGB(P)) + noise(robot().realism.sensorNoise)), 0, 100);
      }),
      rgbi: native('rgbi', (a) => {
        const rgb = readRGB(needSensor(a[0], 'color'));
        const k = 1024 / 255;
        const i = Math.round((rgb[0] + rgb[1] + rgb[2]) / 3 * k);
        return new PyTuple([Math.round(rgb[0] * k), Math.round(rgb[1] * k), Math.round(rgb[2] * k), i]);
      })
    });
    const colorMod = new PyModule('color', Object.assign({}, COLOR));

    const distanceMod = new PyModule('distance_sensor', {
      distance: native('distance', (a) => {
        const P = needSensor(a[0], 'distance');
        const rb = robot();
        const d = rb.device(P);
        const o = rb.sensorWorld(P);
        const dist = world.rayDistance(o, rb.pose.theta + geom.rad(d.angle || 0), 2100);
        if (dist < 40 || dist > 2000) return -1;
        return Math.round(dist + noise(robot().realism.sensorNoise * 2));
      }),
      show: native('show', () => None),
      clear: native('clear', () => None),
      get_pixel: native('get_pixel', () => 0),
      set_pixel: native('set_pixel', () => None)
    });

    const forceMod = new PyModule('force_sensor', {
      force: native('force', (a) => (world.bumperPressed(needSensor(a[0], 'force')) ? 60 : 0)),
      pressed: native('pressed', (a) => world.bumperPressed(needSensor(a[0], 'force'))),
      raw: native('raw', (a) => (world.bumperPressed(needSensor(a[0], 'force')) ? 750 : 380))
    });

    const deviceMod = new PyModule('device', {
      id: native('id', (a) => {
        const d = robot().device(portName(a[0]));
        if (d.type === 'motor') return ({ medium: 48, large: 49, small: 65 })[d.kind || 'medium'];
        return ({ color: 61, distance: 62, force: 63 })[d.type] || 0;
      }),
      ready: native('ready', () => true),
      data: native('data', (a) => new PyTuple([])),
      get_duty_cycle: native('get_duty_cycle', () => 0),
      set_duty_cycle: native('set_duty_cycle', () => None)
    });

    // ---------------------------------------------------------------- runloop
    const runloopMod = new PyModule('runloop', {
      run: native('run', (a, k, ctx) => {
        if (!a.length) throw err('TypeError', 'runloop.run() needs at least one coroutine, e.g. runloop.run(main())');
        const tasks = a.map((c, i) => {
          if (!(c instanceof py.PyCoroutine)) {
            if (c instanceof py.PyFunction) throw err('TypeError', "runloop.run() needs a coroutine: call the function, e.g. runloop.run(" + c.name + "())");
            throw err('TypeError', 'runloop.run() needs coroutines (async functions), got ' + py.typeName(c));
          }
          return ctx.spawn(ctx.runCoroutine(c), c.func.name);
        });
        return new Suspend(() => tasks.every(t => t.done), null, 'runloop.run');
      }),
      sleep_ms: native('sleep_ms', (a) => {
        const end = now() + toNumber(a[0], 'sleep_ms');
        return new Awaitable(() => now() >= end, () => None, 'runloop.sleep_ms');
      }),
      until: native('until', (a, k, ctx) => {
        const fn = a[0];
        const timeout = toNumber(a.length > 1 ? a[1] : kw(k, 'timeout', 0));
        const end = timeout > 0 ? now() + timeout : Infinity;
        let ok = false;
        const check = () => {
          if (ok) return true;
          if (py.truthy(ctx.call(fn, []))) ok = true;
          else if (now() >= end) ok = true;
          return ok;
        };
        return new Awaitable(check, () => None, 'runloop.until');
      })
    });

    // ---------------------------------------------------------------- app (SPIKE App extras; mostly no-ops)
    const appSound = new PyModule('app.sound', {
      play: native('play', (a) => {
        const end = now() + 600;
        if (env.onBeep) env.onBeep(660, 150, 60);
        return new Awaitable(() => now() >= end, () => None, 'app.sound.play');
      }),
      stop: native('stop', () => None),
      set_attributes: native('set_attributes', () => None)
    });
    const appDisplay = new PyModule('app.display', {
      text: native('text', (a) => { if (env.print) env.print('[display] ' + py.str(a[0]) + '\n'); return None; }),
      image: native('image', () => None), show: native('show', () => None), hide: native('hide', () => None)
    });
    const noop = native('noop', () => None);
    const appMod = new PyModule('app', {
      sound: appSound,
      display: appDisplay,
      bargraph: new PyModule('app.bargraph', { change: noop, clear_all: noop, get_value: native('get_value', () => 0), set_value: noop, show: noop, hide: noop }),
      linegraph: new PyModule('app.linegraph', { clear: noop, clear_all: noop, get_average: native('get_average', () => 0), plot: noop, show: noop, hide: noop }),
      music: new PyModule('app.music', { play_drum: noop, play_instrument: noop })
    });

    return {
      hub: hubMod,
      motor: motorMod,
      motor_pair: motorPairMod,
      color_sensor: colorSensorMod,
      color: colorMod,
      distance_sensor: distanceMod,
      force_sensor: forceMod,
      device: deviceMod,
      runloop: runloopMod,
      app: appMod
    };
  }

  FLL.api = FLL.api || {};
  FLL.api.spike = { createModules, classifyColor, reflectionOf, rgbToHsv, COLOR, COLOR_NAMES };
  FLL.api.spikeFont = FONT;
})(typeof globalThis !== 'undefined' ? globalThis : window);
