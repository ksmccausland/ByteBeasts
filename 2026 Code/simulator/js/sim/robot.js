/*
 * Simulated SPIKE Prime robot: motors, synchronized moves, kinematics and sensors.
 * Units: millimetres, degrees, seconds (dt), milliseconds for program time.
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};
  const geom = FLL.geom;

  const PORTS = ['A', 'B', 'C', 'D', 'E', 'F'];
  const MOTOR_SPECS = {
    medium: { maxSpeed: 1110, label: 'Medium motor' },
    large: { maxSpeed: 1050, label: 'Large motor' },
    small: { maxSpeed: 660, label: 'Small motor' }
  };
  // SPIKE 3 command result codes
  const STATUS = { READY: 0, RUNNING: 1, STALLED: 2, CANCELLED: 3, ERROR: 4, DISCONNECTED: 5 };

  const moveToward = (v, target, maxStep) => {
    if (v < target) return Math.min(target, v + maxStep);
    if (v > target) return Math.max(target, v - maxStep);
    return v;
  };

  class MotorCommand {
    constructor(kind) {
      this.kind = kind;
      this.done = false;
      this.status = STATUS.RUNNING;
    }
    finish(status) {
      if (this.done) return;
      this.done = true;
      this.status = status;
    }
  }

  class SimMotor {
    constructor(port, kind, opts) {
      opts = opts || {};
      this.port = port;
      this.kind = kind || 'medium';
      this.maxSpeed = MOTOR_SPECS[this.kind].maxSpeed;
      this.pos = 0;              // relative position (degrees)
      this.speed = 0;            // degrees per second (measured)
      this.absOffset = opts.absOffset || 0;
      this.ctrl = { mode: 'coast' };
      this.cmd = null;
      this.owner = null;         // synced controller owning this motor
      this.desired = 0;          // desired delta for this tick (degrees)
      this.nextSpeed = 0;
      this.stallTime = 0;
      this.limits = opts.limits || null; // [min, max] mechanical limits (attachments)
      this.noise = 0;            // stop-position noise (degrees)
      this.rng = opts.rng || Math.random;
      this.duty = 0;
    }

    get absolutePosition() {
      let a = (this.pos + this.absOffset) % 360;
      if (a < -180) a += 360;
      if (a >= 180) a -= 360;
      return a;
    }

    // Cancel whatever the motor is doing (a new command replaces the old one).
    takeOver() {
      if (this.cmd && !this.cmd.done) this.cmd.finish(STATUS.CANCELLED);
      this.cmd = null;
      if (this.owner) {
        const o = this.owner;
        o.cancel();
      }
      this.stallTime = 0;
    }

    runAt(speed, accel) {
      this.takeOver();
      this.ctrl = { mode: 'speed', v: geom.clamp(speed, -this.maxSpeed, this.maxSpeed), accel: accel || 20000 };
      const c = new MotorCommand('run');
      c.continuous = true;
      this.cmd = c;
      return c;
    }

    runToPosition(target, speed, opts) {
      opts = opts || {};
      this.takeOver();
      const c = new MotorCommand('position');
      this.cmd = c;
      this.ctrl = {
        mode: 'position', target: target, vmax: Math.min(Math.abs(speed), this.maxSpeed),
        accel: opts.accel || 1000, decel: opts.decel || 1000, end: opts.end || 'brake'
      };
      if (Math.abs(target - this.pos) < 0.5 || Math.abs(speed) < 1) {
        this.finishPosition();
      }
      return c;
    }

    runForTime(ms, speed, opts, now) {
      opts = opts || {};
      this.takeOver();
      const c = new MotorCommand('time');
      this.cmd = c;
      this.ctrl = {
        mode: 'time', until: now + ms, v: geom.clamp(speed, -this.maxSpeed, this.maxSpeed),
        accel: opts.accel || 1000, decel: opts.decel || 1000, end: opts.end || 'brake'
      };
      if (ms <= 0) { this.applyEnd(this.ctrl.end); c.finish(STATUS.READY); }
      return c;
    }

    setDuty(duty) {
      this.takeOver();
      this.ctrl = { mode: 'pwm', duty: geom.clamp(duty, -10000, 10000) / 100 };
    }

    stop(endMode) {
      this.takeOver();
      this.applyEnd(endMode || 'brake');
    }

    applyEnd(mode) {
      if (mode === 'hold') this.ctrl = { mode: 'hold', target: this.pos };
      else if (mode === 'coast' || mode === 'continue') this.ctrl = { mode: 'coast' };
      else this.ctrl = { mode: 'brake', target: this.pos };
    }

    finishPosition() {
      const c = this.ctrl;
      if (this.noise) this.pos = c.target + (this.rng() - 0.5) * 2 * this.noise;
      else this.pos = c.target;
      this.speed = 0;
      this.nextSpeed = 0;
      this.applyEnd(c.end);
      if (this.cmd) this.cmd.finish(STATUS.READY);
    }

    // Phase 1: compute the desired movement for this tick.
    plan(dt, now) {
      if (this.owner) return; // a synced controller sets this.desired
      const c = this.ctrl;
      const v0 = this.speed;
      let v1 = v0;
      let delta;
      switch (c.mode) {
        case 'speed':
          v1 = moveToward(v0, c.v, c.accel * dt);
          break;
        case 'time':
          if (now >= c.until) {
            this.applyEnd(c.end);
            if (this.cmd) this.cmd.finish(STATUS.READY);
            return this.plan(dt, now);
          }
          v1 = moveToward(v0, c.v, (Math.abs(c.v) >= Math.abs(v0) ? c.accel : c.decel) * dt);
          break;
        case 'position': {
          const rem = c.target - this.pos;
          const dir = rem >= 0 ? 1 : -1;
          const vStop = Math.sqrt(2 * c.decel * Math.abs(rem));
          const vDes = dir * Math.min(c.vmax, vStop);
          const speedingUp = Math.abs(vDes) > Math.abs(v0) && (v0 === 0 || Math.sign(v0) === Math.sign(vDes));
          v1 = moveToward(v0, vDes, (speedingUp ? c.accel : Math.max(c.decel, 2000)) * dt);
          delta = (v0 + v1) / 2 * dt;
          if (dir > 0 ? delta >= rem - 1e-6 : delta <= rem + 1e-6) {
            delta = rem;
            v1 = 0;
            c.arriving = true;
          }
          break;
        }
        case 'hold': {
          const vDes = geom.clamp((c.target - this.pos) * 25, -this.maxSpeed, this.maxSpeed);
          v1 = moveToward(v0, vDes, 30000 * dt);
          break;
        }
        case 'brake':
          v1 = moveToward(v0, 0, 15000 * dt);
          break;
        case 'pwm':
          v1 = moveToward(v0, c.duty / 100 * this.maxSpeed * 0.92, 8000 * dt);
          break;
        default: // coast
          v1 = moveToward(v0, 0, 2500 * dt);
      }
      if (delta === undefined) delta = (v0 + v1) / 2 * dt;
      this.desired = delta;
      this.nextSpeed = v1;
    }

    // Phase 2: apply what actually happened (the robot may have been blocked).
    commit(actual, dt) {
      if (this.limits) {
        const np = this.pos + actual;
        if (np < this.limits[0]) actual = this.limits[0] - this.pos;
        else if (np > this.limits[1]) actual = this.limits[1] - this.pos;
      }
      const desired = this.desired;
      this.pos += actual;
      const blocked = Math.abs(desired) > 0.05 && Math.abs(actual) < Math.abs(desired) * 0.5;
      if (blocked) {
        this.speed = actual / dt;
        this.stallTime += dt;
      } else {
        this.speed = (this.owner || this.ctrl.mode === 'position' && this.ctrl.arriving) ? actual / dt : this.nextSpeed;
        this.stallTime = 0;
      }
      if (this.owner) return;
      const c = this.ctrl;
      if (c.mode === 'position' && c.arriving && !blocked) {
        c.arriving = false;
        this.finishPosition();
        return;
      }
      if (c.mode === 'position') c.arriving = false;
      if (this.stallTime > 0.5 && this.cmd && !this.cmd.done && !this.cmd.continuous) {
        // SPIKE stall detection: the command ends when the motor cannot move.
        this.cmd.finish(STATUS.STALLED);
        this.applyEnd('brake');
      }
    }
  }

  // Moves two motors along a shared speed profile (motor pairs and drive bases).
  // rates: degrees of each motor per degree of master progress (|rate| <= 1).
  class SyncMove {
    constructor(motors, rates, opts) {
      this.motors = motors;
      this.rates = rates;
      this.cmd = new MotorCommand('sync');
      this.continuous = !!opts.continuous;
      this.distance = Math.abs(opts.distance || 0);
      this.vmax = Math.abs(opts.velocity || 0);
      this.accel = opts.accel || 1000;
      this.decel = opts.decel || 1000;
      this.end = opts.end || 'brake';
      this.until = opts.until !== undefined ? opts.until : null; // time-limited moves
      this.p = 0;
      this.v = 0;
      this.start = motors.map(m => m.pos);
      this.lagTime = 0;
      this.corrector = opts.corrector || null; // optional heading correction (gyro drive)
      this.cancelled = false;
      for (const m of motors) {
        m.takeOver();
        m.owner = this;
        m.cmd = null;
      }
      this.cmd.continuous = this.continuous && this.until === null;
      if (!this.continuous && this.distance < 0.5) this.finish(STATUS.READY);
    }
    cancel() {
      if (this.cancelled) return;
      this.cancelled = true;
      for (const m of this.motors) if (m.owner === this) { m.owner = null; m.ctrl = { mode: 'brake', target: m.pos }; }
      this.cmd.finish(STATUS.CANCELLED);
    }
    finish(status) {
      if (this.cancelled) return;
      this.cancelled = true;
      for (const m of this.motors) {
        if (m.owner === this) {
          m.owner = null;
          m.applyEnd(this.end);
          if (this.end !== 'coast') m.speed = 0;
        }
      }
      this.cmd.finish(status);
    }
    plan(dt, now) {
      if (this.cancelled) return;
      let vDes = this.vmax;
      if (this.until !== null && now >= this.until) {
        this.finish(STATUS.READY);
        return;
      }
      if (!this.continuous) {
        const rem = this.distance - this.p;
        vDes = Math.min(this.vmax, Math.sqrt(2 * this.decel * Math.max(0, rem)));
      }
      const v0 = this.v;
      const v1 = moveToward(v0, vDes, (vDes > v0 ? this.accel : Math.max(this.decel, 2000)) * dt);
      let dp = (v0 + v1) / 2 * dt;
      if (!this.continuous && this.p + dp >= this.distance - 1e-6) dp = this.distance - this.p;
      this.p += dp;
      this.v = v1;
      let corr = [0, 0];
      if (this.corrector) corr = this.corrector(this, dt);
      this.lastCorr = corr;
      for (let i = 0; i < this.motors.length; i++) {
        const m = this.motors[i];
        const setpoint = this.start[i] + this.rates[i] * this.p + (corr[i] || 0);
        const maxStep = m.maxSpeed * dt;
        m.desired = geom.clamp(setpoint - m.pos, -maxStep, maxStep);
        m.nextSpeed = m.desired / dt;
      }
    }
    afterCommit(dt) {
      if (this.cancelled) return;
      if (this.continuous) return;
      if (this.p >= this.distance - 1e-6) {
        let maxLag = 0;
        for (let i = 0; i < this.motors.length; i++) {
          const target = this.start[i] + this.rates[i] * this.distance + (this.lastCorr ? this.lastCorr[i] || 0 : 0);
          maxLag = Math.max(maxLag, Math.abs(target - this.motors[i].pos));
        }
        if (maxLag < 1) { this.finish(STATUS.READY); return; }
      }
      const stalled = this.motors.some(m => m.stallTime > 0.5);
      if (stalled) this.finish(STATUS.STALLED);
    }
  }

  // ------------------------------------------------------------------ robot
  const DEFAULT_CONFIG = {
    name: 'SPIKE Prime driving base',
    body: { front: 112, rear: 68, width: 156 },   // mm from the wheel axle centre
    wheelDiameter: 56,
    trackWidth: 112,
    driveLeft: 'C',
    driveRight: 'D',
    leftReversed: true,   // mirrored mounting: positive left-motor degrees drive the wheel backwards
    devices: {
      A: { type: 'none' },
      B: { type: 'color', x: 92, y: 0 },
      C: { type: 'motor', kind: 'medium' },
      D: { type: 'motor', kind: 'medium' },
      E: { type: 'motor', kind: 'medium', mount: 'front', reach: 90, armLength: 100 },
      F: { type: 'motor', kind: 'medium', mount: 'front', reach: 60, armLength: 70 }
    },
    realism: 'realistic'   // 'ideal' | 'realistic' | 'challenging'
  };

  const REALISM = {
    ideal: { wheelBias: 0, wheelJitter: 0, stopNoise: 0, gyroDrift: 0, gyroNoise: 0, sensorNoise: 0 },
    realistic: { wheelBias: 0.004, wheelJitter: 0.01, stopNoise: 0.8, gyroDrift: 0.15, gyroNoise: 0.15, sensorNoise: 1.5 },
    challenging: { wheelBias: 0.012, wheelJitter: 0.03, stopNoise: 2, gyroDrift: 0.6, gyroNoise: 0.4, sensorNoise: 3 }
  };

  class Robot {
    constructor(config, seed) {
      this.config = JSON.parse(JSON.stringify(config || DEFAULT_CONFIG));
      this.rng = geom.rng(seed || 12345);
      this.pose = { x: 250, y: 120, theta: Math.PI / 2 };
      this.motors = {};
      this.syncs = [];
      this.trail = [];
      this.contacts = [];
      this.cargo = [];
      this.toolEvents = [];
      this.armState = {};
      this.lastPoseForTrail = null;
      this.hub = {
        matrix: new Array(25).fill(0),
        centerColor: 'white',
        buttons: { left: 0, right: 0, center: 0 },
        beep: null,
        orientation: 0
      };
      this.imu = { yawOffset: 0, drift: 0, startTime: 0 };
      this.setup();
    }

    setup() {
      const cfg = this.config;
      this.realism = REALISM[cfg.realism] || REALISM.realistic;
      this.motors = {};
      for (const p of PORTS) {
        const d = cfg.devices[p];
        if (d && d.type === 'motor') {
          this.motors[p] = new SimMotor(p, d.kind || 'medium', { rng: this.rng, absOffset: Math.round(this.rng() * 359) - 180 });
          this.motors[p].noise = this.realism.stopNoise;
          if (d.limits) this.motors[p].limits = d.limits;
          this.armState[p] = { acc: 0, signed: 0, moving: false, quiet: 0, emitted: 0 };
        }
      }
      const r = this.realism;
      this.wheelScale = {
        left: 1 + (this.rng() - 0.5) * 2 * r.wheelBias,
        right: 1 + (this.rng() - 0.5) * 2 * r.wheelBias
      };
      this.imu.drift = (this.rng() - 0.5) * 2 * r.gyroDrift; // degrees per minute
    }

    device(port) { return this.config.devices[port] || { type: 'none' }; }
    isDrivePort(p) { return p === this.config.driveLeft || p === this.config.driveRight; }

    get mmPerDeg() { return Math.PI * this.config.wheelDiameter / 360; }

    // Robot outline (world coordinates).
    bodyPoly(pose) {
      pose = pose || this.pose;
      const b = this.config.body;
      const hw = b.width / 2;
      return [[b.front, hw], [-b.rear, hw], [-b.rear, -hw], [b.front, -hw]].map(([f, l]) => geom.localToWorld(pose, f, l));
    }

    // Area an attachment on `port` can reach (used for mission interactions).
    toolZone(port, pose) {
      pose = pose || this.pose;
      const d = this.device(port);
      const b = this.config.body;
      const reach = d.reach || 80;
      const w = Math.min(b.width, d.toolWidth || b.width) / 2;
      let pts;
      switch (d.mount) {
        case 'back': pts = [[-b.rear, w], [-b.rear - reach, w], [-b.rear - reach, -w], [-b.rear, -w]]; break;
        case 'left': pts = [[b.front * 0.6, b.width / 2], [b.front * 0.6, b.width / 2 + reach], [-b.rear * 0.6, b.width / 2 + reach], [-b.rear * 0.6, b.width / 2]]; break;
        case 'right': pts = [[b.front * 0.6, -b.width / 2], [-b.rear * 0.6, -b.width / 2], [-b.rear * 0.6, -b.width / 2 - reach], [b.front * 0.6, -b.width / 2 - reach]]; break;
        default: pts = [[b.front - 10, w], [b.front + reach, w], [b.front + reach, -w], [b.front - 10, -w]];
      }
      return pts.map(([f, l]) => geom.localToWorld(pose, f, l));
    }

    attachmentPorts() {
      return PORTS.filter(p => this.motors[p] && !this.isDrivePort(p));
    }

    sensorWorld(port, pose) {
      const d = this.device(port);
      return geom.localToWorld(pose || this.pose, d.x || 0, d.y || 0);
    }

    // True heading change in degrees (counter-clockwise positive), unwrapped.
    yawCCW(now) {
      const r = this.realism;
      const drift = this.imu.drift * ((now - this.imu.startTime) / 60000);
      const noise = r.gyroNoise ? (this.rng() - 0.5) * 2 * r.gyroNoise : 0;
      return geom.deg(this.pose.theta) - this.imu.yawOffset + drift + noise;
    }
    resetYaw(ccwValue, now) {
      // After reset, yawCCW() returns ccwValue.
      this.imu.yawOffset = geom.deg(this.pose.theta) - ccwValue;
      this.imu.startTime = now;
    }

    stopAll(mode) {
      for (const s of this.syncs.slice()) s.cancel();
      this.syncs = [];
      for (const p in this.motors) {
        const m = this.motors[p];
        m.owner = null;
        if (m.cmd && !m.cmd.done) m.cmd.finish(STATUS.CANCELLED);
        m.cmd = null;
        m.applyEnd(mode || 'coast');
      }
    }

    addSync(sync) {
      this.syncs = this.syncs.filter(s => !s.cancelled);
      this.syncs.push(sync);
      return sync;
    }

    // Integrate a differential drive motion (wheel travel in mm) from a pose.
    integrate(pose, sL, sR) {
      const track = this.config.trackWidth;
      const ds = (sL + sR) / 2;
      const dth = (sR - sL) / track;
      const mid = pose.theta + dth / 2;
      return { x: pose.x + ds * Math.cos(mid), y: pose.y + ds * Math.sin(mid), theta: pose.theta + dth };
    }

    // One physics step; world provides collision checks.
    step(dt, now, world) {
      for (const s of this.syncs) if (!s.cancelled) s.plan(dt, now);
      for (const p in this.motors) this.motors[p].plan(dt, now);

      const cfg = this.config;
      const L = this.motors[cfg.driveLeft], R = this.motors[cfg.driveRight];
      const dL = L ? L.desired : 0, dR = R ? R.desired : 0;
      const k = this.mmPerDeg;
      const r = this.realism;
      const jitter = () => (r.wheelJitter ? 1 + (this.rng() - 0.5) * 2 * r.wheelJitter : 1);
      const sgnL = cfg.leftReversed ? -1 : 1;
      const sgnR = cfg.rightReversed ? -1 : 1;
      const sL = dL * k * sgnL * this.wheelScale.left * jitter();
      const sR = dR * k * sgnR * this.wheelScale.right * jitter();
      let usedL = 1, usedR = 1;
      this.contacts = [];
      if (Math.abs(sL) > 1e-9 || Math.abs(sR) > 1e-9) {
        const res = this.resolveMotion(sL, sR, world);
        usedL = res.fl;
        usedR = res.fr;
        this.pose = res.pose;
        this.contacts = res.contacts;
      } else if (world) {
        this.contacts = world.contactsFor(this.bodyPoly(), 2);
      }
      if (L) L.commit(dL * usedL, dt);
      if (R) R.commit(dR * usedR, dt);
      for (const p in this.motors) {
        if (p === cfg.driveLeft || p === cfg.driveRight) continue;
        const m = this.motors[p];
        m.commit(m.desired, dt);
        this.trackArm(p, m, dt, now);
      }
      for (const s of this.syncs) s.afterCommit(dt);
      this.syncs = this.syncs.filter(s => !s.cancelled);

      // trail
      const lp = this.lastPoseForTrail;
      if (!lp || Math.hypot(lp.x - this.pose.x, lp.y - this.pose.y) > 6) {
        this.trail.push([this.pose.x, this.pose.y]);
        if (this.trail.length > 4000) this.trail.splice(0, 1000);
        this.lastPoseForTrail = { x: this.pose.x, y: this.pose.y };
      }
    }

    // Detect deliberate attachment movements (>= 45 degrees) for mission interactions.
    trackArm(port, m, dt, now) {
      const st = this.armState[port];
      const moving = Math.abs(m.speed) > 20;
      const delta = m.speed * dt;
      if (moving) {
        if (!st.moving) { st.moving = true; st.acc = 0; st.signed = 0; st.emitted = 0; st.startPose = Object.assign({}, this.pose); }
        st.acc += Math.abs(delta);
        st.signed += delta;
        st.quiet = 0;
        const threshold = st.emitted === 0 ? 45 : 45 + st.emitted * 120;
        if (st.acc >= threshold) {
          st.emitted++;
          this.toolEvents.push({ port: port, dir: Math.sign(st.signed), angle: st.signed, time: now, zone: this.toolZone(port) });
        }
      } else if (st.moving) {
        st.quiet += dt;
        if (st.quiet > 0.08) st.moving = false;
      }
    }

    resolveMotion(sL, sR, world) {
      const start = this.pose;
      const full = this.integrate(start, sL, sR);
      if (!world) return { pose: full, fl: 1, fr: 1, contacts: [] };
      const already = world.collidingSet(this.bodyPoly(start));
      const free = (pose) => !world.collides(this.bodyPoly(pose), already, this.bodyPoly(start));
      if (free(full)) return { pose: full, fl: 1, fr: 1, contacts: world.contactsFor(this.bodyPoly(full), 1.5) };
      const maxFrac = (fn) => {
        let lo = 0, hi = 1;
        for (let i = 0; i < 9; i++) {
          const mid = (lo + hi) / 2;
          if (free(fn(mid))) lo = mid; else hi = mid;
        }
        return lo;
      };
      // Move both wheels as far as possible together, then let each wheel continue on its own
      // (this lets the robot pivot against a wall, which is how wall squaring works).
      const f = maxFrac(t => this.integrate(start, sL * t, sR * t));
      const mid = this.integrate(start, sL * f, sR * f);
      const remL = sL * (1 - f), remR = sR * (1 - f);
      const fl2 = Math.abs(remL) > 1e-6 ? maxFracFrom(mid, (t) => this.integrate(mid, remL * t, 0)) : 0;
      const fr2 = Math.abs(remR) > 1e-6 ? maxFracFrom(mid, (t) => this.integrate(mid, 0, remR * t)) : 0;
      function maxFracFrom(base, fn) {
        let lo = 0, hi = 1;
        for (let i = 0; i < 8; i++) {
          const m2 = (lo + hi) / 2;
          if (free(fn(m2))) lo = m2; else hi = m2;
        }
        return lo;
      }
      let pose = mid, fl = f, fr = f;
      const gainL = Math.abs(remL) * fl2, gainR = Math.abs(remR) * fr2;
      if (gainL > 0.05 || gainR > 0.05) {
        if (gainL >= gainR) { pose = this.integrate(mid, remL * fl2, 0); fl = f + (1 - f) * fl2; }
        else { pose = this.integrate(mid, 0, remR * fr2); fr = f + (1 - f) * fr2; }
      }
      return { pose: pose, fl: fl, fr: fr, contacts: world.contactsFor(this.bodyPoly(pose), 1.5) };
    }
  }

  FLL.robot = { PORTS, MOTOR_SPECS, STATUS, SimMotor, SyncMove, MotorCommand, Robot, DEFAULT_CONFIG, REALISM, moveToward };
})(typeof globalThis !== 'undefined' ? globalThis : window);
