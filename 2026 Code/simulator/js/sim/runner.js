/*
 * Runs a Python program (SPIKE App 3 or Pybricks style) against the simulated world.
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};
  const py = FLL.py;

  class ProgramRunner {
    constructor(world, hooks) {
      this.world = world;
      this.hooks = hooks || {};
      this.state = 'idle';      // idle | running | finished | error | stopped
      this.interp = null;
      this.sched = null;
      this.error = null;
      this.startTime = 0;
      this.filename = 'program.py';
    }

    get running() { return this.state === 'running'; }

    start(source, opts) {
      opts = opts || {};
      this.stop(true);
      const world = this.world;
      const hooks = this.hooks;
      const env = {
        world: world,
        print: (s) => hooks.print && hooks.print(s),
        onBeep: hooks.onBeep,
        font: FLL.api.spikeFont || {}
      };
      const modules = {};
      const add = (mods) => { for (const name of Object.keys(mods)) { const m = mods[name]; modules[name] = () => m; } };
      add(FLL.api.spike.createModules(env));
      add(FLL.api.pybricks.createModules(env));
      this.interp = new py.Interpreter({
        print: (s) => hooks.print && hooks.print(s),
        warn: (s) => hooks.warn && hooks.warn(s),
        modules: modules,
        now: () => world.time
      });
      this.sched = new py.Scheduler(this.interp);
      this.sched.stepsPerTick = opts.stepsPerTick || 250;
      this.error = null;
      this.filename = opts.filename || 'program.py';
      let ast;
      try {
        ast = this.interp.parse(source);
      } catch (e) {
        this.fail(e);
        return false;
      }
      // A fresh program starts with motor counters and the gyro reset (like pressing play on the hub).
      const rb = world.robot;
      rb.stopAll('coast');
      for (const p in rb.motors) rb.motors[p].pos = 0;
      rb.resetYaw(0, world.time);
      rb.hub.matrix.fill(0);
      rb.hub.text = '';
      this.sched.spawn(this.interp.runModule(ast), 'main');
      this.state = 'running';
      this.startTime = world.time;
      world.active = true;
      return true;
    }

    fail(e) {
      const pe = e instanceof py.PyException ? e : null;
      this.state = 'error';
      if (pe) {
        this.error = {
          type: pe.exc.cls.name,
          message: pe.message,
          line: pe.line || (pe.pyTraceback.length ? pe.pyTraceback[pe.pyTraceback.length - 1].line : 0),
          traceback: py.formatTraceback(pe, this.filename)
        };
      } else {
        this.error = { type: 'InternalError', message: String(e && e.message || e), line: 0, traceback: String(e && e.stack || e) };
      }
      this.finishProgram();
      if (this.hooks.onError) this.hooks.onError(this.error);
    }

    finishProgram() {
      if (this.sched) this.sched.stop();
      this.world.robot.stopAll('coast');
    }

    // Run the program for one physics tick.
    tick() {
      if (this.state !== 'running') return;
      const alive = this.sched.tick();
      if (this.sched.error) {
        this.fail(this.sched.error.error);
        return;
      }
      if (!alive) {
        this.state = 'finished';
        this.finishProgram();
        if (this.hooks.onFinish) this.hooks.onFinish(this.sched.exited ? 'exit' : 'done');
      }
    }

    stop(silent) {
      if (this.state === 'running') {
        this.state = 'stopped';
        this.finishProgram();
        if (!silent && this.hooks.onFinish) this.hooks.onFinish('stopped');
      }
    }

    activeLines() {
      if (this.state !== 'running' || !this.sched) return [];
      return this.sched.activeLines();
    }
  }

  FLL.ProgramRunner = ProgramRunner;
})(typeof globalThis !== 'undefined' ? globalThis : window);
