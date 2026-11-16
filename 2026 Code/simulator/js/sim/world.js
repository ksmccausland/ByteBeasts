/*
 * Simulation world: field, robot, missions, loose items, collisions, sensors and scoring.
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};
  const geom = FLL.geom;
  const field = FLL.field;
  const M = FLL.missions;

  const DT = 0.005; // physics step (seconds)

  class World {
    constructor(opts) {
      opts = opts || {};
      this.setup = Object.assign({
        docks: { mine: 13, city: 14, farm: 15 },
        opponentConnections: 0,
        inspectionBonus: true,
        tokens: 6
      }, opts.setup || {});
      this.seed = opts.seed || ((Date.now() & 0xffffff) | 1);
      this.rng = geom.rng(this.seed);
      this.matSampler = opts.matSampler || null;
      this.robot = new FLL.robot.Robot(opts.robotConfig, this.seed + 7);
      this.time = 0;          // simulated milliseconds
      this.events = [];       // mission log events
      this.onLog = opts.onLog || null;
      this.active = false;    // true while a program or manual drive moves the robot
      this.resetField();
    }

    // ------------------------------------------------------------ field state
    resetField() {
      this.items = [];
      this.itemSeq = 0;
      this.missions = M.list.map(def => {
        const m = { def: def, id: def.id, state: null, touch: {}, touchNow: {} };
        m.state = def.reset(this);
        return m;
      });
      // keep cargo the robot is carrying from home (e.g. the keystone species)
      this.robot.cargo = [];
      this.buildGeometry();
      this.events = [];
    }

    buildGeometry() {
      const obstacles = [];
      for (const w of field.walls) obstacles.push({ id: w.name, poly: w.poly, solid: true, wall: true });
      this.fragile = [];
      for (const m of this.missions) {
        const models = m.def.models(m, this.setup) || [];
        m.models = models;
        for (const mod of models) {
          const shape = { id: mod.id, mission: m.id, poly: mod.poly, circle: mod.circle, solid: mod.solid, fragile: mod.fragile };
          if (mod.circle) shape.poly = geom.circlePoly(mod.circle[0], mod.circle[1], mod.circle[2], 16);
          shape.bounds = geom.boundsOf(shape.poly);
          if (mod.solid) obstacles.push(shape);
          else if (mod.fragile) this.fragile.push(shape);
        }
        const zones = m.def.zones(m, this.setup) || [];
        m.zones = zones;
        m.zoneMap = {};
        for (const z of zones) { z.bounds = geom.boundsOf(z.poly); m.zoneMap[z.id] = z; }
      }
      for (const o of obstacles) if (!o.bounds) o.bounds = geom.boundsOf(o.poly);
      this.obstacles = obstacles;
    }

    setDocks(docks) {
      this.setup.docks = Object.assign({}, docks);
      this.buildGeometry();
    }

    // ------------------------------------------------------------ collisions
    static boundsOverlap(a, b, margin) {
      margin = margin || 0;
      return !(a.x1 + margin < b.x0 || b.x1 + margin < a.x0 || a.y1 + margin < b.y0 || b.y1 + margin < a.y0);
    }
    collidingSet(poly) {
      const set = new Set();
      const b = geom.boundsOf(poly);
      for (const o of this.obstacles) {
        if (World.boundsOverlap(b, o.bounds) && geom.convexOverlap(poly, o.poly)) set.add(o.id);
      }
      return set;
    }
    collides(poly, already, startPoly) {
      const b = geom.boundsOf(poly);
      for (const o of this.obstacles) {
        if (!World.boundsOverlap(b, o.bounds)) continue;
        if (!geom.convexOverlap(poly, o.poly)) continue;
        if (already && already.has(o.id)) {
          // Allow moves that do not push further into something we already overlap.
          if (geom.convexPenetration(poly, o.poly) > geom.convexPenetration(startPoly, o.poly) + 0.01) return true;
          continue;
        }
        return true;
      }
      return false;
    }
    contactsFor(poly, margin) {
      const big = geom.expandPoly(poly, margin);
      const b = geom.boundsOf(big);
      const out = [];
      for (const o of this.obstacles) {
        if (World.boundsOverlap(b, o.bounds) && geom.convexOverlap(big, o.poly)) out.push(o.id);
      }
      return out;
    }
    robotOverlaps() {
      return this.collidingSet(this.robot.bodyPoly()).size > 0;
    }

    // ------------------------------------------------------------ sensors
    sampleRGB(x, y) {
      if (x < 0 || x >= field.MAT_W || y < 0 || y >= field.MAT_H) return [28, 28, 30]; // black table surface
      if (this.matSampler) return this.matSampler(x, y);
      return [200, 200, 200];
    }
    // Averaged colour under a sensor (sensor spot radius ~4 mm).
    colorUnder(px, py) {
      let r = 0, g = 0, b = 0, n = 0;
      for (const [dx, dy] of [[0, 0], [3, 0], [-3, 0], [0, 3], [0, -3]]) {
        const c = this.sampleRGB(px + dx, py + dy);
        r += c[0]; g += c[1]; b += c[2]; n++;
      }
      return [r / n, g / n, b / n];
    }
    rayDistance(origin, angle, maxRange) {
      const d = [Math.cos(angle), Math.sin(angle)];
      let best = maxRange || 2500;
      for (const o of this.obstacles.concat(this.fragile)) {
        const t = geom.rayPoly(origin, d, o.poly);
        if (t < best) best = t;
      }
      return best;
    }
    bumperPressed(port) {
      const rb = this.robot;
      const dv = rb.device(port);
      const front = rb.config.body.front;
      const x = Math.max(dv.x || front, front - 2);
      const poly = [[x - 2, (dv.y || 0) + 10], [x + 5, (dv.y || 0) + 10], [x + 5, (dv.y || 0) - 10], [x - 2, (dv.y || 0) - 10]]
        .map(([f, l]) => geom.localToWorld(rb.pose, f, l));
      const b = geom.boundsOf(poly);
      for (const o of this.obstacles.concat(this.fragile)) {
        if (World.boundsOverlap(b, o.bounds) && geom.convexOverlap(poly, o.poly)) return true;
      }
      return false;
    }

    // ------------------------------------------------------------ items
    addItem(it) {
      it.id = 'item' + (++this.itemSeq);
      it.carried = !!it.carried;
      it.placed = false;
      this.items.push(it);
      return it;
    }
    carry(it) {
      it.carried = true;
      it.pickedAt = this.time;
      return this.addItem(it);
    }
    deliver(kind) {
      const it = this.items.find(x => x.carried && x.kind === kind);
      if (!it) return false;
      it.carried = false;
      it.placed = true;
      it.x = this.robot.pose.x;
      it.y = this.robot.pose.y;
      return true;
    }
    cargo() { return this.items.filter(x => x.carried); }

    // ------------------------------------------------------------ mission helpers
    log(m, msg) {
      const e = { time: this.time, mission: m ? m.id : 0, text: msg };
      this.events.push(e);
      if (this.events.length > 200) this.events.shift();
      if (this.onLog) this.onLog(e);
    }
    robotSpeed() { return this.robot.linearSpeed || 0; }
    touching(m, zoneId) {
      const z = m.zoneMap[zoneId];
      if (!z) return false;
      if (m.touchNow[zoneId] === undefined) {
        const body = this._bodyBig;
        m.touchNow[zoneId] = World.boundsOverlap(this._bodyBounds, z.bounds, 3) && geom.convexOverlap(body, z.poly);
      }
      return m.touchNow[zoneId];
    }
    hit(m, zoneId) {
      if (!this.active) return false;
      const z = m.zoneMap[zoneId];
      if (!z) return false;
      if (z.accept.indexOf('tool') >= 0) {
        for (const ev of this._toolEvents) {
          if (geom.convexOverlap(ev.zone, z.poly)) return true;
        }
      }
      if (z.accept.indexOf('push') >= 0) {
        const now = this.touching(m, zoneId);
        if (now && !m.touch[zoneId]) return true;
      }
      return false;
    }

    updateMissions() {
      const rb = this.robot;
      const body = rb.bodyPoly();
      this._bodyBig = geom.expandPoly(body, 4);
      this._bodyBounds = geom.boundsOf(this._bodyBig);
      this._toolEvents = rb.toolEvents;
      rb.toolEvents = [];
      for (const m of this.missions) m.touchNow = {};
      if (this.active) {
        // Pick up loose items with a tool action.
        for (const ev of this._toolEvents) {
          for (const it of this.items) {
            if (it.carried || it.placed) continue;
            if (geom.pointInPoly([it.x, it.y], geom.expandPoly(ev.zone, 8))) {
              it.carried = true;
              it.pickedAt = this.time;
              this.log(null, 'The robot collected a ' + (it.label || it.kind).toLowerCase() + '.');
            }
          }
        }
        for (const m of this.missions) m.def.update(m, this);
      }
      // remember push contacts (rising-edge detection)
      for (const m of this.missions) {
        for (const z of m.zones) {
          if (z.accept.indexOf('push') >= 0 || z.accept.indexOf('touch') >= 0) m.touch[z.id] = this.touching(m, z.id);
        }
      }
      // fragile models are knocked over by contact (they are not obstacles)
    }

    // ------------------------------------------------------------ stepping
    step(dt) {
      dt = dt || DT;
      const rb = this.robot;
      const before = { x: rb.pose.x, y: rb.pose.y };
      rb.step(dt, this.time, this);
      rb.linearSpeed = Math.hypot(rb.pose.x - before.x, rb.pose.y - before.y) / dt;
      this.time += dt * 1000;
      this.updateMissions();
    }

    // ------------------------------------------------------------ scoring
    equipmentTouching(m) {
      const body = geom.expandPoly(this.robot.bodyPoly(), 2);
      const ids = (m.models || []).map(x => x.id).concat(m.def.noEquipmentModels || []);
      const shapes = this.obstacles.concat(this.fragile).filter(o => ids.indexOf(o.id) >= 0);
      return shapes.some(o => geom.convexOverlap(body, o.poly));
    }

    score(opts) {
      opts = opts || {};
      const rows = [];
      let total = 0;
      if (this.setup.inspectionBonus) {
        rows.push({ id: 0, name: 'Equipment inspection', points: 20, items: [{ label: 'Robot and equipment fit in one launch area, under 305 mm', points: 20 }] });
        total += 20;
      }
      for (const m of this.missions) {
        let items = m.def.score(m, this.setup);
        let pts = items.reduce((a, b) => a + b.points, 0);
        let note = '';
        if (M.NO_EQUIPMENT[m.id] && pts > 0 && opts.final !== false && this.equipmentTouching(m)) {
          note = 'The robot touches this model: no points (No Equipment Constraint).';
          pts = 0;
        }
        rows.push({ id: m.id, name: m.def.name, points: pts, items: items, note: note });
        total += pts;
      }
      const tokens = this.setup.tokens;
      const tp = M.TOKEN_POINTS[Math.max(0, Math.min(6, tokens))];
      rows.push({ id: 99, name: 'Precision tokens', points: tp, items: [{ label: tokens + ' token' + (tokens === 1 ? '' : 's') + ' left', points: tp }] });
      total += tp;
      return { total: total, rows: rows };
    }
  }

  FLL.World = World;
  FLL.World.DT = DT;
})(typeof globalThis !== 'undefined' ? globalThis : window);
