/*
 * Canvas renderer for the field, mission models and robot. Handles zoom/pan.
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};
  const geom = FLL.geom;
  const field = FLL.field;

  const COLOR_CSS = {
    black: '#111', magenta: '#d81b91', violet: '#7e3ff2', blue: '#1e63d6', azure: '#3ab7f0', turquoise: '#21c7b0',
    green: '#2aa44a', yellow: '#ffd21f', orange: '#ff8a1f', red: '#e53935', white: '#ffffff', off: '#555', unknown: '#888'
  };

  class FieldView {
    constructor(canvas, app) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.app = app;
      this.scale = 0.4;       // pixels per mm
      this.ox = 0;            // screen x of world x=0
      this.oy = 0;            // screen y of world y=0
      this.options = { grid: false, zones: true, trail: true, labels: true, schematic: false, sensors: true };
      this.measure = null;    // {a:[x,y], b:[x,y]}
      this.hover = null;
      this.matImg = null;
      this.wireImg = null;
      this.dpr = 1;
    }

    setImages(mat, wire) { this.matImg = mat; this.wireImg = wire; }

    resize() {
      const r = this.canvas.getBoundingClientRect();
      this.dpr = window.devicePixelRatio || 1;
      this.cssW = Math.max(100, r.width);
      this.cssH = Math.max(100, r.height);
      this.canvas.width = Math.round(this.cssW * this.dpr);
      this.canvas.height = Math.round(this.cssH * this.dpr);
      // Keep the whole table visible until the user zooms or pans.
      if (!this.fitted || this.autoFit !== false) this.fit();
    }

    fit() {
      const T = field.TABLE;
      const pad = 14;
      const w = T.x1 - T.x0 + 2 * field.wallThickness, h = T.y1 - T.y0 + 2 * field.wallThickness;
      this.scale = Math.min((this.cssW - pad * 2) / w, (this.cssH - pad * 2) / h);
      this.ox = (this.cssW - (T.x1 + T.x0) * this.scale) / 2;
      this.oy = (this.cssH + (T.y1 + T.y0) * this.scale) / 2;
      this.fitted = true;
      this.autoFit = true;
    }

    zoomAt(sx, sy, factor) {
      const [wx, wy] = this.toWorld(sx, sy);
      this.scale = geom.clamp(this.scale * factor, 0.15, 6);
      this.ox = sx - wx * this.scale;
      this.oy = sy + wy * this.scale;
      this.autoFit = false;
    }

    toScreen(x, y) { return [this.ox + x * this.scale, this.oy - y * this.scale]; }
    toWorld(sx, sy) { return [(sx - this.ox) / this.scale, (this.oy - sy) / this.scale]; }

    // ---- drawing helpers (also used by mission draw functions)
    moveTo(ctx, x, y) { const p = this.toScreen(x, y); ctx.moveTo(p[0], p[1]); }
    lineTo(ctx, x, y) { const p = this.toScreen(x, y); ctx.lineTo(p[0], p[1]); }
    path(ctx, poly) {
      ctx.beginPath();
      poly.forEach((p, i) => (i ? this.lineTo(ctx, p[0], p[1]) : this.moveTo(ctx, p[0], p[1])));
      ctx.closePath();
    }
    drawPoly(ctx, poly, fill, stroke, lw) {
      this.path(ctx, poly);
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = Math.max(1, (lw || 1) * Math.min(1.5, this.scale * 2.2)); ctx.stroke(); }
    }
    drawCircle(ctx, cx, cy, r, fill, stroke, lw) {
      const p = this.toScreen(cx, cy);
      ctx.beginPath();
      ctx.arc(p[0], p[1], Math.max(1.5, r * this.scale), 0, Math.PI * 2);
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw || 1.5; ctx.stroke(); }
    }
    drawText(ctx, text, x, y, color, sizePx, opts) {
      opts = opts || {};
      const p = this.toScreen(x, y);
      const size = Math.max(8, Math.min(sizePx * Math.max(0.7, this.scale * 2.4), sizePx * 1.8));
      ctx.font = (opts.bold ? 'bold ' : '') + size + 'px system-ui, Segoe UI, sans-serif';
      ctx.textAlign = opts.align || 'center';
      ctx.textBaseline = 'middle';
      if (opts.outline !== false) {
        ctx.lineWidth = 3;
        ctx.strokeStyle = 'rgba(0,0,0,0.65)';
        ctx.strokeText(text, p[0], p[1]);
      }
      ctx.fillStyle = color || '#fff';
      ctx.fillText(text, p[0], p[1]);
    }

    // ---------------------------------------------------------------- frame
    draw() {
      const ctx = this.ctx;
      const world = this.app.world;
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      ctx.fillStyle = '#1c1f24';
      ctx.fillRect(0, 0, this.cssW, this.cssH);
      this.drawTable(ctx);
      this.drawMat(ctx);
      if (this.options.grid) this.drawGrid(ctx);
      this.drawMissions(ctx, world);
      this.drawItems(ctx, world);
      if (this.options.trail) this.drawTrail(ctx, world.robot);
      if (this.app.ghostPose) this.drawGhost(ctx, world.robot, this.app.ghostPose);
      this.drawRobot(ctx, world.robot, world);
      this.drawMeasure(ctx);
      this.drawHover(ctx, world);
    }

    drawTable(ctx) {
      const T = field.TABLE;
      const wt = field.wallThickness;
      // walls
      const outer = [[T.x0 - wt, T.y0 - wt], [T.x1 + wt, T.y0 - wt], [T.x1 + wt, T.y1 + wt], [T.x0 - wt, T.y1 + wt]];
      this.drawPoly(ctx, outer, '#0b0c0e', null);
      // table surface incl. storage areas
      this.drawPoly(ctx, [[T.x0, T.y0], [T.x1, T.y0], [T.x1, T.y1], [T.x0, T.y1]], '#2a2d33', null);
      // home areas (storage strips)
      ctx.save();
      ctx.setLineDash([6, 5]);
      this.drawPoly(ctx, [[T.x0 + 4, T.y0 + 4], [-4, T.y0 + 4], [-4, T.y1 - 4], [T.x0 + 4, T.y1 - 4]], 'rgba(255,214,0,0.06)', 'rgba(255,214,0,0.55)', 1.2);
      this.drawPoly(ctx, [[field.MAT_W + 4, T.y0 + 4], [T.x1 - 4, T.y0 + 4], [T.x1 - 4, T.y1 - 4], [field.MAT_W + 4, T.y1 - 4]], 'rgba(255,214,0,0.06)', 'rgba(255,214,0,0.55)', 1.2);
      ctx.restore();
      this.drawText(ctx, 'HOME', T.x0 / 2, 900, 'rgba(255,214,0,0.75)', 11, { bold: true });
      this.drawText(ctx, 'HOME', (field.MAT_W + T.x1) / 2, 900, 'rgba(255,214,0,0.75)', 11, { bold: true });
    }

    drawMat(ctx) {
      const img = this.options.schematic ? this.wireImg : this.matImg;
      const p0 = this.toScreen(0, field.MAT_H);
      const w = field.MAT_W * this.scale, h = field.MAT_H * this.scale;
      if (img && img.complete && img.naturalWidth) {
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(img, p0[0], p0[1], w, h);
      } else {
        ctx.fillStyle = '#4f8a4f';
        ctx.fillRect(p0[0], p0[1], w, h);
      }
    }

    drawGrid(ctx) {
      ctx.save();
      ctx.strokeStyle = 'rgba(255,40,40,0.75)';
      ctx.lineWidth = 1;
      for (let x = 0; x <= field.MAT_W; x += 200) {
        ctx.beginPath(); this.moveTo(ctx, x, 0); this.lineTo(ctx, x, field.MAT_H); ctx.stroke();
      }
      for (let y = 0; y <= field.MAT_H; y += 200) {
        ctx.beginPath(); this.moveTo(ctx, 0, y); this.lineTo(ctx, field.MAT_W, y); ctx.stroke();
      }
      field.cols.forEach((c, i) => this.drawText(ctx, c, i * 200 + 100, -18, '#ffb3b3', 11, { bold: true }));
      field.rows.forEach((r, i) => this.drawText(ctx, r, -20, i * 200 + 100, '#ffb3b3', 11, { bold: true }));
      ctx.restore();
    }

    drawMissions(ctx, world) {
      const showZones = this.options.zones;
      for (const m of world.missions) {
        const def = m.def;
        // models
        for (const mod of m.models || []) {
          const poly = mod.circle ? geom.circlePoly(mod.circle[0], mod.circle[1], mod.circle[2], 24) : mod.poly;
          const fill = (mod.fill || def.color || '#888');
          ctx.globalAlpha = 0.72;
          this.drawPoly(ctx, poly, fill, '#101010', 1.4);
          ctx.globalAlpha = 1;
        }
        if (def.draw) def.draw(ctx, m, this, world);
        // target zones
        if (showZones) {
          for (const z of m.zones || []) {
            const avoid = !!z.avoid;
            ctx.save();
            ctx.setLineDash(avoid ? [3, 3] : [5, 4]);
            const col = avoid ? 'rgba(255,60,60,0.95)' : (z.accept.indexOf('tool') >= 0 && z.accept.indexOf('push') < 0 ? 'rgba(255,170,0,0.95)' : 'rgba(0,229,255,0.95)');
            this.drawPoly(ctx, z.poly, avoid ? 'rgba(255,0,0,0.10)' : 'rgba(255,255,255,0.10)', col, 1.4);
            ctx.restore();
          }
        }
      }
      // mission number badges
      if (this.options.labels) {
        for (const m of world.missions) {
          const pos = this.badgePos(m);
          if (!pos) continue;
          const done = this.app.missionDone(m);
          const p = this.toScreen(pos[0], pos[1]);
          const r = Math.max(9, Math.min(15, 26 * this.scale));
          ctx.beginPath();
          ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
          ctx.fillStyle = done ? '#1aa64b' : '#d71920';
          ctx.fill();
          ctx.lineWidth = 2;
          ctx.strokeStyle = '#fff';
          ctx.stroke();
          ctx.fillStyle = '#fff';
          ctx.font = 'bold ' + Math.round(r * 0.95) + 'px system-ui, sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(String(m.id).padStart(2, '0'), p[0], p[1] + 0.5);
        }
      }
    }

    badgePos(m) {
      const fixed = { 1: [900, 150], 2: [560, 470], 3: [80, 760], 4: [210, 1095], 5: [404, 990], 6: [1440, 1100], 7: [1640, 990],
        8: [700, 1110], 9: [470, 960], 10: [800, 1000], 11: [1405, 690], 12: [1860, 820] };
      if (fixed[m.id]) return fixed[m.id];
      const d = FLL.missions.dockOf(m.id, this.app.world.setup);
      if (!d) return null;
      const dock = field.docks[d];
      return [dock.x + (d === 'mine' ? -95 : 0), dock.y + (d === 'city' ? 110 : (d === 'mine' ? -60 : 105))];
    }

    drawItems(ctx, world) {
      for (const it of world.items) {
        if (it.carried || it.placed) continue;
        this.drawItemIcon(ctx, it, it.x, it.y);
      }
    }
    drawItemIcon(ctx, it, x, y) {
      const colors = { seed: '#8d5524', leaf: '#43a047', katydid: '#e53935', tie: '#c62828', keystone: '#ab47bc', fragment: '#ffb300' };
      this.drawCircle(ctx, x, y, 9, colors[it.kind] || '#999', '#fff', 1.5);
    }

    drawTrail(ctx, robot) {
      const t = robot.trail;
      if (t.length < 2) return;
      ctx.save();
      ctx.strokeStyle = 'rgba(255,235,59,0.85)';
      ctx.lineWidth = 2;
      ctx.setLineDash([]);
      ctx.beginPath();
      this.moveTo(ctx, t[0][0], t[0][1]);
      for (let i = 1; i < t.length; i++) this.lineTo(ctx, t[i][0], t[i][1]);
      this.lineTo(ctx, robot.pose.x, robot.pose.y);
      ctx.stroke();
      ctx.restore();
    }

    drawGhost(ctx, robot, pose) {
      ctx.save();
      ctx.globalAlpha = 0.35;
      ctx.setLineDash([4, 4]);
      this.drawPoly(ctx, robot.bodyPoly(pose), null, '#ffffff', 1.5);
      ctx.restore();
    }

    drawRobot(ctx, robot, world) {
      const pose = robot.pose;
      const cfg = robot.config;
      const b = cfg.body;
      const L = (f, l) => geom.localToWorld(pose, f, l);
      const polyL = (pts) => pts.map(([f, l]) => L(f, l));

      // tool zones (attachments)
      for (const p of robot.attachmentPorts()) {
        const m = robot.motors[p];
        const st = robot.armState[p] || {};
        const active = st.moving;
        ctx.save();
        ctx.setLineDash([3, 3]);
        this.drawPoly(ctx, robot.toolZone(p), active ? 'rgba(255,170,0,0.35)' : 'rgba(255,170,0,0.08)', active ? '#ffb300' : 'rgba(255,190,60,0.7)', 1);
        ctx.restore();
        // arm drawing: a bar whose length shows the arm angle (vertical lift arm seen from above)
        const d = robot.device(p);
        const len = (d.armLength || 80) * Math.abs(Math.cos(geom.rad(m.pos)));
        const lat = d.mount === 'left' ? 1 : (d.mount === 'right' ? -1 : 0);
        const offs = p === 'E' ? 22 : -22;
        let pts;
        if (d.mount === 'back') pts = [[-b.rear, offs - 6], [-b.rear - len, offs - 6], [-b.rear - len, offs + 6], [-b.rear, offs + 6]];
        else if (lat) pts = [[offs, lat * b.width / 2], [offs, lat * (b.width / 2 + len)], [offs + 12, lat * (b.width / 2 + len)], [offs + 12, lat * b.width / 2]];
        else pts = [[b.front - 6, offs - 6], [b.front + len, offs - 6], [b.front + len, offs + 6], [b.front - 6, offs + 6]];
        this.drawPoly(ctx, polyL(pts), '#9e9e9e', '#333', 1);
      }

      // wheels
      const wr = cfg.wheelDiameter / 2;
      for (const side of [1, -1]) {
        const y = side * cfg.trackWidth / 2;
        this.drawPoly(ctx, polyL([[wr, y - 7], [-wr, y - 7], [-wr, y + 7], [wr, y + 7]]), '#1b1b1b', '#000', 1);
      }
      // chassis
      this.drawPoly(ctx, robot.bodyPoly(), 'rgba(236,236,236,0.93)', '#333', 1.6);
      // contact highlight
      if (robot.contacts && robot.contacts.length) this.drawPoly(ctx, robot.bodyPoly(), null, '#ff5252', 2.5);

      // hub with light matrix
      const hubF = (b.front - b.rear) / 2 - 5;
      const hw = 34, hl = 52;
      this.drawPoly(ctx, polyL([[hubF + hl / 2, hw], [hubF - hl / 2, hw], [hubF - hl / 2, -hw], [hubF + hl / 2, -hw]]), '#fff7d1', '#d4a800', 1.6);
      const cell = 7.5;
      for (let r = 0; r < 5; r++) {
        for (let c = 0; c < 5; c++) {
          const v = robot.hub.matrix[r * 5 + c] || 0;
          // row 0 is at the hub's front edge, column 0 at its left side
          const f = hubF + 10 + (2 - r) * cell;
          const l = (2 - c) * cell;
          this.drawPoly(ctx, polyL([[f + 3, l + 3], [f - 3, l + 3], [f - 3, l - 3], [f + 3, l - 3]]),
            v > 0 ? 'rgba(255,255,255,' + (0.35 + 0.65 * v / 100) + ')' : 'rgba(70,70,70,0.35)', null);
        }
      }
      const cb = L(hubF - 20, 0);
      this.drawCircle(ctx, cb[0], cb[1], 4.5, COLOR_CSS[robot.hub.centerColor] || '#fff', '#555', 1);

      // sensors
      for (const p of FLL.robot.PORTS) {
        const d = robot.device(p);
        if (d.type === 'color') {
          const w = robot.sensorWorld(p);
          let col = '#888';
          if (this.options.sensors && world) {
            const rgb = world.colorUnder(w[0], w[1]);
            const c = FLL.api.spike.classifyColor(rgb);
            col = COLOR_CSS[FLL.api.spike.COLOR_NAMES[c]] || '#888';
          }
          this.drawCircle(ctx, w[0], w[1], 7, col, '#000', 2);
          this.drawText(ctx, p, w[0], w[1], '#fff', 7, { bold: true });
        } else if (d.type === 'distance') {
          const w = robot.sensorWorld(p);
          for (const s of [-8, 8]) { const e = L((d.x || b.front), (d.y || 0) + s); this.drawCircle(ctx, e[0], e[1], 4.5, '#222', '#00e5ff', 1.5); }
          if (this.options.sensors && world) {
            const ang = pose.theta + geom.rad(d.angle || 0);
            const dist = Math.min(world.rayDistance(w, ang, 2100), 2000);
            ctx.save();
            ctx.strokeStyle = 'rgba(0,229,255,0.45)';
            ctx.setLineDash([2, 4]);
            ctx.beginPath(); this.moveTo(ctx, w[0], w[1]); this.lineTo(ctx, w[0] + Math.cos(ang) * dist, w[1] + Math.sin(ang) * dist); ctx.stroke();
            ctx.restore();
          }
        } else if (d.type === 'force') {
          const fx = Math.max(d.x || b.front, b.front);
          const pressed = world && world.bumperPressed(p);
          this.drawPoly(ctx, polyL([[fx + 6, (d.y || 0) + 9], [fx - 2, (d.y || 0) + 9], [fx - 2, (d.y || 0) - 9], [fx + 6, (d.y || 0) - 9]]), pressed ? '#ff5252' : '#444', '#000', 1);
        }
      }
      // heading arrow
      const a0 = L(b.front - 22, 0), a1 = L(b.front - 2, 0);
      ctx.save();
      ctx.strokeStyle = '#d71920';
      ctx.lineWidth = Math.max(1.5, 3 * this.scale);
      ctx.beginPath(); this.moveTo(ctx, a0[0], a0[1]); this.lineTo(ctx, a1[0], a1[1]); ctx.stroke();
      const h1 = L(b.front - 10, 7), h2 = L(b.front - 10, -7);
      ctx.beginPath(); this.moveTo(ctx, h1[0], h1[1]); this.lineTo(ctx, a1[0], a1[1]); this.lineTo(ctx, h2[0], h2[1]); ctx.stroke();
      ctx.restore();
      // cargo
      const cargo = world ? world.cargo() : [];
      cargo.forEach((it, i) => {
        const p = L(-b.rear + 14, (i - (cargo.length - 1) / 2) * 18);
        this.drawItemIcon(ctx, it, p[0], p[1]);
      });
      // rotate handle (only when editable)
      if (this.app.canEditPose()) {
        const h = L(b.front + 40, 0);
        this.drawCircle(ctx, h[0], h[1], 9, 'rgba(255,255,255,0.85)', '#d71920', 2);
        this.drawText(ctx, '⟳', h[0], h[1], '#d71920', 10, { outline: false, bold: true });
      }
    }

    rotateHandlePos() {
      const r = this.app.world.robot;
      return geom.localToWorld(r.pose, r.config.body.front + 40, 0);
    }

    drawMeasure(ctx) {
      const m = this.measure;
      if (!m || !m.a || !m.b) return;
      ctx.save();
      ctx.strokeStyle = '#00e5ff';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 3]);
      ctx.beginPath(); this.moveTo(ctx, m.a[0], m.a[1]); this.lineTo(ctx, m.b[0], m.b[1]); ctx.stroke();
      ctx.restore();
      this.drawCircle(ctx, m.a[0], m.a[1], 4, '#00e5ff', '#000', 1);
      this.drawCircle(ctx, m.b[0], m.b[1], 4, '#00e5ff', '#000', 1);
      const d = geom.dist(m.a, m.b);
      // compass bearing: 0 = north, clockwise positive
      const bearing = (90 - geom.deg(Math.atan2(m.b[1] - m.a[1], m.b[0] - m.a[0])) + 360) % 360;
      this.drawText(ctx, Math.round(d) + ' mm  (' + (d / 10).toFixed(1) + ' cm)  ' + Math.round(bearing) + '°',
        (m.a[0] + m.b[0]) / 2, (m.a[1] + m.b[1]) / 2 + 22, '#00e5ff', 12, { bold: true });
    }

    drawHover(ctx, world) {
      const h = this.hover;
      if (!h) return;
      const [x, y] = h;
      const T = field.TABLE;
      if (x < T.x0 || x > T.x1 || y < T.y0 || y > T.y1) return;
      const cell = field.cellName(x, y);
      const rgb = world.sampleRGB(x, y);
      const txt = 'x ' + Math.round(x) + '  y ' + Math.round(y) + ' mm' + (cell ? '  ' + cell : '');
      ctx.font = '12px system-ui, sans-serif';
      const w = ctx.measureText(txt).width + 34;
      ctx.fillStyle = 'rgba(0,0,0,0.75)';
      ctx.fillRect(8, this.cssH - 30, w, 22);
      ctx.fillStyle = 'rgb(' + rgb.map(Math.round).join(',') + ')';
      ctx.fillRect(14, this.cssH - 25, 12, 12);
      ctx.fillStyle = '#fff';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(txt, 32, this.cssH - 19);
    }
  }

  FLL.FieldView = FieldView;
  FLL.COLOR_CSS = COLOR_CSS;
})(typeof globalThis !== 'undefined' ? globalThis : window);
