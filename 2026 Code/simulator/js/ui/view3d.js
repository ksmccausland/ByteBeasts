/*
 * 3D view of the field: the mat, walls, mission models, loose items, the path and the robot (with its arms).
 * It only shows the simulation – driving, Record and Run work the same as in the 2D view.
 * three.js (vendor/three) is loaded the first time the 3D view is opened.
 * Axes: simulator x (east) = three x, simulator y (north) = three -z, height = three y.
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};
  const $ = (id) => document.getElementById(id);
  const WALL_H = 70;   // height of the table walls (mm)
  const ITEM_COLORS = { seed: 0x8d5524, leaf: 0x43a047, katydid: 0xe53935, tie: 0xc62828, keystone: 0xab47bc, fragment: 0xffb300 };

  function loadScript(src) {
    return new Promise((ok, fail) => {
      const s = document.createElement('script');
      s.src = src;
      s.onload = ok;
      s.onerror = () => fail(new Error('could not load ' + src));
      document.head.appendChild(s);
    });
  }

  class View3D {
    constructor(app) {
      this.app = app;
      this.on = false;
      this.loading = false;
      this.renderer = null;
    }

    toggle() { this.set(!this.on); }

    set(on) {
      if (on && !this.renderer) {
        if (this.loading) return;
        this.loading = true;
        this.app.toast('Loading the 3D view…', 1500);
        const load = G.THREE && G.THREE.OrbitControls ? Promise.resolve()
          : loadScript('vendor/three/three.min.js').then(() => loadScript('vendor/three/OrbitControls.js'));
        load.then(() => {
          this.loading = false;
          this.init();
          this.set(true);
        }).catch((e) => {
          this.loading = false;
          this.app.toast('The 3D view could not start: ' + e.message, 5000);
          this.app.log('The 3D view could not start: ' + e.message, 'err');
        });
        return;
      }
      this.on = on;
      this.canvas.classList.toggle('show', on);
      $('fieldCanvas').style.visibility = on ? 'hidden' : '';
      $('btn3d').classList.toggle('on', on);
      if (on) {
        this.resize();
        this.app.toast('3D view – drag to turn, right-drag to move, scroll to zoom. Press 3D again for the map.', 4000);
      }
    }

    init() {
      const T = G.THREE;
      this.canvas = document.createElement('canvas');
      this.canvas.id = 'view3d';
      $('fieldCanvas').after(this.canvas);
      try {
        this.renderer = new T.WebGLRenderer({ canvas: this.canvas, antialias: true });
      } catch (e) {
        this.canvas.remove();
        throw new Error('this browser has no WebGL (3D graphics)');
      }
      this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
      this.renderer.outputEncoding = T.sRGBEncoding;
      this.camera = new T.PerspectiveCamera(40, 1, 5, 20000);
      this.controls = new T.OrbitControls(this.camera, this.canvas);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.12;
      this.controls.maxPolarAngle = Math.PI / 2 - 0.04;   // stay above the table
      this.controls.minDistance = 120;
      this.controls.maxDistance = 6000;
      this.canvas.addEventListener('dblclick', () => this.resetCamera());
      this.buildScene();
      this.resetCamera();
    }

    // The default camera: from the south (the side the 2D map is seen from), looking at the whole table.
    resetCamera() {
      if (!this.camera) return;
      const field = FLL.field;
      this.controls.target.set(field.MAT_W / 2, 0, -field.MAT_H / 2);
      this.camera.position.set(field.MAT_W / 2, 1450, 1250);
      this.controls.update();
    }

    resize() {
      if (!this.renderer || !this.on) return;
      const r = this.canvas.getBoundingClientRect();
      const w = Math.max(100, r.width), h = Math.max(100, r.height);
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    }

    // ---------------------------------------------------------------- building blocks
    mat(color, opacity) {
      const m = new G.THREE.MeshLambertMaterial({ color: color });
      if (opacity < 1) { m.transparent = true; m.opacity = opacity; }
      return m;
    }
    box(w, h, d, color) { return new G.THREE.Mesh(new G.THREE.BoxGeometry(w, h, d), this.mat(color, 1)); }
    // A field polygon [[x, y], ...] pushed up from z0 to z0 + h.
    prism(poly, h, color, opacity, z0) {
      const T = G.THREE;
      const shape = new T.Shape(poly.map(p => new T.Vector2(p[0], p[1])));
      const g = new T.ExtrudeGeometry(shape, { depth: h, bevelEnabled: false });
      g.rotateX(-Math.PI / 2);
      const mesh = new T.Mesh(g, this.mat(color, opacity));
      mesh.position.y = z0 || 0;
      return mesh;
    }
    disposeGroup(group) {
      group.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) {
          if (o.material.map) o.material.map.dispose();
          o.material.dispose();
        }
      });
      if (group.parent) group.parent.remove(group);
    }

    buildScene() {
      const T = G.THREE, field = FLL.field, Tb = field.TABLE, wt = field.wallThickness;
      const S = this.scene = new T.Scene();
      S.background = new T.Color(0x1c1f24);
      S.add(new T.HemisphereLight(0xffffff, 0x50555e, 0.8));
      const sun = new T.DirectionalLight(0xffffff, 0.55);
      sun.position.set(700, 2400, 900);
      S.add(sun);

      // table top including the home strips beside the mat
      const top = this.box(Tb.x1 - Tb.x0, 24, Tb.y1 - Tb.y0, 0x2a2d33);
      top.position.set((Tb.x0 + Tb.x1) / 2, -12.5, -(Tb.y0 + Tb.y1) / 2);
      S.add(top);
      // walls
      const wall = (x0, y0, x1, y1) => {
        const m = this.box(x1 - x0, WALL_H, y1 - y0, 0x101114);
        m.position.set((x0 + x1) / 2, WALL_H / 2 - 25, -(y0 + y1) / 2);
        S.add(m);
      };
      wall(Tb.x0 - wt, Tb.y0 - wt, Tb.x1 + wt, Tb.y0);
      wall(Tb.x0 - wt, Tb.y1, Tb.x1 + wt, Tb.y1 + wt);
      wall(Tb.x0 - wt, Tb.y0, Tb.x0, Tb.y1);
      wall(Tb.x1, Tb.y0, Tb.x1 + wt, Tb.y1);

      // the mat
      this.matMesh = new T.Mesh(new T.PlaneGeometry(field.MAT_W, field.MAT_H), new T.MeshLambertMaterial({ color: 0x4f8a4f }));
      this.matMesh.rotation.x = -Math.PI / 2;
      this.matMesh.position.set(field.MAT_W / 2, 0, -field.MAT_H / 2);
      S.add(this.matMesh);
      this.matImg = null;

      this.items = [];
      this.trail = null;
      this.trailKey = '';
      this.modelsKey = null;
      this.badges = [];
      this.robotRef = null;
    }

    // ---------------------------------------------------------------- per-frame updates
    syncMat() {
      const v = this.app.view;
      const img = v.options.schematic ? v.wireImg : v.matImg;
      if (!img || !img.complete || !img.naturalWidth || img === this.matImg) return;
      this.matImg = img;
      const T = G.THREE;
      const tex = new T.Texture(img);
      tex.encoding = T.sRGBEncoding;
      tex.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
      tex.needsUpdate = true;
      const m = this.matMesh.material;
      if (m.map) m.map.dispose();
      m.map = tex;
      m.color.set(0xffffff);
      m.needsUpdate = true;
    }

    // Mission models are rebuilt only when the world builds new ones (field reset, setup changes, scoring states).
    syncModels() {
      const world = this.app.world;
      const refs = world.missions.map(m => m.models);
      if (this.modelsKey && this.modelsKey.length === refs.length && refs.every((r, i) => r === this.modelsKey[i])) return;
      this.modelsKey = refs;
      if (this.models) this.disposeGroup(this.models);
      const T = G.THREE, geom = FLL.geom;
      this.models = new T.Group();
      this.animated = [];
      for (const m of world.missions) {
        // a detailed model of the real mission model, if there is one
        const built = FLL.models3d ? FLL.models3d.build(m, world.setup) : null;
        if (built) {
          if (built.group) {
            this.models.add(built.group);
            this.animated.push({ m: m, built: built });
          }
          continue;
        }
        for (const mod of m.models || []) {
          const poly = mod.circle ? geom.circlePoly(mod.circle[0], mod.circle[1], mod.circle[2], 28) : mod.poly;
          if (!poly || poly.length < 3) continue;
          const h = mod.height || (mod.solid ? 75 : 30);
          this.models.add(this.prism(poly, h, new T.Color(mod.fill || m.def.color || '#888888'), mod.solid ? 1 : 0.85));
        }
      }
      this.scene.add(this.models);
      this.animateModels(1);   // a new or reset field starts in place, without animating
    }

    // Move the moving parts of the mission models towards their state (k = 1 jumps there).
    animateModels(k) {
      const ms = this.app.world.missions;
      const m09 = ms.find(x => x.id === 9);
      for (const a of this.animated) {
        if (a.built.update && a.m.state) a.built.update(a.m.state, k, m09 && m09.state);
      }
    }

    // Mission number badges floating above the models (red = to do, green = scored).
    syncBadges() {
      const app = this.app, show = app.view.options.labels;
      for (const b of this.badges) b.sprite.visible = false;
      if (!show) return;
      const T = G.THREE;
      let i = 0;
      for (const m of app.world.missions) {
        const pos = app.view.badgePos(m);
        if (!pos) continue;
        const done = !!app.missionDone(m);
        let b = this.badges[i];
        if (!b) {
          const c = document.createElement('canvas');
          c.width = c.height = 64;
          const sprite = new T.Sprite(new T.SpriteMaterial({ map: new T.CanvasTexture(c), depthTest: false }));
          sprite.scale.set(46, 46, 1);
          sprite.renderOrder = 10;
          this.scene.add(sprite);
          b = this.badges[i] = { canvas: c, sprite: sprite, key: '' };
        }
        const key = m.id + (done ? 'd' : '');
        if (b.key !== key) {
          b.key = key;
          const ctx = b.canvas.getContext('2d');
          ctx.clearRect(0, 0, 64, 64);
          ctx.beginPath();
          ctx.arc(32, 32, 28, 0, Math.PI * 2);
          ctx.fillStyle = done ? '#1aa64b' : '#d71920';
          ctx.fill();
          ctx.lineWidth = 5;
          ctx.strokeStyle = '#fff';
          ctx.stroke();
          ctx.fillStyle = '#fff';
          ctx.font = 'bold 28px system-ui, sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(String(m.id).padStart(2, '0'), 32, 34);
          b.sprite.material.map.needsUpdate = true;
        }
        b.sprite.position.set(pos[0], 160, -pos[1]);
        b.sprite.visible = true;
        i++;
      }
    }

    syncItems() {
      const T = G.THREE;
      const items = this.app.world.items.filter(it => !it.carried && !it.placed);
      while (this.items.length < items.length) {
        const s = new T.Mesh(new T.SphereGeometry(9, 16, 12), this.mat(0x999999, 1));
        this.scene.add(s);
        this.items.push(s);
      }
      this.items.forEach((s, i) => {
        const it = items[i];
        s.visible = !!it;
        if (!it) return;
        s.position.set(it.x, 9, -it.y);
        s.material.color.setHex(ITEM_COLORS[it.kind] || 0x999999);
      });
    }

    syncTrail() {
      const rb = this.app.world.robot, t = rb.trail;
      const show = this.app.view.options.trail && t.length > 1;
      const key = show ? t.length + ':' + t[t.length - 1] + ':' + Math.round(rb.pose.x) + ',' + Math.round(rb.pose.y) : '';
      if (key === this.trailKey) return;
      this.trailKey = key;
      if (this.trail) { this.disposeGroup(this.trail); this.trail = null; }
      if (!show) return;
      const T = G.THREE;
      const pts = t.map(p => new T.Vector3(p[0], 2, -p[1]));
      pts.push(new T.Vector3(rb.pose.x, 2, -rb.pose.y));
      this.trail = new T.Line(new T.BufferGeometry().setFromPoints(pts), new T.LineBasicMaterial({ color: 0xffeb3b }));
      this.scene.add(this.trail);
    }

    // The robot is built in its own frame: x = forward, -z = left, y = up.
    buildRobot(rb) {
      const T = G.THREE, cfg = rb.config, b = cfg.body, wr = cfg.wheelDiameter / 2;
      if (this.robot) this.disposeGroup(this.robot);
      const g = this.robot = new T.Group();
      const len = b.front + b.rear, mid = (b.front - b.rear) / 2;
      const chassisY = wr * 0.9, chassisH = 58;
      this.chassis = this.box(len, chassisH, b.width, 0xececec);
      this.chassis.position.set(mid, chassisY + chassisH / 2, 0);
      g.add(this.chassis);
      for (const side of [1, -1]) {
        const w = new T.Mesh(new T.CylinderGeometry(wr, wr, 14, 28), this.mat(0x1b1b1b, 1));
        w.rotation.x = Math.PI / 2;
        w.position.set(0, wr, -side * (cfg.trackWidth / 2));
        g.add(w);
      }
      const caster = new T.Mesh(new T.SphereGeometry(Math.min(12, chassisY), 16, 12), this.mat(0x777777, 1));
      caster.position.set(-b.rear + 18, Math.min(12, chassisY), 0);
      g.add(caster);
      const top = chassisY + chassisH;
      const hub = this.box(52, 20, 68, 0xfff7d1);
      hub.position.set(mid - 5, top + 10, 0);
      g.add(hub);
      const arrow = new T.Mesh(new T.ConeGeometry(9, 26, 16), this.mat(0xd71920, 1));
      arrow.rotation.z = -Math.PI / 2;
      arrow.position.set(b.front - 16, top + 6, 0);
      g.add(arrow);
      // arms: pos 0 = straight out, the motor angle lifts them
      this.arms = [];
      for (const p of rb.attachmentPorts()) {
        const d = rb.device(p), alen = d.armLength || 80;
        const offs = p === 'E' ? 22 : -22;
        const pivot = new T.Group();
        const bar = this.box(alen, 10, 12, 0x9e9e9e);
        bar.position.x = alen / 2;
        pivot.add(bar);
        if (d.mount === 'back') { pivot.position.set(-b.rear, top - 10, -offs); pivot.rotation.y = Math.PI; }
        else if (d.mount === 'left') { pivot.position.set(offs, top - 10, -b.width / 2); pivot.rotation.y = Math.PI / 2; }
        else if (d.mount === 'right') { pivot.position.set(offs, top - 10, b.width / 2); pivot.rotation.y = -Math.PI / 2; }
        else pivot.position.set(b.front - 6, top - 10, -offs);
        g.add(pivot);
        this.arms.push({ port: p, pivot: pivot });
      }
      this.scene.add(g);
      this.robotRef = rb;
    }

    syncRobot() {
      const rb = this.app.world.robot;
      if (rb !== this.robotRef) this.buildRobot(rb);
      const p = rb.pose;
      this.robot.position.set(p.x, 0, -p.y);
      this.robot.rotation.y = p.theta;
      for (const a of this.arms) {
        const m = rb.motors[a.port];
        if (m) a.pivot.rotation.z = FLL.geom.rad(m.pos);
      }
      this.chassis.material.color.setHex(rb.contacts && rb.contacts.length ? 0xff8a80 : 0xececec);
    }

    draw() {
      if (!this.on || !this.renderer) return;
      this.syncMat();
      this.syncModels();
      this.animateModels(0.12);
      this.syncBadges();
      this.syncItems();
      this.syncTrail();
      this.syncRobot();
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    }
  }

  FLL.View3D = View3D;
})(typeof globalThis !== 'undefined' ? globalThis : window);
