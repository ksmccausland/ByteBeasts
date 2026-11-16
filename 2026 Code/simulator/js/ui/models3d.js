/*
 * 3D mission models for the 3D view: simplified versions of the real BIOGLOW LEGO models.
 * Every builder returns { group, update(state, k) }: update() moves the moving parts towards the
 * position that matches the mission state (k = smoothing step, 1 = jump), so solved missions animate.
 * Field coordinates in mm: x east, y north, z up. They are mapped to three.js as (x, z, -y).
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};
  const R = (d) => d * Math.PI / 180;

  // LEGO-like colours
  const C = {
    lbg: 0xa0a5a9, dbg: 0x6c6e68, black: 0x27292b, white: 0xf4f4f4, tan: 0xe4cd9e, brown: 0x583927, rbrown: 0x7c4a2a,
    green: 0x237841, bgreen: 0x4b9f4a, lime: 0xa5ca18, sand: 0xd9b26f, red: 0xc91a09, yellow: 0xf2cd37,
    azure: 0x36aebf, teal: 0x069d9f, purple: 0x7b5ea7, magenta: 0xa83d79, glass: 0x9fd8ee, stone: 0x7d8471
  };

  // ---------------------------------------------------------------- building kit
  function kit() {
    const T = G.THREE;
    const mats = {};
    const K = {
      T: T,
      V: (p) => new T.Vector3(p[0], p[2] || 0, -p[1]),
      mat(color, opacity) {
        const key = color + ':' + (opacity || 1);
        if (!mats[key]) {
          mats[key] = new T.MeshLambertMaterial({ color: color });
          if (opacity < 1) { mats[key].transparent = true; mats[key].opacity = opacity; }
        }
        return mats[key];
      },
      // own material (for parts that change colour)
      ownMat: (color) => new T.MeshLambertMaterial({ color: color }),
      put(obj, x, y, z) { obj.position.set(x, z || 0, -y); return obj; },
      group(x, y, z, yawDeg) { const g = new T.Group(); K.put(g, x || 0, y || 0, z || 0); if (yawDeg) g.rotation.y = R(yawDeg); return g; },
      // box with its bottom at z (w along x, d along y, h up)
      box(w, d, h, color, x, y, z) { const m = new T.Mesh(new T.BoxGeometry(w, h, d), typeof color === 'number' ? K.mat(color) : color); return K.put(m, x || 0, y || 0, (z || 0) + h / 2); },
      cyl(r, h, color, x, y, z, rTop, seg) { const m = new T.Mesh(new T.CylinderGeometry(rTop == null ? r : rTop, r, h, seg || 20), typeof color === 'number' ? K.mat(color) : color); return K.put(m, x || 0, y || 0, (z || 0) + h / 2); },
      sphere(r, color, x, y, z, sx, sy, sz) {
        const m = new T.Mesh(new T.SphereGeometry(r, 18, 12), typeof color === 'number' ? K.mat(color) : color);
        m.scale.set(sx || 1, sz || 1, sy || 1);
        return K.put(m, x || 0, y || 0, z || 0);
      },
      cone(r, h, color, x, y, z) { const m = new T.Mesh(new T.ConeGeometry(r, h, 16), K.mat(color)); return K.put(m, x || 0, y || 0, (z || 0) + h / 2); },
      torus(r, tube, color, x, y, z, standing) {
        const m = new T.Mesh(new T.TorusGeometry(r, tube, 8, 24), K.mat(color));
        if (!standing) m.rotation.x = Math.PI / 2;
        return K.put(m, x || 0, y || 0, z || 0);
      },
      // round rod between two points [x, y, z]
      rod(a, b, r, color) {
        const A = K.V(a), B = K.V(b);
        const m = new T.Mesh(new T.CylinderGeometry(r, r, A.distanceTo(B), 8), K.mat(color));
        m.position.copy(A).add(B).multiplyScalar(0.5);
        m.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), B.clone().sub(A).normalize());
        return m;
      },
      // square beam between two points (LEGO Technic beam look)
      beam(a, b, w, color) {
        const A = K.V(a), B = K.V(b);
        const m = new T.Mesh(new T.BoxGeometry(w, A.distanceTo(B), w), K.mat(color));
        m.position.copy(A).add(B).multiplyScalar(0.5);
        m.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), B.clone().sub(A).normalize());
        return m;
      },
      tube(points, r, color) {
        const curve = new T.CatmullRomCurve3(points.map(K.V));
        return new T.Mesh(new T.TubeGeometry(curve, Math.max(24, points.length * 12), r, 8, false), K.mat(color));
      },
      // field polygon pushed up from z to z + h
      prism(poly, h, color, z) {
        const shape = new T.Shape(poly.map(p => new T.Vector2(p[0], p[1])));
        const g = new T.ExtrudeGeometry(shape, { depth: h, bevelEnabled: false });
        g.rotateX(-Math.PI / 2);
        const m = new T.Mesh(g, typeof color === 'number' ? K.mat(color) : color);
        m.position.y = z || 0;
        return m;
      },
      // open frame of beams: w along x, d along y, h up, centred on (x, y)
      frame(w, d, h, t, color, x, y, z) {
        const g = new T.Group();
        const x0 = (x || 0) - w / 2, x1 = (x || 0) + w / 2, y0 = (y || 0) - d / 2, y1 = (y || 0) + d / 2, z0 = z || 0, z1 = z0 + h;
        const c = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
        for (let i = 0; i < 4; i++) {
          const a = c[i], b = c[(i + 1) % 4];
          g.add(K.beam([a[0], a[1], z0 + t / 2], [b[0], b[1], z0 + t / 2], t, color));
          g.add(K.beam([a[0], a[1], z1 - t / 2], [b[0], b[1], z1 - t / 2], t, color));
          g.add(K.beam([a[0], a[1], z0], [a[0], a[1], z1], t, color));
        }
        return g;
      },
      minifig(x, y, z, shirt) {
        const g = K.group(x, y, z);
        g.add(K.box(10, 6, 12, C.black, 0, 0, 0));
        g.add(K.box(12, 7, 12, shirt || C.azure, 0, 0, 12));
        g.add(K.cyl(4.5, 9, C.yellow, 0, 0, 24));
        return g;
      },
      ease: (cur, target, k) => cur + (target - cur) * k
    };
    return K;
  }

  // ---------------------------------------------------------------- builders (by mission id)
  const B = {};

  // M01 Drone Survey: the drone sits on top of the cables between scan marker and pilot.
  B[1] = (K, m) => {
    const g = new K.T.Group();
    g.add(K.cyl(31, 10, C.tan, 1151, 57, 0));
    g.add(K.minifig(1151, 57, 10, C.red));
    g.add(K.box(52, 34, 40, C.red, 626, 57, 0));
    g.add(K.box(30, 20, 14, C.black, 1110, 57, 10));
    const apex = [888, 57, 210];
    for (const y of [49, 65]) {
      g.add(K.rod([652, y, 38], [apex[0], y, apex[2]], 2.5, C.lbg));
      g.add(K.rod([1122, y, 22], [apex[0], y, apex[2]], 2.5, C.lbg));
    }
    g.add(K.box(30, 26, 8, C.dbg, apex[0], apex[1], apex[2]));
    const drone = K.group(apex[0], apex[1], apex[2] + 8);
    drone.add(K.box(22, 22, 10, C.white, 0, 0, 0));
    drone.add(K.cyl(4, 6, C.black, 0, 0, -6));
    const rotors = [];
    for (const [dx, dy] of [[-15, -15], [15, -15], [15, 15], [-15, 15]]) {
      drone.add(K.beam([0, 0, 6], [dx, dy, 8], 3, C.dbg));
      const r = K.cyl(9, 1.5, C.black, dx, dy, 9);
      drone.add(r);
      rotors.push(r);
    }
    g.add(drone);
    // LiDAR map: a tile that flips (grey side up -> green side up) into the survey area
    const lidar = K.group(1033, 251, 0, 30);
    const tile = new K.T.Group();
    tile.add(K.box(70, 32, 5, C.lbg, 0, 0, -5));
    tile.add(K.box(70, 32, 5, C.bgreen, 0, 0, 0));
    tile.add(K.box(14, 14, 2, C.black, -18, 0, -7));
    tile.position.y = 5;
    tile.rotation.x = Math.PI;   // grey side up
    lidar.add(tile);
    g.add(lidar);
    let spin = 0;
    return {
      group: g,
      update(st, k) {
        const up = st.drone;
        drone.position.y = K.ease(drone.position.y, apex[2] + (up ? 140 : 8), k);
        spin += up ? 0.5 : 0;
        drone.rotation.y = spin * 0.05;
        rotors.forEach(r => { r.rotation.y += up ? 0.9 : 0; });
        tile.rotation.x = K.ease(tile.rotation.x, st.lidar ? 0 : Math.PI, k);
        const tx = st.lidar ? 1057.6 : 1033, ty = st.lidar ? 220.2 : 251;
        lidar.position.x = K.ease(lidar.position.x, tx, k);
        lidar.position.z = K.ease(lidar.position.z, -ty, k);
      }
    };
  };

  // M02 Exploding Seeds: a stalk with three seed pods on arching stems.
  B[2] = (K) => {
    const g = new K.T.Group(), cx = 637, cy = 540;
    g.add(K.box(62, 62, 12, C.dbg, cx, cy, 0));
    g.add(K.box(14, 14, 172, C.purple, cx, cy, 12));
    g.add(K.box(44, 10, 10, C.black, cx, cy, 178));
    const pods = [];
    [90, 210, 330].forEach((a) => {
      const ex = cx + Math.cos(R(a)) * 72, ey = cy + Math.sin(R(a)) * 72;
      g.add(K.tube([[cx, cy, 186], [cx + Math.cos(R(a)) * 42, cy + Math.sin(R(a)) * 42, 205], [ex, ey, 150], [ex, ey, 92]], 3, C.purple));
      const pod = K.group(ex, ey, 60);
      pod.add(K.cyl(10, 26, C.azure, 0, 0, 6, 12));
      pod.add(K.torus(9, 2.5, C.teal, 0, 0, 2));
      g.add(pod);
      pods.push(pod);
    });
    return { group: g, update(st) { pods.forEach((p, i) => { p.visible = i >= st.released; }); } };
  };

  // M03 Flip the Rock: a rock with a research flag; the flag goes down when the rock is flipped.
  B[3] = (K) => {
    const g = new K.T.Group();
    const rock = K.group(150, 657, 0);   // hinge on the east edge
    const poly = [[70, 689], [30, 681], [14, 655], [14, 635], [22, 625], [142, 625], [150, 655], [142, 670], [126, 681], [94, 689]];
    const body = K.prism(poly.map(p => [p[0] - 150, p[1] - 657]), 30, C.stone, 0);
    rock.add(body);
    rock.add(K.sphere(42, C.stone, -70, 0, 28, 1.4, 0.75, 0.55));
    rock.add(K.sphere(22, 0x8e9682, -100, 8, 40, 1, 1, 0.7));
    const flag = K.group(-30, 0, 45);
    flag.add(K.cyl(2, 95, C.lbg, 0, 0, 0));
    flag.add(K.box(2, 30, 20, C.red, 0, 15, 72));
    rock.add(flag);
    g.add(rock);
    return {
      group: g,
      update(st, k) {
        flag.rotation.x = K.ease(flag.rotation.x, st.flag ? -Math.PI / 2 : 0, k);
        rock.rotation.z = K.ease(rock.rotation.z, st.flag && !st.returned ? R(-14) : 0, k);
      }
    };
  };

  // M04 Lucky Leaves: three pods, each with a leaf on a loop – one hides the katydid.
  B[4] = (K) => {
    const g = new K.T.Group();
    const geom = FLL.geom, field = FLL.field;
    const pods = [
      { id: 'top', poly: [[140, 1036], [186, 1036], [186, 1083], [140, 1083]], yaw: 0 },
      { id: 'diag', poly: geom.rectPoly(106, 1033, 74, 36, R(-45)), yaw: -45 },
      { id: 'left', poly: [[58, 955], [105, 955], [105, 1001], [58, 1001]], yaw: 0 }
    ];
    const outline = field.leafHabitat.concat([field.leafHabitat[0]]).map(p => K.V([p[0], p[1], 1]));
    g.add(new K.T.Line(new K.T.BufferGeometry().setFromPoints(outline), new K.T.LineBasicMaterial({ color: 0x111111 })));
    const parts = {};
    for (const p of pods) {
      g.add(K.prism(p.poly, 10, C.green, 0));
      const [cx, cy] = geom.centroid(p.poly);
      const leaf = K.group(cx, cy, 10, p.yaw);
      leaf.add(K.sphere(22, C.lime, 0, 0, 12, 1.3, 0.7, 0.22));
      leaf.add(K.torus(13, 2, C.teal, 0, 0, 44, true));
      leaf.add(K.rod([0, 0, 14], [0, 0, 31], 2, C.teal));
      const bug = K.group(cx, cy, 10, p.yaw);
      bug.add(K.sphere(9, 0x6f8f2a, 0, 0, 8, 2.2, 0.8, 0.8));
      bug.add(K.sphere(6, 0x6f8f2a, 17, 0, 12));
      bug.add(K.rod([20, 2, 14], [40, 14, 26], 0.8, C.black));
      bug.add(K.rod([20, -2, 14], [40, -14, 26], 0.8, C.black));
      for (const s of [-1, 1]) bug.add(K.rod([-6, 0, 10], [-22, s * 12, 0], 1.2, 0x55701f));
      g.add(leaf);
      g.add(bug);
      parts[p.id] = { leaf: leaf, bug: bug };
    }
    return {
      group: g,
      update(st) {
        for (const id in parts) {
          parts[id].leaf.visible = st.pods[id] === 'leaf';
          parts[id].bug.visible = st.pods[id] === 'katydid';
        }
      }
    };
  };

  // M05 Reaching Roots: a root housing by the back wall; the root grows out towards the south.
  B[5] = (K) => {
    const g = new K.T.Group();
    g.add(K.box(40, 70, 64, C.rbrown, 404, 1098, 0));
    g.add(K.box(48, 20, 22, C.tan, 404, 1060, 0));
    g.add(K.box(48, 6, 70, C.dbg, 404, 1066, 22));
    const gear = K.cyl(18, 6, C.lbg, 427, 1078, 30);
    gear.rotation.z = Math.PI / 2;
    g.add(gear);
    const rootGeom = new K.T.CylinderGeometry(6, 8, 1, 10);
    rootGeom.translate(0, 0.5, 0);
    rootGeom.rotateX(Math.PI / 2);   // grows towards the south
    const root = new K.T.Mesh(rootGeom, K.mat(C.brown));
    K.put(root, 404, 1062, 9);
    root.scale.z = 18;
    g.add(root);
    const tip = K.sphere(7, C.brown, 404, 1044, 9);
    g.add(tip);
    return {
      group: g,
      update(st, k) {
        const len = [18, 40, 70][st.ext];
        root.scale.z = K.ease(root.scale.z, len, k);
        tip.position.z = root.position.z + root.scale.z;
      }
    };
  };

  // M06 Leafcutter Frenzy: an ant carrying leaf fragments back to the nest mound.
  B[6] = (K) => {
    const g = new K.T.Group();
    const poly = FLL.geom.rectPoly(1522, 1074, 104, 50, R(45));
    g.add(K.prism(poly, 14, C.sand, 0));
    const mound = K.sphere(40, 0xc8a15f, 1522, 1074, 14, 1.4, 0.7, 0.75);
    mound.rotation.y = R(45);
    g.add(mound);
    g.add(K.sphere(9, 0x3b2a1a, 1497, 1050, 18, 1, 1, 0.6));
    const ant = K.group(1478, 1022, 0, 45);
    ant.add(K.sphere(5, 0x7a1d12, -10, 0, 7));
    ant.add(K.sphere(4, 0x7a1d12, 0, 0, 7));
    ant.add(K.sphere(6, 0x7a1d12, 11, 0, 8, 1.3, 1, 1));
    for (const s of [-1, 1]) for (const dx of [-6, 0, 6]) ant.add(K.rod([dx, 0, 6], [dx * 1.4, s * 11, 0], 0.8, C.black));
    const frags = [];
    for (let i = 0; i < 4; i++) {
      const f = K.box(14, 12, 2, C.lime, -4 + (i % 2) * 4, 0, 12 + i * 3);
      f.rotation.y = R(i * 37);
      ant.add(f);
      frags.push(f);
    }
    g.add(ant);
    const spread = [[1470, 1060], [1540, 1010], [1455, 1030], [1560, 1040]];
    const loose = spread.map(([x, y], i) => { const f = K.box(14, 12, 2, C.lime, x, y, 0); f.rotation.y = R(i * 50); g.add(f); return f; });
    return {
      group: g,
      update(st, k) {
        const home = st.antHome;
        ant.position.x = K.ease(ant.position.x, home ? 1500 : 1478, k);
        ant.position.z = K.ease(ant.position.z, home ? -1045 : -1022, k);
        frags.forEach((f, i) => { f.visible = i < st.fragments; });
        loose.forEach((f, i) => { f.visible = home && i < 4 - st.fragments; });
      }
    };
  };

  // M07 Humongous Fungus: a frame with a mycelium arm that swings out to the south.
  B[7] = (K) => {
    const g = new K.T.Group();
    g.add(K.frame(14, 33, 62, 5, C.dbg, 1574, 1055.5, 0));
    g.add(K.frame(14, 33, 62, 5, C.dbg, 1609, 1055.5, 0));
    g.add(K.box(49, 12, 6, C.azure, 1591.5, 1060, 62));
    const arm = K.group(1591.5, 1052, 66);
    arm.add(K.box(10, 70, 8, C.white, 0, -35, 0));
    arm.add(K.beam([-16, -40, 4], [16, -40, 4], 5, C.white));
    arm.add(K.sphere(15, 0xf2efe6, 0, -76, 4, 1, 1, 0.55));
    arm.add(K.cone(7, 18, C.tan, 0, -76, -14));
    g.add(arm);
    return { group: g, update(st, k) { arm.rotation.x = K.ease(arm.rotation.x, st.extended ? -0.08 : -Math.PI / 2, k); } };
  };

  // M08 + M09: the big tree with the vine, research platform, camera trap and seed.
  B[8] = (K) => {
    const g = new K.T.Group();
    const TREE = [[592, 996], [598, 997], [653, 1085], [652, 1091], [628, 1113], [560, 1123], [528, 1092], [502, 1050], [503, 1044], [527, 1023]];
    g.add(K.prism(TREE, 12, C.brown, 0));
    g.add(K.cyl(34, 250, C.rbrown, 578, 1062, 12, 26));
    const crowns = [[578, 1062, 285, 55], [530, 1080, 255, 42], [625, 1050, 260, 44], [575, 1015, 245, 38], [600, 1105, 255, 40]];
    for (const [x, y, z, r] of crowns) {
      g.add(K.rod([578, 1062, 220], [x, y, z], 4, C.rbrown));
      g.add(K.sphere(r, z > 270 ? C.green : C.bgreen, x, y, z, 1, 1, 0.45));
    }
    // M08 vine: tangled up in the tree, or lying on the mat
    const tangled = K.tube([[612, 1060, 200], [640, 1045, 170], [655, 1060, 150], [638, 1072, 130], [652, 1052, 110], [642, 1040, 90]], 4, C.bgreen);
    const loose = K.tube([[612, 1060, 200], [650, 1052, 120], [660, 1040, 6], [680, 1030, 4], [690, 1055, 4], [672, 1072, 4], [695, 1080, 4]], 4, C.bgreen);
    g.add(tangled, loose);
    // M09 platform on the south-east side of the trunk
    const plat = K.group(620, 1018, 50);
    plat.add(K.box(82, 58, 7, C.dbg, 0, 0, 0));
    plat.add(K.minifig(-15, 5, 7, C.azure));
    plat.add(K.minifig(15, -8, 7, C.yellow));
    plat.add(K.beam([-30, 26, 0], [-38, 40, -40], 5, C.black));
    g.add(plat);
    // camera trap on the west side
    const cam = K.group(512, 1082, 135);
    cam.add(K.box(18, 16, 14, C.black, 0, 0, -14));
    const lens = K.cyl(4, 4, C.azure, -11, 0, -8);
    lens.rotation.z = Math.PI / 2;
    cam.add(lens);
    g.add(cam);
    const seed = K.group(530, 1015, 95);
    seed.add(K.cyl(8, 20, C.azure, 0, 0, -10, 10));
    g.add(seed);
    return {
      group: g,
      // m09 is the Research Platform mission; its state is passed by the 3D view
      update(st, k, m09) {
        tangled.visible = !st.vine;
        loose.visible = !!st.vine;
        if (!m09) return;
        plat.position.y = K.ease(plat.position.y, m09.platform ? 150 : 50, k);
        cam.position.y = K.ease(cam.position.y, m09.camera ? 60 : 135, k);
        cam.position.x = K.ease(cam.position.x, m09.camera ? 492 : 512, k);
        seed.visible = !m09.seed;
      }
    };
  };

  // M10 Fragile Microhabitats: a spider in an upright frame and a snail on a slanted stem.
  B[10] = (K) => {
    const g = new K.T.Group();
    const spider = K.group(1040, 721, 0, -15);
    spider.add(K.box(52, 14, 4, C.black, 0, 0, 0));
    const fr = [[-23, 0, 4], [23, 0, 4], [23, 0, 62], [-23, 0, 62]];
    for (let i = 0; i < 4; i++) spider.add(K.beam(fr[i], fr[(i + 1) % 4], 4, C.black));
    spider.add(K.beam([-23, 0, 62], [-10, 0, 78], 4, C.teal));
    spider.add(K.beam([-10, 0, 78], [30, 0, 70], 4, C.teal));
    spider.add(K.sphere(5, C.tan, 0, -1, 34, 1, 0.6, 1));
    for (const s of [-1, 1]) for (const dz of [-6, -2, 2, 6]) spider.add(K.rod([0, -1, 34], [s * 12, -1, 34 + dz * 1.6], 0.7, C.tan));
    g.add(spider);
    const snail = K.group(749, 1047, 0, 30);
    snail.add(K.box(30, 20, 8, C.dbg, 0, 0, 0));
    snail.add(K.beam([-10, 0, 8], [10, 0, 50], 6, C.teal));
    snail.add(K.torus(10, 5, C.yellow, 12, 0, 58, true));
    snail.add(K.sphere(5, 0x9e9e9e, 2, 0, 50, 1.6, 0.8, 0.8));
    g.add(snail);
    return {
      group: g,
      update(st, k) {
        spider.rotation.x = K.ease(spider.rotation.x, st.spider ? R(82) : 0, k);
        snail.rotation.z = K.ease(snail.rotation.z, st.snail ? R(-80) : 0, k);
      }
    };
  };

  // M11 Window to the Past: a small building whose root cover drops down to show the roots.
  B[11] = (K) => {
    const g = K.group(1406, 594, 0, 45);
    g.add(K.box(76, 76, 8, C.teal, 0, 0, 0));
    for (const [x, y] of [[-31, -31], [31, -31], [31, 31], [-31, 31]]) g.add(K.box(14, 14, 82, C.teal, x, y, 8));
    g.add(K.box(80, 80, 8, C.magenta, 0, 0, 90));
    g.add(K.cone(10, 26, C.bgreen, 0, 0, 98));
    for (const a of [0, 120, 240]) { const l = K.sphere(12, C.lime, Math.cos(R(a)) * 14, Math.sin(R(a)) * 14, 104, 1, 0.4, 0.15); l.rotation.y = R(a); g.add(l); }
    for (const [x, y] of [[-12, -6], [4, 8], [14, -10]]) g.add(K.tube([[x, y, 90], [x + 4, y - 3, 60], [x - 3, y + 2, 30], [x + 6, y - 6, 9]], 2, C.brown));
    const cover = K.group(0, -38, 8);
    cover.add(K.box(60, 4, 76, C.rbrown, 0, -2, 0));
    g.add(cover);
    return { group: g, update(st, k) { cover.rotation.x = K.ease(cover.rotation.x, st.cover ? R(88) : 0, k); } };
  };

  // M12 Forest Elder: the elder tree, the cane that can be raised and the support tie.
  B[12] = (K) => {
    const g = new K.T.Group();
    g.add(K.cyl(32, 10, C.magenta, 1936, 756, 0));
    g.add(K.tube([[1936, 756, 10], [1934, 754, 90], [1928, 750, 150], [1905, 742, 200], [1895, 738, 228]], 9, C.purple));
    g.add(K.tube([[1928, 750, 150], [1950, 762, 190], [1962, 772, 222]], 8, C.purple));
    g.add(K.sphere(30, C.green, 1895, 738, 232, 1, 1, 0.3));
    g.add(K.sphere(26, C.green, 1962, 772, 226, 1, 1, 0.3));
    g.add(K.prism([[1932, 576], [1924, 540], [1924, 516], [1940, 516], [1948, 532], [1948, 556], [1940, 576]], 80, C.lbg, 0));
    // cane: lies on the mat, raised it leans against the trunk
    const cane = K.group(1852, 686, 2, 43);
    cane.add(K.box(84, 6, 6, C.dbg, 42, 0, 0));
    cane.add(K.box(6, 18, 6, C.dbg, 84, 7, 0));
    g.add(cane);
    const tieOnMat = K.torus(10, 2.5, C.red, 1897, 768, 3);
    const tieOnPost = K.torus(20, 2.5, C.red, 1936, 546, 45);
    g.add(tieOnMat, tieOnPost);
    return {
      group: g,
      update(st, k) {
        cane.rotation.z = K.ease(cane.rotation.z, st.cane ? R(46) : 0, k);
        tieOnMat.visible = !st.tieTaken;
        tieOnPost.visible = !!st.tie;
      }
    };
  };

  // Docks (M13-M15): built in the dock frame (x = short side with the rail at -x, y = long side).
  function dockBase(K, g) {
    g.add(K.box(88, 136, 5, C.dbg, 0, 0, 0));
    g.add(K.box(8, 136, 14, C.black, -44, 0, 0));
  }

  // M13 Keystone Species: an open frame; the young trees stand up, the keystone animal sits on the platform.
  B[13] = (K) => {
    const g = new K.T.Group();
    dockBase(K, g);
    g.add(K.frame(80, 128, 72, 6, C.tan, 0, 0, 5));
    g.add(K.box(40, 50, 5, C.dbg, -18, -20, 70));
    const animal = K.group(-18, -20, 75);
    animal.add(K.sphere(12, 0xc87137, 0, 0, 10, 1.5, 1, 1));
    animal.add(K.sphere(8, 0xc87137, 18, 0, 18));
    animal.add(K.cone(3, 8, C.black, 22, 4, 24));
    g.add(animal);
    const trees = [];
    for (const x of [-22, 0, 22]) {
      const t = K.group(x, 52, 8);
      t.add(K.cyl(3, 30, C.brown, 0, 0, 0));
      t.add(K.cone(12, 30, C.bgreen, 0, 0, 22));
      g.add(t);
      trees.push(t);
    }
    return {
      group: g,
      update(st, k) {
        animal.visible = !!st.keystone;
        trees.forEach(t => { t.rotation.x = K.ease(t.rotation.x, st.trees ? 0 : R(-85), k); });
      }
    };
  };

  // M14 Seeds of Renewal: a replantation station; planted seeds appear in its tray.
  B[14] = (K) => {
    const g = new K.T.Group();
    dockBase(K, g);
    g.add(K.frame(80, 128, 66, 6, C.tan, 0, 0, 5));
    g.add(K.box(70, 116, 4, C.rbrown, 0, 0, 30));
    for (const y of [-40, -15, 10, 35]) g.add(K.beam([-34, y, 66], [34, y + 10, 50], 5, C.teal));
    const seeds = [];
    for (let i = 0; i < 12; i++) {
      const s = K.cyl(6, 14, C.azure, -24 + (i % 3) * 24, -48 + Math.floor(i / 3) * 30, 34, 8);
      g.add(s);
      seeds.push(s);
    }
    return { group: g, update(st) { seeds.forEach((s, i) => { s.visible = i < st.seeds; }); } };
  };

  // M15 Biocentric Architecture: a building with a nesting canopy, a garden skylight and a compost hatch.
  B[15] = (K) => {
    const g = new K.T.Group();
    dockBase(K, g);
    g.add(K.box(76, 124, 62, C.purple, 0, 0, 5));
    for (const y of [-40, 0, 40]) g.add(K.box(78, 24, 20, C.glass, 0, y, 28));
    g.add(K.box(80, 128, 6, C.bgreen, 0, 0, 67));
    // canopy on the rail side: folded flat on the roof, raised it stands up
    const canopy = K.group(-38, 0, 73);
    canopy.add(K.frame(60, 70, 4, 4, C.teal, 30, 0, 0));
    canopy.add(K.box(56, 66, 2, C.lime, 30, 0, 2));
    g.add(canopy);
    // skylight at the far end: sticks out, pushed in it sits on the roof
    const sky = K.group(0, 92, 73);
    sky.add(K.box(50, 44, 5, new K.T.MeshLambertMaterial({ color: C.glass, transparent: true, opacity: 0.7 }), 0, 0, 0));
    g.add(sky);
    // compost hatch on the side opposite the rail: drops open onto the mat
    const hatch = K.group(40, 0, 5);
    hatch.add(K.box(4, 60, 44, C.rbrown, 2, 0, 0));
    g.add(hatch);
    return {
      group: g,
      update(st, k) {
        canopy.rotation.z = K.ease(canopy.rotation.z, st.canopy ? R(70) : 0, k);
        sky.position.z = K.ease(sky.position.z, st.skylight ? -40 : -92, k);
        hatch.rotation.z = K.ease(hatch.rotation.z, st.hatch ? R(-88) : 0, k);
      }
    };
  };

  // Build the 3D model of a mission, or null when there is no detailed model (the view then extrudes the footprint).
  function build(m, setup) {
    const f = B[m.id];
    if (!f) return m.id === 9 ? { group: null } : null;   // M09 lives in the tree (M08)
    const K = kit();
    const out = f(K, m);
    if (m.def.dockModel) {
      const d = FLL.missions.dockOf(m.id, setup);
      if (!d) return { group: null };
      const fr = FLL.field.dockFrame(d);
      const holder = new K.T.Group();
      holder.position.set(fr.x, 0, -fr.y);
      holder.rotation.y = fr.angle;
      holder.add(out.group);
      out.group = holder;
    }
    return out;
  }

  FLL.models3d = { build: build, colors: C };
})(typeof globalThis !== 'undefined' ? globalThis : window);
