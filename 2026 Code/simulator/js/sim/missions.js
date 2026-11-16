/*
 * BIOGLOW missions: mission models (obstacles), interaction zones, loose items and scoring.
 *
 * Interactions are a simplified model of the real mechanisms:
 *   - "push": the robot body drives into the target zone
 *   - "tool": an attachment motor turns at least 45 degrees while its tool zone covers the target
 *   - "avoid": touching the model knocks it over
 * Scoring follows the official Robot Game Rulebook and Challenge Updates (Oct 2026).
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};
  const geom = FLL.geom;
  const field = FLL.field;

  const rect = (x0, y0, x1, y1) => [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  const rot = (cx, cy, w, h, deg) => geom.rectPoly(cx, cy, w, h, geom.rad(deg));
  // Zone in a dock's local frame (x: short side, rail at -x; y: long side).
  const dockZone = (dockId, x0, y0, x1, y1) => geom.placePoly(rect(x0, y0, x1, y1), field.dockFrame(dockId));

  // Missions with the "no equipment" constraint score nothing if equipment touches the model at the end.
  const NO_EQUIPMENT = { 1: true, 4: true, 5: true, 7: true, 9: true, 10: true, 12: true, 15: true };

  const MISSIONS = [];
  function def(m) { MISSIONS.push(m); return m; }

  // ---------------------------------------------------------------- M01
  def({
    id: 1, name: 'Drone Survey', color: '#7d8a99',
    description: 'Launch the drone so it no longer touches the mat. Bonus: flip the LiDAR map so the scan marker is in the survey area.',
    simHint: 'Use an attachment (tool) at the drone between the cables to launch it. Use a tool at the grey LiDAR map next to the black survey square to flip it.',
    models: () => [
      { id: 'm01-pilot', label: 'Pilot', circle: [1151, 57, 31], solid: true, fill: '#9aa3ad' },
      { id: 'm01-anchor', label: 'Scan marker', poly: rect(600, 40, 652, 74), solid: true, fill: '#d23a32' },
      { id: 'm01-lidar', label: 'LiDAR map', poly: rot(1033, 251, 70, 32, 30), solid: true, fill: '#8e979f' }
    ],
    zones: (m) => [
      { id: 'drone', label: 'Launch drone', poly: rect(852, 22, 928, 96), accept: ['tool'] },
      { id: 'lidar', label: 'Flip LiDAR map', poly: geom.expandPoly(rot(1033, 251, 70, 32, 30), 28), accept: ['tool'] }
    ],
    reset: () => ({ drone: false, lidar: false }),
    update(m, w) {
      if (!m.state.drone && w.hit(m, 'drone')) { m.state.drone = true; w.log(m, 'The drone is launched!'); }
      if (!m.state.lidar && w.hit(m, 'lidar')) { m.state.lidar = true; w.log(m, 'The LiDAR map flipped into the survey area.'); }
    },
    score(m) {
      const out = [{ label: 'Drone no longer touching the mat', points: m.state.drone ? 20 : 0 }];
      out.push({ label: 'Bonus: LiDAR map flipped, scan marker in survey area', points: m.state.drone && m.state.lidar ? 10 : 0 });
      return out;
    },
    draw(ctx, m, v) {
      // cables between the scan marker and the pilot
      ctx.strokeStyle = '#222'; ctx.lineWidth = Math.max(1, 3 * v.scale);
      for (const y of [49, 65]) { ctx.beginPath(); v.moveTo(ctx, 652, y); v.lineTo(ctx, 1122, y); ctx.stroke(); }
      const dz = m.state.drone;
      v.drawPoly(ctx, rect(878, 47, 899, 67), dz ? '#ffffff' : '#e8e8e8', '#333', dz ? 2 : 1);
      if (dz) v.drawText(ctx, '▲ drone up', 889, 105, '#ffffff', 11);
      v.drawPoly(ctx, field.surveyArea, null, m.state.lidar ? '#1aa64b' : '#111', 2.5);
    }
  });

  // ---------------------------------------------------------------- M02
  def({
    id: 2, name: 'Exploding Seeds', color: '#9ccc45',
    description: 'Release the seeds from the seed pod so they no longer touch the stalk (10 points each).',
    simHint: 'Touch the stalk (drive into it or use a tool). Every new touch releases one seed. Released seeds can be collected with a tool for Mission 14.',
    models: () => [
      { id: 'm02-stalk', label: 'Stalk', poly: [[631, 573], [603, 545], [603, 534], [631, 506], [642, 506], [670, 534], [670, 546], [643, 573]], solid: true, fill: '#8bc34a' }
    ],
    zones: () => [
      { id: 'stalk', label: 'Release seeds', poly: geom.expandPoly([[631, 573], [603, 545], [603, 534], [631, 506], [642, 506], [670, 534], [670, 546], [643, 573]], 18), accept: ['tool', 'push'] }
    ],
    reset: () => ({ released: 0 }),
    update(m, w) {
      if (m.state.released < 3 && w.hit(m, 'stalk')) {
        m.state.released++;
        const a = w.rng() * Math.PI * 2;
        const d = 75 + w.rng() * 45;
        w.addItem({ kind: 'seed', from: 'M02', x: 637 + Math.cos(a) * d, y: 540 + Math.sin(a) * d, label: 'Seed' });
        w.log(m, 'A seed was released (' + m.state.released + '/3).');
      }
    },
    score: (m) => [{ label: 'Seeds no longer touching the stalk (' + m.state.released + ')', points: 10 * m.state.released }]
  });

  // ---------------------------------------------------------------- M03
  def({
    id: 3, name: 'Flip the Rock', color: '#6d7f3a',
    description: 'Flip the rock so the research flag is down. Bonus: return the rock to its original position.',
    simHint: 'Use a tool on the east side of the rock: the first action flips it (flag down), a second action puts it back (bonus).',
    models: () => [
      { id: 'm03-rock', label: 'Rock', poly: [[70, 689], [30, 681], [14, 655], [14, 635], [22, 625], [142, 625], [150, 655], [142, 670], [126, 681], [94, 689]], solid: true, fill: '#7b8a52' }
    ],
    zones: () => [{ id: 'rock', label: 'Flip rock', poly: rect(150, 612, 205, 702), accept: ['tool'] }],
    reset: () => ({ flag: false, returned: false }),
    update(m, w) {
      if (w.hit(m, 'rock')) {
        if (!m.state.flag) { m.state.flag = true; w.log(m, 'The rock flipped and the research flag is down.'); }
        else if (!m.state.returned) { m.state.returned = true; w.log(m, 'The rock is back in its original position.'); }
      }
    },
    score: (m) => [
      { label: 'Research flag is down', points: m.state.flag ? 20 : 0 },
      { label: 'Bonus: rock returned to its starting position', points: m.state.flag && m.state.returned ? 10 : 0 }
    ],
    draw(ctx, m, v) {
      v.drawText(ctx, m.state.flag ? '⚑ down' : '⚑ up', 168, 700, m.state.flag ? '#ffe082' : '#ffffff', 11);
    }
  });

  // ---------------------------------------------------------------- M04
  const PODS = [
    { id: 'top', poly: rect(140, 1036, 186, 1083) },
    { id: 'diag', poly: rot(106, 1033, 74, 36, -45) },
    { id: 'left', poly: rect(58, 955, 105, 1001) }
  ];
  def({
    id: 4, name: 'Lucky Leaves', color: '#2e9e4f',
    description: 'Remove leaves from the nest without disturbing the katydid. 1 leaf: 10. Bonus: second leaf with the katydid in its original position: +20.',
    simHint: 'The katydid hides in one of the three pods (randomised every reset, see the red dot). Use a tool on the other pods to collect the leaves.',
    models: () => PODS.map(p => ({ id: 'm04-' + p.id, label: 'Nest', poly: p.poly, solid: true, fill: '#3fae5a' })),
    zones: () => PODS.map(p => ({ id: p.id, label: 'Pod', poly: geom.expandPoly(p.poly, 22), accept: ['tool'] })),
    reset(w) {
      const contents = ['leaf', 'leaf', 'katydid'];
      for (let i = contents.length - 1; i > 0; i--) {
        const j = Math.floor(w.rng() * (i + 1));
        [contents[i], contents[j]] = [contents[j], contents[i]];
      }
      const pods = {};
      PODS.forEach((p, i) => { pods[p.id] = contents[i]; });
      return { pods: pods, removed: 0, katydid: 'home', leftHabitat: false };
    },
    update(m, w) {
      for (const p of PODS) {
        if (!w.hit(m, p.id)) continue;
        const c = m.state.pods[p.id];
        if (c === 'leaf') {
          m.state.pods[p.id] = null;
          m.state.removed++;
          w.carry({ kind: 'leaf', from: 'M04', label: 'Leaf' });
          w.log(m, 'A leaf was removed from the nest.');
        } else if (c === 'katydid') {
          m.state.pods[p.id] = null;
          m.state.katydid = 'carried';
          w.carry({ kind: 'katydid', from: 'M04', label: 'Katydid' });
          w.log(m, 'Oops, the katydid was disturbed!');
        }
      }
      // The katydid leaves the habitat when the robot carries it away.
      const k = w.items.find(it => it.kind === 'katydid');
      if (k) {
        const pos = k.carried ? [w.robot.pose.x, w.robot.pose.y] : [k.x, k.y];
        const inside = geom.pointInPoly(pos, field.leafHabitat) || PODS.some(p => geom.pointInPoly(pos, geom.expandPoly(p.poly, 10)));
        m.state.katydidInside = inside;
        if (!inside) m.state.leftHabitat = true;
      }
    },
    score(m) {
      const inHabitat = m.state.katydid === 'home' || m.state.katydidInside;
      if (!inHabitat) return [{ label: 'Katydid outside the leaf habitat: mission scores zero', points: 0 }];
      return [
        { label: 'One leaf removed (' + m.state.removed + ' removed)', points: m.state.removed >= 1 ? 10 : 0 },
        { label: 'Bonus: second leaf removed, katydid untouched', points: m.state.removed >= 2 && m.state.katydid === 'home' && !m.state.leftHabitat ? 20 : 0 }
      ];
    },
    draw(ctx, m, v) {
      v.drawPoly(ctx, field.leafHabitat, 'rgba(255,255,255,0.08)', '#111', 2);
      for (const p of PODS) {
        const c = m.state.pods[p.id];
        const [cx, cy] = geom.centroid(p.poly);
        if (c === 'leaf') v.drawText(ctx, '🍃', cx, cy, '#fff', 14);
        if (c === 'katydid') { v.drawCircle(ctx, cx, cy, 7, '#e53935', '#fff'); }
      }
    }
  });

  // ---------------------------------------------------------------- M05
  def({
    id: 5, name: 'Reaching Roots', color: '#8d5a2b',
    description: 'Extend the plant root across the forest boundary: partially 10, completely 20.',
    simHint: 'Use a tool (or push) at the front of the root model by the back wall. Each action extends the root further (2 actions = completely).',
    models: () => [{ id: 'm05-root', label: 'Root', poly: [[384, 1133], [384, 1077], [416, 1045], [424, 1045], [424, 1133]], solid: true, fill: '#8d5a2b' }],
    zones: () => [{ id: 'root', label: 'Extend root', poly: rect(366, 1000, 444, 1047), accept: ['tool', 'push'] }],
    reset: () => ({ ext: 0 }),
    update(m, w) {
      if (m.state.ext < 2 && w.hit(m, 'root')) {
        m.state.ext++;
        w.log(m, m.state.ext === 1 ? 'The root is partially extended.' : 'The root is completely extended.');
      }
    },
    score: (m) => [{ label: 'Plant root ' + ['not extended', 'partially extended', 'completely extended'][m.state.ext], points: [0, 10, 20][m.state.ext] }]
  });

  // ---------------------------------------------------------------- M06
  def({
    id: 6, name: 'Leafcutter Frenzy', color: '#d9b26f',
    description: 'Guide the ant back to the nest. 10 points per leaf fragment still in the nest – move too fast and fragments scatter.',
    simHint: 'Push the ant (red dot) slowly into the nest. Pushing faster than about 200 mm/s scatters leaf fragments.',
    models: () => [{ id: 'm06-nest', label: 'Nest', poly: rot(1522, 1074, 104, 50, 45), solid: true, fill: '#d9b26f' }],
    zones: () => [{ id: 'ant', label: 'Push ant', poly: geom.circlePoly(1478, 1022, 26, 12), accept: ['push', 'tool'] }],
    reset: () => ({ antHome: false, fragments: 4 }),
    update(m, w) {
      if (!m.state.antHome && w.hit(m, 'ant')) {
        m.state.antHome = true;
        const sp = w.robotSpeed();
        let lost = 0;
        if (sp > 320) lost = 3; else if (sp > 260) lost = 2; else if (sp > 200) lost = 1;
        m.state.fragments -= lost;
        w.log(m, 'The ant reached the nest' + (lost ? ' but ' + lost + ' leaf fragment' + (lost > 1 ? 's' : '') + ' scattered (too fast: ' + Math.round(sp) + ' mm/s).' : '.'));
      }
    },
    score: (m) => [{ label: 'Ant touching the nest, fragments in nest: ' + (m.state.antHome ? m.state.fragments : 0), points: m.state.antHome ? 10 * m.state.fragments : 0 }],
    draw(ctx, m, v) {
      const p = m.state.antHome ? [1500, 1045] : [1478, 1022];
      v.drawCircle(ctx, p[0], p[1], 8, '#d32f2f', '#000');
      v.drawText(ctx, '🍂×' + m.state.fragments, 1530, 1120, '#fff', 11);
    }
  });

  // ---------------------------------------------------------------- M07
  def({
    id: 7, name: 'Humongous Fungus', color: '#f2efe6',
    description: 'Completely extend the mycelium: 20. Bonus +10 for each connection with an opposing team\'s extended plant root (needs a second table).',
    simHint: 'Use a tool (or push) on the south side of the white mycelium arm. The bonus can be set in the Score panel when you practise with another team.',
    models: () => [{ id: 'm07-mycelium', label: 'Mycelium', poly: rect(1567, 1039, 1616, 1072), solid: true, fill: '#f2efe6' }],
    zones: () => [{ id: 'myc', label: 'Extend mycelium', poly: rect(1560, 994, 1626, 1040), accept: ['tool', 'push'] }],
    reset: () => ({ extended: false }),
    update(m, w) {
      if (!m.state.extended && w.hit(m, 'myc')) { m.state.extended = true; w.log(m, 'The mycelium is completely extended.'); }
    },
    score(m, setup) {
      const conn = Math.max(0, Math.min(2, setup.opponentConnections || 0));
      return [
        { label: 'Mycelium completely extended', points: m.state.extended ? 20 : 0 },
        { label: 'Bonus: connections with opposing team (' + conn + ')', points: m.state.extended ? 10 * conn : 0 }
      ];
    }
  });

  // ---------------------------------------------------------------- M08 + M09 (shared tree)
  const TREE = [[592, 996], [598, 997], [653, 1085], [652, 1091], [628, 1113], [560, 1123], [528, 1092], [502, 1050], [503, 1044], [527, 1023]];
  def({
    id: 8, name: 'Tangled', color: '#7a4a2a',
    description: 'Untangle the vine so it touches the mat: 30.',
    simHint: 'Use a tool on the east side of the big tree to pull the vine down.',
    models: () => [{ id: 'm08-tree', label: 'Tree', poly: TREE, solid: true, fill: '#6d4c2f' }],
    zones: () => [{ id: 'vine', label: 'Pull vine', poly: rect(640, 1018, 702, 1082), accept: ['tool'] }],
    reset: () => ({ vine: false }),
    update(m, w) {
      if (!m.state.vine && w.hit(m, 'vine')) { m.state.vine = true; w.log(m, 'The vine touches the mat.'); }
    },
    score: (m) => [{ label: 'Vine touching the mat', points: m.state.vine ? 30 : 0 }]
  });

  def({
    id: 9, name: 'Research Platform', color: '#c7a26b',
    description: 'Raise the research platform (10), deploy the camera trap (10), knock the seed off the tree (10).',
    simHint: 'Platform: push the south-east side of the tree. Camera trap: tool on the west side. Seed: tool on the south-west side (the seed can be collected for Mission 14).',
    models: () => [],
    zones: () => [
      { id: 'platform', label: 'Raise platform', poly: rect(598, 952, 662, 998), accept: ['push', 'tool'] },
      { id: 'camera', label: 'Camera trap', poly: rect(455, 1055, 506, 1112), accept: ['tool'] },
      { id: 'seed', label: 'Seed', poly: rect(478, 985, 540, 1030), accept: ['tool'] }
    ],
    reset: () => ({ platform: false, camera: false, seed: false }),
    update(m, w) {
      if (!m.state.platform && w.hit(m, 'platform')) { m.state.platform = true; w.log(m, 'The research platform is raised.'); }
      if (!m.state.camera && w.hit(m, 'camera')) { m.state.camera = true; w.log(m, 'The camera trap is deployed.'); }
      if (!m.state.seed && w.hit(m, 'seed')) {
        m.state.seed = true;
        w.carry({ kind: 'seed', from: 'M09', label: 'Seed' });
        w.log(m, 'The seed was taken from the tree (the robot carries it).');
      }
    },
    score: (m) => [
      { label: 'Research platform raised', points: m.state.platform ? 10 : 0 },
      { label: 'Camera trap deployed', points: m.state.camera ? 10 : 0 },
      { label: 'Seed no longer touching the tree', points: m.state.seed ? 10 : 0 }
    ],
    noEquipmentModels: ['m08-tree']
  });

  // ---------------------------------------------------------------- M10
  const SPIDER = [[1014, 719], [1026, 707], [1054, 702], [1065, 713], [1065, 725], [1060, 730], [1048, 741], [1025, 730]];
  const SNAIL = [[729, 1051], [730, 1036], [751, 1024], [764, 1031], [768, 1037], [769, 1055], [766, 1066], [759, 1070], [748, 1067], [733, 1058]];
  def({
    id: 10, name: 'Fragile Microhabitats', color: '#f5f5f5',
    description: 'Keep the spider habitat and the snail habitat in their original positions (10 each).',
    simHint: 'Do not touch these two small models. They start with 20 points – touching them knocks them over.',
    models: (m) => [
      { id: 'm10-spider', label: 'Spider', poly: SPIDER, solid: false, fragile: 'spider', fill: m && m.state && m.state.spider ? '#9e9e9e' : '#fafafa' },
      { id: 'm10-snail', label: 'Snail', poly: SNAIL, solid: false, fragile: 'snail', fill: m && m.state && m.state.snail ? '#9e9e9e' : '#ffd54f' }
    ],
    zones: () => [
      { id: 'spider', label: 'Avoid!', poly: geom.expandPoly(SPIDER, 3), accept: ['touch'], avoid: true },
      { id: 'snail', label: 'Avoid!', poly: geom.expandPoly(SNAIL, 3), accept: ['touch'], avoid: true }
    ],
    reset: () => ({ spider: false, snail: false }),
    update(m, w) {
      if (!m.state.spider && w.touching(m, 'spider')) { m.state.spider = true; w.log(m, 'The spider habitat was knocked over!'); }
      if (!m.state.snail && w.touching(m, 'snail')) { m.state.snail = true; w.log(m, 'The snail habitat was knocked over!'); }
    },
    score: (m) => [
      { label: 'Spider habitat in original position', points: m.state.spider ? 0 : 10 },
      { label: 'Snail habitat in original position', points: m.state.snail ? 0 : 10 }
    ]
  });

  // ---------------------------------------------------------------- M11
  const WINDOW = [[1351, 592], [1405, 538], [1461, 595], [1407, 649]];
  def({
    id: 11, name: 'Window to the Past', color: '#6b3f1f',
    description: 'Open the root window: the root cover is down, touching the mat (20).',
    simHint: 'Push or use a tool on the south-east side of the model.',
    models: () => [{ id: 'm11-window', label: 'Window', poly: WINDOW, solid: true, fill: '#7b4a27' }],
    zones: () => [{ id: 'cover', label: 'Root cover', poly: [[1405, 538], [1461, 595], [1484, 572], [1428, 515]], accept: ['tool', 'push'] }],
    reset: () => ({ cover: false }),
    update(m, w) {
      if (!m.state.cover && w.hit(m, 'cover')) { m.state.cover = true; w.log(m, 'The root cover is down.'); }
    },
    score: (m) => [{ label: 'Root cover down, touching the mat', points: m.state.cover ? 20 : 0 }]
  });

  // ---------------------------------------------------------------- M12
  const POST = [[1932, 576], [1924, 540], [1924, 516], [1940, 516], [1948, 532], [1948, 556], [1940, 576]];
  def({
    id: 12, name: 'Forest Elder', color: '#c9b48a',
    description: 'Raise the cane so it touches the tree (20). Put the support tie around the post (10).',
    simHint: 'Cane: tool at the cane end south-west of the tree. Tie: tool on the west side of the tree picks up the tie, then a tool action at the post places it.',
    models: () => [
      { id: 'm12-tree', label: 'Elder', circle: [1936, 756, 32], solid: true, fill: '#c9b48a' },
      { id: 'm12-post', label: 'Post', poly: POST, solid: true, fill: '#9e9e9e' }
    ],
    zones: () => [
      { id: 'cane', label: 'Raise cane', poly: rect(1822, 656, 1882, 716), accept: ['tool'] },
      { id: 'tie', label: 'Take tie', poly: rect(1868, 735, 1904, 795), accept: ['tool'] },
      { id: 'post', label: 'Place tie', poly: geom.expandPoly(POST, 26), accept: ['tool'] }
    ],
    reset: () => ({ cane: false, tieTaken: false, tie: false }),
    update(m, w) {
      if (!m.state.cane && w.hit(m, 'cane')) { m.state.cane = true; w.log(m, 'The cane is raised against the tree.'); }
      if (!m.state.tieTaken && w.hit(m, 'tie')) {
        m.state.tieTaken = true;
        w.carry({ kind: 'tie', from: 'M12', label: 'Support tie' });
        w.log(m, 'The robot took the support tie.');
      }
      if (!m.state.tie && w.hit(m, 'post') && w.deliver('tie')) {
        m.state.tie = true;
        w.log(m, 'The support tie is around the post.');
      }
    },
    score: (m) => [
      { label: 'Cane completely raised, touching the tree', points: m.state.cane ? 20 : 0 },
      { label: 'Support tie around the post', points: m.state.tie ? 10 : 0 }
    ],
    draw(ctx, m, v) {
      if (!m.state.cane) { ctx.strokeStyle = '#8d6e63'; ctx.lineWidth = Math.max(1.5, 5 * v.scale); ctx.beginPath(); v.moveTo(ctx, 1912, 742); v.lineTo(ctx, 1852, 686); ctx.stroke(); }
      if (!m.state.tieTaken) v.drawCircle(ctx, 1897, 768, 9, null, '#e53935', 3);
      if (m.state.tie) v.drawCircle(ctx, 1936, 546, 20, null, '#e53935', 3);
    }
  });

  // ---------------------------------------------------------------- M13-M15 (interchangeable models on docks)
  function dockOf(id, setup) {
    for (const d of ['mine', 'city', 'farm']) if (setup.docks[d] === id) return d;
    return null;
  }

  def({
    id: 13, name: 'Keystone Species', color: '#2f7d32', dockModel: true,
    description: 'Deliver your keystone species onto the restoration platform and raise the young trees (30).',
    simHint: 'Launch with the keystone species loaded (Launch options). Use a tool at the rail side (marked) of the dock to place it, and push the far end of the model to raise the trees.',
    models: (m, setup) => {
      const d = dockOf(13, setup);
      return d ? [{ id: 'm13-model', label: 'Keystone', poly: field.dockPoly(d), solid: true, fill: '#2f7d32' }] : [];
    },
    zones: (m, setup) => {
      const d = dockOf(13, setup);
      if (!d) return [];
      return [
        { id: 'platform', label: 'Place keystone', poly: dockZone(d, -96, -46, -44, 46), accept: ['tool'] },
        { id: 'trees', label: 'Raise trees', poly: dockZone(d, -40, 68, 40, 108), accept: ['push', 'tool'] }
      ];
    },
    reset: () => ({ keystone: false, trees: false }),
    update(m, w) {
      if (!m.state.keystone && w.hit(m, 'platform') && w.deliver('keystone')) { m.state.keystone = true; w.log(m, 'The keystone species is on the restoration platform.'); }
      if (!m.state.trees && w.hit(m, 'trees')) { m.state.trees = true; w.log(m, 'The young trees are raised.'); }
    },
    score: (m) => [{ label: 'Keystone species on platform AND young trees raised', points: m.state.keystone && m.state.trees ? 30 : 0 }]
  });

  def({
    id: 14, name: 'Seeds of Renewal', color: '#7b4a2a', dockModel: true,
    description: 'Place collected seeds in the replantation station: 5 each, +5 each if they touch the mat.',
    simHint: 'Collect seeds from Missions 2 and 9 (tool action on a seed), then use a tool at the rail side of the station to plant all carried seeds.',
    models: (m, setup) => {
      const d = dockOf(14, setup);
      return d ? [{ id: 'm14-model', label: 'Station', poly: field.dockPoly(d), solid: true, fill: '#7b4a2a' }] : [];
    },
    zones: (m, setup) => {
      const d = dockOf(14, setup);
      return d ? [{ id: 'station', label: 'Plant seeds', poly: dockZone(d, -96, -46, -44, 46), accept: ['tool'] }] : [];
    },
    reset: () => ({ seeds: 0 }),
    update(m, w) {
      if (w.hit(m, 'station')) {
        let n = 0;
        while (w.deliver('seed')) n++;
        if (n) { m.state.seeds += n; w.log(m, n + ' seed' + (n > 1 ? 's' : '') + ' planted in the replantation station.'); }
      }
    },
    score: (m) => [
      { label: 'Seeds in the replantation station (' + m.state.seeds + ')', points: 5 * m.state.seeds },
      { label: 'Bonus: seeds touching the mat', points: 5 * m.state.seeds }
    ]
  });

  def({
    id: 15, name: 'Biocentric Architecture', color: '#8bc34a', dockModel: true,
    description: 'Raise the nesting canopy (10), slide the garden skylight in (10), open the compost hatch (10). Environmental bonus +10 for the dock\'s greatest need (mine: canopy, city: skylight, farm: hatch).',
    simHint: 'Canopy: tool at the rail side. Skylight: push the far end. Hatch: tool at the side opposite the rail.',
    models: (m, setup) => {
      const d = dockOf(15, setup);
      return d ? [{ id: 'm15-model', label: 'Building', poly: field.dockPoly(d), solid: true, fill: '#9ccc65' }] : [];
    },
    zones: (m, setup) => {
      const d = dockOf(15, setup);
      if (!d) return [];
      return [
        { id: 'canopy', label: 'Raise canopy', poly: dockZone(d, -96, -46, -44, 46), accept: ['tool'] },
        { id: 'skylight', label: 'Skylight in', poly: dockZone(d, -40, 68, 40, 108), accept: ['push', 'tool'] },
        { id: 'hatch', label: 'Open hatch', poly: dockZone(d, 44, -46, 96, 46), accept: ['tool'] }
      ];
    },
    reset: () => ({ canopy: false, skylight: false, hatch: false }),
    update(m, w) {
      if (!m.state.canopy && w.hit(m, 'canopy')) { m.state.canopy = true; w.log(m, 'The nesting canopy is raised.'); }
      if (!m.state.skylight && w.hit(m, 'skylight')) { m.state.skylight = true; w.log(m, 'The garden skylight is in.'); }
      if (!m.state.hatch && w.hit(m, 'hatch')) { m.state.hatch = true; w.log(m, 'The compost hatch is open.'); }
    },
    score(m, setup) {
      const d = dockOf(15, setup);
      const need = { mine: 'canopy', city: 'skylight', farm: 'hatch' }[d];
      return [
        { label: 'Nesting canopy raised', points: m.state.canopy ? 10 : 0 },
        { label: 'Garden skylight completely in', points: m.state.skylight ? 10 : 0 },
        { label: 'Compost hatch opened, touching the mat', points: m.state.hatch ? 10 : 0 },
        { label: 'Environmental bonus (' + (d || '-') + ' dock: ' + (need || '-') + ')', points: need && m.state[need] ? 10 : 0 }
      ];
    }
  });

  // ---------------------------------------------------------------- precision tokens
  const TOKEN_POINTS = [0, 10, 15, 25, 35, 50, 50];

  FLL.missions = {
    list: MISSIONS,
    NO_EQUIPMENT,
    TOKEN_POINTS,
    dockOf
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
