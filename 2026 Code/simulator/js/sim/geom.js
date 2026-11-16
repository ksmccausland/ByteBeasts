/*
 * 2D geometry helpers (millimetres, y axis pointing up / north).
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};

  const geom = {};

  geom.deg = (r) => r * 180 / Math.PI;
  geom.rad = (d) => d * Math.PI / 180;
  geom.clamp = (v, lo, hi) => (v < lo ? lo : (v > hi ? hi : v));
  // Normalize an angle in degrees to [-180, 180).
  geom.wrapDeg = (a) => {
    a = ((a + 180) % 360 + 360) % 360 - 180;
    return a;
  };
  geom.wrapRad = (a) => {
    a = ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    return a;
  };
  geom.dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

  // Transform a robot-local point (forward f, left l) to world coordinates.
  geom.localToWorld = (pose, f, l) => {
    const c = Math.cos(pose.theta), s = Math.sin(pose.theta);
    return [pose.x + f * c - l * s, pose.y + f * s + l * c];
  };
  geom.worldToLocal = (pose, x, y) => {
    const c = Math.cos(pose.theta), s = Math.sin(pose.theta);
    const dx = x - pose.x, dy = y - pose.y;
    return [dx * c + dy * s, -dx * s + dy * c];
  };

  // Rectangle polygon from centre, size and rotation (radians, CCW).
  geom.rectPoly = (cx, cy, w, h, angle) => {
    const c = Math.cos(angle || 0), s = Math.sin(angle || 0);
    const hw = w / 2, hh = h / 2;
    return [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]].map(([x, y]) => [cx + x * c - y * s, cy + x * s + y * c]);
  };
  // Polygon from a local polygon placed with a frame {x, y, angle}.
  geom.placePoly = (local, frame) => {
    const c = Math.cos(frame.angle || 0), s = Math.sin(frame.angle || 0);
    return local.map(([x, y]) => [frame.x + x * c - y * s, frame.y + x * s + y * c]);
  };
  geom.circlePoly = (cx, cy, r, n) => {
    n = n || 16;
    const out = [];
    for (let i = 0; i < n; i++) {
      const a = i / n * Math.PI * 2;
      out.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
    }
    return out;
  };
  geom.boundsOf = (poly) => {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const [x, y] of poly) {
      if (x < x0) x0 = x; if (x > x1) x1 = x;
      if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    return { x0, y0, x1, y1 };
  };
  geom.centroid = (poly) => {
    let x = 0, y = 0;
    for (const p of poly) { x += p[0]; y += p[1]; }
    return [x / poly.length, y / poly.length];
  };
  geom.expandPoly = (poly, margin) => {
    // Offset a convex polygon outward from its centroid (approximate, good for zones).
    const [cx, cy] = geom.centroid(poly);
    return poly.map(([x, y]) => {
      const dx = x - cx, dy = y - cy;
      const d = Math.hypot(dx, dy) || 1;
      return [x + dx / d * margin, y + dy / d * margin];
    });
  };

  geom.pointInPoly = (pt, poly) => {
    const x = pt[0], y = pt[1];
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
      if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) inside = !inside;
    }
    return inside;
  };

  // Separating axis test for two convex polygons.
  function project(poly, ax, ay) {
    let min = Infinity, max = -Infinity;
    for (const p of poly) {
      const d = p[0] * ax + p[1] * ay;
      if (d < min) min = d;
      if (d > max) max = d;
    }
    return [min, max];
  }
  function axesOf(poly, out) {
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i], b = poly[(i + 1) % poly.length];
      const ex = b[0] - a[0], ey = b[1] - a[1];
      const len = Math.hypot(ex, ey);
      if (len > 1e-9) out.push([-ey / len, ex / len]);
    }
    return out;
  }
  geom.convexOverlap = (A, B) => {
    const axes = axesOf(B, axesOf(A, []));
    for (const [ax, ay] of axes) {
      const pa = project(A, ax, ay), pb = project(B, ax, ay);
      if (pa[1] <= pb[0] || pb[1] <= pa[0]) return false;
    }
    return true;
  };
  // Penetration depth (minimum translation distance) between convex polygons, or 0.
  geom.convexPenetration = (A, B) => {
    const axes = axesOf(B, axesOf(A, []));
    let best = Infinity;
    for (const [ax, ay] of axes) {
      const pa = project(A, ax, ay), pb = project(B, ax, ay);
      const o = Math.min(pa[1], pb[1]) - Math.max(pa[0], pb[0]);
      if (o <= 0) return 0;
      if (o < best) best = o;
    }
    return best;
  };
  geom.circleOverlapPoly = (cx, cy, r, poly) => {
    if (geom.pointInPoly([cx, cy], poly)) return true;
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i], b = poly[(i + 1) % poly.length];
      if (geom.segPointDist(a, b, [cx, cy]) < r) return true;
    }
    return false;
  };
  geom.segPointDist = (a, b, p) => {
    const vx = b[0] - a[0], vy = b[1] - a[1];
    const len2 = vx * vx + vy * vy;
    let t = len2 > 0 ? ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / len2 : 0;
    t = t < 0 ? 0 : (t > 1 ? 1 : t);
    return Math.hypot(a[0] + t * vx - p[0], a[1] + t * vy - p[1]);
  };
  // Shape helpers: shape = {poly} or {circle: [cx, cy, r]}
  geom.shapeOverlapPoly = (shape, poly) => {
    if (shape.circle) return geom.circleOverlapPoly(shape.circle[0], shape.circle[1], shape.circle[2], poly);
    return geom.convexOverlap(shape.poly, poly);
  };
  geom.polyOverlap = (A, B) => geom.convexOverlap(A, B);

  // Ray (origin o, unit direction d) against segment ab: returns distance or Infinity.
  geom.raySegment = (o, d, a, b) => {
    const ex = b[0] - a[0], ey = b[1] - a[1];
    const den = d[0] * ey - d[1] * ex;
    if (Math.abs(den) < 1e-12) return Infinity;
    const ax = a[0] - o[0], ay = a[1] - o[1];
    const t = (ax * ey - ay * ex) / den;
    const u = (ax * d[1] - ay * d[0]) / den;
    if (t >= 0 && u >= 0 && u <= 1) return t;
    return Infinity;
  };
  geom.rayPoly = (o, d, poly) => {
    let best = Infinity;
    for (let i = 0; i < poly.length; i++) {
      const t = geom.raySegment(o, d, poly[i], poly[(i + 1) % poly.length]);
      if (t < best) best = t;
    }
    return best;
  };
  geom.rayCircle = (o, d, c, r) => {
    const fx = o[0] - c[0], fy = o[1] - c[1];
    const b = fx * d[0] + fy * d[1];
    const cc = fx * fx + fy * fy - r * r;
    const disc = b * b - cc;
    if (disc < 0) return Infinity;
    const s = Math.sqrt(disc);
    const t1 = -b - s, t2 = -b + s;
    if (t1 >= 0) return t1;
    if (t2 >= 0) return 0;
    return Infinity;
  };

  // Sample points along the outline of a polygon (for "completely inside" checks).
  geom.outlineSamples = (poly, step) => {
    step = step || 10;
    const out = [];
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i], b = poly[(i + 1) % poly.length];
      const n = Math.max(1, Math.ceil(geom.dist(a, b) / step));
      for (let k = 0; k < n; k++) out.push([a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n]);
    }
    return out;
  };

  // Seedable random generator with gaussian helper.
  geom.rng = (seed) => {
    let a = (seed >>> 0) || 1;
    const next = () => {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    next.gauss = () => {
      let u = 0, v = 0;
      while (u === 0) u = next();
      while (v === 0) v = next();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    return next;
  };

  FLL.geom = geom;
})(typeof globalThis !== 'undefined' ? globalThis : window);
