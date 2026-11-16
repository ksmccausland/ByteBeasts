/*
 * BIOGLOW field geometry (FIRST LEGO League Challenge 2026/27).
 * Coordinates in millimetres with the origin in the bottom-left corner of the mat
 * (grid cell A1), x to the right (east) and y up (north, towards the back wall).
 * Measured from the official "Wireframe & Path Diagram" (20 cm grid).
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};
  const geom = FLL.geom;

  const MAT_W = 2000, MAT_H = 1140;
  const TABLE = { x0: -181, x1: 2181, y0: 0, y1: 1143 };
  const LAUNCH_R = 487; // outer edge of the launch-area arc (lines count as "in")

  const field = {
    MAT_W, MAT_H, TABLE, LAUNCH_R,
    wallThickness: 38,
    gridSize: 200,
    cols: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    rows: ['1', '2', '3', '4', '5', '6'],

    launchAreas: {
      left: { name: 'Left launch area', cx: 0, cy: 0, r: LAUNCH_R, color: '#e32b2b' },
      right: { name: 'Right launch area', cx: MAT_W, cy: 0, r: LAUNCH_R, color: '#1f6fd1' }
    },

    // Interchangeable model docks (missions 13-15). Local frame: x = short side
    // (88 mm, rail on -x), y = long side (136 mm). angle = rotation of the frame (degrees, CCW).
    docks: {
      mine: { name: 'Mine dock', x: 1885.5, y: 1031.5, angle: -45 },
      city: { name: 'City dock', x: 1279, y: 93, angle: 0 },
      farm: { name: 'Farm dock', x: 1000, y: 626, angle: 0 }
    },
    dockSize: { w: 88, h: 136 },

    // Thick black lines printed on the mat (useful for line following / squaring).
    blackLines: [
      { name: 'North-west line', pts: [[470, 815], [497, 864], [519, 893], [752, 893]] },
      { name: 'North-east line', pts: [[1335, 968], [1438, 876], [1680, 876]] },
      { name: 'Centre line', pts: [[946, 418], [1006, 309]] }
    ],

    // Printed marked areas used by missions.
    surveyArea: geom.rectPoly(1057.6, 220.2, 60, 52, geom.rad(30)),
    leafHabitat: [[105, 955], [186, 955], [186, 1035], [142, 1035], [105, 998]]
  };

  // Static obstacles: the four table walls.
  field.walls = [
    { name: 'south wall', poly: [[TABLE.x0 - 50, TABLE.y0 - 50], [TABLE.x1 + 50, TABLE.y0 - 50], [TABLE.x1 + 50, TABLE.y0], [TABLE.x0 - 50, TABLE.y0]] },
    { name: 'north wall', poly: [[TABLE.x0 - 50, TABLE.y1], [TABLE.x1 + 50, TABLE.y1], [TABLE.x1 + 50, TABLE.y1 + 50], [TABLE.x0 - 50, TABLE.y1 + 50]] },
    { name: 'west wall', poly: [[TABLE.x0 - 50, TABLE.y0 - 50], [TABLE.x0, TABLE.y0 - 50], [TABLE.x0, TABLE.y1 + 50], [TABLE.x0 - 50, TABLE.y1 + 50]] },
    { name: 'east wall', poly: [[TABLE.x1, TABLE.y0 - 50], [TABLE.x1 + 50, TABLE.y0 - 50], [TABLE.x1 + 50, TABLE.y1 + 50], [TABLE.x1, TABLE.y1 + 50]] }
  ];

  field.dockFrame = (dockId) => {
    const d = field.docks[dockId];
    return { x: d.x, y: d.y, angle: geom.rad(d.angle) };
  };
  field.dockPoly = (dockId) => geom.placePoly(geom.rectPoly(0, 0, field.dockSize.w, field.dockSize.h, 0), field.dockFrame(dockId));

  // Is the point inside the launch area (quarter disc incl. its line)?
  field.inLaunchArea = (side, p) => {
    const a = field.launchAreas[side];
    const dx = p[0] - a.cx, dy = p[1] - a.cy;
    if (p[1] < -1) return false;
    if (side === 'left' && p[0] < -1) return false;
    if (side === 'right' && p[0] > MAT_W + 1) return false;
    return dx * dx + dy * dy <= a.r * a.r;
  };
  // Home = launch area + the storage strip beside the mat on that side.
  field.inHome = (side, p) => {
    if (p[1] < TABLE.y0 - 1 || p[1] > TABLE.y1 + 1) return false;
    if (side === 'left' && p[0] <= 0 && p[0] >= TABLE.x0 - 1) return true;
    if (side === 'right' && p[0] >= MAT_W && p[0] <= TABLE.x1 + 1) return true;
    return field.inLaunchArea(side, p);
  };
  field.polyInLaunchArea = (side, poly) => geom.outlineSamples(poly, 8).every(p => field.inLaunchArea(side, p));
  field.polyInHome = (side, poly) => geom.outlineSamples(poly, 8).every(p => field.inHome(side, p));
  field.polyPartlyInHome = (side, poly) => geom.outlineSamples(poly, 8).some(p => field.inHome(side, p));
  field.homeOf = (poly) => {
    if (field.polyInHome('left', poly)) return 'left';
    if (field.polyInHome('right', poly)) return 'right';
    return null;
  };

  // Grid cell name for a point, e.g. "D3".
  field.cellName = (x, y) => {
    if (x < 0 || x >= MAT_W || y < 0 || y >= MAT_H) return '';
    const c = field.cols[Math.min(9, Math.floor(x / 200))];
    const r = field.rows[Math.min(5, Math.floor(y / 200))];
    return c + r;
  };

  FLL.field = field;
})(typeof globalThis !== 'undefined' ? globalThis : window);
