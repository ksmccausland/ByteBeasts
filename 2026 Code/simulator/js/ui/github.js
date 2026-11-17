/*
 * The ByteBeasts repository on GitHub (github.com/ksmccausland/ByteBeasts): the programs the GitHub menu offers.
 * The list of files comes from the GitHub API. A browser may ask it only 60 times an hour, so the list is kept for
 * 10 minutes, and the last saved list is shown when GitHub cannot be asked. The files come from raw.githubusercontent.com,
 * which has no limit. Both send CORS headers for any page, so this also works when the simulator is opened as a local file.
 * settingsOf / robotFor read the robot a program was written for (the settings at the top of the team's programs), so the
 * simulated robot can be set up like it.
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};
  const REPO = 'ksmccausland/ByteBeasts';
  const BRANCH = 'main';
  const CACHE_KEY = 'fll-bioglow-github-tree-v1';
  const MAX_AGE_MS = 10 * 60 * 1000;   // a saved list is used without asking GitHub again for this long
  const MAX_BYTES = 512 * 1024;        // the biggest program in the repository is about 150 KB
  // Not programs for the simulator: the simulator itself, and desktop programs with a camera or a window
  const SKIP = /(^|\/)(simulator|Captcha|SquareDetection)\/|label_lidar/;
  const OFFLINE = 'Could not reach GitHub. Check the internet connection (a firewall or an ad blocker can block it too).';

  function load() {
    try {
      const saved = JSON.parse(localStorage.getItem(CACHE_KEY));
      return saved && Array.isArray(saved.files) ? saved : null;
    } catch (e) { return null; }
  }
  function save(list) {
    try { localStorage.setItem(CACHE_KEY, JSON.stringify(list)); } catch (e) { /* storage full or blocked: the list is just not kept */ }
  }

  // The files of the repository: { time, truncated, files: [{ path, size }] }.
  // force = ask GitHub even when the saved list is fresh. If GitHub cannot be asked, the saved list is returned
  // with a problem message, so the menu can say why the list is old.
  async function list(force) {
    const saved = load();
    if (!force && saved && Date.now() - saved.time < MAX_AGE_MS) return saved;
    try {
      let res;
      try { res = await fetch('https://api.github.com/repos/' + REPO + '/git/trees/' + BRANCH + '?recursive=1'); }
      catch (e) { throw new Error(OFFLINE); }
      if (!res.ok) throw new Error(res.status === 403 || res.status === 429
        ? 'GitHub is refusing requests from this browser for now (it allows only a few an hour).'
        : 'GitHub answered with an error (HTTP ' + res.status + ').');
      const data = await res.json();
      if (!Array.isArray(data.tree)) throw new Error('GitHub sent an answer the simulator does not understand.');
      const fresh = { time: Date.now(), truncated: !!data.truncated, files: data.tree.filter(e => e.type === 'blob').map(e => ({ path: e.path, size: e.size })) };
      save(fresh);
      return fresh;
    } catch (e) {
      if (!saved) throw e;
      return Object.assign({}, saved, { problem: e.message });
    }
  }

  // SPIKE programs: .py, .llsp3 (SPIKE app projects) and SPIKE Python saved without an extension, as the older years have them.
  function isProgram(f) {
    const name = f.path.split('/').pop();
    if (f.size > MAX_BYTES || SKIP.test(f.path)) return false;
    if (/\.(py|llsp3)$/i.test(name)) return true;
    return !name.includes('.') && f.size >= 100;   // tiny files are placeholders
  }

  // The programs, newest year first, then by folder and name. Each one carries its folder, for the menu headings.
  function programs(files) {
    return files
      .filter(f => /^20\d\d Code\//.test(f.path) && isProgram(f))
      .map(f => {
        const parts = f.path.split('/');
        const name = parts.pop();
        return { path: f.path, name: name, size: f.size, year: parts[0], folder: parts.join(' / ') };
      })
      .sort((a, b) => b.year.localeCompare(a.year) || a.folder.localeCompare(b.folder) || a.name.localeCompare(b.name, undefined, { numeric: true }));
  }

  // The address of one file in the repository (each part of the path is encoded, the slashes stay).
  function rawUrl(path) {
    return 'https://raw.githubusercontent.com/' + REPO + '/' + BRANCH + '/' + path.split('/').map(p => encodeURIComponent(p)).join('/');
  }

  // The bytes of one file: a SPIKE project (a zip, or in a few cases plain text) or a Python file.
  async function fetchBytes(path) {
    let res;
    try { res = await fetch(rawUrl(path)); }
    catch (e) { throw new Error(OFFLINE); }
    if (!res.ok) throw new Error('GitHub answered with an error (HTTP ' + res.status + ') for this file.');
    return res.arrayBuffer();
  }

  // The robot a Python program was written for, read from its settings (the BaseFunctions settings at the top of the file,
  // or the 2025 programs' WHEEL_DIAMETER in cm). Anything the program does not say is null (or an empty list).
  // Only the four BaseFunctions motor names count as motors: COLOR_SENSOR = port.E is a sensor.
  function settingsOf(code) {
    const first = (re) => { const m = re.exec(code); return m ? m[1] : null; };
    const portOf = (name) => first(new RegExp('^[ \\t]*' + name + '[ \\t]*=[ \\t]*port\\.([A-F])\\b', 'm'));
    const s = { left: portOf('LEFT_DRIVE'), right: portOf('RIGHT_DRIVE'), attach: [], sensors: [], wheelMm: null, invert: null };
    if (!s.left || !s.right) {   // no settings: a program that pairs its wheel motors directly
      const m = /motor_pair\.pair\(\s*motor_pair\.PAIR_\d\s*,\s*port\.([A-F])\s*,\s*port\.([A-F])\s*\)/.exec(code);
      if (m) { s.left = m[1]; s.right = m[2]; }
    }
    for (const name of ['LEFT_ATTACH', 'RIGHT_ATTACH']) {
      const p = portOf(name);
      if (p) s.attach.push(p);
    }
    const sensorRe = /^[ \t]*[A-Z_]*(COLOR|DISTANCE|FORCE)[A-Z_]*[ \t]*=[ \t]*port\.([A-F])\b/gm;
    for (let m; (m = sensorRe.exec(code));) s.sensors.push({ port: m[2], type: m[1].toLowerCase() });
    const invert = first(/^[ \t]*MOTOR_INVERT[ \t]*=[ \t]*(-?1)\b/m);
    if (invert) s.invert = +invert;
    // WHEEL_DIAMETER_IN = 5.6 / 2.54 is centimetres divided by 2.54; without the division it is inches. WHEEL_DIAMETER = 6.24 is centimetres.
    const inch = /^[ \t]*WHEEL_DIAMETER_IN[ \t]*=[ \t]*(\d+(?:\.\d+)?)([ \t]*\/[ \t]*2\.54\b)?/m.exec(code);
    const cm = first(/^[ \t]*WHEEL_DIAMETER[ \t]*=[ \t]*(\d+(?:\.\d+)?)/m);
    const mm = inch ? (inch[2] ? +inch[1] * 10 : +inch[1] * 25.4) : (cm ? +cm * 10 : null);
    if (mm >= 20 && mm <= 150) s.wheelMm = Math.round(mm * 10) / 10;
    return s;
  }

  // The simulated robot for a program: a copy of the robot design `cfg` with the motors, sensors and wheel size the program
  // says it has, and what is different from `cfg` in plain words. `changes` is empty (and `cfg` null) when the robot already
  // fits, or when the program does not say which ports its wheels are on. A robot has at most 4 motors, so the motors
  // the program never mentions are taken off.
  function robotFor(code, cfg) {
    const s = settingsOf(code);
    const out = { settings: s, cfg: null, changes: [] };
    if (!s.left || !s.right) return out;
    const next = JSON.parse(JSON.stringify(cfg));
    const dev = next.devices;
    const typeOf = (p) => (dev[p] && dev[p].type) || 'none';
    const scan = (re, fn) => { for (let m; (m = re.exec(code));) fn(m); };
    const need = {};        // port -> the device the program has there
    const mentioned = {};   // every port the program names
    scan(/\bport\.([A-F])\b/g, m => { mentioned[m[1]] = true; });
    scan(/\bmotor\.[a-z_]+\(\s*port\.([A-F])\b/g, m => { need[m[1]] = 'motor'; });
    for (const p of [s.left, s.right].concat(s.attach)) need[p] = 'motor';
    for (const sn of s.sensors) if (!need[sn.port]) need[sn.port] = sn.type;

    const name = { color: 'color sensor', distance: 'distance sensor', force: 'force sensor' };
    const motors = [], sensors = [], removed = [];
    for (const p of FLL.robot.PORTS) {
      const have = typeOf(p), want = need[p];
      if (want === 'motor') {
        if (have !== 'motor') {
          dev[p] = { type: 'motor', kind: 'medium', mount: 'front', reach: 80, armLength: 90 };
          motors.push(p + (have === 'none' ? '' : ' (instead of its ' + name[have] + ')'));
        } else if (p !== s.left && p !== s.right && !dev[p].mount) {
          Object.assign(dev[p], { mount: 'front', reach: 80, armLength: 90 });   // was a wheel motor of the robot, is an attachment in the program
        }
      } else if (want) {
        if (have !== want) {
          dev[p] = want === 'color' ? { type: 'color', x: 92, y: 0 } : { type: want, x: next.body.front, y: 0 };
          if (want === 'distance') dev[p].angle = 0;
          sensors.push(p + ' (' + name[want] + ')');
        }
      } else if (have === 'motor' && !mentioned[p]) {
        dev[p] = { type: 'none' };
        removed.push(p);
      }
    }

    const list = (a) => a.length < 2 ? a.join('') : a.slice(0, -1).join(', ') + ' and ' + a[a.length - 1];
    if (s.left !== cfg.driveLeft || s.right !== cfg.driveRight) {
      next.driveLeft = s.left;
      next.driveRight = s.right;
      out.changes.push('Wheel motors: left ' + s.left + ', right ' + s.right + ' (now ' + cfg.driveLeft + ' and ' + cfg.driveRight + ')');
    }
    if (motors.length) out.changes.push('Motor' + (motors.length > 1 ? 's' : '') + ' added on port ' + list(motors));
    if (sensors.length) out.changes.push('Sensor' + (sensors.length > 1 ? 's' : '') + ' added on port ' + list(sensors));
    if (removed.length) out.changes.push('Motor' + (removed.length > 1 ? 's' : '') + ' on port ' + list(removed) + ' taken off (the program does not use ' + (removed.length > 1 ? 'them' : 'it') + ')');
    if (s.wheelMm && Math.abs(s.wheelMm - cfg.wheelDiameter) > 0.5) {
      next.wheelDiameter = s.wheelMm;
      out.changes.push('Wheel diameter ' + s.wheelMm + ' mm (now ' + cfg.wheelDiameter + ' mm)');
    }
    if (out.changes.length) out.cfg = next;
    return out;
  }

  FLL.github = { REPO: REPO, BRANCH: BRANCH, URL: 'https://github.com/' + REPO, list: list, programs: programs, rawUrl: rawUrl, fetchBytes: fetchBytes, settingsOf: settingsOf, robotFor: robotFor };
})(typeof globalThis !== 'undefined' ? globalThis : window);
