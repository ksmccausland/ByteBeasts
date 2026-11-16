/*
 * ZIP reading/writing and SPIKE App 3 (.llsp3) project import/export.
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};

  // ---------------------------------------------------------------- ZIP
  async function inflateRaw(data) {
    if (typeof DecompressionStream === 'undefined') throw new Error('This browser cannot unzip files (DecompressionStream missing).');
    const ds = new DecompressionStream('deflate-raw');
    const stream = new Blob([data]).stream().pipeThrough(ds);
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  async function unzip(buffer) {
    const u8 = new Uint8Array(buffer);
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    let eocd = -1;
    for (let i = u8.length - 22; i >= Math.max(0, u8.length - 65557); i--) {
      if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
    }
    if (eocd < 0) throw new Error('Not a ZIP file');
    const count = dv.getUint16(eocd + 10, true);
    let p = dv.getUint32(eocd + 16, true);
    const files = new Map();
    const dec = new TextDecoder();
    for (let n = 0; n < count; n++) {
      if (dv.getUint32(p, true) !== 0x02014b50) throw new Error('Corrupt ZIP directory');
      const method = dv.getUint16(p + 10, true);
      const csize = dv.getUint32(p + 20, true);
      const nameLen = dv.getUint16(p + 28, true);
      const extraLen = dv.getUint16(p + 30, true);
      const commentLen = dv.getUint16(p + 32, true);
      const local = dv.getUint32(p + 42, true);
      const name = dec.decode(u8.subarray(p + 46, p + 46 + nameLen));
      p += 46 + nameLen + extraLen + commentLen;
      const lNameLen = dv.getUint16(local + 26, true);
      const lExtraLen = dv.getUint16(local + 28, true);
      const start = local + 30 + lNameLen + lExtraLen;
      const raw = u8.subarray(start, start + csize);
      if (name.endsWith('/')) continue;
      if (method === 0) files.set(name, raw.slice());
      else if (method === 8) files.set(name, await inflateRaw(raw));
      else throw new Error('Unsupported ZIP compression in ' + name);
    }
    return files;
  }

  const CRC_TABLE = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
      t[n] = c >>> 0;
    }
    return t;
  })();
  function crc32(u8) {
    let c = 0xFFFFFFFF;
    for (let i = 0; i < u8.length; i++) c = CRC_TABLE[(c ^ u8[i]) & 0xFF] ^ (c >>> 8);
    return (c ^ 0xFFFFFFFF) >>> 0;
  }

  // Build a ZIP (stored, no compression). files: [{name, data: Uint8Array|string}]
  function zip(files) {
    const enc = new TextEncoder();
    const parts = [];
    const central = [];
    let offset = 0;
    for (const f of files) {
      const data = typeof f.data === 'string' ? enc.encode(f.data) : f.data;
      const name = enc.encode(f.name);
      const crc = crc32(data);
      const lh = new DataView(new ArrayBuffer(30));
      lh.setUint32(0, 0x04034b50, true); lh.setUint16(4, 20, true); lh.setUint16(6, 0, true); lh.setUint16(8, 0, true);
      lh.setUint16(10, 0, true); lh.setUint16(12, 0x21, true); lh.setUint32(14, crc, true);
      lh.setUint32(18, data.length, true); lh.setUint32(22, data.length, true); lh.setUint16(26, name.length, true); lh.setUint16(28, 0, true);
      parts.push(new Uint8Array(lh.buffer), name, data);
      const ch = new DataView(new ArrayBuffer(46));
      ch.setUint32(0, 0x02014b50, true); ch.setUint16(4, 20, true); ch.setUint16(6, 20, true); ch.setUint16(8, 0, true);
      ch.setUint16(10, 0, true); ch.setUint16(12, 0, true); ch.setUint16(14, 0x21, true); ch.setUint32(16, crc, true);
      ch.setUint32(20, data.length, true); ch.setUint32(24, data.length, true); ch.setUint16(28, name.length, true);
      ch.setUint16(30, 0, true); ch.setUint16(32, 0, true); ch.setUint16(34, 0, true); ch.setUint16(36, 0, true);
      ch.setUint32(38, 0, true); ch.setUint32(42, offset, true);
      central.push(new Uint8Array(ch.buffer), name);
      offset += 30 + name.length + data.length;
    }
    const cdSize = central.reduce((a, b) => a + b.length, 0);
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true); end.setUint16(8, files.length, true); end.setUint16(10, files.length, true);
    end.setUint32(12, cdSize, true); end.setUint32(16, offset, true);
    return new Blob(parts.concat(central, [new Uint8Array(end.buffer)]), { type: 'application/zip' });
  }

  // ---------------------------------------------------------------- Scratch -> Blockly
  const DEFS = {};
  function defsIndex() {
    if (Object.keys(DEFS).length) return DEFS;
    for (const d of FLL.blocks.BLOCK_DEFS) {
      const info = { fields: {}, values: {}, statements: {}, output: d.output !== undefined };
      for (const key of Object.keys(d)) {
        if (!/^args\d+$/.test(key)) continue;
        for (const a of d[key]) {
          if (a.type === 'input_value') info.values[a.name] = true;
          else if (a.type === 'input_statement') info.statements[a.name] = true;
          else if (a.type && a.type.indexOf('field_') === 0) info.fields[a.name] = a;
        }
      }
      DEFS[d.type] = info;
    }
    return DEFS;
  }

  function fieldAccepts(fieldDef, value) {
    if (!fieldDef || fieldDef.type !== 'field_dropdown') return true;
    const opts = typeof fieldDef.options === 'function' ? fieldDef.options() : fieldDef.options;
    return opts.some(o => o[1] === value);
  }

  const PROC_NAME = (proccode) => proccode.replace(/%[snb]/g, ' ').replace(/\s+/g, ' ').trim() || 'my block';

  function convertScratchProject(project) {
    const defs = defsIndex();
    const target = project.targets.find(t => !t.isStage) || project.targets[0];
    const stage = project.targets.find(t => t.isStage);
    const blocks = target.blocks || {};
    const variables = [];
    const varIds = new Set();
    const addVar = (id, name) => { if (!varIds.has(id)) { varIds.add(id); variables.push({ id: id, name: name }); } };
    for (const t of [stage, target]) {
      if (!t || !t.variables) continue;
      for (const id of Object.keys(t.variables)) addVar(id, t.variables[id][0]);
    }
    const warnings = new Set();
    const procs = {}; // proccode -> {name, argNames, argIds}
    const paramVarId = (name) => {
      const v = variables.find(x => x.name === name);
      if (v) return v.id;
      const id = 'param_' + name.replace(/\W/g, '_') + '_' + variables.length;
      addVar(id, name);
      return id;
    };
    for (const id of Object.keys(blocks)) {
      const b = blocks[id];
      if (b && b.opcode === 'procedures_prototype' && b.mutation) {
        const argNames = JSON.parse(b.mutation.argumentnames || '[]');
        const argIds = JSON.parse(b.mutation.argumentids || '[]');
        procs[b.mutation.proccode] = { name: PROC_NAME(b.mutation.proccode), argNames: argNames, argIds: argIds };
        argNames.forEach(paramVarId);
      }
    }

    const primitive = (p) => {
      if (!Array.isArray(p)) return null;
      const code = p[0];
      if (code >= 4 && code <= 8) {
        const n = parseFloat(p[1]);
        return { type: 'math_number', fields: { NUM: isNaN(n) ? 0 : n } };
      }
      if (code === 10) {
        const s = String(p[1]);
        if (s !== '' && !isNaN(Number(s))) return { type: 'math_number', fields: { NUM: Number(s) } };
        return { type: 'text', fields: { TEXT: s } };
      }
      if (code === 9) return { type: 'text', fields: { TEXT: String(p[1]) } };
      return null;
    };
    const menuValue = (inp) => {
      if (!inp) return null;
      const ref = inp[1];
      if (Array.isArray(ref)) return String(ref[1]);
      const sb = blocks[ref];
      if (!sb || !sb.fields) return null;
      const keys = Object.keys(sb.fields);
      return keys.length ? String(sb.fields[keys[0]][0]) : null;
    };
    const convertValue = (inp) => {
      if (!inp) return null;
      const kind = inp[0];
      const out = {};
      const shadowSpec = kind === 1 ? inp[1] : inp[2];
      const sh = primitive(shadowSpec);
      if (sh) out.shadow = sh;
      if (kind !== 1) {
        const ref = inp[1];
        if (Array.isArray(ref)) {
          if (ref[0] === 12) out.block = { type: 'variables_get', fields: { VAR: { id: ref[2] } } };
          else { const p2 = primitive(ref); if (p2) out.shadow = p2; }
        } else if (ref) {
          const blk = convertBlock(ref, true);
          if (blk) out.block = blk;
        }
      } else if (!sh && inp[1] && !Array.isArray(inp[1])) {
        // a shadow block that is not a primitive: treat as a reporter
        const blk = convertBlock(inp[1], true);
        if (blk) out.block = blk;
      }
      return (out.shadow || out.block) ? out : null;
    };
    const chain = (id) => {
      if (!id) return null;
      return convertBlock(id, false);
    };

    function convertBlock(id, asValue) {
      const b = blocks[id];
      if (!b) return null;
      const op = b.opcode;
      let out = null;
      switch (op) {
        case 'procedures_definition': {
          const proto = blocks[b.inputs.custom_block && b.inputs.custom_block[1]];
          const pc = proto && proto.mutation ? procs[proto.mutation.proccode] : null;
          out = {
            type: 'procedures_defnoreturn',
            fields: { NAME: pc ? pc.name : 'my block' },
            extraState: { params: (pc ? pc.argNames : []).map(n => ({ name: n, id: paramVarId(n) })) }
          };
          const body = chain(b.next);
          if (body) out.inputs = { STACK: { block: body } };
          return out; // body consumed the chain
        }
        case 'procedures_call': {
          const pc = procs[b.mutation && b.mutation.proccode];
          out = { type: 'procedures_callnoreturn', extraState: { name: pc ? pc.name : PROC_NAME(b.mutation ? b.mutation.proccode : ''), params: pc ? pc.argNames : [] } };
          if (pc) {
            out.inputs = {};
            pc.argIds.forEach((aid, i) => {
              const v = convertValue(b.inputs[aid]);
              if (v) out.inputs['ARG' + i] = v;
            });
          }
          break;
        }
        case 'argument_reporter_string_number':
        case 'argument_reporter_boolean':
          return { type: 'variables_get', fields: { VAR: { id: paramVarId(b.fields.VALUE[0]) } } };
        case 'data_setvariableto':
        case 'data_changevariableby': {
          const vref = b.fields.VARIABLE;
          addVar(vref[1], vref[0]);
          out = { type: op === 'data_setvariableto' ? 'variables_set' : 'data_changevariableby', fields: { VAR: { id: vref[1] } } };
          const v = convertValue(b.inputs.VALUE);
          if (v) out.inputs = { VALUE: v };
          break;
        }
        case 'data_variable':
          addVar(b.fields.VARIABLE[1], b.fields.VARIABLE[0]);
          return { type: 'variables_get', fields: { VAR: { id: b.fields.VARIABLE[1] } } };
        case 'control_stop': {
          const opt = b.fields.STOP_OPTION ? b.fields.STOP_OPTION[0] : 'all';
          out = { type: 'flippercontrol_stop', fields: { STOP_OPTION: opt === 'all' ? 'all' : 'this script' } };
          break;
        }
        default: {
          const def = defs[op];
          if (!def) {
            warnings.add(op);
            out = asValue ? { type: 'unsupported_reporter', fields: { LABEL: op } } : { type: 'unsupported_block', fields: { LABEL: op } };
            break;
          }
          out = { type: op, fields: {}, inputs: {} };
          for (const name of Object.keys(b.inputs || {})) {
            const inp = b.inputs[name];
            if (def.fields[name]) {
              const val = menuValue(inp);
              if (val !== null && fieldAccepts(def.fields[name], val)) out.fields[name] = val;
              else if (val !== null) warnings.add(op + ' (' + name + '=' + val + ')');
            } else if (def.statements[name]) {
              const body = chain(inp[1]);
              if (body) out.inputs[name] = { block: body };
            } else if (def.values[name]) {
              const v = convertValue(inp);
              if (v) out.inputs[name] = v;
            }
          }
          for (const name of Object.keys(b.fields || {})) {
            if (def.fields[name]) {
              const val = String(b.fields[name][0]);
              if (fieldAccepts(def.fields[name], val)) out.fields[name] = val;
            }
          }
        }
      }
      if (out && !asValue && b.next) {
        const nx = chain(b.next);
        if (nx) out.next = { block: nx };
      }
      return out;
    }

    const top = [];
    for (const id of Object.keys(blocks)) {
      const b = blocks[id];
      if (!b || Array.isArray(b) || !b.topLevel || b.shadow) continue;
      const blk = convertBlock(id, false);
      if (!blk) continue;
      blk.x = Math.round((b.x || 0) * 1.0);
      blk.y = Math.round((b.y || 0) * 1.0);
      top.push(blk);
    }
    return {
      workspace: { blocks: { languageVersion: 0, blocks: top }, variables: variables },
      warnings: Array.from(warnings)
    };
  }

  // ---------------------------------------------------------------- .llsp3
  async function readLlsp3(buffer) {
    const files = await unzip(buffer);
    const dec = new TextDecoder();
    const manifest = files.has('manifest.json') ? JSON.parse(dec.decode(files.get('manifest.json'))) : {};
    const name = manifest.name || 'SPIKE project';
    if (manifest.type === 'python' || files.has('projectbody.json')) {
      let code = '';
      if (files.has('projectbody.json')) {
        const body = JSON.parse(dec.decode(files.get('projectbody.json')));
        code = body.main || '';
      }
      return { name: name, mode: 'python', python: code };
    }
    const sb3 = files.get('scratch.sb3');
    if (!sb3) throw new Error('This .llsp3 file has no word-block or Python program inside.');
    const inner = await unzip(sb3.buffer.slice(sb3.byteOffset, sb3.byteOffset + sb3.byteLength));
    const project = JSON.parse(dec.decode(inner.get('project.json')));
    const conv = convertScratchProject(project);
    return { name: name, mode: 'blocks', blocks: conv.workspace, warnings: conv.warnings };
  }

  function writeLlsp3Python(name, code) {
    const now = new Date().toISOString();
    const manifest = {
      type: 'python', appType: 'llsp3', autoDelete: false, created: now, lastsaved: now,
      id: Math.random().toString(36).slice(2, 14), name: name, slotIndex: 0, size: code.length,
      workspaceX: 120, workspaceY: 120, zoomLevel: 0.5, showAllBlocks: false, version: 38,
      hardware: {}, state: { canvasDrawerOpen: false }, extraFiles: [], lastConnectedHubType: 'flipper'
    };
    const icon = '<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60"><rect width="60" height="60" rx="8" fill="#ffd400"/><text x="30" y="38" font-size="20" text-anchor="middle" font-family="sans-serif">py</text></svg>';
    return zip([
      { name: 'manifest.json', data: JSON.stringify(manifest) },
      { name: 'projectbody.json', data: JSON.stringify({ main: code }) },
      { name: 'icon.svg', data: icon }
    ]);
  }

  FLL.files = { unzip, zip, crc32, readLlsp3, writeLlsp3Python, convertScratchProject };
})(typeof globalThis !== 'undefined' ? globalThis : window);
