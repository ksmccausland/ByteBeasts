/*
 * Python tokenizer for the FLL simulator's Python interpreter.
 * Produces NAME, NUMBER, STRING, OP, NEWLINE, INDENT, DEDENT and EOF tokens.
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};
  const py = FLL.py = FLL.py || {};

  const KEYWORDS = new Set([
    'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break',
    'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for',
    'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or',
    'pass', 'raise', 'return', 'try', 'while', 'with', 'yield'
  ]);

  // Longest operators first so that greedy matching works.
  const OPERATORS = [
    '**=', '//=', '>>=', '<<=', '...', '->', ':=',
    '**', '//', '<<', '>>', '<=', '>=', '==', '!=', '+=', '-=', '*=', '/=', '%=',
    '&=', '|=', '^=', '@=',
    '+', '-', '*', '/', '%', '@', '&', '|', '^', '~', '<', '>', '(', ')', '[', ']',
    '{', '}', ',', ':', '.', ';', '='
  ];

  class SyntaxErr extends Error {
    constructor(msg, line, col, kind) {
      super(msg);
      this.line = line;
      this.col = col;
      this.kind = kind || 'SyntaxError';
    }
  }

  function isIdStart(ch) {
    return /[A-Za-z_À-￿]/.test(ch);
  }
  function isIdChar(ch) {
    return /[A-Za-z0-9_À-￿]/.test(ch);
  }

  // Decode the escapes of a (non-raw) string literal body.
  function decodeEscapes(body, line) {
    let out = '';
    for (let i = 0; i < body.length; i++) {
      const c = body[i];
      if (c !== '\\') { out += c; continue; }
      i++;
      if (i >= body.length) { out += '\\'; break; }
      const e = body[i];
      switch (e) {
        case '\n': break; // line continuation inside string
        case '\\': out += '\\'; break;
        case '\'': out += '\''; break;
        case '"': out += '"'; break;
        case 'n': out += '\n'; break;
        case 't': out += '\t'; break;
        case 'r': out += '\r'; break;
        case 'a': out += '\x07'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'v': out += '\v'; break;
        case 'x': {
          const hex = body.substr(i + 1, 2);
          if (!/^[0-9a-fA-F]{2}$/.test(hex)) throw new SyntaxErr('invalid \\x escape', line, 0);
          out += String.fromCharCode(parseInt(hex, 16));
          i += 2;
          break;
        }
        case 'u': {
          const hex = body.substr(i + 1, 4);
          if (!/^[0-9a-fA-F]{4}$/.test(hex)) throw new SyntaxErr('invalid \\u escape', line, 0);
          out += String.fromCharCode(parseInt(hex, 16));
          i += 4;
          break;
        }
        case 'U': {
          const hex = body.substr(i + 1, 8);
          if (!/^[0-9a-fA-F]{8}$/.test(hex)) throw new SyntaxErr('invalid \\U escape', line, 0);
          out += String.fromCodePoint(parseInt(hex, 16));
          i += 8;
          break;
        }
        default:
          if (/[0-7]/.test(e)) {
            let oct = e;
            while (oct.length < 3 && /[0-7]/.test(body[i + 1] || '')) oct += body[++i];
            out += String.fromCharCode(parseInt(oct, 8));
          } else {
            out += '\\' + e; // unknown escapes are kept as-is (like Python)
          }
      }
    }
    return out;
  }

  function tokenize(src) {
    src = src.replace(/\r\n?/g, '\n');
    if (!src.endsWith('\n')) src += '\n';
    const tokens = [];
    const indents = [0];
    let i = 0;
    let line = 1;
    let lineStart = 0;
    let depth = 0;        // bracket nesting depth
    let atLineStart = true;
    const n = src.length;

    function push(type, value, l, c) {
      tokens.push({ type: type, value: value, line: l, col: c });
    }

    while (i < n) {
      if (atLineStart) {
        // Measure indentation of a new logical line.
        let col = 0;
        let j = i;
        while (j < n && (src[j] === ' ' || src[j] === '\t' || src[j] === '\f')) {
          if (src[j] === '\t') col = (Math.floor(col / 8) + 1) * 8;
          else if (src[j] === ' ') col++;
          j++;
        }
        // Blank lines and comment-only lines do not affect indentation.
        if (src[j] === '\n' || src[j] === '#' || (src[j] === '\\' && src[j + 1] === '\n')) {
          if (src[j] === '#') { while (j < n && src[j] !== '\n') j++; }
          if (src[j] === '\\') { j += 2; line++; lineStart = j; i = j; continue; }
          i = j + 1;
          line++;
          lineStart = i;
          continue;
        }
        if (j >= n) { i = j; break; }
        const cur = indents[indents.length - 1];
        if (col > cur) {
          indents.push(col);
          push('INDENT', col, line, col);
        } else if (col < cur) {
          while (indents.length > 1 && col < indents[indents.length - 1]) {
            indents.pop();
            push('DEDENT', col, line, col);
          }
          if (col !== indents[indents.length - 1]) {
            throw new SyntaxErr('unindent does not match any outer indentation level', line, col, 'IndentationError');
          }
        }
        i = j;
        atLineStart = false;
        continue;
      }

      const ch = src[i];
      const col = i - lineStart;

      if (ch === '\n') {
        if (depth === 0) {
          const last = tokens[tokens.length - 1];
          if (last && last.type !== 'NEWLINE') push('NEWLINE', '\n', line, col);
          atLineStart = true;
        }
        i++;
        line++;
        lineStart = i;
        continue;
      }
      if (ch === ' ' || ch === '\t' || ch === '\f') { i++; continue; }
      if (ch === '#') {
        while (i < n && src[i] !== '\n') i++;
        continue;
      }
      if (ch === '\\') {
        if (src[i + 1] === '\n') { i += 2; line++; lineStart = i; continue; }
        throw new SyntaxErr('unexpected character after line continuation character', line, col);
      }

      // Strings (with optional prefixes r, b, f, u and combinations).
      let m = /^([rRbBfFuU]{0,2})('''|"""|'|")/.exec(src.substr(i, 5));
      if (m && (m[1] === '' || /^(r|b|f|u|rb|br|fr|rf)$/i.test(m[1]))) {
        const prefix = m[1].toLowerCase();
        const quote = m[2];
        const startLine = line;
        let j = i + m[0].length;
        let body = '';
        let closed = false;
        while (j < n) {
          const c = src[j];
          if (c === '\\') {
            body += c + (src[j + 1] || '');
            if (src[j + 1] === '\n') { line++; lineStart = j + 2; }
            j += 2;
            continue;
          }
          if (src.startsWith(quote, j)) { j += quote.length; closed = true; break; }
          if (c === '\n') {
            if (quote.length === 1) break;
            line++;
            lineStart = j + 1;
          }
          body += c;
          j++;
        }
        if (!closed) {
          throw new SyntaxErr(quote.length === 3 ? 'unterminated triple-quoted string literal' : 'unterminated string literal', startLine, col);
        }
        const raw = prefix.indexOf('r') >= 0;
        const isF = prefix.indexOf('f') >= 0;
        const isB = prefix.indexOf('b') >= 0;
        tokens.push({
          type: 'STRING',
          value: isF ? body : (raw ? body : decodeEscapes(body, startLine)),
          fstring: isF,
          raw: raw,
          bytes: isB,
          line: startLine,
          col: col
        });
        i = j;
        continue;
      }

      // Numbers.
      if (/[0-9]/.test(ch) || (ch === '.' && /[0-9]/.test(src[i + 1] || ''))) {
        const rest = src.substr(i, 80);
        let mm = /^0[xX][0-9a-fA-F_]+/.exec(rest) || /^0[oO][0-7_]+/.exec(rest) || /^0[bB][01_]+/.exec(rest);
        if (mm) {
          const t = mm[0].replace(/_/g, '');
          const base = { x: 16, o: 8, b: 2 }[t[1].toLowerCase()];
          push('NUMBER', { kind: 'int', value: parseInt(t.slice(2), base) }, line, col);
          i += mm[0].length;
          continue;
        }
        mm = /^(?:[0-9][0-9_]*)?(?:\.[0-9_]*)?(?:[eE][+-]?[0-9_]+)?[jJ]?/.exec(rest);
        const text = mm[0];
        const clean = text.replace(/_/g, '');
        if (/[jJ]$/.test(clean)) throw new SyntaxErr('complex numbers are not supported', line, col);
        if (/[.eE]/.test(clean)) push('NUMBER', { kind: 'float', value: parseFloat(clean) }, line, col);
        else {
          if (/^0[0-9]+$/.test(clean) && !/^0+$/.test(clean)) {
            throw new SyntaxErr('leading zeros in decimal integer literals are not permitted', line, col);
          }
          push('NUMBER', { kind: 'int', value: parseInt(clean, 10) }, line, col);
        }
        i += text.length;
        if (isIdStart(src[i] || '')) throw new SyntaxErr('invalid decimal literal', line, col);
        continue;
      }

      // Names and keywords.
      if (isIdStart(ch)) {
        let j = i + 1;
        while (j < n && isIdChar(src[j])) j++;
        const word = src.slice(i, j);
        push(KEYWORDS.has(word) ? 'KEYWORD' : 'NAME', word, line, col);
        i = j;
        continue;
      }

      // Operators and delimiters.
      let op = null;
      for (const o of OPERATORS) {
        if (src.startsWith(o, i)) { op = o; break; }
      }
      if (op) {
        if (op === '(' || op === '[' || op === '{') depth++;
        else if (op === ')' || op === ']' || op === '}') {
          depth--;
          if (depth < 0) throw new SyntaxErr("unmatched '" + op + "'", line, col);
        }
        push('OP', op, line, col);
        i += op.length;
        continue;
      }
      if (ch === '!' ) throw new SyntaxErr("invalid syntax ('!' must be followed by '=')", line, col);
      throw new SyntaxErr("invalid character '" + ch + "'", line, col);
    }

    if (depth > 0) {
      throw new SyntaxErr("'(' was never closed", line - 1, 0);
    }
    const last = tokens[tokens.length - 1];
    if (last && last.type !== 'NEWLINE' && last.type !== 'DEDENT') push('NEWLINE', '\n', line, 0);
    while (indents.length > 1) {
      indents.pop();
      push('DEDENT', 0, line, 0);
    }
    push('EOF', null, line, 0);
    return tokens;
  }

  py.tokenize = tokenize;
  py.SyntaxErr = SyntaxErr;
  py.decodeEscapes = decodeEscapes;
  py.KEYWORDS = KEYWORDS;
})(typeof globalThis !== 'undefined' ? globalThis : window);
