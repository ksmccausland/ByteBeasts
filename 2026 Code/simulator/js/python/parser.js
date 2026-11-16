/*
 * Recursive-descent parser for the Python subset used by the FLL simulator.
 * Produces a plain-object AST (node.type, node.line) and annotates functions
 * with scope information and expressions with a "pure" flag (no calls/awaits).
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};
  const py = FLL.py = FLL.py || {};
  const SyntaxErr = py.SyntaxErr;

  const AUG_OPS = {
    '+=': '+', '-=': '-', '*=': '*', '/=': '/', '//=': '//', '%=': '%', '**=': '**',
    '>>=': '>>', '<<=': '<<', '&=': '&', '|=': '|', '^=': '^', '@=': '@'
  };
  const COMPARE_OPS = new Set(['<', '>', '==', '>=', '<=', '!=']);

  class Parser {
    constructor(tokens) {
      this.t = tokens;
      this.p = 0;
    }
    peek(k) { return this.t[this.p + (k || 0)]; }
    next() { return this.t[this.p++]; }
    isOp(v, k) { const t = this.peek(k); return t.type === 'OP' && t.value === v; }
    isKw(v, k) { const t = this.peek(k); return t.type === 'KEYWORD' && t.value === v; }
    isType(type) { return this.peek().type === type; }
    eatOp(v) { if (this.isOp(v)) return this.next(); return null; }
    eatKw(v) { if (this.isKw(v)) return this.next(); return null; }
    error(msg, tok, kind) {
      tok = tok || this.peek();
      throw new SyntaxErr(msg, tok.line, tok.col, kind);
    }
    describe(tok) {
      if (tok.type === 'EOF') return 'end of program';
      if (tok.type === 'NEWLINE') return 'end of line';
      if (tok.type === 'INDENT') return 'indent';
      if (tok.type === 'DEDENT') return 'dedent';
      if (tok.type === 'STRING') return 'string';
      if (tok.type === 'NUMBER') return 'number';
      return "'" + tok.value + "'";
    }
    expectOp(v) {
      if (!this.isOp(v)) this.error("expected '" + v + "' but found " + this.describe(this.peek()));
      return this.next();
    }
    expectKw(v) {
      if (!this.isKw(v)) this.error("expected '" + v + "' but found " + this.describe(this.peek()));
      return this.next();
    }
    expectName() {
      const t = this.peek();
      if (t.type !== 'NAME') {
        if (t.type === 'KEYWORD') this.error("'" + t.value + "' is a reserved word and can't be used as a name");
        this.error('expected a name but found ' + this.describe(t));
      }
      return this.next().value;
    }
    expectNewline() {
      const t = this.peek();
      if (t.type === 'NEWLINE') { this.next(); return; }
      if (t.type === 'EOF') return;
      this.error('invalid syntax: unexpected ' + this.describe(t));
    }

    // ---------------------------------------------------------------- statements
    parseModule() {
      const body = [];
      while (!this.isType('EOF')) {
        if (this.isType('NEWLINE')) { this.next(); continue; }
        if (this.isType('INDENT')) this.error('unexpected indent', null, 'IndentationError');
        const s = this.parseStatement();
        for (const x of s) body.push(x);
      }
      return { type: 'Module', body: body, line: 1 };
    }

    parseStatement() {
      const t = this.peek();
      if (t.type === 'KEYWORD') {
        switch (t.value) {
          case 'if': return [this.parseIf()];
          case 'while': return [this.parseWhile()];
          case 'for': return [this.parseFor(false)];
          case 'try': return [this.parseTry()];
          case 'with': return [this.parseWith(false)];
          case 'def': return [this.parseDef([], false)];
          case 'class': return [this.parseClass([])];
          case 'async': {
            this.next();
            if (this.isKw('def')) return [this.parseDef([], true)];
            if (this.isKw('for')) return [this.parseFor(true)];
            if (this.isKw('with')) return [this.parseWith(true)];
            this.error("expected 'def' after 'async'");
          }
        }
      }
      if (t.type === 'OP' && t.value === '@') return [this.parseDecorated()];
      return this.parseSimpleStatements();
    }

    parseSimpleStatements() {
      const out = [this.parseSmall()];
      while (this.eatOp(';')) {
        if (this.isType('NEWLINE') || this.isType('EOF')) break;
        out.push(this.parseSmall());
      }
      this.expectNewline();
      return out;
    }

    parseSmall() {
      const t = this.peek();
      const line = t.line;
      if (t.type === 'KEYWORD') {
        switch (t.value) {
          case 'pass': this.next(); return { type: 'Pass', line: line };
          case 'break': this.next(); return { type: 'Break', line: line };
          case 'continue': this.next(); return { type: 'Continue', line: line };
          case 'return': {
            this.next();
            let value = null;
            if (this.canStartExpr()) value = this.parseTestListStarExpr();
            return { type: 'Return', value: value, line: line };
          }
          case 'raise': {
            this.next();
            let exc = null, cause = null;
            if (this.canStartExpr()) {
              exc = this.parseTest();
              if (this.eatKw('from')) cause = this.parseTest();
            }
            return { type: 'Raise', exc: exc, cause: cause, line: line };
          }
          case 'global':
          case 'nonlocal': {
            this.next();
            const names = [this.expectName()];
            while (this.eatOp(',')) names.push(this.expectName());
            return { type: t.value === 'global' ? 'Global' : 'Nonlocal', names: names, line: line };
          }
          case 'import': return this.parseImport();
          case 'from': return this.parseFromImport();
          case 'assert': {
            this.next();
            const test = this.parseTest();
            let msg = null;
            if (this.eatOp(',')) msg = this.parseTest();
            return { type: 'Assert', test: test, msg: msg, line: line };
          }
          case 'del': {
            this.next();
            const targets = [this.parseExpr()];
            while (this.eatOp(',')) {
              if (!this.canStartExpr()) break;
              targets.push(this.parseExpr());
            }
            for (const tg of targets) this.checkTarget(tg, 'delete');
            return { type: 'Delete', targets: targets, line: line };
          }
          case 'yield':
            this.error("'yield' (generators) is not supported in this simulator");
        }
      }
      // Expression statement or assignment.
      const first = this.parseTestListStarExpr();
      if (this.isOp('=')) {
        const targets = [first];
        let value = null;
        while (this.eatOp('=')) {
          if (this.isKw('yield')) this.error("'yield' is not supported in this simulator");
          const v = this.parseTestListStarExpr();
          targets.push(v);
        }
        value = targets.pop();
        for (const tg of targets) this.checkTarget(tg, 'assign');
        return { type: 'Assign', targets: targets, value: value, line: line };
      }
      const pt = this.peek();
      if (pt.type === 'OP' && AUG_OPS[pt.value]) {
        this.next();
        if (first.type !== 'Name' && first.type !== 'Attribute' && first.type !== 'Subscript') {
          this.error("'" + this.exprName(first) + "' is an illegal expression for augmented assignment", pt);
        }
        const value = this.parseTestList();
        return { type: 'AugAssign', target: first, op: AUG_OPS[pt.value], value: value, line: line };
      }
      if (pt.type === 'OP' && pt.value === ':') {
        // Annotated assignment: x: int = 5
        this.next();
        const annotation = this.parseTest();
        let value = null;
        if (this.eatOp('=')) value = this.parseTestListStarExpr();
        this.checkTarget(first, 'assign');
        if (!value) return { type: 'Pass', line: line };
        return { type: 'Assign', targets: [first], value: value, annotation: annotation, line: line };
      }
      if (first.type === 'Name' && (first.id === 'print' || first.id === 'exec') && this.canStartExpr()) {
        this.error("Missing parentheses in call to '" + first.id + "'. Did you mean " + first.id + '(...)?');
      }
      return { type: 'Expr', value: first, line: line };
    }

    exprName(node) {
      switch (node.type) {
        case 'Call': return 'function call';
        case 'Constant': return 'literal';
        case 'BinOp': case 'UnaryOp': return 'expression';
        case 'Compare': return 'comparison';
        default: return node.type.toLowerCase();
      }
    }

    checkTarget(node, what) {
      switch (node.type) {
        case 'Name': case 'Attribute': case 'Subscript': return;
        case 'Starred': this.checkTarget(node.value, what); return;
        case 'Tuple': case 'List':
          for (const e of node.elts) this.checkTarget(e, what);
          return;
        default: {
          const t = { line: node.line, col: node.col || 0 };
          let hint = '';
          if (node.type === 'Compare' && what === 'assign') hint = " Maybe you meant '==' instead of '='?";
          throw new SyntaxErr('cannot ' + what + ' to ' + this.exprName(node) + '.' + hint, t.line, t.col);
        }
      }
    }

    parseImport() {
      const line = this.next().line;
      const names = [];
      do {
        let name = this.expectName();
        while (this.eatOp('.')) name += '.' + this.expectName();
        let asname = null;
        if (this.eatKw('as')) asname = this.expectName();
        names.push({ name: name, asname: asname });
      } while (this.eatOp(','));
      return { type: 'Import', names: names, line: line };
    }

    parseFromImport() {
      const line = this.next().line;
      let level = 0;
      while (this.isOp('.') || this.isOp('...')) level += this.next().value.length;
      let module = '';
      if (this.isType('NAME')) {
        module = this.expectName();
        while (this.eatOp('.')) module += '.' + this.expectName();
      }
      this.expectKw('import');
      if (this.eatOp('*')) return { type: 'ImportFrom', module: module, names: '*', level: level, line: line };
      const paren = !!this.eatOp('(');
      const names = [];
      do {
        if (paren && this.isOp(')')) break;
        const name = this.expectName();
        let asname = null;
        if (this.eatKw('as')) asname = this.expectName();
        names.push({ name: name, asname: asname });
      } while (this.eatOp(','));
      if (paren) this.expectOp(')');
      return { type: 'ImportFrom', module: module, names: names, level: level, line: line };
    }

    parseBlock(what, headerTok) {
      if (this.isOp('=')) this.error("invalid syntax: use '==' to compare two values ('=' stores a value in a variable)");
      if (!this.isOp(':')) this.error("expected ':' at the end of the " + what.replace(/'/g, '') + ' but found ' + this.describe(this.peek()));
      this.next();
      if (this.isType('NEWLINE')) {
        this.next();
        if (!this.isType('INDENT')) {
          this.error("expected an indented block after '" + what + "' on line " + headerTok.line, null, 'IndentationError');
        }
        this.next();
        const body = [];
        while (!this.isType('DEDENT') && !this.isType('EOF')) {
          if (this.isType('NEWLINE')) { this.next(); continue; }
          const s = this.parseStatement();
          for (const x of s) body.push(x);
        }
        if (this.isType('DEDENT')) this.next();
        return body;
      }
      return this.parseSimpleStatements();
    }

    parseIf() {
      const tok = this.next();
      const test = this.parseNamedExprTest();
      const body = this.parseBlock(tok.value + ' statement', tok);
      let orelse = [];
      if (this.isKw('elif')) {
        orelse = [this.parseIf()];
      } else if (this.isKw('else')) {
        const et = this.next();
        orelse = this.parseBlock("'else' statement", et);
      }
      return { type: 'If', test: test, body: body, orelse: orelse, line: tok.line };
    }

    parseWhile() {
      const tok = this.next();
      const test = this.parseNamedExprTest();
      const body = this.parseBlock("'while' statement", tok);
      let orelse = [];
      if (this.isKw('else')) { const et = this.next(); orelse = this.parseBlock("'else' statement", et); }
      return { type: 'While', test: test, body: body, orelse: orelse, line: tok.line };
    }

    parseFor(isAsync) {
      const tok = this.next();
      const target = this.parseExprList();
      this.checkTarget(target, 'assign');
      this.expectKw('in');
      const iter = this.parseTestList();
      const body = this.parseBlock("'for' statement", tok);
      let orelse = [];
      if (this.isKw('else')) { const et = this.next(); orelse = this.parseBlock("'else' statement", et); }
      return { type: 'For', target: target, iter: iter, body: body, orelse: orelse, isAsync: isAsync, line: tok.line };
    }

    parseTry() {
      const tok = this.next();
      const body = this.parseBlock("'try' statement", tok);
      const handlers = [];
      let orelse = [], finalbody = [];
      while (this.isKw('except')) {
        const et = this.next();
        let typ = null, name = null;
        if (!this.isOp(':')) {
          typ = this.parseTest();
          if (this.isOp(',')) {
            const elts = [typ];
            while (this.eatOp(',')) elts.push(this.parseTest());
            typ = { type: 'Tuple', elts: elts, line: typ.line };
          }
          if (this.eatKw('as')) name = this.expectName();
        }
        const hbody = this.parseBlock("'except' statement", et);
        handlers.push({ excType: typ, name: name, body: hbody, line: et.line });
      }
      if (this.isKw('else')) {
        const et = this.next();
        orelse = this.parseBlock("'else' statement", et);
      }
      if (this.isKw('finally')) {
        const ft = this.next();
        finalbody = this.parseBlock("'finally' statement", ft);
      }
      if (!handlers.length && !finalbody.length) this.error("expected 'except' or 'finally' block");
      return { type: 'Try', body: body, handlers: handlers, orelse: orelse, finalbody: finalbody, line: tok.line };
    }

    parseWith(isAsync) {
      const tok = this.next();
      const items = [];
      do {
        const expr = this.parseTest();
        let target = null;
        if (this.eatKw('as')) { target = this.parseExpr(); this.checkTarget(target, 'assign'); }
        items.push({ expr: expr, target: target });
      } while (this.eatOp(','));
      const body = this.parseBlock("'with' statement", tok);
      return { type: 'With', items: items, body: body, isAsync: isAsync, line: tok.line };
    }

    parseDecorated() {
      const decorators = [];
      while (this.isOp('@')) {
        this.next();
        decorators.push(this.parseNamedExprTest());
        this.expectNewline();
      }
      if (this.isKw('def')) return this.parseDef(decorators, false);
      if (this.isKw('async')) { this.next(); return this.parseDef(decorators, true); }
      if (this.isKw('class')) return this.parseClass(decorators);
      this.error('expected a function or class definition after decorator');
    }

    parseParams(closer, allowAnnotations) {
      const params = [], kwonly = [];
      let vararg = null, kwarg = null, sawDefault = false, sawStar = false;
      while (!this.isOp(closer)) {
        if (this.eatOp('/')) { /* positional-only marker: ignored */ }
        else if (this.eatOp('**')) {
          kwarg = this.expectName();
          if (allowAnnotations && this.eatOp(':')) this.parseTest();
        } else if (this.eatOp('*')) {
          sawStar = true;
          if (this.isType('NAME')) {
            vararg = this.expectName();
            if (allowAnnotations && this.eatOp(':')) this.parseTest();
          }
        } else {
          const nameTok = this.peek();
          const name = this.expectName();
          if (allowAnnotations && this.eatOp(':')) this.parseTest();
          let def = null;
          if (this.eatOp('=')) { def = this.parseTest(); sawDefault = true; }
          else if (sawDefault && !sawStar) this.error('non-default argument follows default argument', nameTok);
          const all = params.concat(kwonly).map(p => p.name);
          if (all.indexOf(name) >= 0) this.error("duplicate argument '" + name + "' in function definition", nameTok);
          (sawStar ? kwonly : params).push({ name: name, default: def });
        }
        if (!this.eatOp(',')) break;
      }
      return { params: params, vararg: vararg, kwonly: kwonly, kwarg: kwarg };
    }

    parseDef(decorators, isAsync) {
      const tok = this.next(); // 'def'
      const name = this.expectName();
      this.expectOp('(');
      const args = this.parseParams(')', true);
      this.expectOp(')');
      if (this.eatOp('->')) this.parseTest();
      const body = this.parseBlock("function definition", tok);
      return {
        type: 'FunctionDef', name: name, args: args, body: body, isAsync: isAsync,
        decorators: decorators, line: tok.line
      };
    }

    parseClass(decorators) {
      const tok = this.next();
      const name = this.expectName();
      let bases = [];
      if (this.eatOp('(')) {
        while (!this.isOp(')')) {
          if (this.isType('NAME') && this.isOp('=', 1)) { // metaclass=... etc: ignored
            this.next(); this.next(); this.parseTest();
          } else bases.push(this.parseTest());
          if (!this.eatOp(',')) break;
        }
        this.expectOp(')');
      }
      const body = this.parseBlock('class definition', tok);
      return { type: 'ClassDef', name: name, bases: bases, body: body, decorators: decorators, line: tok.line };
    }

    // --------------------------------------------------------------- expressions
    canStartExpr() {
      const t = this.peek();
      if (t.type === 'NAME' || t.type === 'NUMBER' || t.type === 'STRING') return true;
      if (t.type === 'KEYWORD') return ['None', 'True', 'False', 'not', 'lambda', 'await'].indexOf(t.value) >= 0;
      if (t.type === 'OP') return ['(', '[', '{', '-', '+', '~', '*', '...'].indexOf(t.value) >= 0;
      return false;
    }

    parseTestList() {
      const first = this.parseTest();
      if (!this.isOp(',')) return first;
      const elts = [first];
      while (this.eatOp(',')) {
        if (!this.canStartExpr()) break;
        elts.push(this.parseTest());
      }
      return { type: 'Tuple', elts: elts, line: first.line, col: first.col };
    }

    parseTestOrStar() {
      if (this.isOp('*')) {
        const t = this.next();
        return { type: 'Starred', value: this.parseExpr(), line: t.line, col: t.col };
      }
      return this.parseNamedExprTest();
    }

    parseTestListStarExpr() {
      const first = this.parseTestOrStar();
      if (!this.isOp(',')) return first;
      const elts = [first];
      while (this.eatOp(',')) {
        if (!this.canStartExpr()) break;
        elts.push(this.parseTestOrStar());
      }
      return { type: 'Tuple', elts: elts, line: first.line, col: first.col };
    }

    parseExprList() {
      const one = () => {
        if (this.isOp('*')) {
          const t = this.next();
          return { type: 'Starred', value: this.parseExpr(), line: t.line, col: t.col };
        }
        return this.parseExpr();
      };
      const first = one();
      if (!this.isOp(',')) return first;
      const elts = [first];
      while (this.eatOp(',')) {
        if (!this.canStartExpr()) break;
        elts.push(one());
      }
      return { type: 'Tuple', elts: elts, line: first.line, col: first.col };
    }

    parseNamedExprTest() {
      const t = this.peek();
      if (t.type === 'NAME' && this.isOp(':=', 1)) {
        this.next(); this.next();
        const value = this.parseTest();
        return { type: 'NamedExpr', target: { type: 'Name', id: t.value, line: t.line, col: t.col }, value: value, line: t.line, col: t.col };
      }
      return this.parseTest();
    }

    parseTest() {
      if (this.isKw('lambda')) return this.parseLambda();
      const t = this.peek();
      const cond = this.parseOrTest();
      if (this.isKw('if')) {
        this.next();
        const test = this.parseOrTest();
        if (!this.isKw('else')) this.error("expected 'else' after 'if' expression");
        this.next();
        const orelse = this.parseTest();
        return { type: 'IfExp', test: test, body: cond, orelse: orelse, line: t.line, col: t.col };
      }
      return cond;
    }

    parseTestNoCond() {
      if (this.isKw('lambda')) return this.parseLambda();
      return this.parseOrTest();
    }

    parseLambda() {
      const t = this.next();
      const args = this.parseParams(':', false);
      this.expectOp(':');
      const body = this.parseTest();
      return { type: 'Lambda', args: args, body: body, line: t.line, col: t.col };
    }

    parseOrTest() {
      const t = this.peek();
      let left = this.parseAndTest();
      if (!this.isKw('or')) return left;
      const values = [left];
      while (this.eatKw('or')) values.push(this.parseAndTest());
      return { type: 'BoolOp', op: 'or', values: values, line: t.line, col: t.col };
    }

    parseAndTest() {
      const t = this.peek();
      let left = this.parseNotTest();
      if (!this.isKw('and')) return left;
      const values = [left];
      while (this.eatKw('and')) values.push(this.parseNotTest());
      return { type: 'BoolOp', op: 'and', values: values, line: t.line, col: t.col };
    }

    parseNotTest() {
      if (this.isKw('not')) {
        const t = this.next();
        return { type: 'UnaryOp', op: 'not', operand: this.parseNotTest(), line: t.line, col: t.col };
      }
      return this.parseComparison();
    }

    parseComparison() {
      const t = this.peek();
      const left = this.parseExpr();
      const ops = [], comparators = [];
      for (;;) {
        const p = this.peek();
        let op = null;
        if (p.type === 'OP' && COMPARE_OPS.has(p.value)) { this.next(); op = p.value; }
        else if (p.type === 'KEYWORD' && p.value === 'in') { this.next(); op = 'in'; }
        else if (p.type === 'KEYWORD' && p.value === 'not' && this.isKw('in', 1)) { this.next(); this.next(); op = 'not in'; }
        else if (p.type === 'KEYWORD' && p.value === 'is') {
          this.next();
          op = this.eatKw('not') ? 'is not' : 'is';
        }
        if (!op) break;
        ops.push(op);
        comparators.push(this.parseExpr());
      }
      if (!ops.length) return left;
      return { type: 'Compare', left: left, ops: ops, comparators: comparators, line: t.line, col: t.col };
    }

    parseBinLevel(next, ops) {
      const t = this.peek();
      let left = next.call(this);
      for (;;) {
        const p = this.peek();
        if (p.type === 'OP' && ops.indexOf(p.value) >= 0) {
          this.next();
          const right = next.call(this);
          left = { type: 'BinOp', op: p.value, left: left, right: right, line: t.line, col: t.col };
        } else break;
      }
      return left;
    }
    parseExpr() { return this.parseBinLevel(this.parseXor, ['|']); }
    parseXor() { return this.parseBinLevel(this.parseAndExpr, ['^']); }
    parseAndExpr() { return this.parseBinLevel(this.parseShift, ['&']); }
    parseShift() { return this.parseBinLevel(this.parseArith, ['<<', '>>']); }
    parseArith() { return this.parseBinLevel(this.parseTerm, ['+', '-']); }
    parseTerm() { return this.parseBinLevel(this.parseFactor, ['*', '/', '//', '%', '@']); }

    parseFactor() {
      const p = this.peek();
      if (p.type === 'OP' && (p.value === '-' || p.value === '+' || p.value === '~')) {
        this.next();
        const operand = this.parseFactor();
        // Fold negative numeric literals.
        if (p.value === '-' && operand.type === 'Constant' && (operand.kind === 'int' || operand.kind === 'float') && !operand.folded) {
          return { type: 'Constant', kind: operand.kind, value: -operand.value, folded: true, line: p.line, col: p.col };
        }
        return { type: 'UnaryOp', op: p.value, operand: operand, line: p.line, col: p.col };
      }
      return this.parsePower();
    }

    parsePower() {
      const t = this.peek();
      let base;
      if (this.isKw('await')) {
        this.next();
        base = { type: 'Await', value: this.parsePrimary(), line: t.line, col: t.col };
      } else base = this.parsePrimary();
      if (this.eatOp('**')) {
        const exp = this.parseFactor();
        return { type: 'BinOp', op: '**', left: base, right: exp, line: t.line, col: t.col };
      }
      return base;
    }

    parsePrimary() {
      let node = this.parseAtom();
      for (;;) {
        const p = this.peek();
        if (p.type !== 'OP') break;
        if (p.value === '(') {
          this.next();
          node = this.parseCall(node, p);
        } else if (p.value === '[') {
          this.next();
          const slice = this.parseSubscriptList();
          this.expectOp(']');
          node = { type: 'Subscript', value: node, slice: slice, line: p.line, col: p.col };
        } else if (p.value === '.') {
          this.next();
          const nt = this.peek();
          if (nt.type !== 'NAME' && nt.type !== 'KEYWORD') this.error('expected attribute name after "."');
          this.next();
          node = { type: 'Attribute', value: node, attr: nt.value, line: p.line, col: p.col };
        } else break;
      }
      return node;
    }

    parseCall(func, tok) {
      const args = [], keywords = [];
      let sawKw = false;
      while (!this.isOp(')')) {
        if (this.eatOp('*')) {
          args.push({ type: 'Starred', value: this.parseTest(), line: tok.line });
        } else if (this.eatOp('**')) {
          keywords.push({ name: null, value: this.parseTest() });
          sawKw = true;
        } else if (this.isType('NAME') && this.isOp('=', 1)) {
          const name = this.next().value;
          this.next();
          keywords.push({ name: name, value: this.parseTest() });
          sawKw = true;
        } else {
          const at = this.peek();
          let arg = this.parseNamedExprTest();
          if (this.isKw('for')) {
            arg = this.parseComprehension('GeneratorExp', arg, null, arg);
          }
          if (sawKw) this.error('positional argument follows keyword argument', at);
          args.push(arg);
        }
        if (!this.eatOp(',')) break;
      }
      if (!this.isOp(')')) this.error("expected ')' or ',' in function call but found " + this.describe(this.peek()));
      this.next();
      return { type: 'Call', func: func, args: args, keywords: keywords, line: tok.line, col: tok.col };
    }

    parseSubscriptList() {
      const one = () => {
        const t = this.peek();
        let lower = null, upper = null, step = null;
        if (!this.isOp(':')) {
          lower = this.parseTest();
          if (!this.isOp(':')) return lower;
        }
        this.expectOp(':');
        if (!this.isOp(']') && !this.isOp(',') && !this.isOp(':')) upper = this.parseTest();
        if (this.eatOp(':')) {
          if (!this.isOp(']') && !this.isOp(',')) step = this.parseTest();
        }
        return { type: 'Slice', lower: lower, upper: upper, step: step, line: t.line, col: t.col };
      };
      const first = one();
      if (!this.isOp(',')) return first;
      const elts = [first];
      while (this.eatOp(',')) {
        if (this.isOp(']')) break;
        elts.push(one());
      }
      return { type: 'Tuple', elts: elts, line: first.line, col: first.col };
    }

    parseComprehension(kind, elt, value, startTok) {
      const generators = [];
      while (this.isKw('for') || (this.isKw('async') && this.isKw('for', 1))) {
        let isAsync = false;
        if (this.eatKw('async')) isAsync = true;
        this.expectKw('for');
        const target = this.parseExprList();
        this.checkTarget(target, 'assign');
        this.expectKw('in');
        const iter = this.parseOrTest();
        const ifs = [];
        while (this.isKw('if')) {
          this.next();
          ifs.push(this.parseTestNoCond());
        }
        generators.push({ target: target, iter: iter, ifs: ifs, isAsync: isAsync });
      }
      const node = { type: kind, elt: elt, generators: generators, line: startTok.line, col: startTok.col };
      if (kind === 'DictComp') { node.key = elt; node.value = value; delete node.elt; }
      return node;
    }

    parseAtom() {
      const t = this.peek();
      switch (t.type) {
        case 'NAME':
          this.next();
          return { type: 'Name', id: t.value, line: t.line, col: t.col };
        case 'NUMBER':
          this.next();
          return { type: 'Constant', kind: t.value.kind, value: t.value.value, line: t.line, col: t.col };
        case 'STRING':
          return this.parseStrings();
        case 'KEYWORD':
          if (t.value === 'None') { this.next(); return { type: 'Constant', kind: 'None', value: null, line: t.line, col: t.col }; }
          if (t.value === 'True') { this.next(); return { type: 'Constant', kind: 'bool', value: true, line: t.line, col: t.col }; }
          if (t.value === 'False') { this.next(); return { type: 'Constant', kind: 'bool', value: false, line: t.line, col: t.col }; }
          if (t.value === 'yield') this.error("'yield' (generators) is not supported in this simulator");
          this.error('invalid syntax: unexpected ' + this.describe(t));
          break;
        case 'OP':
          if (t.value === '(') {
            this.next();
            if (this.eatOp(')')) return { type: 'Tuple', elts: [], line: t.line, col: t.col };
            if (this.isKw('yield')) this.error("'yield' (generators) is not supported in this simulator");
            const first = this.parseTestOrStar();
            if (this.isKw('for')) {
              const g = this.parseComprehension('GeneratorExp', first, null, t);
              this.expectOp(')');
              return g;
            }
            if (this.isOp(')')) {
              this.next();
              if (first.type === 'Starred') this.error('cannot use starred expression here', t);
              first.parenthesized = true;
              return first;
            }
            const elts = [first];
            while (this.eatOp(',')) {
              if (this.isOp(')')) break;
              elts.push(this.parseTestOrStar());
            }
            if (!this.isOp(')')) this.error("expected ')' but found " + this.describe(this.peek()));
            this.next();
            return { type: 'Tuple', elts: elts, line: t.line, col: t.col, parenthesized: true };
          }
          if (t.value === '[') {
            this.next();
            if (this.eatOp(']')) return { type: 'List', elts: [], line: t.line, col: t.col };
            const first = this.parseTestOrStar();
            if (this.isKw('for')) {
              const g = this.parseComprehension('ListComp', first, null, t);
              this.expectOp(']');
              return g;
            }
            const elts = [first];
            while (this.eatOp(',')) {
              if (this.isOp(']')) break;
              elts.push(this.parseTestOrStar());
            }
            if (!this.isOp(']')) this.error("expected ']' or ',' but found " + this.describe(this.peek()));
            this.next();
            return { type: 'List', elts: elts, line: t.line, col: t.col };
          }
          if (t.value === '{') {
            this.next();
            if (this.eatOp('}')) return { type: 'Dict', keys: [], values: [], line: t.line, col: t.col };
            if (this.eatOp('**')) {
              const keys = [null], values = [this.parseExpr()];
              while (this.eatOp(',')) {
                if (this.isOp('}')) break;
                if (this.eatOp('**')) { keys.push(null); values.push(this.parseExpr()); continue; }
                keys.push(this.parseTest()); this.expectOp(':'); values.push(this.parseTest());
              }
              this.expectOp('}');
              return { type: 'Dict', keys: keys, values: values, line: t.line, col: t.col };
            }
            const first = this.parseTestOrStar();
            if (this.eatOp(':')) {
              const v = this.parseTest();
              if (this.isKw('for')) {
                const g = this.parseComprehension('DictComp', first, v, t);
                this.expectOp('}');
                return g;
              }
              const keys = [first], values = [v];
              while (this.eatOp(',')) {
                if (this.isOp('}')) break;
                if (this.eatOp('**')) { keys.push(null); values.push(this.parseExpr()); continue; }
                keys.push(this.parseTest());
                this.expectOp(':');
                values.push(this.parseTest());
              }
              this.expectOp('}');
              return { type: 'Dict', keys: keys, values: values, line: t.line, col: t.col };
            }
            if (this.isKw('for')) {
              const g = this.parseComprehension('SetComp', first, null, t);
              this.expectOp('}');
              return g;
            }
            const elts = [first];
            while (this.eatOp(',')) {
              if (this.isOp('}')) break;
              elts.push(this.parseTestOrStar());
            }
            this.expectOp('}');
            return { type: 'Set', elts: elts, line: t.line, col: t.col };
          }
          if (t.value === '...') { this.next(); return { type: 'Constant', kind: 'ellipsis', value: null, line: t.line, col: t.col }; }
          break;
      }
      if (t.type === 'NEWLINE' || t.type === 'EOF') this.error('invalid syntax: the line ends too early', t);
      if (t.type === 'INDENT') this.error('unexpected indent', t, 'IndentationError');
      this.error('invalid syntax: unexpected ' + this.describe(t), t);
    }

    parseStrings() {
      const first = this.peek();
      const parts = [];
      let anyF = false, isBytes = false;
      while (this.isType('STRING')) {
        const s = this.next();
        if (s.bytes) isBytes = true;
        if (s.fstring) {
          anyF = true;
          for (const part of parseFString(s.value, s.raw, s.line)) parts.push(part);
        } else parts.push(s.value);
      }
      if (!anyF) {
        return { type: 'Constant', kind: isBytes ? 'bytes' : 'str', value: parts.join(''), line: first.line, col: first.col };
      }
      // Merge adjacent literal parts.
      const merged = [];
      for (const p of parts) {
        if (typeof p === 'string' && typeof merged[merged.length - 1] === 'string') merged[merged.length - 1] += p;
        else merged.push(p);
      }
      return { type: 'JoinedStr', parts: merged, line: first.line, col: first.col };
    }
  }

  // Split an f-string body into literal strings and {expression} parts.
  function parseFString(body, raw, line) {
    const parts = [];
    let lit = '';
    let i = 0;
    const flushLit = () => {
      if (lit) parts.push(raw ? lit : py.decodeEscapes(lit, line));
      lit = '';
    };
    while (i < body.length) {
      const c = body[i];
      if (c === '{') {
        if (body[i + 1] === '{') { lit += '{'; i += 2; continue; }
        flushLit();
        // Find the end of the expression, respecting brackets and strings.
        let j = i + 1, depth = 0, quote = null;
        let exprEnd = -1, convEnd = -1, specStart = -1;
        for (; j < body.length; j++) {
          const d = body[j];
          if (quote) { if (d === quote) quote = null; continue; }
          if (d === '\'' || d === '"') { quote = d; continue; }
          if (d === '(' || d === '[' || d === '{') { depth++; continue; }
          if ((d === ')' || d === ']' || d === '}') && depth > 0) { depth--; continue; }
          if (depth === 0) {
            if (d === '!' && body[j + 1] !== '=' && exprEnd < 0) { exprEnd = j; continue; }
            if (d === ':' && specStart < 0) {
              if (exprEnd < 0) exprEnd = j;
              convEnd = j;
              specStart = j + 1;
              // format spec: runs until the matching '}' (may contain nested {..})
              let k = j + 1, nd = 0;
              for (; k < body.length; k++) {
                if (body[k] === '{') nd++;
                else if (body[k] === '}') { if (nd === 0) break; nd--; }
              }
              j = k;
              break;
            }
            if (d === '}') break;
          }
        }
        if (j >= body.length) throw new SyntaxErr("f-string: expecting '}'", line, 0);
        if (exprEnd < 0) exprEnd = j;
        let exprText = body.slice(i + 1, exprEnd);
        let conversion = null;
        if (body[exprEnd] === '!') {
          const convText = body.slice(exprEnd + 1, convEnd >= 0 ? convEnd : j).trim();
          if (!/^[rsa]$/.test(convText)) throw new SyntaxErr("f-string: invalid conversion character: expected 's', 'r', or 'a'", line, 0);
          conversion = convText;
        }
        let spec = null;
        if (specStart >= 0) {
          spec = parseFString(body.slice(specStart, j), true, line);
        }
        let debugPrefix = null;
        const trimmed = exprText.replace(/\s+$/, '');
        if (/[^=!<>]=$/.test(trimmed) || trimmed === '=') {
          debugPrefix = exprText;
          exprText = trimmed.slice(0, -1);
          if (!conversion && !spec) conversion = 'r';
        }
        if (!exprText.trim()) throw new SyntaxErr('f-string: empty expression not allowed', line, 0);
        const tokens = py.tokenize('(' + exprText + ')');
        tokens.forEach(t => { t.line = line; });
        const p = new Parser(tokens);
        const expr = p.parseTestListStarExpr();
        if (debugPrefix !== null) parts.push(debugPrefix);
        parts.push({ expr: expr, conversion: conversion, spec: spec });
        i = j + 1;
        continue;
      }
      if (c === '}') {
        if (body[i + 1] === '}') { lit += '}'; i += 2; continue; }
        throw new SyntaxErr("f-string: single '}' is not allowed", line, 0);
      }
      lit += c;
      i++;
    }
    flushLit();
    return parts;
  }

  // ------------------------------------------------------------ annotations
  // Collect names bound in a scope (without descending into nested scopes).
  function collectTargets(node, out) {
    if (!node) return;
    switch (node.type) {
      case 'Name': out.add(node.id); break;
      case 'Tuple': case 'List': node.elts.forEach(e => collectTargets(e, out)); break;
      case 'Starred': collectTargets(node.value, out); break;
    }
  }

  function scanScope(stmts, assigned, globals, nonlocals) {
    for (const s of stmts) scanStmt(s, assigned, globals, nonlocals);
  }

  function scanStmt(s, assigned, globals, nonlocals) {
    switch (s.type) {
      case 'Assign': s.targets.forEach(t => collectTargets(t, assigned)); scanExprForWalrus(s.value, assigned); break;
      case 'AugAssign': collectTargets(s.target, assigned); break;
      case 'For': collectTargets(s.target, assigned); scanScope(s.body, assigned, globals, nonlocals); scanScope(s.orelse, assigned, globals, nonlocals); break;
      case 'While': scanExprForWalrus(s.test, assigned); scanScope(s.body, assigned, globals, nonlocals); scanScope(s.orelse, assigned, globals, nonlocals); break;
      case 'If': scanExprForWalrus(s.test, assigned); scanScope(s.body, assigned, globals, nonlocals); scanScope(s.orelse, assigned, globals, nonlocals); break;
      case 'Try':
        scanScope(s.body, assigned, globals, nonlocals);
        s.handlers.forEach(h => { if (h.name) assigned.add(h.name); scanScope(h.body, assigned, globals, nonlocals); });
        scanScope(s.orelse, assigned, globals, nonlocals);
        scanScope(s.finalbody, assigned, globals, nonlocals);
        break;
      case 'With': s.items.forEach(it => collectTargets(it.target, assigned)); scanScope(s.body, assigned, globals, nonlocals); break;
      case 'FunctionDef': case 'ClassDef': assigned.add(s.name); break;
      case 'Import': s.names.forEach(n => assigned.add(n.asname || n.name.split('.')[0])); break;
      case 'ImportFrom': if (s.names !== '*') s.names.forEach(n => assigned.add(n.asname || n.name)); break;
      case 'Global': s.names.forEach(n => globals.add(n)); break;
      case 'Nonlocal': s.names.forEach(n => nonlocals.add(n)); break;
      case 'Delete': s.targets.forEach(t => collectTargets(t, assigned)); break;
      case 'Expr': scanExprForWalrus(s.value, assigned); break;
    }
  }

  function scanExprForWalrus(e, assigned) {
    if (!e || typeof e !== 'object') return;
    if (e.type === 'NamedExpr') assigned.add(e.target.id);
    if (e.type === 'Lambda' || e.type === 'ListComp' || e.type === 'SetComp' || e.type === 'DictComp' || e.type === 'GeneratorExp') return;
    for (const k in e) {
      const v = e[k];
      if (k === 'line' || k === 'col') continue;
      if (Array.isArray(v)) v.forEach(x => scanExprForWalrus(x, assigned));
      else if (v && typeof v === 'object' && v.type) scanExprForWalrus(v, assigned);
    }
  }

  function makeScope(kind, body, args) {
    const assigned = new Set(), globals = new Set(), nonlocals = new Set();
    if (args) {
      args.params.forEach(p => assigned.add(p.name));
      args.kwonly.forEach(p => assigned.add(p.name));
      if (args.vararg) assigned.add(args.vararg);
      if (args.kwarg) assigned.add(args.kwarg);
    }
    scanScope(body, assigned, globals, nonlocals);
    const locals = new Set();
    assigned.forEach(n => { if (!globals.has(n) && !nonlocals.has(n)) locals.add(n); });
    return { kind: kind, locals: locals, globals: globals, nonlocals: nonlocals };
  }

  // Annotate: scopes for functions/classes/lambdas, purity for expressions.
  function annotate(node) {
    if (!node || typeof node !== 'object') return true;
    if (Array.isArray(node)) {
      let pure = true;
      for (const n of node) if (!annotate(n)) pure = false;
      return pure;
    }
    let pure = true;
    switch (node.type) {
      case 'Module':
        node.scope = makeScope('module', node.body, null);
        annotate(node.body);
        return true;
      case 'FunctionDef':
        node.scope = makeScope('function', node.body, node.args);
        annotateArgs(node.args);
        annotate(node.decorators);
        annotate(node.body);
        return true;
      case 'ClassDef':
        node.scope = makeScope('class', node.body, null);
        annotate(node.bases);
        annotate(node.decorators);
        annotate(node.body);
        return true;
      case 'Lambda': {
        const body = [{ type: 'Return', value: node.body, line: node.line }];
        node.scope = makeScope('function', body, node.args);
        annotateArgs(node.args);
        annotate(node.body);
        node.pure = true;
        return true;
      }
      case 'Call': case 'Await':
        pure = false;
        break;
    }
    for (const k in node) {
      if (k === 'line' || k === 'col' || k === 'scope' || k === 'pure') continue;
      const v = node[k];
      if (Array.isArray(v)) {
        for (const x of v) {
          if (x && typeof x === 'object') {
            if (x.type) { if (!annotate(x)) pure = false; }
            else {
              // keyword args / comprehension generators / f-string parts / handlers
              for (const kk in x) {
                const vv = x[kk];
                if (Array.isArray(vv)) { if (!annotate(vv)) pure = false; }
                else if (vv && typeof vv === 'object' && vv.type) { if (!annotate(vv)) pure = false; }
                else if (Array.isArray(vv)) { if (!annotate(vv)) pure = false; }
              }
            }
          }
        }
      } else if (v && typeof v === 'object' && v.type) {
        if (!annotate(v)) pure = false;
      }
    }
    if (node.type === 'JoinedStr') {
      for (const part of node.parts) {
        if (typeof part !== 'string') {
          if (!annotate(part.expr)) pure = false;
          if (part.spec) for (const sp of part.spec) if (typeof sp !== 'string' && !annotate(sp.expr)) pure = false;
        }
      }
    }
    node.pure = pure;
    return pure;
  }

  function annotateArgs(args) {
    args.params.forEach(p => annotate(p.default));
    args.kwonly.forEach(p => annotate(p.default));
  }

  // Report misplaced return/break/continue/await like CPython does at compile time.
  function findAwait(node) {
    if (!node || typeof node !== 'object') return null;
    if (Array.isArray(node)) {
      for (const x of node) { const r = findAwait(x); if (r) return r; }
      return null;
    }
    if (node.type === 'Await') return node;
    if (node.type === 'Lambda') return null;
    for (const k in node) {
      if (k === 'line' || k === 'col' || k === 'scope' || k === 'pure') continue;
      const v = node[k];
      if (v && typeof v === 'object') { const r = findAwait(v); if (r) return r; }
    }
    return null;
  }

  function validate(stmts, ctx) {
    for (const s of stmts) validateStmt(s, ctx);
  }

  function validateStmt(s, ctx) {
    const fail = (msg, node) => { throw new SyntaxErr(msg, (node || s).line, (node || s).col || 0); };
    const checkAwait = (expr) => {
      const a = findAwait(expr);
      if (a && !ctx.inAsync) fail("'await' outside async function", a);
    };
    switch (s.type) {
      case 'FunctionDef':
        checkAwait(s.decorators);
        s.args.params.forEach(p => checkAwait(p.default));
        s.args.kwonly.forEach(p => checkAwait(p.default));
        validate(s.body, { inFunc: true, inAsync: s.isAsync, inLoop: false });
        return;
      case 'ClassDef':
        checkAwait(s.bases);
        validate(s.body, { inFunc: false, inAsync: false, inLoop: false });
        return;
      case 'Return':
        if (!ctx.inFunc) fail("'return' outside function");
        checkAwait(s.value);
        return;
      case 'Break':
        if (!ctx.inLoop) fail("'break' outside loop");
        return;
      case 'Continue':
        if (!ctx.inLoop) fail("'continue' not properly in loop");
        return;
      case 'While':
        checkAwait(s.test);
        validate(s.body, Object.assign({}, ctx, { inLoop: true }));
        validate(s.orelse, ctx);
        return;
      case 'For':
        if (s.isAsync && !ctx.inAsync) fail("'async for' outside async function");
        checkAwait(s.iter);
        validate(s.body, Object.assign({}, ctx, { inLoop: true }));
        validate(s.orelse, ctx);
        return;
      case 'If':
        checkAwait(s.test);
        validate(s.body, ctx);
        validate(s.orelse, ctx);
        return;
      case 'Try':
        validate(s.body, ctx);
        s.handlers.forEach(h => { checkAwait(h.excType); validate(h.body, ctx); });
        validate(s.orelse, ctx);
        validate(s.finalbody, ctx);
        return;
      case 'With':
        s.items.forEach(it => checkAwait(it.expr));
        validate(s.body, ctx);
        return;
      default:
        checkAwait(s);
    }
  }

  function parse(source) {
    const tokens = py.tokenize(source);
    const parser = new Parser(tokens);
    const mod = parser.parseModule();
    annotate(mod);
    validate(mod.body, { inFunc: false, inAsync: false, inLoop: false });
    return mod;
  }

  py.parse = parse;
  py.Parser = Parser;
})(typeof globalThis !== 'undefined' ? globalThis : window);
