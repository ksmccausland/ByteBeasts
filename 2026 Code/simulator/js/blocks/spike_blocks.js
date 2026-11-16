/*
 * SPIKE-style word blocks for Blockly and a generator that turns them into
 * SPIKE App 3 Python (which runs in the simulator and on a real hub).
 * Block types and input names follow the SPIKE app (Scratch "flipper" opcodes)
 * so .llsp3 word-block projects can be imported.
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};

  const C = {
    motors: '#0090f5', movement: '#ff4ccd', light: '#9966ff', sound: '#cf63cf', events: '#ffbf00',
    control: '#ffab19', sensors: '#4cbfe6', operators: '#59c059', variables: '#ff8c1a', myblocks: '#ff6680', other: '#9e9e9e'
  };

  const PORTS = ['A', 'B', 'C', 'D', 'E', 'F'];
  const portOptions = () => {
    const out = PORTS.map(p => [p, p]);
    for (let i = 0; i < 6; i++) for (let j = i + 1; j < 6; j++) out.push([PORTS[i] + '+' + PORTS[j], PORTS[i] + PORTS[j]]);
    for (let i = 0; i < 6; i++) for (let j = i + 1; j < 6; j++) for (let k = j + 1; k < 6; k++) {
      out.push([PORTS[i] + '+' + PORTS[j] + '+' + PORTS[k], PORTS[i] + PORTS[j] + PORTS[k]]);
    }
    return out;
  };
  const singlePort = () => PORTS.map(p => [p, p]);
  const pairOptions = () => {
    const out = [];
    for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) if (i !== j) out.push([PORTS[i] + '+' + PORTS[j], PORTS[i] + PORTS[j]]);
    // put the common driving pairs first
    const first = ['CD', 'AB', 'AE', 'EF', 'BA', 'DC'];
    out.sort((a, b) => (first.indexOf(a[1]) < 0 ? 99 : first.indexOf(a[1])) - (first.indexOf(b[1]) < 0 ? 99 : first.indexOf(b[1])));
    return out;
  };
  const COLOR_OPTIONS = [['🔴 red', '9'], ['⚫ black', '0'], ['⚪ white', '10'], ['🟢 green', '6'], ['🟡 yellow', '7'], ['🔵 blue', '3'],
    ['🟠 orange', '8'], ['🟣 violet', '2'], ['💗 magenta', '1'], ['🩵 azure', '4'], ['turquoise', '5'], ['no color', '-1']];
  const COLOR_PY = { '-1': 'color.UNKNOWN', 0: 'color.BLACK', 1: 'color.MAGENTA', 2: 'color.PURPLE', 3: 'color.BLUE', 4: 'color.AZURE', 5: 'color.TURQUOISE', 6: 'color.GREEN', 7: 'color.YELLOW', 8: 'color.ORANGE', 9: 'color.RED', 10: 'color.WHITE' };
  const IMAGE_OPTIONS = [['❤ heart', 'HEART'], ['♡ small heart', 'HEART_SMALL'], ['☺ happy', 'HAPPY'], ['☹ sad', 'SAD'], ['😐 meh', 'MEH'],
    ['😠 angry', 'ANGRY'], ['😲 surprised', 'SURPRISED'], ['✔ yes', 'YES'], ['✖ no', 'NO'], ['↑ arrow N', 'ARROW_N'], ['→ arrow E', 'ARROW_E'],
    ['↓ arrow S', 'ARROW_S'], ['← arrow W', 'ARROW_W'], ['◆ diamond', 'DIAMOND'], ['■ square', 'SQUARE'], ['▲ triangle', 'TRIANGLE'],
    ['◎ target', 'TARGET'], ['⌂ house', 'HOUSE'], ['♜ chessboard', 'CHESSBOARD'], ['👻 ghost', 'GHOST'], ['☠ skull', 'SKULL']];

  const num = (name) => ({ type: 'input_value', name: name });
  const bool = (name) => ({ type: 'input_value', name: name, check: 'Boolean' });
  const dd = (name, options) => ({ type: 'field_dropdown', name: name, options: options });

  const BLOCKS = [
    // ------------------------------------------------------------ events
    { type: 'flipperevents_whenProgramStarts', message0: '▶ when program starts', nextStatement: null, colour: C.events, tooltip: 'Runs when the program starts.' },
    { type: 'flipperevents_whenButton', message0: 'when %1 button %2', args0: [dd('BUTTON', [['left', 'left'], ['right', 'right']]), dd('EVENT', [['pressed', 'pressed'], ['released', 'released']])], nextStatement: null, colour: C.events },
    { type: 'flipperevents_whenColor', message0: 'when %1 color is %2', args0: [dd('PORT', singlePort), dd('OPTION', COLOR_OPTIONS)], nextStatement: null, colour: C.events },
    { type: 'flipperevents_whenPressed', message0: 'when %1 is %2', args0: [dd('PORT', singlePort), dd('OPTION', [['pressed', 'pressed'], ['released', 'released']])], nextStatement: null, colour: C.events },
    { type: 'flipperevents_whenTimer', message0: 'when timer > %1', args0: [num('VALUE')], nextStatement: null, colour: C.events },
    { type: 'flipperevents_whenCondition', message0: 'when %1', args0: [bool('CONDITION')], nextStatement: null, colour: C.events },

    // ------------------------------------------------------------ motors
    { type: 'flippermotor_motorTurnForDirection', message0: '%1 run %2 for %3 %4', args0: [dd('PORT', portOptions), dd('DIRECTION', [['↻ clockwise', 'clockwise'], ['↺ counterclockwise', 'counterclockwise']]), num('VALUE'), dd('UNIT', [['rotations', 'rotations'], ['degrees', 'degrees'], ['seconds', 'seconds']])], previousStatement: null, nextStatement: null, colour: C.motors, inputsInline: true, tooltip: 'Turn a motor and wait until it is done.' },
    { type: 'flippermotor_motorGoDirectionToPosition', message0: '%1 go %2 to position %3', args0: [dd('PORT', portOptions), dd('DIRECTION', [['shortest path', 'shortest'], ['↻ clockwise', 'clockwise'], ['↺ counterclockwise', 'counterclockwise']]), num('POSITION')], previousStatement: null, nextStatement: null, colour: C.motors, inputsInline: true },
    { type: 'flippermotor_motorStartDirection', message0: '%1 start motor %2', args0: [dd('PORT', portOptions), dd('DIRECTION', [['↻ clockwise', 'clockwise'], ['↺ counterclockwise', 'counterclockwise']])], previousStatement: null, nextStatement: null, colour: C.motors, inputsInline: true },
    { type: 'flippermotor_motorStop', message0: '%1 stop motor', args0: [dd('PORT', portOptions)], previousStatement: null, nextStatement: null, colour: C.motors },
    { type: 'flippermotor_motorSetSpeed', message0: '%1 set speed to %2 %%', args0: [dd('PORT', portOptions), num('SPEED')], previousStatement: null, nextStatement: null, colour: C.motors, inputsInline: true },
    { type: 'flippermoremotor_motorSetDegreeCounted', message0: '%1 set relative position to %2', args0: [dd('PORT', portOptions), num('VALUE')], previousStatement: null, nextStatement: null, colour: C.motors, inputsInline: true },
    { type: 'flippermotor_absolutePosition', message0: '%1 position', args0: [dd('PORT', singlePort)], output: 'Number', colour: C.motors },
    { type: 'flippermoremotor_position', message0: '%1 relative position', args0: [dd('PORT', singlePort)], output: 'Number', colour: C.motors },
    { type: 'flippermotor_speed', message0: '%1 speed', args0: [dd('PORT', singlePort)], output: 'Number', colour: C.motors },

    // ------------------------------------------------------------ movement
    { type: 'flippermove_move', message0: 'move %1 for %2 %3', args0: [dd('DIRECTION', [['⬆ forward', 'forward'], ['⬇ back', 'back']]), num('VALUE'), dd('UNIT', [['cm', 'cm'], ['in', 'inches'], ['rotations', 'rotations'], ['degrees', 'degrees'], ['seconds', 'seconds']])], previousStatement: null, nextStatement: null, colour: C.movement, inputsInline: true, tooltip: 'Drive straight and wait until the move is done.' },
    { type: 'flippermove_steer', message0: 'move ↻ %1 for %2 %3', args0: [num('STEERING'), num('VALUE'), dd('UNIT', [['cm', 'cm'], ['in', 'inches'], ['rotations', 'rotations'], ['degrees', 'degrees'], ['seconds', 'seconds']])], previousStatement: null, nextStatement: null, colour: C.movement, inputsInline: true, tooltip: 'Steering -100..100: 0 = straight, 50 = pivot on one wheel, 100 = spin on the spot.' },
    { type: 'flippermove_startMove', message0: 'start moving %1', args0: [dd('DIRECTION', [['⬆ forward', 'forward'], ['⬇ back', 'back']])], previousStatement: null, nextStatement: null, colour: C.movement },
    { type: 'flippermove_startSteer', message0: 'start moving ↻ %1', args0: [num('STEERING')], previousStatement: null, nextStatement: null, colour: C.movement, inputsInline: true },
    { type: 'flippermove_stopMove', message0: 'stop moving', previousStatement: null, nextStatement: null, colour: C.movement },
    { type: 'flippermove_movementSpeed', message0: 'set movement speed to %1 %%', args0: [num('SPEED')], previousStatement: null, nextStatement: null, colour: C.movement, inputsInline: true },
    { type: 'flippermove_setMovementPair', message0: 'set movement motors to %1', args0: [dd('PAIR', pairOptions)], previousStatement: null, nextStatement: null, colour: C.movement },
    { type: 'flippermove_setDistance', message0: 'set 1 motor rotation to %1 %2', args0: [num('DISTANCE'), dd('UNIT', [['cm', 'cm'], ['in', 'inches']])], previousStatement: null, nextStatement: null, colour: C.movement, inputsInline: true },
    { type: 'flippermoremove_startDualSpeed', message0: 'start moving at %1 %% %2 %%', args0: [num('LEFT'), num('RIGHT')], previousStatement: null, nextStatement: null, colour: C.movement, inputsInline: true, tooltip: 'Tank drive: left and right wheel speed in %.' },
    { type: 'flippermoremove_moveDualSpeed', message0: 'move for %1 %2 at %3 %% %4 %%', args0: [num('VALUE'), dd('UNIT', [['cm', 'cm'], ['in', 'inches'], ['rotations', 'rotations'], ['degrees', 'degrees'], ['seconds', 'seconds']]), num('LEFT'), num('RIGHT')], previousStatement: null, nextStatement: null, colour: C.movement, inputsInline: true },

    // ------------------------------------------------------------ light
    { type: 'flipperlight_lightDisplayImageOnForTime', message0: 'turn on %1 for %2 seconds', args0: [dd('MATRIX', IMAGE_OPTIONS), num('VALUE')], previousStatement: null, nextStatement: null, colour: C.light, inputsInline: true },
    { type: 'flipperlight_lightDisplayImageOn', message0: 'turn on %1', args0: [dd('MATRIX', IMAGE_OPTIONS)], previousStatement: null, nextStatement: null, colour: C.light },
    { type: 'flipperlight_lightDisplayText', message0: 'write %1', args0: [num('TEXT')], previousStatement: null, nextStatement: null, colour: C.light, inputsInline: true },
    { type: 'flipperlight_lightDisplayOff', message0: 'turn off pixels', previousStatement: null, nextStatement: null, colour: C.light },
    { type: 'flipperlight_lightDisplaySetPixel', message0: 'set pixel %1 , %2 to %3 %%', args0: [num('X'), num('Y'), num('BRIGHTNESS')], previousStatement: null, nextStatement: null, colour: C.light, inputsInline: true },
    { type: 'flipperlight_centerButtonLight', message0: 'set center button light to %1', args0: [dd('COLOR', [['red', '9'], ['green', '6'], ['blue', '3'], ['yellow', '7'], ['orange', '8'], ['violet', '2'], ['white', '10'], ['off', '0']])], previousStatement: null, nextStatement: null, colour: C.light },

    // ------------------------------------------------------------ sound
    { type: 'flippersound_beepForTime', message0: 'play beep %1 for %2 seconds', args0: [num('NOTE'), num('DURATION')], previousStatement: null, nextStatement: null, colour: C.sound, inputsInline: true },
    { type: 'flippersound_beep', message0: 'start playing beep %1', args0: [num('NOTE')], previousStatement: null, nextStatement: null, colour: C.sound, inputsInline: true },
    { type: 'flippersound_stopSound', message0: 'stop all sounds', previousStatement: null, nextStatement: null, colour: C.sound },

    // ------------------------------------------------------------ control
    { type: 'control_wait', message0: 'wait %1 seconds', args0: [num('DURATION')], previousStatement: null, nextStatement: null, colour: C.control, inputsInline: true },
    { type: 'control_repeat', message0: 'repeat %1', args0: [num('TIMES')], message1: '%1', args1: [{ type: 'input_statement', name: 'SUBSTACK' }], previousStatement: null, nextStatement: null, colour: C.control, inputsInline: true },
    { type: 'control_forever', message0: 'forever', message1: '%1', args1: [{ type: 'input_statement', name: 'SUBSTACK' }], previousStatement: null, colour: C.control },
    { type: 'control_if', message0: 'if %1 then', args0: [bool('CONDITION')], message1: '%1', args1: [{ type: 'input_statement', name: 'SUBSTACK' }], previousStatement: null, nextStatement: null, colour: C.control },
    { type: 'control_if_else', message0: 'if %1 then', args0: [bool('CONDITION')], message1: '%1', args1: [{ type: 'input_statement', name: 'SUBSTACK' }], message2: 'else', message3: '%1', args3: [{ type: 'input_statement', name: 'SUBSTACK2' }], previousStatement: null, nextStatement: null, colour: C.control },
    { type: 'control_wait_until', message0: 'wait until %1', args0: [bool('CONDITION')], previousStatement: null, nextStatement: null, colour: C.control },
    { type: 'control_repeat_until', message0: 'repeat until %1', args0: [bool('CONDITION')], message1: '%1', args1: [{ type: 'input_statement', name: 'SUBSTACK' }], previousStatement: null, nextStatement: null, colour: C.control },
    { type: 'flippercontrol_stop', message0: 'stop %1', args0: [dd('STOP_OPTION', [['and exit program', 'all'], ['this stack', 'this script']])], previousStatement: null, colour: C.control },

    // ------------------------------------------------------------ sensors
    { type: 'flippersensors_isColor', message0: '%1 is color %2 ?', args0: [dd('PORT', singlePort), dd('VALUE', COLOR_OPTIONS)], output: 'Boolean', colour: C.sensors },
    { type: 'flippersensors_color', message0: '%1 color', args0: [dd('PORT', singlePort)], output: 'Number', colour: C.sensors },
    { type: 'flippersensors_isReflectivity', message0: '%1 is reflected light %2 %3 %% ?', args0: [dd('PORT', singlePort), dd('COMPARATOR', [['<', '<'], ['>', '>'], ['=', '=']]), num('VALUE')], output: 'Boolean', colour: C.sensors, inputsInline: true },
    { type: 'flippersensors_reflectivity', message0: '%1 reflected light', args0: [dd('PORT', singlePort)], output: 'Number', colour: C.sensors },
    { type: 'flippersensors_isDistance', message0: '%1 is distance %2 %3 %4 ?', args0: [dd('PORT', singlePort), dd('COMPARATOR', [['<', '<'], ['>', '>'], ['=', '=']]), num('VALUE'), dd('UNIT', [['cm', 'cm'], ['in', 'inches'], ['mm', 'mm']])], output: 'Boolean', colour: C.sensors, inputsInline: true },
    { type: 'flippersensors_distance', message0: '%1 distance in %2', args0: [dd('PORT', singlePort), dd('UNIT', [['cm', 'cm'], ['in', 'inches'], ['mm', 'mm']])], output: 'Number', colour: C.sensors },
    { type: 'flippersensors_isPressed', message0: '%1 is %2 ?', args0: [dd('PORT', singlePort), dd('OPTION', [['pressed', 'pressed'], ['released', 'released']])], output: 'Boolean', colour: C.sensors },
    { type: 'flippersensors_force', message0: '%1 pressure in N', args0: [dd('PORT', singlePort)], output: 'Number', colour: C.sensors },
    { type: 'flippersensors_orientationAxis', message0: '%1 angle', args0: [dd('AXIS', [['yaw', 'yaw'], ['pitch', 'pitch'], ['roll', 'roll']])], output: 'Number', colour: C.sensors, tooltip: 'Yaw: turning right (clockwise) makes the angle bigger, like in the SPIKE app.' },
    { type: 'flippersensors_resetYaw', message0: 'set yaw angle to 0', previousStatement: null, nextStatement: null, colour: C.sensors },
    { type: 'flippersensors_timer', message0: 'timer', output: 'Number', colour: C.sensors },
    { type: 'flippersensors_resetTimer', message0: 'reset timer', previousStatement: null, nextStatement: null, colour: C.sensors },
    { type: 'flippersensors_buttonIsPressed', message0: '%1 button %2 ?', args0: [dd('BUTTON', [['left', 'left'], ['right', 'right']]), dd('EVENT', [['pressed', 'pressed'], ['released', 'released']])], output: 'Boolean', colour: C.sensors },

    // ------------------------------------------------------------ operators
    { type: 'operator_add', message0: '%1 + %2', args0: [num('NUM1'), num('NUM2')], output: 'Number', colour: C.operators, inputsInline: true },
    { type: 'operator_subtract', message0: '%1 - %2', args0: [num('NUM1'), num('NUM2')], output: 'Number', colour: C.operators, inputsInline: true },
    { type: 'operator_multiply', message0: '%1 * %2', args0: [num('NUM1'), num('NUM2')], output: 'Number', colour: C.operators, inputsInline: true },
    { type: 'operator_divide', message0: '%1 / %2', args0: [num('NUM1'), num('NUM2')], output: 'Number', colour: C.operators, inputsInline: true },
    { type: 'operator_random', message0: 'pick random %1 to %2', args0: [num('FROM'), num('TO')], output: 'Number', colour: C.operators, inputsInline: true },
    { type: 'operator_gt', message0: '%1 > %2', args0: [num('OPERAND1'), num('OPERAND2')], output: 'Boolean', colour: C.operators, inputsInline: true },
    { type: 'operator_lt', message0: '%1 < %2', args0: [num('OPERAND1'), num('OPERAND2')], output: 'Boolean', colour: C.operators, inputsInline: true },
    { type: 'operator_equals', message0: '%1 = %2', args0: [num('OPERAND1'), num('OPERAND2')], output: 'Boolean', colour: C.operators, inputsInline: true },
    { type: 'flipperoperator_isInBetween', message0: 'is %1 between %2 and %3 ?', args0: [num('VALUE'), num('LOW'), num('HIGH')], output: 'Boolean', colour: C.operators, inputsInline: true },
    { type: 'operator_and', message0: '%1 and %2', args0: [bool('OPERAND1'), bool('OPERAND2')], output: 'Boolean', colour: C.operators, inputsInline: true },
    { type: 'operator_or', message0: '%1 or %2', args0: [bool('OPERAND1'), bool('OPERAND2')], output: 'Boolean', colour: C.operators, inputsInline: true },
    { type: 'operator_not', message0: 'not %1', args0: [bool('OPERAND')], output: 'Boolean', colour: C.operators, inputsInline: true },
    { type: 'operator_join', message0: 'join %1 %2', args0: [num('STRING1'), num('STRING2')], output: null, colour: C.operators, inputsInline: true },
    { type: 'operator_mod', message0: '%1 mod %2', args0: [num('NUM1'), num('NUM2')], output: 'Number', colour: C.operators, inputsInline: true },
    { type: 'operator_round', message0: 'round %1', args0: [num('NUM')], output: 'Number', colour: C.operators, inputsInline: true },
    { type: 'operator_mathop', message0: '%1 of %2', args0: [dd('OPERATOR', [['abs', 'abs'], ['floor', 'floor'], ['ceiling', 'ceiling'], ['sqrt', 'sqrt'], ['sin', 'sin'], ['cos', 'cos'], ['tan', 'tan'], ['asin', 'asin'], ['acos', 'acos'], ['atan', 'atan'], ['ln', 'ln'], ['log', 'log'], ['e ^', 'e ^'], ['10 ^', '10 ^']]), num('NUM')], output: 'Number', colour: C.operators, inputsInline: true },

    // ------------------------------------------------------------ variables (change by)
    { type: 'data_changevariableby', message0: 'change %1 by %2', args0: [{ type: 'field_variable', name: 'VAR', variable: 'my variable' }, num('VALUE')], previousStatement: null, nextStatement: null, colour: C.variables, inputsInline: true },

    // ------------------------------------------------------------ misc
    { type: 'python_comment', message0: '# %1', args0: [{ type: 'field_input', name: 'TEXT', text: 'note' }], previousStatement: null, nextStatement: null, colour: '#7a7a7a', tooltip: 'A note in your program (does nothing).' },
    { type: 'unsupported_block', message0: '⚠ %1', args0: [{ type: 'field_label_serializable', name: 'LABEL', text: 'unsupported' }], previousStatement: null, nextStatement: null, colour: C.other, tooltip: 'This SPIKE block is not supported by the simulator yet.' },
    { type: 'unsupported_reporter', message0: '⚠ %1', args0: [{ type: 'field_label_serializable', name: 'LABEL', text: 'unsupported' }], output: null, colour: C.other }
  ];

  // ------------------------------------------------------------ toolbox
  const N = (v) => ({ shadow: { type: 'math_number', fields: { NUM: v } } });
  const T = (v) => ({ shadow: { type: 'text', fields: { TEXT: v } } });
  const b = (type, inputs, fields) => {
    const o = { kind: 'block', type: type };
    if (inputs) o.inputs = inputs;
    if (fields) o.fields = fields;
    return o;
  };
  function toolbox(defaults) {
    defaults = defaults || {};
    const drive = defaults.pair || 'CD';
    const arm = defaults.arm || 'E';
    const colorPort = defaults.colorPort || 'B';
    return {
      kind: 'categoryToolbox',
      contents: [
        { kind: 'category', name: 'Motors', colour: C.motors, contents: [
          b('flippermotor_motorTurnForDirection', { VALUE: N(1) }, { PORT: arm }),
          b('flippermotor_motorGoDirectionToPosition', { POSITION: N(0) }, { PORT: arm }),
          b('flippermotor_motorStartDirection', null, { PORT: arm }),
          b('flippermotor_motorStop', null, { PORT: arm }),
          b('flippermotor_motorSetSpeed', { SPEED: N(75) }, { PORT: arm }),
          b('flippermoremotor_motorSetDegreeCounted', { VALUE: N(0) }, { PORT: arm }),
          b('flippermotor_absolutePosition', null, { PORT: arm }),
          b('flippermoremotor_position', null, { PORT: arm }),
          b('flippermotor_speed', null, { PORT: arm })
        ] },
        { kind: 'category', name: 'Movement', colour: C.movement, contents: [
          b('flippermove_move', { VALUE: N(10) }),
          b('flippermove_steer', { STEERING: N(30), VALUE: N(10) }),
          b('flippermove_startMove'),
          b('flippermove_startSteer', { STEERING: N(30) }),
          b('flippermove_stopMove'),
          b('flippermove_movementSpeed', { SPEED: N(50) }),
          b('flippermove_setMovementPair', null, { PAIR: drive }),
          b('flippermove_setDistance', { DISTANCE: N(17.5) }),
          { kind: 'label', text: 'More movement' },
          b('flippermoremove_startDualSpeed', { LEFT: N(50), RIGHT: N(50) }),
          b('flippermoremove_moveDualSpeed', { VALUE: N(10), LEFT: N(50), RIGHT: N(50) })
        ] },
        { kind: 'category', name: 'Light', colour: C.light, contents: [
          b('flipperlight_lightDisplayImageOnForTime', { VALUE: N(2) }),
          b('flipperlight_lightDisplayImageOn'),
          b('flipperlight_lightDisplayText', { TEXT: T('Hello') }),
          b('flipperlight_lightDisplayOff'),
          b('flipperlight_lightDisplaySetPixel', { X: N(1), Y: N(1), BRIGHTNESS: N(100) }),
          b('flipperlight_centerButtonLight')
        ] },
        { kind: 'category', name: 'Sound', colour: C.sound, contents: [
          b('flippersound_beepForTime', { NOTE: N(60), DURATION: N(0.2) }),
          b('flippersound_beep', { NOTE: N(60) }),
          b('flippersound_stopSound')
        ] },
        { kind: 'category', name: 'Events', colour: C.events, contents: [
          b('flipperevents_whenProgramStarts'),
          b('flipperevents_whenButton'),
          b('flipperevents_whenColor', null, { PORT: colorPort }),
          b('flipperevents_whenPressed'),
          b('flipperevents_whenTimer', { VALUE: N(5) }),
          b('flipperevents_whenCondition')
        ] },
        { kind: 'category', name: 'Control', colour: C.control, contents: [
          b('control_wait', { DURATION: N(1) }),
          b('control_repeat', { TIMES: N(10) }),
          b('control_forever'),
          b('control_if'),
          b('control_if_else'),
          b('control_wait_until'),
          b('control_repeat_until'),
          b('flippercontrol_stop')
        ] },
        { kind: 'category', name: 'Sensors', colour: C.sensors, contents: [
          b('flippersensors_isColor', null, { PORT: colorPort }),
          b('flippersensors_color', null, { PORT: colorPort }),
          b('flippersensors_isReflectivity', { VALUE: N(50) }, { PORT: colorPort }),
          b('flippersensors_reflectivity', null, { PORT: colorPort }),
          b('flippersensors_isDistance', { VALUE: N(15) }),
          b('flippersensors_distance'),
          b('flippersensors_isPressed'),
          b('flippersensors_force'),
          b('flippersensors_orientationAxis'),
          b('flippersensors_resetYaw'),
          b('flippersensors_timer'),
          b('flippersensors_resetTimer'),
          b('flippersensors_buttonIsPressed')
        ] },
        { kind: 'category', name: 'Operators', colour: C.operators, contents: [
          b('operator_add', { NUM1: N(''), NUM2: N('') }),
          b('operator_subtract', { NUM1: N(''), NUM2: N('') }),
          b('operator_multiply', { NUM1: N(''), NUM2: N('') }),
          b('operator_divide', { NUM1: N(''), NUM2: N('') }),
          b('operator_random', { FROM: N(1), TO: N(10) }),
          b('operator_gt', { OPERAND1: N(''), OPERAND2: N(50) }),
          b('operator_lt', { OPERAND1: N(''), OPERAND2: N(50) }),
          b('operator_equals', { OPERAND1: N(''), OPERAND2: N(50) }),
          b('flipperoperator_isInBetween', { VALUE: N(''), LOW: N(-10), HIGH: N(10) }),
          b('operator_and'),
          b('operator_or'),
          b('operator_not'),
          b('operator_join', { STRING1: T('apple '), STRING2: T('banana') }),
          b('operator_mod', { NUM1: N(''), NUM2: N('') }),
          b('operator_round', { NUM: N('') }),
          b('operator_mathop', { NUM: N('') })
        ] },
        { kind: 'category', name: 'Variables', colour: C.variables, custom: 'VARIABLE_SPIKE' },
        { kind: 'category', name: 'My Blocks', colour: C.myblocks, custom: 'PROCEDURE' },
        { kind: 'category', name: 'Notes', colour: '#7a7a7a', contents: [b('python_comment')] }
      ]
    };
  }

  // ------------------------------------------------------------ registration
  let registered = false;
  function register(Blockly) {
    if (registered) return;
    registered = true;
    Blockly.common.defineBlocksWithJsonArray(BLOCKS);
  }

  function variablesFlyout(Blockly) {
    return function (workspace) {
      const xml = [];
      xml.push({ kind: 'button', text: 'Make a Variable', callbackkey: 'CREATE_VARIABLE' });
      const vars = workspace.getVariableMap ? workspace.getVariableMap().getAllVariables() : workspace.getAllVariables();
      if (vars.length) {
        const v = vars[vars.length - 1];
        const ref = { name: v.getName ? v.getName() : v.name, id: v.getId ? v.getId() : v.id };
        xml.push({ kind: 'block', type: 'variables_set', fields: { VAR: { id: ref.id } }, inputs: { VALUE: N(0) } });
        xml.push({ kind: 'block', type: 'data_changevariableby', fields: { VAR: { id: ref.id } }, inputs: { VALUE: N(1) } });
        for (const vv of vars) {
          xml.push({ kind: 'block', type: 'variables_get', fields: { VAR: { id: vv.getId ? vv.getId() : vv.id } } });
        }
      }
      return xml;
    };
  }

  function theme(Blockly) {
    return Blockly.Theme.defineTheme('spike', {
      base: Blockly.Themes.Classic,
      startHats: true,
      blockStyles: {
        variable_blocks: { colourPrimary: C.variables, colourSecondary: '#db6e00', colourTertiary: '#db6e00' },
        procedure_blocks: { colourPrimary: C.myblocks, colourSecondary: '#ff4d6a', colourTertiary: '#ff3355' },
        math_blocks: { colourPrimary: '#ffffff', colourSecondary: '#ffffff', colourTertiary: '#cccccc' },
        text_blocks: { colourPrimary: '#ffffff', colourSecondary: '#ffffff', colourTertiary: '#cccccc' }
      },
      componentStyles: {
        workspaceBackgroundColour: '#f9f9fb',
        toolboxBackgroundColour: '#ffffff',
        flyoutBackgroundColour: '#eef0f4',
        flyoutOpacity: 0.97,
        scrollbarColour: '#c5c9d2',
        insertionMarkerColour: '#000'
      },
      fontStyle: { family: 'system-ui, "Segoe UI", Helvetica, sans-serif', weight: '600', size: 11 }
    });
  }

  // ------------------------------------------------------------ Python generation
  function makeGenerator(Blockly) {
    const PG = G.python;
    const Order = PG.Order;
    const gen = new PG.PythonGenerator('SPIKE');
    gen.INDENT = '    ';
    gen.addReservedWords('velocity,timer,timer_start,yaw_angle,move_pair,move_speed,cm_per_rotation,motor_speed,note_freq,' +
      'distance_cm,distance_mm,distance_in,port,motor,motor_pair,color,color_sensor,distance_sensor,force_sensor,runloop,' +
      'light_matrix,light,sound,button,motion_sensor,time,math,random,hub,main');
    // reuse a few built-in generators
    const base = PG.pythonGenerator.forBlock;
    for (const t of ['math_number', 'text', 'logic_boolean']) gen.forBlock[t] = base[t];

    const needs = () => gen.needs;
    const need = (...what) => what.forEach(w => needs().add(w));
    const v = (block, name, order) => gen.valueToCode(block, name, order === undefined ? Order.NONE : order) || '0';
    const stmt = (block, name) => gen.statementToCode(block, name) || gen.PASS;
    const portsOf = (val) => Array.from(String(val || 'A').replace(/[^A-F]/g, '')).map(p => 'port.' + p);
    const numLit = (code) => /^-?\d+(\.\d+)?$/.test(code) ? parseFloat(code) : null;
    const asInt = (code) => {
      const n = numLit(code);
      if (n !== null) return String(Math.round(n));
      return 'int(' + code + ')';
    };
    const ms = (code) => {
      const n = numLit(code);
      if (n !== null) return String(Math.round(n * 1000));
      return 'int((' + code + ') * 1000)';
    };
    // Convert a distance in the given unit to wheel degrees (expression).
    const degreesFor = (code, unit) => {
      need('move');
      const n = numLit(code);
      const c = n !== null ? code : '(' + code + ')';
      switch (unit) {
        case 'rotations': return n !== null ? String(Math.round(n * 360)) : 'int(' + c + ' * 360)';
        case 'degrees': return asInt(code);
        case 'inches': return 'int(' + c + ' * 2.54 / cm_per_rotation * 360)';
        default: return 'int(' + c + ' / cm_per_rotation * 360)';
      }
    };
    const motorSpeed = (port) => { need('motor', 'velocity', 'motor_speed'); return 'velocity(motor_speed.get(' + port + ', 75))'; };

    const F = gen.forBlock;
    // events are handled by generateProgram()
    // ---- motors
    F.flippermotor_motorTurnForDirection = (block) => {
      need('motor', 'port', 'runloop');
      const ports = portsOf(block.getFieldValue('PORT'));
      const sign = block.getFieldValue('DIRECTION') === 'counterclockwise' ? '-' : '';
      const val = v(block, 'VALUE', Order.MULTIPLICATIVE);
      const unit = block.getFieldValue('UNIT');
      const lines = ports.map((p, i) => {
        const aw = i === ports.length - 1 ? 'await ' : '';
        if (unit === 'seconds') return aw + 'motor.run_for_time(' + p + ', ' + ms(val) + ', ' + sign + motorSpeed(p) + ')';
        const deg = unit === 'rotations' ? (numLit(val) !== null ? String(Math.round(numLit(val) * 360)) : 'int(' + val + ' * 360)') : asInt(val);
        return aw + 'motor.run_for_degrees(' + p + ', ' + deg + ', ' + sign + motorSpeed(p) + ')';
      });
      return lines.join('\n') + '\n';
    };
    F.flippermotor_motorGoDirectionToPosition = (block) => {
      need('motor', 'port');
      const ports = portsOf(block.getFieldValue('PORT'));
      const dir = { shortest: 'motor.SHORTEST_PATH', clockwise: 'motor.CLOCKWISE', counterclockwise: 'motor.COUNTERCLOCKWISE' }[block.getFieldValue('DIRECTION')] || 'motor.SHORTEST_PATH';
      const pos = v(block, 'POSITION');
      return ports.map((p, i) => (i === ports.length - 1 ? 'await ' : '') + 'motor.run_to_absolute_position(' + p + ', ' + asInt(pos) + ', ' + motorSpeed(p) + ', direction=' + dir + ')').join('\n') + '\n';
    };
    F.flippermotor_motorStartDirection = (block) => {
      need('motor', 'port');
      const sign = block.getFieldValue('DIRECTION') === 'counterclockwise' ? '-' : '';
      return portsOf(block.getFieldValue('PORT')).map(p => 'motor.run(' + p + ', ' + sign + motorSpeed(p) + ')').join('\n') + '\n';
    };
    F.flippermotor_motorStop = (block) => { need('motor', 'port'); return portsOf(block.getFieldValue('PORT')).map(p => 'motor.stop(' + p + ')').join('\n') + '\n'; };
    F.flippermotor_motorSetSpeed = (block) => {
      need('motor_speed', 'port');
      const s = v(block, 'SPEED');
      return portsOf(block.getFieldValue('PORT')).map(p => 'motor_speed[' + p + '] = ' + s).join('\n') + '\n';
    };
    F.flippermoremotor_motorSetDegreeCounted = (block) => { need('motor', 'port'); return portsOf(block.getFieldValue('PORT')).map(p => 'motor.reset_relative_position(' + p + ', ' + asInt(v(block, 'VALUE')) + ')').join('\n') + '\n'; };
    F.flippermotor_absolutePosition = (block) => { need('motor', 'port'); return ['motor.absolute_position(' + portsOf(block.getFieldValue('PORT'))[0] + ') % 360', Order.MULTIPLICATIVE]; };
    F.flippermoremotor_position = (block) => { need('motor', 'port'); return ['motor.relative_position(' + portsOf(block.getFieldValue('PORT'))[0] + ')', Order.FUNCTION_CALL]; };
    F.flippermotor_speed = (block) => { need('motor', 'port'); return ['int(motor.velocity(' + portsOf(block.getFieldValue('PORT'))[0] + ') / 11.1)', Order.FUNCTION_CALL]; };

    // ---- movement
    F.flippermove_move = (block) => {
      need('move', 'velocity');
      const back = block.getFieldValue('DIRECTION') === 'back' ? '-' : '';
      const val = v(block, 'VALUE', Order.MULTIPLICATIVE);
      const unit = block.getFieldValue('UNIT');
      if (unit === 'seconds') return 'await motor_pair.move_for_time(move_pair, ' + ms(val) + ', 0, velocity=' + back + 'velocity(move_speed))\n';
      return 'await motor_pair.move_for_degrees(move_pair, ' + degreesFor(val, unit) + ', 0, velocity=' + back + 'velocity(move_speed))\n';
    };
    F.flippermove_steer = (block) => {
      need('move', 'velocity');
      const steer = asInt(v(block, 'STEERING'));
      const val = v(block, 'VALUE', Order.MULTIPLICATIVE);
      const unit = block.getFieldValue('UNIT');
      if (unit === 'seconds') return 'await motor_pair.move_for_time(move_pair, ' + ms(val) + ', ' + steer + ', velocity=velocity(move_speed))\n';
      return 'await motor_pair.move_for_degrees(move_pair, ' + degreesFor(val, unit) + ', ' + steer + ', velocity=velocity(move_speed))\n';
    };
    F.flippermove_startMove = (block) => {
      need('move', 'velocity');
      const back = block.getFieldValue('DIRECTION') === 'back' ? '-' : '';
      return 'motor_pair.move(move_pair, 0, velocity=' + back + 'velocity(move_speed))\n';
    };
    F.flippermove_startSteer = (block) => { need('move', 'velocity'); return 'motor_pair.move(move_pair, ' + asInt(v(block, 'STEERING')) + ', velocity=velocity(move_speed))\n'; };
    F.flippermove_stopMove = () => { need('move'); return 'motor_pair.stop(move_pair)\n'; };
    F.flippermove_movementSpeed = (block) => { need('move'); return 'move_speed = ' + v(block, 'SPEED') + '\n'; };
    F.flippermove_setMovementPair = (block) => {
      need('move', 'port');
      const ps = portsOf(block.getFieldValue('PAIR'));
      return 'motor_pair.pair(move_pair, ' + (ps[0] || 'port.A') + ', ' + (ps[1] || 'port.B') + ')\n';
    };
    F.flippermove_setDistance = (block) => {
      need('move');
      const d = v(block, 'DISTANCE', Order.MULTIPLICATIVE);
      return 'cm_per_rotation = ' + (block.getFieldValue('UNIT') === 'inches' ? d + ' * 2.54' : d) + '\n';
    };
    F.flippermoremove_startDualSpeed = (block) => { need('move', 'velocity'); return 'motor_pair.move_tank(move_pair, velocity(' + v(block, 'LEFT') + '), velocity(' + v(block, 'RIGHT') + '))\n'; };
    F.flippermoremove_moveDualSpeed = (block) => {
      need('move', 'velocity');
      const val = v(block, 'VALUE', Order.MULTIPLICATIVE);
      const unit = block.getFieldValue('UNIT');
      const L = 'velocity(' + v(block, 'LEFT') + ')', R = 'velocity(' + v(block, 'RIGHT') + ')';
      if (unit === 'seconds') return 'await motor_pair.move_tank_for_time(move_pair, ' + L + ', ' + R + ', ' + ms(val) + ')\n';
      return 'await motor_pair.move_tank_for_degrees(move_pair, ' + degreesFor(val, unit) + ', ' + L + ', ' + R + ')\n';
    };

    // ---- light & sound
    F.flipperlight_lightDisplayImageOnForTime = (block) => {
      need('light_matrix', 'runloop');
      return 'light_matrix.show_image(light_matrix.IMAGE_' + block.getFieldValue('MATRIX') + ')\nawait runloop.sleep_ms(' + ms(v(block, 'VALUE', Order.MULTIPLICATIVE)) + ')\nlight_matrix.clear()\n';
    };
    F.flipperlight_lightDisplayImageOn = (block) => { need('light_matrix'); return 'light_matrix.show_image(light_matrix.IMAGE_' + block.getFieldValue('MATRIX') + ')\n'; };
    F.flipperlight_lightDisplayText = (block) => { need('light_matrix'); return 'await light_matrix.write(str(' + v(block, 'TEXT') + '))\n'; };
    F.flipperlight_lightDisplayOff = () => { need('light_matrix'); return 'light_matrix.clear()\n'; };
    F.flipperlight_lightDisplaySetPixel = (block) => {
      need('light_matrix');
      return 'light_matrix.set_pixel(' + asInt(v(block, 'X')) + ' - 1, ' + asInt(v(block, 'Y')) + ' - 1, ' + asInt(v(block, 'BRIGHTNESS')) + ')\n';
    };
    F.flipperlight_centerButtonLight = (block) => { need('light', 'color'); return 'light.color(light.POWER, ' + COLOR_PY[block.getFieldValue('COLOR')] + ')\n'; };
    F.flippersound_beepForTime = (block) => { need('sound', 'note'); return 'await sound.beep(note_freq(' + v(block, 'NOTE') + '), ' + ms(v(block, 'DURATION', Order.MULTIPLICATIVE)) + ', 100)\n'; };
    F.flippersound_beep = (block) => { need('sound', 'note'); return 'sound.beep(note_freq(' + v(block, 'NOTE') + '), 60000, 100)\n'; };
    F.flippersound_stopSound = () => { need('sound'); return 'sound.stop()\n'; };

    // ---- control
    F.control_wait = (block) => { need('runloop'); return 'await runloop.sleep_ms(' + ms(v(block, 'DURATION', Order.MULTIPLICATIVE)) + ')\n'; };
    F.control_repeat = (block) => 'for _ in range(' + asInt(v(block, 'TIMES')) + '):\n' + stmt(block, 'SUBSTACK');
    F.control_forever = (block) => 'while True:\n' + stmt(block, 'SUBSTACK');
    F.control_if = (block) => 'if ' + (gen.valueToCode(block, 'CONDITION', Order.NONE) || 'False') + ':\n' + stmt(block, 'SUBSTACK');
    F.control_if_else = (block) => 'if ' + (gen.valueToCode(block, 'CONDITION', Order.NONE) || 'False') + ':\n' + stmt(block, 'SUBSTACK') + 'else:\n' + stmt(block, 'SUBSTACK2');
    F.control_wait_until = (block) => { need('runloop'); return 'await runloop.until(lambda: ' + (gen.valueToCode(block, 'CONDITION', Order.LAMBDA) || 'False') + ')\n'; };
    F.control_repeat_until = (block) => 'while not (' + (gen.valueToCode(block, 'CONDITION', Order.NONE) || 'False') + '):\n' + stmt(block, 'SUBSTACK');
    F.flippercontrol_stop = (block) => (block.getFieldValue('STOP_OPTION') === 'all' ? 'raise SystemExit\n' : 'return\n');
    F.control_stop = (block) => (block.getFieldValue('STOP_OPTION') === 'all' ? 'raise SystemExit\n' : 'return\n');

    // ---- sensors
    const cmp = (block, left, right) => {
      const c = block.getFieldValue('COMPARATOR');
      return [left + ' ' + (c === '=' ? '==' : c) + ' ' + right, Order.RELATIONAL];
    };
    F.flippersensors_isColor = (block) => { need('color_sensor', 'color', 'port'); return ['color_sensor.color(port.' + block.getFieldValue('PORT') + ') == ' + COLOR_PY[block.getFieldValue('VALUE')], Order.RELATIONAL]; };
    F.flippersensors_color = (block) => { need('color_sensor', 'port'); return ['color_sensor.color(port.' + block.getFieldValue('PORT') + ')', Order.FUNCTION_CALL]; };
    F.flippersensors_isReflectivity = (block) => { need('color_sensor', 'port'); return cmp(block, 'color_sensor.reflection(port.' + block.getFieldValue('PORT') + ')', v(block, 'VALUE', Order.RELATIONAL)); };
    F.flippersensors_reflectivity = (block) => { need('color_sensor', 'port'); return ['color_sensor.reflection(port.' + block.getFieldValue('PORT') + ')', Order.FUNCTION_CALL]; };
    const distExpr = (block) => {
      need('distance', 'port');
      const unit = block.getFieldValue('UNIT');
      return 'distance_' + (unit === 'inches' ? 'in' : (unit === 'mm' ? 'mm' : 'cm')) + '(port.' + block.getFieldValue('PORT') + ')';
    };
    F.flippersensors_isDistance = (block) => cmp(block, distExpr(block), v(block, 'VALUE', Order.RELATIONAL));
    F.flippersensors_distance = (block) => [distExpr(block), Order.FUNCTION_CALL];
    F.flippersensors_isPressed = (block) => {
      need('force_sensor', 'port');
      const e = 'force_sensor.pressed(port.' + block.getFieldValue('PORT') + ')';
      return block.getFieldValue('OPTION') === 'released' ? ['not ' + e, Order.LOGICAL_NOT] : [e, Order.FUNCTION_CALL];
    };
    F.flippersensors_force = (block) => { need('force_sensor', 'port'); return ['force_sensor.force(port.' + block.getFieldValue('PORT') + ') / 10', Order.MULTIPLICATIVE]; };
    F.flippersensors_orientationAxis = (block) => {
      need('yaw');
      const ax = block.getFieldValue('AXIS');
      if (ax === 'yaw') return ['yaw_angle()', Order.FUNCTION_CALL];
      need('motion');
      return ['int(motion_sensor.tilt_angles()[' + (ax === 'pitch' ? 1 : 2) + '] / 10)', Order.FUNCTION_CALL];
    };
    F.flippersensors_resetYaw = () => { need('motion'); return 'motion_sensor.reset_yaw(0)\n'; };
    F.flippersensors_timer = () => { need('timer'); return ['timer()', Order.FUNCTION_CALL]; };
    F.flippersensors_resetTimer = () => { need('timer'); return 'timer_start = time.ticks_ms()\n'; };
    F.flippersensors_buttonIsPressed = (block) => {
      need('button');
      const e = 'button.pressed(button.' + block.getFieldValue('BUTTON').toUpperCase() + ')';
      return [block.getFieldValue('EVENT') === 'released' ? e + ' == 0' : e + ' > 0', Order.RELATIONAL];
    };

    // ---- operators
    const arith = (op, order) => (block) => [v(block, 'NUM1', order) + ' ' + op + ' ' + v(block, 'NUM2', order + 0.01), order];
    F.operator_add = arith('+', Order.ADDITIVE);
    F.operator_subtract = arith('-', Order.ADDITIVE);
    F.operator_multiply = arith('*', Order.MULTIPLICATIVE);
    F.operator_divide = arith('/', Order.MULTIPLICATIVE);
    F.operator_mod = arith('%', Order.MULTIPLICATIVE);
    F.operator_random = (block) => { need('random'); return ['random.randint(' + asInt(v(block, 'FROM')) + ', ' + asInt(v(block, 'TO')) + ')', Order.FUNCTION_CALL]; };
    const rel = (op) => (block) => [v(block, 'OPERAND1', Order.RELATIONAL) + ' ' + op + ' ' + v(block, 'OPERAND2', Order.RELATIONAL), Order.RELATIONAL];
    F.operator_gt = rel('>');
    F.operator_lt = rel('<');
    F.operator_equals = rel('==');
    F.flipperoperator_isInBetween = (block) => [v(block, 'LOW', Order.RELATIONAL) + ' <= ' + v(block, 'VALUE', Order.RELATIONAL) + ' <= ' + v(block, 'HIGH', Order.RELATIONAL), Order.RELATIONAL];
    F.operator_and = (block) => [(gen.valueToCode(block, 'OPERAND1', Order.LOGICAL_AND) || 'False') + ' and ' + (gen.valueToCode(block, 'OPERAND2', Order.LOGICAL_AND) || 'False'), Order.LOGICAL_AND];
    F.operator_or = (block) => [(gen.valueToCode(block, 'OPERAND1', Order.LOGICAL_OR) || 'False') + ' or ' + (gen.valueToCode(block, 'OPERAND2', Order.LOGICAL_OR) || 'False'), Order.LOGICAL_OR];
    F.operator_not = (block) => ['not ' + (gen.valueToCode(block, 'OPERAND', Order.LOGICAL_NOT) || 'False'), Order.LOGICAL_NOT];
    F.operator_join = (block) => ['str(' + v(block, 'STRING1') + ') + str(' + v(block, 'STRING2') + ')', Order.ADDITIVE];
    F.operator_round = (block) => ['round(' + v(block, 'NUM') + ')', Order.FUNCTION_CALL];
    F.operator_mathop = (block) => {
      const x = v(block, 'NUM');
      const op = block.getFieldValue('OPERATOR');
      if (op === 'abs') return ['abs(' + x + ')', Order.FUNCTION_CALL];
      need('math');
      const map = {
        floor: 'math.floor(' + x + ')', ceiling: 'math.ceil(' + x + ')', sqrt: 'math.sqrt(' + x + ')',
        sin: 'math.sin(math.radians(' + x + '))', cos: 'math.cos(math.radians(' + x + '))', tan: 'math.tan(math.radians(' + x + '))',
        asin: 'math.degrees(math.asin(' + x + '))', acos: 'math.degrees(math.acos(' + x + '))', atan: 'math.degrees(math.atan(' + x + '))',
        ln: 'math.log(' + x + ')', log: 'math.log10(' + x + ')', 'e ^': 'math.exp(' + x + ')', '10 ^': '10 ** (' + x + ')'
      };
      return [map[op] || x, Order.FUNCTION_CALL];
    };

    // ---- variables
    const varName = (block) => gen.getVariableName(block.getFieldValue('VAR'));
    F.variables_get = (block) => [varName(block), Order.ATOMIC];
    F.variables_set = (block) => varName(block) + ' = ' + v(block, 'VALUE') + '\n';
    F.data_changevariableby = (block) => varName(block) + ' += ' + v(block, 'VALUE', Order.ADDITIVE) + '\n';
    F.math_change = (block) => varName(block) + ' += ' + v(block, 'DELTA', Order.ADDITIVE) + '\n';

    // ---- my blocks (procedures become async functions)
    const procDef = (block, withReturn) => {
      const name = gen.getProcedureName(block.getFieldValue('NAME'));
      const params = block.getVars().map(n => gen.getVariableName(n));
      let body = gen.statementToCode(block, 'STACK');
      if (withReturn) {
        const ret = gen.valueToCode(block, 'RETURN', Order.NONE) || 'None';
        body += gen.INDENT + 'return ' + ret + '\n';
      }
      const globals = globalsFor(gen, body).filter(g => params.indexOf(g) < 0);
      const glob = globals.length ? gen.INDENT + 'global ' + globals.join(', ') + '\n' : '';
      gen.procedures.push('async def ' + name + '(' + params.join(', ') + '):\n' + glob + (body || gen.PASS));
      return null;
    };
    F.procedures_defnoreturn = (block) => procDef(block, false);
    F.procedures_defreturn = (block) => procDef(block, true);
    const procCall = (block) => {
      const name = gen.getProcedureName(block.getFieldValue('NAME'));
      const args = block.getVars().map((_, i) => gen.valueToCode(block, 'ARG' + i, Order.NONE) || 'None');
      return 'await ' + name + '(' + args.join(', ') + ')';
    };
    F.procedures_callnoreturn = (block) => procCall(block) + '\n';
    F.procedures_callreturn = (block) => ['(' + procCall(block) + ')', Order.ATOMIC];
    F.procedures_ifreturn = (block) => {
      const cond = gen.valueToCode(block, 'CONDITION', Order.NONE) || 'False';
      let code = 'if ' + cond + ':\n';
      if (block.hasReturnValue_) code += gen.INDENT + 'return ' + (gen.valueToCode(block, 'VALUE', Order.NONE) || 'None') + '\n';
      else code += gen.INDENT + 'return\n';
      return code;
    };

    // ---- misc
    F.python_comment = (block) => '# ' + String(block.getFieldValue('TEXT') || '').replace(/\n/g, ' ') + '\n';
    F.unsupported_block = (block) => '# unsupported SPIKE block: ' + block.getFieldValue('LABEL') + '\n';
    F.unsupported_reporter = () => ['0', Order.ATOMIC];
    return gen;
  }

  // Names a generated function must declare global: workspace variables plus
  // the word-block settings it assigns.
  function globalsFor(gen, body) {
    const out = gen.globalVars.slice();
    for (const name of ['move_speed', 'cm_per_rotation', 'timer_start']) {
      if (new RegExp('^\\s*' + name + ' (=|\\+=)', 'm').test(body) && out.indexOf(name) < 0) out.push(name);
    }
    return out;
  }

  const HAT_TYPES = ['flipperevents_whenProgramStarts', 'flipperevents_whenButton', 'flipperevents_whenColor', 'flipperevents_whenPressed', 'flipperevents_whenTimer', 'flipperevents_whenCondition'];

  // Build the complete SPIKE Python program for a workspace.
  function generateProgram(Blockly, gen, workspace, robotDefaults) {
    robotDefaults = robotDefaults || {};
    gen.init(workspace);
    gen.needs = new Set();
    gen.procedures = [];
    const vars = (workspace.getVariableMap ? workspace.getVariableMap().getAllVariables() : workspace.getAllVariables());
    gen.globalVars = vars.map(vv => gen.getVariableName(vv.getId ? vv.getId() : vv.id));
    const tops = workspace.getTopBlocks(true);
    const stacks = [];
    let count = 0;
    // procedures first so their definitions are collected
    for (const blk of tops) {
      if (blk.type.indexOf('procedures_def') === 0) gen.blockToCode(blk);
    }
    for (const blk of tops) {
      if (HAT_TYPES.indexOf(blk.type) < 0) continue;
      count++;
      let body = gen.blockToCode(blk.getNextBlock()) || '';
      body = gen.prefixLines(body, gen.INDENT);
      const name = { flipperevents_whenProgramStarts: 'program_start', flipperevents_whenButton: 'when_button', flipperevents_whenColor: 'when_color', flipperevents_whenPressed: 'when_pressed', flipperevents_whenTimer: 'when_timer', flipperevents_whenCondition: 'when_condition' }[blk.type] + '_' + count;
      let cond = null;
      switch (blk.type) {
        case 'flipperevents_whenButton':
          gen.needs.add('button');
          cond = 'button.pressed(button.' + blk.getFieldValue('BUTTON').toUpperCase() + ')' + (blk.getFieldValue('EVENT') === 'released' ? ' == 0' : ' > 0');
          break;
        case 'flipperevents_whenColor':
          gen.needs.add('color_sensor'); gen.needs.add('color'); gen.needs.add('port');
          cond = 'color_sensor.color(port.' + blk.getFieldValue('PORT') + ') == ' + COLOR_PY[blk.getFieldValue('OPTION')];
          break;
        case 'flipperevents_whenPressed':
          gen.needs.add('force_sensor'); gen.needs.add('port');
          cond = (blk.getFieldValue('OPTION') === 'released' ? 'not ' : '') + 'force_sensor.pressed(port.' + blk.getFieldValue('PORT') + ')';
          break;
        case 'flipperevents_whenTimer':
          gen.needs.add('timer');
          cond = 'timer() > ' + (gen.valueToCode(blk, 'VALUE', G.python.Order.RELATIONAL) || '0');
          break;
        case 'flipperevents_whenCondition':
          cond = gen.valueToCode(blk, 'CONDITION', G.python.Order.LAMBDA) || 'False';
          break;
      }
      const glob = globalsFor(gen, body);
      const globLine = glob.length ? gen.INDENT + 'global ' + glob.join(', ') + '\n' : '';
      let code = 'async def ' + name + '():\n' + globLine;
      if (cond === null) code += (body || gen.PASS);
      else if (blk.type === 'flipperevents_whenTimer') {
        gen.needs.add('runloop');
        code += gen.INDENT + 'await runloop.until(lambda: ' + cond + ')\n' + (body || gen.PASS);
      } else {
        gen.needs.add('runloop');
        code += gen.INDENT + 'while True:\n' +
          gen.INDENT + gen.INDENT + 'await runloop.until(lambda: ' + cond + ')\n' +
          gen.prefixLines(body || gen.PASS, gen.INDENT) +
          gen.INDENT + gen.INDENT + 'await runloop.until(lambda: not (' + cond + '))\n';
      }
      stacks.push({ name: name, code: code });
    }
    return assemble(gen, stacks, robotDefaults);
  }

  function assemble(gen, stacks, robotDefaults) {
    const n = gen.needs;
    if (n.has('velocity') || n.has('move')) n.add('motor_pair_mod');
    if (n.has('move')) { n.add('port'); n.add('velocity'); }
    if (n.has('distance')) n.add('distance_sensor');
    if (n.has('yaw')) n.add('motion');
    if (n.has('timer')) n.add('time');
    if (n.has('motor_speed')) n.add('port');
    const hubParts = [];
    if (n.has('port')) hubParts.push('port');
    if (n.has('motion')) hubParts.push('motion_sensor');
    if (n.has('light_matrix')) hubParts.push('light_matrix');
    if (n.has('light')) hubParts.push('light');
    if (n.has('sound')) hubParts.push('sound');
    if (n.has('button')) hubParts.push('button');
    const lines = ['# Generated from SPIKE word blocks by the FLL BIOGLOW simulator.', '# This is SPIKE App 3 Python: it also runs on a real SPIKE Prime hub.'];
    if (hubParts.length) lines.push('from hub import ' + hubParts.join(', '));
    const mods = ['runloop'];
    if (n.has('motor')) mods.push('motor');
    if (n.has('motor_pair_mod')) mods.push('motor_pair');
    if (n.has('color_sensor')) mods.push('color_sensor');
    if (n.has('color')) mods.push('color');
    if (n.has('distance_sensor')) mods.push('distance_sensor');
    if (n.has('force_sensor')) mods.push('force_sensor');
    if (n.has('time')) mods.push('time');
    if (n.has('math')) mods.push('math');
    if (n.has('random')) mods.push('random');
    lines.push('import ' + mods.join(', '));
    lines.push('');
    const vars = gen.globalVars;
    if (vars.length) {
      lines.push('# variables');
      vars.forEach(vv => lines.push(vv + ' = 0'));
      lines.push('');
    }
    const helpers = [];
    if (n.has('move')) {
      const pair = robotDefaults.pair || 'CD';
      helpers.push('# movement settings (like the SPIKE app defaults)',
        'move_pair = motor_pair.PAIR_1',
        'motor_pair.pair(move_pair, port.' + pair[0] + ', port.' + pair[1] + ')',
        'move_speed = 50          # %',
        'cm_per_rotation = 17.5   # 56 mm wheel', '');
    }
    if (n.has('motor_speed')) helpers.push('motor_speed = {}         # motor speed in % per port (default 75)', '');
    if (n.has('velocity')) helpers.push('def velocity(percent):', '    # speed in % -> degrees per second (100% = 1110 deg/s)', '    return int(max(-100, min(100, percent)) * 11.1)', '');
    if (n.has('yaw')) helpers.push('def yaw_angle():', '    # word blocks count turning right (clockwise) as positive', '    return int(-motion_sensor.tilt_angles()[0] / 10)', '');
    if (n.has('timer')) helpers.push('timer_start = time.ticks_ms()', '', 'def timer():', '    return time.ticks_diff(time.ticks_ms(), timer_start) / 1000', '');
    if (n.has('distance')) helpers.push('def distance_mm(p):', '    d = distance_sensor.distance(p)', '    return d if d >= 0 else 2000', '',
      'def distance_cm(p):', '    return distance_mm(p) / 10', '', 'def distance_in(p):', '    return distance_mm(p) / 25.4', '');
    if (n.has('note')) helpers.push('def note_freq(note):', '    return int(440 * 2 ** ((note - 69) / 12))', '');
    const out = lines.concat(helpers);
    for (const p of gen.procedures) out.push(p);
    for (const s of stacks) out.push(s.code);
    if (stacks.length) out.push('runloop.run(' + stacks.map(s => s.name + '()').join(', ') + ')');
    else out.push('# Add a "when program starts" block to make a program.');
    return out.join('\n').replace(/\n{3,}/g, '\n\n') + '\n';
  }

  FLL.blocks = { register, toolbox, theme, makeGenerator, generateProgram, variablesFlyout, COLORS: C, HAT_TYPES, BLOCK_DEFS: BLOCKS };
})(typeof globalThis !== 'undefined' ? globalThis : window);
