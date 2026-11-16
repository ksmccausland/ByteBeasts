/*
 * Program templates, example programs and the help page.
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};

  // Port placeholders: {L} left wheel, {R} right wheel, {ARM} attachment motor, {COLOR} colour sensor.
  function ports(cfg) {
    const dev = cfg.devices;
    const arm = ['E', 'F', 'A', 'B', 'C', 'D'].find(p => dev[p] && dev[p].type === 'motor' && p !== cfg.driveLeft && p !== cfg.driveRight) || 'E';
    const color = ['B', 'A', 'C', 'D', 'E', 'F'].find(p => dev[p] && dev[p].type === 'color') || 'B';
    return { L: cfg.driveLeft, R: cfg.driveRight, ARM: arm, COLOR: color, WHEEL: cfg.wheelDiameter, TRACK: cfg.trackWidth };
  }
  function fill(text, cfg) {
    const p = ports(cfg);
    return text.replace(/\{(L|R|ARM|COLOR|WHEEL|TRACK)\}/g, (m, k) => p[k]);
  }
  function fillBlocks(obj, cfg) {
    return JSON.parse(fill(JSON.stringify(obj), cfg));
  }

  // ---------------------------------------------------------------- python sources
  const SPIKE_TEMPLATE = `from hub import port, motion_sensor, light_matrix
import motor, motor_pair, runloop

async def main():
    # Pair the wheel motors: left = {L}, right = {R}
    motor_pair.pair(motor_pair.PAIR_1, port.{L}, port.{R})
    light_matrix.show_image(light_matrix.IMAGE_HAPPY)

    # Drive forward 720 degrees (2 wheel rotations, about 35 cm)
    await motor_pair.move_for_degrees(motor_pair.PAIR_1, 720, 0, velocity=400)

    # Move the attachment motor on port {ARM} down and up again
    await motor.run_for_degrees(port.{ARM}, 90, 500)
    await motor.run_for_degrees(port.{ARM}, -90, 500)

    # Drive back home
    await motor_pair.move_for_degrees(motor_pair.PAIR_1, -720, 0, velocity=400)

runloop.run(main())
`;

  const PYBRICKS_TEMPLATE = `from pybricks.hubs import PrimeHub
from pybricks.pupdevices import Motor, ColorSensor
from pybricks.parameters import Port, Direction, Stop
from pybricks.robotics import DriveBase
from pybricks.tools import wait

hub = PrimeHub()
left_motor = Motor(Port.{L}, Direction.COUNTERCLOCKWISE)
right_motor = Motor(Port.{R})
arm = Motor(Port.{ARM})
robot = DriveBase(left_motor, right_motor, wheel_diameter={WHEEL}, axle_track={TRACK})
robot.use_gyro(True)   # use the hub's gyro for straight driving and exact turns

robot.straight(300)        # millimetres
robot.turn(90)             # degrees, positive = turn right
arm.run_angle(500, 90)     # speed (deg/s), angle
arm.run_angle(500, -90)
robot.turn(-90)
robot.straight(-300)
`;

  const SQUARE_PY = `from hub import port, motion_sensor
import motor_pair, runloop

async def turn(angle, speed=200):
    """Turn on the spot using the gyro. Positive angle = turn right."""
    motion_sensor.reset_yaw(0)
    if angle > 0:
        motor_pair.move_tank(motor_pair.PAIR_1, speed, -speed)
        # SPIKE 3 Python: yaw is in tenths of a degree and turning right makes it negative
        await runloop.until(lambda: motion_sensor.tilt_angles()[0] <= -angle * 10)
    else:
        motor_pair.move_tank(motor_pair.PAIR_1, -speed, speed)
        await runloop.until(lambda: motion_sensor.tilt_angles()[0] >= -angle * 10)
    motor_pair.stop(motor_pair.PAIR_1)

async def main():
    motor_pair.pair(motor_pair.PAIR_1, port.{L}, port.{R})
    for side in range(4):
        await motor_pair.move_for_degrees(motor_pair.PAIR_1, 500, 0, velocity=450)
        await turn(90)
        print('side', side + 1, 'done, yaw =', motion_sensor.tilt_angles()[0] / 10)

runloop.run(main())
`;

  const LINE_PY = `from hub import port
import motor, motor_pair, color_sensor, runloop

# Follow the bottom edge of the black line in the north-west of the mat.
# The robot starts just left of the line, facing east.
TARGET = 50   # halfway between black (about 10) and white (about 95)
KP = 1.6      # how hard to steer

async def main():
    motor_pair.pair(motor_pair.PAIR_1, port.{L}, port.{R})
    motor.reset_relative_position(port.{R}, 0)
    while motor.relative_position(port.{R}) < 400:
        error = color_sensor.reflection(port.{COLOR}) - TARGET
        motor_pair.move(motor_pair.PAIR_1, int(-KP * error), velocity=250)
        await runloop.sleep_ms(10)
    motor_pair.stop(motor_pair.PAIR_1)
    print('End of the line!')

runloop.run(main())
`;

  const M03_PY = `from hub import port, motion_sensor
import motor, motor_pair, runloop

# Mission 03 Flip the Rock: start in the LEFT launch area, facing north.
DEG_PER_MM = 360 / (3.1416 * {WHEEL})

async def straight(mm, speed=450):
    await motor_pair.move_for_degrees(motor_pair.PAIR_1, int(mm * DEG_PER_MM), 0, velocity=speed)

async def turn(angle, speed=180):
    """Gyro turn, positive = right (clockwise)."""
    motion_sensor.reset_yaw(0)
    if angle > 0:
        motor_pair.move_tank(motor_pair.PAIR_1, speed, -speed)
        await runloop.until(lambda: motion_sensor.tilt_angles()[0] <= -angle * 10)
    else:
        motor_pair.move_tank(motor_pair.PAIR_1, -speed, speed)
        await runloop.until(lambda: motion_sensor.tilt_angles()[0] >= -angle * 10)
    motor_pair.stop(motor_pair.PAIR_1)
    await runloop.sleep_ms(150)

async def main():
    motor_pair.pair(motor_pair.PAIR_1, port.{L}, port.{R})
    await straight(517)                               # drive up next to the rock
    await turn(-90)                                   # face west, towards the rock
    await motor.run_for_degrees(port.{ARM}, 90, 500)    # flip the rock: flag goes down
    await motor.run_for_degrees(port.{ARM}, -90, 500)   # put the rock back: bonus!
    await turn(-90)                                   # face south
    await straight(517)                               # back into the launch area

runloop.run(main())
`;

  const PB_M12_M11 = `from pybricks.hubs import PrimeHub
from pybricks.pupdevices import Motor
from pybricks.parameters import Port, Direction
from pybricks.robotics import DriveBase

# Missions 12 (Forest Elder cane) and 11 (Window to the Past).
# Start in the RIGHT launch area, facing north (see Setup > Start position).
hub = PrimeHub()
left_motor = Motor(Port.{L}, Direction.COUNTERCLOCKWISE)
right_motor = Motor(Port.{R})
arm = Motor(Port.{ARM})
robot = DriveBase(left_motor, right_motor, wheel_diameter={WHEEL}, axle_track={TRACK})
robot.use_gyro(True)

# --- Mission 12: raise the cane against the old tree
robot.straight(400)
arm.run_angle(600, 90)
arm.run_angle(600, -90)
robot.straight(-400)

# --- Mission 11: push the root cover down
robot.straight(19)
robot.turn(-45)            # face north-west
robot.settings(straight_speed=150)
robot.straight(475)        # drive into the window model
robot.straight(-475)
robot.turn(45)
robot.straight(-19)
hub.display.text('OK')
`;

  const DIST_PY = `from hub import port
import motor_pair, distance_sensor, runloop

# Needs the "Distance + force sensor" robot preset (distance sensor on port A).
async def main():
    motor_pair.pair(motor_pair.PAIR_1, port.{L}, port.{R})
    motor_pair.move(motor_pair.PAIR_1, 0, velocity=400)
    # drive until something is closer than 10 cm
    await runloop.until(lambda: 0 < distance_sensor.distance(port.A) < 100)
    motor_pair.stop(motor_pair.PAIR_1)
    print('Stopped', distance_sensor.distance(port.A), 'mm in front of an object')

runloop.run(main())
`;

  // ---------------------------------------------------------------- blocks
  const N = (v) => ({ shadow: { type: 'math_number', fields: { NUM: v } } });
  const chain = (blocks) => {
    for (let i = blocks.length - 2; i >= 0; i--) blocks[i].next = { block: blocks[i + 1] };
    return blocks[0];
  };
  const hat = (body, x, y) => Object.assign({ type: 'flipperevents_whenProgramStarts', x: x || 30, y: y || 30 }, body.length ? { next: { block: chain(body) } } : {});
  const move = (dir, val, unit) => ({ type: 'flippermove_move', fields: { DIRECTION: dir, UNIT: unit || 'cm' }, inputs: { VALUE: N(val) } });
  const steer = (s, val, unit) => ({ type: 'flippermove_steer', fields: { UNIT: unit || 'degrees' }, inputs: { STEERING: N(s), VALUE: N(val) } });

  const FIRST_BLOCKS = {
    blocks: { languageVersion: 0, blocks: [hat([
      { type: 'flippermove_setMovementPair', fields: { PAIR: '{L}{R}' } },
      { type: 'flippermove_movementSpeed', inputs: { SPEED: N(50) } },
      { type: 'flipperlight_lightDisplayImageOn', fields: { MATRIX: 'HAPPY' } },
      move('forward', 30),
      { type: 'flippermotor_motorTurnForDirection', fields: { PORT: '{ARM}', DIRECTION: 'clockwise', UNIT: 'degrees' }, inputs: { VALUE: N(90) } },
      { type: 'flippermotor_motorTurnForDirection', fields: { PORT: '{ARM}', DIRECTION: 'counterclockwise', UNIT: 'degrees' }, inputs: { VALUE: N(90) } },
      move('back', 30)
    ])] }
  };
  const SQUARE_BLOCKS = {
    blocks: { languageVersion: 0, blocks: [hat([
      { type: 'flippermove_setMovementPair', fields: { PAIR: '{L}{R}' } },
      { type: 'control_repeat', inputs: { TIMES: N(4), SUBSTACK: { block: chain([move('forward', 25), steer(100, 180, 'degrees')]) } } },
      { type: 'flipperlight_lightDisplayImageOnForTime', fields: { MATRIX: 'YES' }, inputs: { VALUE: N(2) } }
    ])] }
  };
  const GYRO_BLOCKS = {
    blocks: { languageVersion: 0, blocks: [hat([
      { type: 'flippermove_setMovementPair', fields: { PAIR: '{L}{R}' } },
      { type: 'flippersensors_resetYaw' },
      { type: 'flippermoremove_startDualSpeed', inputs: { LEFT: N(25), RIGHT: N(-25) } },
      { type: 'control_wait_until', inputs: { CONDITION: { block: { type: 'operator_gt', inputs: { OPERAND1: { shadow: { type: 'math_number', fields: { NUM: 0 } }, block: { type: 'flippersensors_orientationAxis', fields: { AXIS: 'yaw' } } }, OPERAND2: N(88) } } } } },
      { type: 'flippermove_stopMove' },
      move('forward', 20)
    ])] }
  };
  const LINE_BLOCKS = {
    blocks: { languageVersion: 0, blocks: [hat([
      { type: 'flippermove_setMovementPair', fields: { PAIR: '{L}{R}' } },
      { type: 'flippermoremotor_motorSetDegreeCounted', fields: { PORT: '{R}' }, inputs: { VALUE: N(0) } },
      { type: 'control_repeat_until', inputs: {
        CONDITION: { block: { type: 'operator_gt', inputs: { OPERAND1: { shadow: { type: 'math_number', fields: { NUM: 0 } }, block: { type: 'flippermoremotor_position', fields: { PORT: '{R}' } } }, OPERAND2: N(400) } } },
        SUBSTACK: { block: { type: 'control_if_else', inputs: {
          CONDITION: { block: { type: 'flippersensors_isReflectivity', fields: { PORT: '{COLOR}', COMPARATOR: '<' }, inputs: { VALUE: N(50) } } },
          SUBSTACK: { block: { type: 'flippermoremove_startDualSpeed', inputs: { LEFT: N(25), RIGHT: N(8) } } },
          SUBSTACK2: { block: { type: 'flippermoremove_startDualSpeed', inputs: { LEFT: N(8), RIGHT: N(25) } } }
        } } }
      } },
      { type: 'flippermove_stopMove' }
    ])] }
  };

  // ---------------------------------------------------------------- catalogue
  const LIST = [
    { id: 'blocks-first', group: 'Word blocks', name: 'First moves', mode: 'blocks', blocks: FIRST_BLOCKS, description: 'Drive forward, move the arm, drive back.', start: { x: 220, y: 140, heading: 0 } },
    { id: 'blocks-square', group: 'Word blocks', name: 'Drive a square', mode: 'blocks', blocks: SQUARE_BLOCKS, description: 'Repeat 4×: drive 25 cm and spin 90°.', start: { x: 220, y: 140, heading: 0 } },
    { id: 'blocks-gyro', group: 'Word blocks', name: 'Gyro turn', mode: 'blocks', blocks: GYRO_BLOCKS, description: 'Turn exactly 90° using the yaw angle.', start: { x: 220, y: 140, heading: 0 } },
    { id: 'blocks-line', group: 'Word blocks', name: 'Simple line follower', mode: 'blocks', blocks: LINE_BLOCKS, description: 'Zig-zag along the black line using reflected light.', start: { x: 448, y: 883, heading: 90 } },
    { id: 'blocks-guided', group: 'Word blocks', name: 'Official guided mission (SPIKE app)', mode: 'blocks', blocksRef: 'guided', description: 'The BIOGLOW guided mission from the SPIKE app, imported from the .llsp3 file.', note: 'This is the guided mission from the SPIKE app (guided-mission-bioglow-11.llsp3). It drives until the colour sensor sees black, then uses the gyro. Try different start positions to see where it goes!' },
    { id: 'py-template', group: 'SPIKE Python', name: 'Drive and arm', mode: 'python', python: SPIKE_TEMPLATE, description: 'motor_pair.move_for_degrees and motor.run_for_degrees.', start: { x: 220, y: 140, heading: 0 } },
    { id: 'py-square', group: 'SPIKE Python', name: 'Square with gyro turns', mode: 'python', python: SQUARE_PY, description: 'A reusable turn() function using the yaw angle.', start: { x: 220, y: 140, heading: 0 } },
    { id: 'py-line', group: 'SPIKE Python', name: 'Line follower (P-controller)', mode: 'python', python: LINE_PY, description: 'Smooth line following with reflected light.', start: { x: 448, y: 883, heading: 90 } },
    { id: 'py-m03', group: 'SPIKE Python', name: 'Mission 03: Flip the Rock', mode: 'python', python: M03_PY, description: 'A complete mission run from the left launch area (30 points).', start: { x: 300, y: 140, heading: 0 } },
    { id: 'py-distance', group: 'SPIKE Python', name: 'Stop before a wall (distance sensor)', mode: 'python', python: DIST_PY, description: 'Needs the “Distance + force sensor” robot preset.', start: { x: 220, y: 140, heading: 0 } },
    { id: 'pb-template', group: 'Pybricks', name: 'DriveBase basics', mode: 'python', python: PYBRICKS_TEMPLATE, description: 'straight(), turn() and run_angle() with the gyro.', start: { x: 220, y: 140, heading: 0 } },
    { id: 'pb-m12-m11', group: 'Pybricks', name: 'Missions 12 + 11 (right side)', mode: 'python', python: PB_M12_M11, description: 'Two missions in one run from the right launch area (40 points).', start: { x: 1840, y: 140, heading: 0 } }
  ];

  function instantiate(ex, cfg) {
    const out = { name: ex.name, mode: ex.mode };
    if (ex.mode === 'python') out.python = fill(ex.python, cfg);
    else if (ex.blocksRef === 'guided') out.blocks = JSON.parse(JSON.stringify(FLL.guidedMission || { blocks: { languageVersion: 0, blocks: [] } }));
    else out.blocks = fillBlocks(ex.blocks, cfg);
    if (ex.start) out.start = ex.start;
    return out;
  }

  function template(kind, cfg) {
    if (kind === 'blocks') return { name: 'Word blocks program', mode: 'blocks', blocks: fillBlocks({ blocks: { languageVersion: 0, blocks: [hat([move('forward', 10)])] } }, cfg) };
    if (kind === 'pybricks') return { name: 'Pybricks program', mode: 'python', python: fill(PYBRICKS_TEMPLATE, cfg) };
    return { name: 'SPIKE Python program', mode: 'python', python: fill(SPIKE_TEMPLATE, cfg) };
  }

  function defaultPrograms(cfg) {
    return [
      Object.assign(instantiate(LIST[0], cfg), { name: 'My first blocks' }),
      Object.assign(instantiate(LIST[5], cfg), { name: 'My first SPIKE Python' }),
      Object.assign(instantiate(LIST[10], cfg), { name: 'My first Pybricks' })
    ];
  }

  // ---------------------------------------------------------------- help
  function helpHtml() {
    return `<div class="help">
<div class="cols">
<div>
<h2>🚀 Getting started</h2>
<ol>
  <li>Choose a program in the list at the top left (or 📚 <b>Examples</b>, or <b>＋ New</b>).</li>
  <li>Press <b>▶ Run</b> (F5). The robot drives on the BIOGLOW mat and scores points.</li>
  <li>Drag the robot to move it, drag the red ⟳ handle to turn it. Arrow keys nudge it (Shift = 10 mm), <code>[</code> <code>]</code> turn it.</li>
  <li>Scroll to zoom, drag the mat to pan, double-click to fit the table.</li>
  <li>Open the SPIKE app file of your team with 📂 <b>Open</b> (.llsp3 word blocks or Python).</li>
</ol>
<h2>🏁 Match mode</h2>
<p>Press <b>Start match</b> for a real 2:30 match: the timer starts at your first launch, the robot must start completely inside a launch area, and picking up the robot outside home costs a precision token. When you pick the robot up outside home, things it collected during that run go to the referee.</p>
<h2>🔴 Record your driving</h2>
<p>Press <b>R</b> (or the <b>● Record</b> button) and drive with the arrow keys. A new program called “Recorded drive” is written while you drive: every time you press or let go of a key, a step is added (move, turn, or arm). Choose what it writes in the box next to the Record button: <b>🧩 Blocks</b> (SPIKE word blocks), <b>🐍 Python</b> (SPIKE Python), or <b>🧩+🐍 Both</b> (two programs from the same drive). 💾 Save can download either one for the SPIKE app. Press <b>R</b> again (or <b>Esc</b>) to finish. To replay it, put the robot back where you started recording and press ▶ Run. Only keyboard driving is recorded; moving the robot with the mouse is written as a comment block.</p>
<h2>🧊 3D view</h2>
<p>Press <b>🧊 3D</b> to see the field, the mission models and the robot in 3D. Drag to turn the view, right-drag to move it, scroll to zoom, and double-click (or <b>⤢ Fit</b>) to go back to the start view. Running programs, driving and recording all work in 3D. To move the robot with the mouse or to measure, switch back to the map by pressing <b>🧊 3D</b> again.</p>
<h2>🎯 Mission targets</h2>
<p>The dashed shapes on the field are where the robot has to act:</p>
<ul>
  <li><span style="color:#00acc1">■ cyan</span>: <b>push</b> – drive the robot into it (some also accept an attachment).</li>
  <li><span style="color:#f9a825">■ orange</span>: <b>attachment</b> – turn an attachment motor (port E or F) at least 45° while the orange tool zone in front of the robot covers the target.</li>
  <li><span style="color:#e53935">■ red</span>: <b>do not touch</b> (Fragile Microhabitats).</li>
</ul>
<p>Seeds (M02, M09), leaves (M04) and the support tie (M12) can be collected with an attachment action and delivered at another target. The real mission models are 3D mechanisms – the simulator uses this simpler model so you can practise navigation and programming.</p>
</div>
<div>
<h2>🧭 Field coordinates</h2>
<p>Positions are in millimetres from the <b>bottom-left corner of the mat</b>: <i>x</i> to the right (0–2000), <i>y</i> towards the back wall (0–1140). Turn on the <b>Grid</b> to see the official 20 cm grid (A–J, 1–6). Use 📏 <b>Measure</b> to measure distances and angles (0° = north, 90° = east).</p>
<h2>⌨ Keyboard</h2>
<table>
<tr><td>F5</td><td>Run</td></tr><tr><td>Esc</td><td>Stop</td></tr><tr><td>Home</td><td>Robot to start position</td></tr>
<tr><td>D</td><td>Keyboard drive mode (arrows, Q/A = arm E, W/S = arm F)</td></tr>
<tr><td>R</td><td>Start / stop recording your driving (as blocks, Python or both)</td></tr>
<tr><td>M</td><td>Measure tool</td></tr><tr><td>Ctrl+Space</td><td>Python autocomplete</td></tr><tr><td>Ctrl+/</td><td>Comment lines</td></tr>
</table>
<h2>🤖 Default robot</h2>
<p>Like the SPIKE app guided mission: wheel motors <b>C</b> (left) and <b>D</b> (right), colour sensor <b>B</b>, attachment motors <b>E</b> and <b>F</b>, 56 mm wheels, 112 mm axle track. Change it in <b>Setup → Robot design</b>.</p>
</div>
</div>

<h2>🐍 SPIKE App 3 Python (cheat sheet)</h2>
<pre>from hub import port, motion_sensor, light_matrix, sound, button
import motor, motor_pair, color_sensor, color, distance_sensor, force_sensor, runloop

async def main():
    motor_pair.pair(motor_pair.PAIR_1, port.C, port.D)
    await motor_pair.move_for_degrees(motor_pair.PAIR_1, 360, 0, velocity=500)   # steering -100..100
    await motor_pair.move_for_time(motor_pair.PAIR_1, 1000, 0, velocity=-300)    # ms
    motor_pair.move_tank(motor_pair.PAIR_1, 300, -300)    # start spinning (no await: keeps going)
    await runloop.until(lambda: motion_sensor.tilt_angles()[0] &lt;= -900)        # yaw in 1/10°, right = negative
    motor_pair.stop(motor_pair.PAIR_1)
    await motor.run_for_degrees(port.E, 90, 500)          # attachment
    motion_sensor.reset_yaw(0)
    if color_sensor.color(port.B) == color.BLACK: ...
    light = color_sensor.reflection(port.B)               # 0..100
    await runloop.sleep_ms(500)

runloop.run(main())</pre>
<p><b>Remember:</b> put <code>await</code> in front of commands that should finish before the next line runs. Velocities are in degrees per second (medium motor max 1110).</p>

<h2>🐍 Pybricks (cheat sheet)</h2>
<pre>from pybricks.hubs import PrimeHub
from pybricks.pupdevices import Motor, ColorSensor, UltrasonicSensor, ForceSensor
from pybricks.parameters import Port, Direction, Stop, Color
from pybricks.robotics import DriveBase
from pybricks.tools import wait, StopWatch

hub = PrimeHub()
robot = DriveBase(Motor(Port.C, Direction.COUNTERCLOCKWISE), Motor(Port.D), wheel_diameter=56, axle_track=112)
robot.use_gyro(True)
robot.straight(200)     # mm
robot.turn(90)          # degrees, positive = right
robot.curve(150, 90)    # radius, angle
print(hub.imu.heading(), robot.distance(), ColorSensor(Port.B).reflection())</pre>

<h2>🧩 Word blocks</h2>
<p>The blocks work like the SPIKE app word blocks (Motors, Movement, Light, Sound, Events, Control, Sensors, Operators, Variables, My Blocks). The 🐍 Python tab shows the SPIKE Python made from your blocks – you can copy it into the SPIKE app, or use <b>Save → Turn blocks into an editable Python program</b>. Yaw angle in blocks counts turning right as positive (like the SPIKE app).</p>

<h2>📏 How the simulation works</h2>
<ul>
  <li>The mat image and model positions come from the official BIOGLOW field setup guide and wireframe (2000 × 1140 mm mat, 487 mm launch areas).</li>
  <li>Motors accelerate and brake like SPIKE motors; a blocked robot stalls (commands end after ~0.5 s). Driving into a wall at an angle pivots the robot, so wall squaring works.</li>
  <li><b>Realism</b> (Setup) adds small wheel and gyro errors – like a real robot, odometry alone drifts. Use the gyro, lines and walls to be precise.</li>
  <li>The colour sensor reads the actual mat picture under the sensor.</li>
  <li>Not simulated: 3D mechanics, robots pushing models, the opposing team.</li>
</ul>
<p class="small-note">FIRST®, FIRST® LEGO® League and BIOGLOW™ are trademarks of FIRST and the LEGO Group. This simulator is an unofficial practice tool. Blockly (Apache 2.0) and CodeMirror (MIT) are included in the vendor folder.</p>
</div>`;
  }

  FLL.examples = { list: LIST, instantiate, template, defaultPrograms, helpHtml, fill };
})(typeof globalThis !== 'undefined' ? globalThis : window);
