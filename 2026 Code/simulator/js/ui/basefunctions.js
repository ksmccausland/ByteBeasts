/*
 * BaseFunctions for the recorder.
 * This is the code of "2026 Code/BaseFunctions/example_program.py" (drive_straight, pivot_turn, tank_turn) with the
 * recommendations of BaseFunctions/Suggestions.md applied, so a recorded drive calls the same functions the team uses
 * on the real robot. build(cfg) returns the imports, the robot settings (taken from the Robot panel) and the functions.
 *
 * Changes from example_program.py:
 *   item 1   import time (the file called time.ticks_ms() without importing it)
 *   item 3   timeout_ms: 0 or None means no timeout in all three functions
 *   item 4   MOTOR_INVERT is applied to the finished wheel speeds, so steering and turns follow it too
 *   item 4b  YAW_SIGN says which way the hub's yaw counts
 *   item 6   the faster wheel keeps headroom, so the heading correction still works at full speed
 *   item 8   typos fixed; pivot_turn and tank_turn print their progress only with debug=True
 */
(function (G) {
  'use strict';
  const FLL = G.FLL = G.FLL || {};
  const PORTS = ['A', 'B', 'C', 'D', 'E', 'F'];

  const HEADER = `# Recorded in the simulator with the team's BaseFunctions (2026 Code/BaseFunctions) and the
# recommendations of BaseFunctions/Suggestions.md. To use it on the real robot, replace the settings
# below with that robot's own (ports, wheel size, MOTOR_INVERT and YAW_SIGN, see Suggestions.md).
# Differences from example_program.py: items 1, 3, 4, 4b, 6 and 8 of Suggestions.md, and
# pivot_turn / tank_turn only print their progress when called with debug=True.
`;

  const FUNCTIONS = `async def drive_straight(
    distance_in,          # distance to drive in inches
    speed=400,            # speed of robot **NOTE max speed is 1050
    *,                    # optional values
    acceleration=2000,    # how fast robot will accelerate
    kp=1.2,               # correction strength: higher number means robot will correct more strongly (change only for heavy attachments)
    beep=True,            # robot will beep when function has completed
    timeout_ms=0          # Set to 0 to disable the timeout completely (0 or None = no timeout)
):
    if distance_in == 0:
        motor_pair.stop(DRIVE_PAIR)
        return

    # Zero sensors.
    motion_sensor.reset_yaw(0)
    motor.reset_relative_position(LEFT_DRIVE, 0)
    motor.reset_relative_position(RIGHT_DRIVE, 0)

    target_heading = 0
    current_deg = 0

    # Distance sign determines travel direction.
    direction = 1 if distance_in > 0 else -1
    drive_speed = min(abs(speed), MAX_SPEED) * direction

    # Convert travel distance in inches to wheel rotation.
    total_deg = (abs(distance_in) * 360 / (3.14159 * WHEEL_DIAMETER_IN))

    # start timer
    start_time = time.ticks_ms()

    try:
        # loop until distance is travelled
        while current_deg < total_deg:
            # Check timeout only if timeout_ms is provided and not 0
            if timeout_ms and timeout_ms > 0:
                elapsed_time = time.ticks_diff(
                    time.ticks_ms(), start_time
                )
                if elapsed_time >= timeout_ms:
                    print("DRIVE TIMEOUT!")
                    break

            # Yaw is measured in tenths of a degree
            current_heading = YAW_SIGN * motion_sensor.tilt_angles()[0]
            error = target_heading - current_heading

            # Wrap error to +/-180 degrees
            error = (error + 1800) % 3600 - 1800

            # Proportional correction only
            correction = kp * error

            # leave room for the correction so the faster wheel is not clipped at MAX_SPEED
            correction = max(-MAX_SPEED, min(MAX_SPEED, correction))
            limit = MAX_SPEED - abs(correction)
            base_speed = max(-limit, min(limit, drive_speed))
            left_speed = int((base_speed + correction) * MOTOR_INVERT)
            right_speed = int((base_speed - correction) * MOTOR_INVERT)

            # drive robot
            motor_pair.move_tank(
                DRIVE_PAIR,
                left_speed,
                right_speed,
                acceleration=acceleration
            )

            # timer to prevent taxing the controller and allow other functions
            await runloop.sleep_ms(10)

            # update distance travelled based on average of both drive motors
            current_deg = (abs(motor.relative_position(LEFT_DRIVE)) + abs(motor.relative_position(RIGHT_DRIVE))) / 2

    finally:
        motor_pair.stop(DRIVE_PAIR)

        if beep:
            sound.beep(1000)

async def pivot_turn(
    target_angle,        # number of degrees to turn (positive = clockwise, to the right)
    drive_wheel="l",     # select which wheel will be driven, either "l" or "r"
    *,
    min_speed=200,       # Minimum speed to overcome friction (0-1000 scale)
    acceleration=500,    # how fast robot will accelerate
    kp=2.5,              # correction strength: higher number means robot will correct more strongly (change only for heavy attachments)
    accuracy=2,          # how close to target angle is acceptable
    beep=True,           # if True robot will beep when function has completed
    timeout_ms=5000,     # how long the function has to run in milliseconds (0 or None = no timeout)
    debug=False          # if True, print the angle, the target and the correction on every loop
):
    drive_wheel = drive_wheel.lower()
    if drive_wheel not in ("l", "r"):
        raise ValueError("drive_wheel must be 'l' or 'r'")

    # Convert degrees to decidegrees
    target_angle *= 10
    accuracy *= 10

    # Reset heading
    motion_sensor.reset_yaw(0)

    # Start timeout timer
    start_time = time.ticks_ms()

    try:
        while True:
            # Store current yaw and remaining angle to turn
            current_angle = YAW_SIGN * motion_sensor.tilt_angles()[0]
            error = target_angle - current_angle

            # Wrap error to take the shortest turn
            error = (error + 1800) % 3600 - 1800

            if abs(error) <= accuracy:
                break

            # Check timeout only if timeout_ms is provided and not 0
            if timeout_ms and timeout_ms > 0:
                elapsed_time = time.ticks_diff(
                    time.ticks_ms(), start_time
                )

                if elapsed_time >= timeout_ms:
                    print("PIVOT TIMEOUT!")
                    break

            # Calculate Proportional correction
            correction = kp * error

            # Inject minimum speed so the motors actually turn
            if correction > 0:
                correction = max(min_speed, correction)
            elif correction < 0:
                correction = min(-min_speed, correction)

            # Limit motor speed to maximum allowed
            correction = max(
                -MAX_SPEED, min(MAX_SPEED, correction)
            )

            # Use the same turn direction as tank_turn
            if drive_wheel == "l":
                left_speed = int(correction) * MOTOR_INVERT
                right_speed = 0
            else:
                left_speed = 0
                right_speed = -int(correction) * MOTOR_INVERT

            motor_pair.move_tank(
                DRIVE_PAIR,
                left_speed,
                right_speed,
                acceleration=acceleration
            )

            if debug:
                print(
                    str(current_angle / 10)
                    + " | "
                    + str(target_angle / 10)
                    + " | "
                    + str(int(correction))
                )

            # Allow for other functions to execute
            await runloop.sleep_ms(10)

    finally:
        motor_pair.stop(DRIVE_PAIR)

        if beep:
            sound.beep(3000)

async def tank_turn(
    target_angle,        # number of degrees to turn the robot (positive = clockwise, to the right)
    *,
    min_speed=100,       # Minimum speed to overcome friction (0-1000 scale)
    acceleration=500,    # how fast robot will accelerate
    kp=2.5,              # correction strength: higher number means robot will correct more strongly (change only for heavy attachments)
    accuracy=2,          # how close to target angle is acceptable
    beep=True,           # if True robot will beep when function has completed
    timeout_ms=5000,     # how long the function has to run in milliseconds (0 or None = no timeout)
    debug=False          # if True, print the angle, the target and the correction on every loop
):
    # Convert degrees to decidegrees
    target_angle *= 10
    accuracy *= 10

    # Reset heading
    motion_sensor.reset_yaw(0)

    # Start timeout timer
    start_time = time.ticks_ms()

    try:
        while True:
            # Store current yaw and remaining angle to turn
            current_angle = YAW_SIGN * motion_sensor.tilt_angles()[0]
            error = target_angle - current_angle

            # Wrap error to take the shortest turn
            error = (error + 1800) % 3600 - 1800

            if abs(error) <= accuracy:
                break

            # Check timeout only if timeout_ms is provided and not 0
            if timeout_ms and timeout_ms > 0:
                elapsed_time = time.ticks_diff(
                    time.ticks_ms(), start_time
                )

                if elapsed_time >= timeout_ms:
                    print("TURN TIMEOUT!")
                    break

            # Calculate proportional correction
            correction = kp * error

            # Inject minimum speed so the motors actually turn
            if correction > 0:
                correction = max(min_speed, correction)
            elif correction < 0:
                correction = min(-min_speed, correction)

            # Limit motor speed
            correction = max(
                -MAX_SPEED, min(MAX_SPEED, correction)
            )

            # Turn robot
            motor_pair.move_tank(
                DRIVE_PAIR,
                int(correction) * MOTOR_INVERT,
                -int(correction) * MOTOR_INVERT,
                acceleration=acceleration
            )

            if debug:
                print(
                    str(current_angle / 10)
                    + " | "
                    + str(target_angle / 10)
                    + " | "
                    + str(int(correction))
                )

            # Allow for other functions to execute
            await runloop.sleep_ms(10)

    finally:
        motor_pair.stop(DRIVE_PAIR)

        if beep:
            sound.beep(2000)
`;

  // Motors that are not driving a wheel: the first one is LEFT_ATTACH, the second RIGHT_ATTACH
  // (a robot may have at most four motors, so there are never more than two).
  function attachPorts(cfg) {
    const dev = cfg.devices || {};
    return PORTS.filter(p => dev[p] && dev[p].type === 'motor' && p !== cfg.driveLeft && p !== cfg.driveRight).slice(0, 2);
  }

  // How the recorded program names a motor port: the BaseFunctions setting for an attachment motor, else port.X.
  function portRef(cfg, p) {
    const i = cfg ? attachPorts(cfg).indexOf(p) : -1;
    return i < 0 ? 'port.' + p : (i === 0 ? 'LEFT_ATTACH' : 'RIGHT_ATTACH');
  }

  // Rebuilt on every frame while recording, so the text for one robot setup is kept.
  let memo = { key: '', text: '' };

  function build(cfg) {
    const attach = attachPorts(cfg);
    const key = [cfg.wheelDiameter, cfg.driveLeft, cfg.driveRight, attach.join('')].join('|');
    if (memo.key === key) return memo.text;
    const cm = +(cfg.wheelDiameter / 10).toFixed(3);
    const row = (code, note) => (code.length < 38 ? code.padEnd(38) : code + '  ') + '# ' + note;
    const lines = [
      HEADER + 'import runloop, motor_pair, motor, color, color_sensor, time',
      'from hub import light, motion_sensor, port, sound, button, light_matrix',
      '',
      row('WHEEL_DIAMETER_IN = ' + cm + ' / 2.54', 'diameter of wheels in inches (' + cfg.wheelDiameter + ' mm in the Robot panel)'),
      row('MOTOR_INVERT = 1', 'use -1 if positive speeds do not drive the robot forward (the simulator drives forward)'),
      row('YAW_SIGN = -1', '1 if the hub\'s yaw goes UP when the robot turns clockwise (seen from above), -1 if it goes DOWN (the simulator\'s does)'),
      row('MAX_SPEED = 1050', 'max speed of the robot (DO NOT CHANGE)'),
      row('LEFT_DRIVE = port.' + cfg.driveLeft, 'port of left drive motor'),
      row('RIGHT_DRIVE = port.' + cfg.driveRight, 'port of right drive motor')
    ];
    if (attach[0]) lines.push(row('LEFT_ATTACH = port.' + attach[0], 'port of left attachment motor'));
    if (attach[1]) lines.push(row('RIGHT_ATTACH = port.' + attach[1], 'port of right attachment motor'));
    lines.push('', 'motor_pair.pair(motor_pair.PAIR_1, LEFT_DRIVE, RIGHT_DRIVE)', 'DRIVE_PAIR = motor_pair.PAIR_1', '');
    memo = { key: key, text: lines.join('\n') + '\n' + FUNCTIONS };
    return memo.text;
  }

  FLL.baseFunctions = { build: build, portRef: portRef, attachPorts: attachPorts, FUNCTIONS: FUNCTIONS };
})(typeof globalThis !== 'undefined' ? globalThis : window);
