import runloop, motor_pair, motor, color, color_sensor
from hub import light, motion_sensor, port, sound, button, light_matrix

WHEEL_DIAMETER_IN = 5.6 / 2.54        # diameter of wheels in measured in inches
MOTOR_INVERT = -1                     # use to invert motors if positve drive directions do no result in forward motion
MAX_SPEED = 1050                      # max speed of the robot (DO NOT CHANGE)
LEFT_DRIVE = port.A                   # port of left drive motor
RIGHT_DRIVE = port.C                  # port of right drive motor

motor_pair.pair(motor_pair.PAIR_1, LEFT_DRIVE, RIGHT_DRIVE)
DRIVE_PAIR = motor_pair.PAIR_1

async def drive_straight(
    distance_in,          # distance to drive in inches
    speed=400,            # speed of robot **NOTE max speed is 1050
    *,                    # optional values
    acceleration=2000,    # how fast robot will accelerate
    kp=1.2,               # correction strength: higher number means robot will correct more strongly (change only for heavy attachments)
    beep=True,            # robot will beep when function has completed
    timeout_ms=0          # Set to 0 to disable the timeout completely
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
    drive_speed = min(abs(speed), MAX_SPEED) * direction * MOTOR_INVERT

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
            current_heading = motion_sensor.tilt_angles()[0]
            error = target_heading - current_heading

            # Wrap error to +/-180 degrees
            error = (error + 1800) % 3600 - 1800

            # Proportional correction only
            correction = kp * error

            # set speeds for wheels to correct for heading error
            left_speed = int(max(-MAX_SPEED, min(MAX_SPEED, drive_speed - correction)))
            right_speed = int(max(-MAX_SPEED, min(MAX_SPEED, drive_speed + correction)))

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
    target_angle,        # number of degrees to turn
    drive_wheel="l",     # select which wheel will be driven, either "l" or "r"
    *,
    min_speed=200,       # Minimum speed to overcome friction (0-1000 scale)
    acceleration=500,    # how fast robot will accelerate
    kp=2.5,              # correction strength: higher number means robot will correct more strongly (change only for heavy attachments)
    accuracy=2,          # how close to target angle is acceptable
    beep=True,           # if True robot will beep when function has completed
    timeout_ms=5000      # how long the function has to run in milliseconds
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
            current_angle = motion_sensor.tilt_angles()[0]
            error = target_angle - current_angle

            # Wrap error to take the shortest turn
            error = (error + 1800) % 3600 - 1800

            if abs(error) <= accuracy:
                break

            # Check timeout (None disables it)
            if timeout_ms is not None:
                elapsed_time = time.ticks_diff(
                    time.ticks_ms(), start_time
                )

                if elapsed_time >= timeout_ms:
                    print("PIVOT TIMEOUT!")
                    break

            # Calculate Proportional correction
            correction = kp * error

            # NEW: Inject minimum speed so the motors actually turn
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
                left_speed = -int(correction)
                right_speed = 0
            else:
                left_speed = 0
                right_speed = int(correction)

            motor_pair.move_tank(
                DRIVE_PAIR,
                left_speed,
                right_speed,
                acceleration=acceleration
            )

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
    target_angle,        # number of degress to turn the robot
    *,
    min_speed=100,       # Minimum speed to overcome friction (0-1000 scale)
    acceleration=500,    # how fast robot will accelerate
    kp=2.5,              # correction strength: higher number means robot will correct more strongly (change only for heavy attachments)
    accuracy=2,          # how close to target angle is acceptable
    beep=True,           # if True robot will beep when function has completed
    timeout_ms=5000      # how long the function has to run in milliseconds
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
            current_angle = motion_sensor.tilt_angles()[0]
            error = target_angle - current_angle

            # Wrap error to take the shortest turn
            error = (error + 1800) % 3600 - 1800

            if abs(error) <= accuracy:
                break

            # Check timeout only if timeout_ms is provided and not 0
            if timeout_ms > 0:
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
                -int(correction),
                int(correction),
                acceleration=acceleration
            )

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

async def main():
  # your function goes here
  await drive_straight('distance in inches', 'speed (1050=max)')
  await tank_turn('degrees to turn')
  await runloop.run(drive_straight(distance, speed), motor.run_for_degrees('motor to spin', 'rotate', 'rate'))

runloop.run(main())
