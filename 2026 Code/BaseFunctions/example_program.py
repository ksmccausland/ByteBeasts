import runloop, motor_pair, motor, color, color_sensor
from hub import light, motion_sensor, port, sound, button, light_matrix

WHEEL_DIAMETER_IN = 5.6 / 2.54
MOTOR_INVERT = -1
MAX_SPEED = 1050
LEFT_DRIVE = port.A
RIGHT_DRIVE = port.C

motor_pair.pair(motor_pair.PAIR_1, LEFT_DRIVE, RIGHT_DRIVE)
DRIVE_PAIR = motor_pair.PAIR_1

# 
async def drive_straight(
    distance_in,
    speed=400,
    *,
    acceleration=2000,
    kp=1.2,
    use_timeout=False,
    beep=True,
    timeout_ms=5000
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
            # timeout checking
            if use_timeout:
                elapsed_time = time.ticks_diff(
                    time.ticks_ms(), start_time
                )
                if elapsed_time >= timeout_ms:
                    print("TIMEOUT!")
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
    target_angle,
    drive_wheel="l",      # select which wheel will turn, either "l" or "r"
    *,
    max_speed=1000,
    acceleration=500,
    kp=0.7,
    accuracy=2,
    beep=True,
    timeout_ms=5000
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

            # Proportional correction only
            correction = kp * error

            # Limit motor speed
            correction = max(
                -max_speed, min(max_speed, correction)
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
    target_angle,
    *,
    max_speed=1000,
    acceleration=500,
    kp=0.7,
    accuracy=2,
    beep=True,
    timeout_ms=5000
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

            # Check timeout
            if timeout_ms is not None:
                elapsed_time = time.ticks_diff(
                    time.ticks_ms(), start_time
                )

                if elapsed_time >= timeout_ms:
                    print("TURN TIMEOUT!")
                    break

            # Proportional correction only
            correction = kp * error

            # Limit motor speed
            correction = max(
                -max_speed, min(max_speed, correction)
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
