from hub import port, motion_sensor
import runloop
import motor_pair
import motor
import time

WHEEL_DIAMETER_IN = 5.6 / 2.54
MOTOR_INVERT = -1
MAX_SPEED = 1050
LEFT_DRIVE = port.A
RIGHT_DRIVE = port.C

motor_pair.pair(motor_pair.PAIR_1, LEFT_DRIVE, RIGHT_DRIVE)
DRIVE_PAIR = motor_pair.PAIR_1

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
