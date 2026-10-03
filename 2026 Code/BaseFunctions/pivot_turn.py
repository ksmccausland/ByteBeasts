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
