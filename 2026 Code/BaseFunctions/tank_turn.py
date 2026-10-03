import runloop, motor_pair
from hub import motion_sensor, port

LEFT_DRIVE = port.A
RIGHT_DRIVE = port.B

motor_pair.pair(motor_pair.PAIR_1, LEFT_DRIVE, RIGHT_DRIVE)
DRIVE_PAIR = motor_pair.PAIR_1


async def proportional_turn(
    target_angle,
    *,
    max_speed=1000,
    acceleration=500,
    kp=0.7,
    accuracy=2,
    beep=True
):
    # Convert degrees to decidegrees
    target_angle *= 10
    accuracy *= 10

    # Reset heading
    motion_sensor.reset_yaw(0)

    try:
        while True:
            # Store current yaw and remaining angle to turn
            current_angle = motion_sensor.tilt_angles()[0]
            error = target_angle - current_angle

            # Wrap error to take the shortest turn
            error = (error + 1800) % 3600 - 1800

            if abs(error) <= accuracy:
                break

            # Proportional correction only
            correction = kp * error

            # Limit motor speed
            correction = max(-max_speed, min(max_speed, correction))

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
