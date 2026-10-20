import runloop, motor_pair, time
from hub import motion_sensor, port, sound

LEFT_DRIVE = port.A
RIGHT_DRIVE = port.B

motor_pair.pair(motor_pair.PAIR_1, LEFT_DRIVE, RIGHT_DRIVE)
DRIVE_PAIR = motor_pair.PAIR_1


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
