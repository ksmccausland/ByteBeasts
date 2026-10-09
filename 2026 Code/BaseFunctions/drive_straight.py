from hub import port, motion_sensor,sound
import runloop
import motor_pair
import motor
import time

WHEEL_DIAMETER_IN = 6 / 2.54
MOTOR_INVERT = -1
MAX_SPEED = 1050
LEFT_DRIVE = port.A
RIGHT_DRIVE = port.C

motor_pair.pair(motor_pair.PAIR_1, LEFT_DRIVE, RIGHT_DRIVE)
DRIVE_PAIR = motor_pair.PAIR_1


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


async def main():
    await drive_straight(
        20,
        500,
        use_timeout=True,
        timeout_ms=5000
    )


runloop.run(main())
