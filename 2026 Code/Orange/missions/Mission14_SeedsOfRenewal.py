import runloop, motor_pair, motor, color, color_sensor, time
from hub import light, motion_sensor, port, sound, button, light_matrix

WHEEL_DIAMETER_IN = 5.6 / 2.54        # diameter of wheels in measured in inches
MOTOR_INVERT = 1                    # use to invert motors if positve drive directions do no result in forward motion
MAX_SPEED = 1050                    # max speed of the robot (DO NOT CHANGE)
LEFT_DRIVE = port.A                # port of left drive motor
RIGHT_DRIVE = port.E                # port of right drive motor
LEFT_ATTACH = port.B                # port of left attachment motor
RIGHT_ATTACH = port.D                # port of right attachment motor

motor_pair.pair(motor_pair.PAIR_1, LEFT_DRIVE, RIGHT_DRIVE)
DRIVE_PAIR = motor_pair.PAIR_1

async def drive_straight(
    distance_in,        # distance to drive in inches
    speed=400,            # speed of robot **NOTE max speed is 1050
    *,                    # optional values
    acceleration=2000,    # how fast robot will accelerate
    kp=1.2,            # correction strength: higher number means robot will correct more strongly (change only for heavy attachments)
    beep=True,            # robot will beep when function has completed
    timeout_ms=0        # Set to 0 to disable the timeout completely
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
    drive_wheel="l",    # select which wheel will be driven, either "l" or "r"
    *,
    min_speed=200,    # Minimum speed to overcome friction (0-1000 scale)
    acceleration=500,    # how fast robot will accelerate
    kp=2.5,            # correction strength: higher number means robot will correct more strongly (change only for heavy attachments)
    accuracy=2,        # how close to target angle is acceptable
    beep=True,        # if True robot will beep when function has completed
    timeout_ms=5000    # how long the function has to run in milliseconds
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
    min_speed=100,    # Minimum speed to overcome friction (0-1000 scale)
    acceleration=500,    # how fast robot will accelerate
    kp=2.5,            # correction strength: higher number means robot will correct more strongly (change only for heavy attachments)
    accuracy=2,        # how close to target angle is acceptable
    beep=True,        # if True robot will beep when function has completed
    timeout_ms=5000    # how long the function has to run in milliseconds
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

# Mission 14 - Seeds of Renewal
# Plant collected seeds in the replantation station: 5 points each, +5 each if they touch the mat.
# This run collects the seed from the Mission 9 tree (the tree seed is always in the same place) and plants it.
# The Mission 2 seeds are NOT collected: they land at a random spot every time, so a fixed route cannot pick them up.
#
# START: robot completely inside the LEFT launch area, wheels at x = 220 mm, y = 140 mm, nose pointing NORTH
# (the same start as Mission 1). The attachment motor on port D (RIGHT_ATTACH) is on the front.
#
# DOCK: written for the CITY dock (the simulator default: Mission 13 on the mine dock, Mission 14 on the city dock,
# Mission 15 on the farm dock). The station is at x = 1279 mm, y = 93 mm. If Mission 14 sits on another dock
# the second half of the route has to change.
#
# ROUTE: east to x = 480 mm, north to y = 840 mm, tool on the seed at the south-west of the tree, back south to
# y = 260 mm, east to x = 1060 mm, south to y = 160 mm and face east, tool on the station, north to y = 260 mm and
# west to the launch area. y = 260 mm is the same lane Mission 1 uses: it passes above the LiDAR scan marker.
# The station is approached at y = 160 mm, not at its centre (y = 93 mm): the robot's nose is 112 mm ahead of the
# wheels, so driving south any further than y = 160 mm would push it into the south wall.
#
# Clearance: at x = 480 mm the robot's right edge is about 45 mm from the Mission 2 stalk (the stalk starts at x = 603 mm).
# Do not drive further north than y = 840 mm: the Mission 9 tree starts at y = 996 mm.
#
# It has NOT been run in the simulator or on the real robot: tune the numbers on the table.
MM = 1 / 25.4                       # drive_straight takes inches

TOOL = RIGHT_ATTACH                 # attachment motor that picks up and plants the seeds
TOOL_DEGREES = 90                   # how far the attachment turns for one action (use -90 to turn the other way)
TOOL_SPEED = 500                    # attachment speed
SPEED = 500                         # drive speed
SETTLE_MS = 400                     # pause after every drive and turn

TO_TREE_X_MM = 260                  # east from the start (x = 220) to x = 480
TO_TREE_Y_MM = 700                  # north from y = 140 to y = 840
BACK_TO_LANE_MM = 580               # south from y = 840 to y = 260
ACROSS_MM = 580                     # east along y = 260 from x = 480 to x = 1060
DOWN_TO_STATION_MM = 100            # south from y = 260 to y = 160
HOME_MM = 840                       # west along y = 260 from x = 1060 to x = 220

# tank_turn: with MOTOR_INVERT = 1 (positive speeds drive forward) a positive angle turns the robot to the LEFT and a
# negative angle to the RIGHT. If your robot turns the other way, check MOTOR_INVERT and the hub's yaw direction.
LEFT = 90
RIGHT = -90

async def drive(mm, timeout_ms):
    await drive_straight(mm * MM, SPEED, timeout_ms=timeout_ms)
    # Let the wheels stop completely so the next turn starts from standstill.
    await runloop.sleep_ms(SETTLE_MS)

async def turn(angle):
    await tank_turn(angle)
    await runloop.sleep_ms(SETTLE_MS)

async def act(degrees):
    # one tool action, then back to the start position so the attachment is not in the way
    await motor.run_for_degrees(TOOL, degrees, TOOL_SPEED)
    await motor.run_for_degrees(TOOL, -degrees, TOOL_SPEED)

async def main():
    await turn(RIGHT)                               # facing east
    await drive(TO_TREE_X_MM, 5000)
    await turn(LEFT)                                # facing north
    await drive(TO_TREE_Y_MM, 8000)

    await act(TOOL_DEGREES)                         # take the seed from the tree (Mission 9 seed, +10)

    await drive(-BACK_TO_LANE_MM, 8000)             # back down to the y = 260 mm lane
    await turn(RIGHT)                               # facing east
    await drive(ACROSS_MM, 8000)
    await turn(RIGHT)                               # facing south
    await drive(DOWN_TO_STATION_MM, 5000)
    await turn(LEFT)                                # facing east, towards the station

    await act(TOOL_DEGREES)                         # plant the seed (Mission 14)

    await turn(LEFT)                                # facing north
    await drive(DOWN_TO_STATION_MM, 5000)
    await turn(LEFT)                                # facing west, towards home
    await drive(HOME_MM, 12000)

runloop.run(main())
