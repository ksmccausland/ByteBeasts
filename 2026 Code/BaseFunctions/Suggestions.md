# BaseFunctions: suggestions

Review of the functions in this folder (`example_program.py`, `drive_straight.py`, `pivot_turn.py`, `tank_turn.py`) and of the mission files that carry copies of them (`Orange/missions/7and11.py`, `Orange/missions/Mission6_LeafCutter_Eloise_v1.py`, `Orange/katydid-mission-4.llsp3`).

Written with Claude Code on 2026-11-15, against commit `e617a21`. Line numbers are for `example_program.py` at that commit unless another file is named.

**How far to trust it.** Every finding comes from reading the code, or from running it on a PC against stand-in hub modules and a simulated robot (see [How this was checked](#how-this-was-checked)). **Nothing was run on a robot**, and no code file was changed. Where the table says *robot first*, try the change on a robot before pasting it into missions.

## Summary

| # | Suggestion | Kind | Who is affected | Robot first? |
|---|---|---|---|---|
| 1 | Add `import time` to `example_program.py` | Crash | Anyone starting a new mission from the example | No |
| 2 | Fix or delete the three standalone files | Crash | Anyone running them on their own | No |
| 3 | Give `timeout_ms` one meaning | Inconsistent | No current mission (only shows with `0` or `None` on a turn) | No |
| 4 | Make `MOTOR_INVERT` safe to change | Wrong behaviour | Any robot whose `MOTOR_INVERT` doesn't match its hub's yaw direction; check Katydid (`MOTOR_INVERT = 1`) | Yes |
| 5 | Settle the wheel diameter for each robot | Team decision | `drive_straight.py` and Katydid use 6.5 cm, the rest 5.6 cm | Yes |
| 6 | Keep full steering strength at high speed | Improvement | Drives at or near `MAX_SPEED` | Yes |
| 7 | Stop pasting the functions by hand | Structure | Anyone who fixes a function: there are up to 7 copies | No |
| 8 | Small tidy-ups | Tidy | | No |

Suggested order: items 1 and 3 now (they change nothing for a mission that works today), then the yaw check at the end of this document, then item 4. Decide items 2 and 5 as a team, try item 6 on one robot, and build item 7 when you want to change how missions are made.

---

## 1. `example_program.py` is missing `import time`

`drive_straight`, `pivot_turn` and `tank_turn` all call `time.ticks_ms()` (lines 44, 115 and 203), but line 1 does not import `time`. The first call stops the program with `NameError: name 'time' is not defined`, before the robot moves. `7and11.py`, the Mission 6 file and Katydid all import it, so this only bites someone who starts a new mission from the example file.

```diff
@@ line 1 @@
-import runloop, motor_pair, motor, color, color_sensor
+import runloop, motor_pair, motor, color, color_sensor, time
```

*Simulation:* `drive_straight(10)` raises the NameError as shipped and completes with the import added.

## 2. The three standalone files don't run on their own

| File | What happens when it runs alone | Cause |
|---|---|---|
| `tank_turn.py` | `NameError: name 'MAX_SPEED' is not defined` on the first pass through the loop (line 64), even with `beep=False` | `MAX_SPEED` is never defined. Line 5 also sets `RIGHT_DRIVE = port.B`, which is the left attachment motor in the example's wiring, so on that robot it would drive the wrong motor |
| `pivot_turn.py` | The turn works and the motors stop, then `NameError: name 'sound' is not defined` (line 107). With `beep=False` it completes | Line 1 imports `port, motion_sensor` but not `sound` |
| `drive_straight.py` | Runs | Uses `6.5 / 2.54` for the wheel diameter (line 7); the other files use `5.6 / 2.54`. See item 5 |

**Suggestion:** `example_program.py` already contains all three functions, so make it the single source and delete the three small files. If you want them as teaching files, generate them from the example (item 7) instead of editing them by hand, because hand-edited copies drift.

*Simulation:* each file was run alone with and without `beep=False`. Supplying the missing names made `tank_turn.py` and `pivot_turn.py` complete.

## 3. `timeout_ms` means different things in the three functions

| `timeout_ms=` | `drive_straight` | `pivot_turn` | `tank_turn` |
|---|---|---|---|
| `0` | No timeout | Times out at once, so the robot never moves | No timeout |
| `None` | No timeout | No timeout | `TypeError` |
| `1000` | Stops after 1 s | Stops after 1 s | Stops after 1 s |

The comments disagree as well: `drive_straight` says "0 disables" (line 22), `pivot_turn` says "None disables it" (line 129). No current mission passes `0` or `None` to a turn, so nothing is broken today.

**Suggestion:** treat `0` and `None` as "no timeout" everywhere. Keep the defaults as they are (drives: off, turns: 5000 ms) if that is on purpose.

```diff
@@ pivot_turn, lines 129-130 @@
-            # Check timeout (None disables it)
-            if timeout_ms is not None:
+            # Check timeout only if timeout_ms is provided and not 0
+            if timeout_ms and timeout_ms > 0:
```

```diff
@@ tank_turn, line 218 @@
-            if timeout_ms > 0:
+            if timeout_ms and timeout_ms > 0:
```

Also change the three `timeout_ms` parameter comments (lines 22, 101, 193) to end with `(0 or None = no timeout)`.

*Simulation (wheels held still for 4 s):* after the change, `0` and `None` never time out in all three functions and `1000` stops at 1000 ms in all three.

## 4. `MOTOR_INVERT` changes the drive direction but not the steering

The comment on line 5 says to flip `MOTOR_INVERT` if positive speeds don't drive the robot forward. The code uses it in only one place, line 38 (`drive_speed`). The signs of the heading correction (lines 69-70) and of the two turns (lines 155, 159, 244-245) are written once and never look at it. So the shipped code steers correctly for only two pairings of `MOTOR_INVERT` and the hub's yaw direction, and in those two a positive angle means opposite turns (table below). With any other pairing the robot *fights* itself: in the simulation a 5° nudge during `drive_straight` grew to about 180° off course instead of shrinking, and `tank_turn(90)` and `pivot_turn(90, 'l')` ran until their 5 s timeout and ended near -90° instead of +90°.

| | Yaw goes **up** when the robot turns clockwise | Yaw goes **up** when the robot turns counter-clockwise |
|---|---|---|
| Shipped code, `MOTOR_INVERT = -1` | works; a positive angle is a right turn | fights the robot |
| Shipped code, `MOTOR_INVERT = 1` | fights the robot | works; a positive angle is a left turn |
| With the patch below, either `MOTOR_INVERT` | works; a positive angle is a right turn | fights the robot |
| With the patch and `YAW_SIGN` (4b), either `MOTOR_INVERT` | works; a positive angle is a right turn | works; a positive angle is a right turn |

*Works* means that in the simulation `drive_straight` returned to its heading after a nudge and both turns finished inside the 2° accuracy window.

Which column is a given robot in? That is a property of its hub, so run the yaw check at the end of this document.

- Mission 6 and Mission 7/11 use `MOTOR_INVERT = -1`. If they hold a line and turn correctly today, their hubs are in the first column.
- Katydid uses `MOTOR_INVERT = 1`. If it holds a line and its turns reach their targets, it is in the second column, where a positive angle already means a left turn. If its turns spin or time out, it is in the first column and this item fixes it.

**Patch.** Apply the motor sign to the finished wheel speeds instead of only to the drive speed:

```diff
@@ drive_straight, line 38 @@
-    drive_speed = min(abs(speed), MAX_SPEED) * direction * MOTOR_INVERT
+    drive_speed = min(abs(speed), MAX_SPEED) * direction
```

```diff
@@ drive_straight, lines 69-70 @@
-            left_speed = int(max(-MAX_SPEED, min(MAX_SPEED, drive_speed - correction)))
-            right_speed = int(max(-MAX_SPEED, min(MAX_SPEED, drive_speed + correction)))
+            left_speed = int(max(-MAX_SPEED, min(MAX_SPEED, drive_speed + correction)) * MOTOR_INVERT)
+            right_speed = int(max(-MAX_SPEED, min(MAX_SPEED, drive_speed - correction)) * MOTOR_INVERT)
```

```diff
@@ pivot_turn, lines 154-159 @@
             if drive_wheel == "l":
-                left_speed = -int(correction)
+                left_speed = int(correction) * MOTOR_INVERT
                 right_speed = 0
             else:
                 left_speed = 0
-                right_speed = int(correction)
+                right_speed = -int(correction) * MOTOR_INVERT
```

```diff
@@ tank_turn, lines 242-247 @@
             motor_pair.move_tank(
                 DRIVE_PAIR,
-                -int(correction),
-                int(correction),
+                int(correction) * MOTOR_INVERT,
+                -int(correction) * MOTOR_INVERT,
                 acceleration=acceleration
             )
```

**Nothing changes for robots with `MOTOR_INVERT = -1`** (the example, Mission 6, Mission 7/11). The patched code sent exactly the same wheel commands as the shipped code in all 14 simulated runs (7 moves, each with the yaw reading both ways round). Only robots with `MOTOR_INVERT = 1` behave differently: their steering and turns flip, which is the fix. The sign of `MOTOR_INVERT` still sets which way a positive distance drives.

### 4b. Optional: `YAW_SIGN`, for a hub that reads yaw the other way

The patch above removes the dependence on the motor wiring but still assumes yaw goes up clockwise. If the yaw check shows a robot where it goes down (for example because its hub is mounted the other way up), add one setting and apply it where yaw is read:

```diff
@@ settings, between lines 5 and 6 @@
+YAW_SIGN = 1                          # 1 if the hub's yaw goes UP when the robot turns clockwise (seen from above), -1 if it goes DOWN
 MAX_SPEED = 1050                      # max speed of the robot (DO NOT CHANGE)
```

```diff
@@ drive_straight, line 59 @@
-            current_heading = motion_sensor.tilt_angles()[0]
+            current_heading = YAW_SIGN * motion_sensor.tilt_angles()[0]
```

```diff
@@ pivot_turn line 120 and tank_turn line 208 (same change in both) @@
-            current_angle = motion_sensor.tilt_angles()[0]
+            current_angle = YAW_SIGN * motion_sensor.tilt_angles()[0]
```

*Simulation:* with the item 4 patch and `YAW_SIGN` set to match the hub, all 9 test moves (drives at 400 and 1050 with nudges in both directions, `tank_turn` 90 and -135, `pivot_turn` 90 left, 90 right and -45 right) ended on target for both `MOTOR_INVERT = -1` and `1`, with yaw going up in either direction. Without `YAW_SIGN`, those 9 moves passed only when yaw goes up clockwise.

## 5. Two wheel diameters are in use

| `WHEEL_DIAMETER_IN` | Files |
|---|---|
| `5.6 / 2.54` | `example_program.py`, `pivot_turn.py`, Mission 6, Mission 7/11 |
| `6.5 / 2.54` | `drive_straight.py`, Katydid |

Drive distance depends directly on this number (line 41). A robot with 5.6 cm wheels running the 6.5 cm value turns its wheels 895.6° for `drive_straight(20)` instead of the 1039.5° that 20 inches needs, and travels about 17.2 inches. If Katydid really has 6.5 cm wheels its value is right; the risk is that the number is typed separately into every file.

**Suggestion:** keep each robot's settings (`WHEEL_DIAMETER_IN`, `MOTOR_INVERT`, `YAW_SIGN` if you add it, and the ports) together in one block at the top of its program, with the robot's name in a comment. To calibrate a robot, drive a measured distance and set `new diameter = old diameter x (distance it travelled / distance you asked for)`, assuming the wheels don't slip.

## 6. Heading correction loses strength near full speed

Each wheel gets `drive_speed +/- correction`, then is clipped to +/-1050. At full speed one wheel is already at the limit and can't speed up, so only the other wheel's slow-down counts and the robot steers at half strength. For a 5° error:

| Speed | Shipped wheel speeds (`MOTOR_INVERT = -1`) | Speed difference | With the patch below | Speed difference |
|---|---|---|---|---|
| 400 | left -340, right -460 | 120 | left -340, right -460 | 120 |
| 1050 | left -990, right -1050 | 60 | left -930, right -1050 | 120 |

The loss starts when `speed + 1.2 x error` passes 1050: above 4.2° of error at speed 1000, above 20.8° at 800, and for any error at 1050. Mission 6's second leg runs about 4.85 inches at `MAX_SPEED`, and Katydid drives 23 inches at 1000.

**Patch.** Reduce the base speed by the correction so the faster wheel never hits the limit. This replaces the lines 69-70 version from item 4 and includes the same `MOTOR_INVERT` handling:

```diff
@@ drive_straight, lines 69-70 (use instead of the version in item 4) @@
-            left_speed = int(max(-MAX_SPEED, min(MAX_SPEED, drive_speed - correction)))
-            right_speed = int(max(-MAX_SPEED, min(MAX_SPEED, drive_speed + correction)))
+            # leave room for the correction so the faster wheel is not clipped at MAX_SPEED
+            correction = max(-MAX_SPEED, min(MAX_SPEED, correction))
+            limit = MAX_SPEED - abs(correction)
+            base_speed = max(-limit, min(limit, drive_speed))
+            left_speed = int((base_speed + correction) * MOTOR_INVERT)
+            right_speed = int((base_speed - correction) * MOTOR_INVERT)
```

The trade-off is that the robot slows slightly while it is correcting, by up to the size of the correction.

*Simulation:* the wheel commands matched the shipped code in every run where the shipped code never reached the limit (all nudges at speeds 200 and 400, and nudges up to 20° at 800). They differ at 800 with a 30° nudge and at 1050 for any nudge, which is the intended change. The real-robot effect is unmeasured.

## 7. Each function exists in up to seven places

All copies have the same code (compared as syntax trees, so comments and spacing are ignored):

| Function | Copies | Where |
|---|---|---|
| `drive_straight` | 7 | `BaseFunctions/example_program.py`, `BaseFunctions/drive_straight.py`, `Orange/Mission6_LeafCutter_Eloise_v1.llsp3`, `Orange/missions/Mission6_LeafCutter_Eloise_v1.py`, `Orange/mission-7and11.llsp3`, `Orange/missions/7and11.py`, `Orange/katydid-mission-4.llsp3` |
| `pivot_turn` | 7 | the same seven files, with `pivot_turn.py` in place of `drive_straight.py` |
| `tank_turn` | 6 | the same files, without Katydid, with `tank_turn.py` in place of `drive_straight.py` |

Every fix in this document has to be pasted into all of them, and two of the places are `.llsp3` files, where the code sits inside a zip and can't be diffed. (`katydid-mission-4.llsp3` is plain Python text with an `.llsp3` extension rather than a zip.)

A Python `.llsp3` holds a single `main` program, so as far as the file format shows, a mission can't import shared code and pasting has to stay. What can change is who does the pasting:

- **Build script.** Each mission keeps only its settings block and `main()`. A small script joins them with the base block from `example_program.py` and writes both the `.py` and the `.llsp3` (an uncompressed zip, using `Orange/mission-7and11.llsp3` as the template). The Mission 6 `.llsp3` and `.py` were produced this way by hand.
- **Cheaper first step.** A script that only reports which mission files still carry an out-of-date copy of a function.

## 8. Small tidy-ups

- Typos in comments: line 4 `diameter of wheels in measured in inches`, line 5 `positve ... do no result`, line 186 `degress`.
- Line 142 still says `# NEW:`; the same comment in `tank_turn` doesn't.
- Both turn loops print a line every 10 ms (lines 168-174 and 249-255). That is useful while tuning. Whether it slows the loop on the hub is unmeasured; a `debug=False` parameter would let missions keep it quiet.
- Unused imports: `color`, `color_sensor`, `light`, `button`, `light_matrix`. Harmless.
- Mission 6 defines `ROTATION_IN = 3.14159 * WHEEL_DIAMETER_IN` for distances given in wheel rotations. Line 41 repeats the same product, so if more missions want it, it could be defined once next to `WHEEL_DIAMETER_IN`.

---

## How this was checked

- Stand-in modules replaced `runloop`, `motor_pair`, `motor` and `hub`. A virtual clock advanced only when the code called `runloop.sleep_ms`.
- A toy robot turned at a rate set by the difference between the wheel speeds (one fixed constant, for 5.6 cm wheels about 11 cm apart). Its motors respond instantly, with no friction, acceleration limit or gyro drift.
- That is enough to find crashes (`NameError`), wrong signs, whether a change alters the commands sent to the wheels, and which way the robot ends up pointing. Figures such as "ends at 88.4°" are the toy robot's, not predictions for a real one.
- It can't say whether `kp` or `min_speed` are well tuned, how the real motors and `acceleration` behave, or which way the real hub counts yaw.
- Hub API details came from the unofficial Tufts CEEO SPIKE Python docs (https://tuftsceeo.github.io/SPIKEPythonDocs/SPIKE3.html). The page was only partly read.
- The check scripts are in a temporary folder, not in the repo. They can be added, for example under a `tools/` folder, if you want them.

## What still needs a robot

Things I could not confirm:

- **Which way yaw counts on each robot's hub.** This decides item 4 (see the check below).
- **`await runloop.run(drive_straight(...), motor.run_for_degrees(...))` inside `main()`** (line 270 of the example). The docs I found only show `runloop.run` at the top level of a program, where it runs several functions in parallel. I found nothing showing it awaited inside a function. No mission uses this form, but the example teaches it.
- **Whether the end-of-function beeps play.** Lines 90, 183 and 264 call `sound.beep(...)` without `await`. The docs say it returns an awaitable and don't say whether it plays or blocks when not awaited. If a beep is silent, use `await sound.beep(...)`.
- **Mission 6's `MOTOR_INVERT`, and that port A is its left wheel.** The Mission 6 file uses `LEFT_DRIVE = port.A`, `RIGHT_DRIVE = port.E` and `MOTOR_INVERT = -1`, copied from the block version.
- **All patches above, on a real robot.**

### Quick checks

1. **Yaw check.** Run this, then turn the robot by hand **clockwise as seen from above** and watch the numbers. Rising means `YAW_SIGN = 1` (the code's assumption holds); falling means the robot needs `YAW_SIGN = -1` (item 4b).

   ```python
   from hub import motion_sensor
   import runloop

   async def main():
       motion_sensor.reset_yaw(0)
       for _ in range(100):
           print(motion_sensor.tilt_angles()[0])
           await runloop.sleep_ms(100)

   runloop.run(main())
   ```

2. **Beep check.** Run `drive_straight(5)`, `pivot_turn(30)` and `tank_turn(30)`. Does each one beep when it finishes?
3. **Steering check, after item 4.** Run `drive_straight(20, 400)` and tap the nose sideways while it drives. It should swing back onto its line, not spin away. Then run `tank_turn(90)` and `pivot_turn(90, 'l')`: both should turn right.
4. **Distance check.** Run `drive_straight(20)` and measure. If it is off, see the calibration rule in item 5.
5. **Full-speed check, before and after item 6.** Run `drive_straight(20, MAX_SPEED)` along a tape line and compare how straight it ends.
