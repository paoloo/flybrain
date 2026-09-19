"""tracking.py - make the fly brain follow a colored point with two virtual servos.

The servos are NOT real. The program only prints commands like:

    move: up-right
    move: left
    move: centered

You test it by moving your laptop (or the colored object) and watching the
printed commands follow the point.

Run:    conda run -n mcp python tracking.py
Quit:   press q in the video window

You need a colored object to track. Green or red works best; pick a color
that is rare in the room (the detector is a plain color score, see TARGET
below). Change CAMERA_INDEX if your webcam is not device 0.

TEACHING NOTES - how the brain is used here
-------------------------------------------
There is no training and no servo controller in the usual sense. Per camera
frame the program:

  1. pastes the camera image into the FRONT face of a 384x256 "cubemap",
     the six-face spherical image format the fly's eye expects
     (faces: front/right/back on the top row, left/up/down on the bottom).
     The other five faces stay black: the fly only sees what the camera sees.
  2. advances the whole 166,700-neuron connectome brain by one 20 ms tick
     (brain.step). Inside that tick the eye module samples the cubemap into
     6,006 photoreceptors and injects current into them; the rest of the
     network reacts through the measured wiring.
  3. reads TWO things back out of the brain:
       the EYE:  brain.previous_rgb holds the color each photoreceptor just
                 sampled. Each photoreceptor also knows its place in a
                 48x64 "column atlas" that maps to a viewing direction
                 (brain.visual_pixels: left eye = cols 0-31, right eye =
                 cols 32-63, rows top = up). A weighted average of the
                 photoreceptors that see the target color is therefore
                 "where the fly sees the point".
       the MOTOR pools: spike rates of descending neurons (shown in the
                 window as fwd/turn). Those are wired for the flying game,
                 so the servo decision here uses the eye signal only.
  4. converts the eye position into a servo command:
       atlas column left of center  -> "move left",   right -> "move right"
       atlas row    above center    -> "move up",     below -> "move down"
     A weighted average near the center is "centered" (servos hold).

The color score reuses the same green-opponency style math the brain's own
visual encoder uses (see flybrain/brain.py, encode_retina).

To swap in the tiny offline brain (no 1 GB data/cache needed, same API):
    brain = FlyBrain(fixture=True)
"""

import cv2
import numpy as np

from flybrain.brain import FlyBrain

CAMERA_INDEX = 0          # try 1, 2, ... if the wrong camera opens
TARGET = "green"          # "green", "red" or "blue": the point to follow

# Color-opponency weights (r, g, b): the target channel wins, the other two
# are subtracted. The -0.15 bias inside color_score() rejects weak background.
WEIGHTS = {
    "green": (-0.5, 1.0, -0.5),
    "red": (1.0, -0.5, -0.5),
    "blue": (-0.5, -0.5, 1.0),
}

# Atlas geometry: 48 rows x 64 cols of photoreceptor "columns".
# Azimuth 0 (straight ahead) sits at col ~33.8 of the RIGHT eye; the left eye
# covers -135..-8.5 degrees, the right eye -8.5..+135 degrees. There is a
# small mapping seam near the center, hence the wider left threshold below.
ROW_CENTER = 23.5         # elevation 0 degrees (the horizon)
COL_CENTER = 33.8         # azimuth 0 degrees
RIGHT_T, LEFT_T, VERT_T = 3.0, -6.0, 4.0   # dead zones in atlas units


def color_score(eye, target):
    """eye: (6006, 3) RGB in 0..1 sampled by the photoreceptors.
    Returns a score per photoreceptor; > 0 means 'this receptor sees the
    target color'."""
    wr, wg, wb = WEIGHTS[target]
    r, g, b = eye[:, 0], eye[:, 1], eye[:, 2]
    return np.clip(wr * r + wg * g + wb * b - 0.15, 0, None)


def eye_command(brain):
    """Turn the eye signal into a servo command string.

    Returns (command, row, col): the weighted average atlas position of the
    photoreceptors that see the target, or ("searching", None, None)."""
    score = color_score(brain.previous_rgb, TARGET)
    if score.sum() < 0.5:                       # nothing clearly colored seen
        return "searching", None, None
    w = score / score.sum()
    row = float((brain.visual_pixels[:, 0] * w).sum())
    col = float((brain.visual_pixels[:, 1] * w).sum())
    dx, dy = col - COL_CENTER, row - ROW_CENTER

    parts = []
    if dx > RIGHT_T:
        parts.append("right")
    elif dx < LEFT_T:
        parts.append("left")
    if dy > VERT_T:                             # larger row = lower in view
        parts.append("down")
    elif dy < -VERT_T:
        parts.append("up")
    return ("-".join(parts) if parts else "centered"), row, col


def draw_crosshair(view, row, col):
    """Draw where the FLY's eye thinks the point is, on the camera image."""
    # invert the atlas mapping: (row, col) -> azimuth/elevation -> pixels
    if col >= 32:
        az = -8.5 + (col - 32) / 31 * 143.5
    else:
        az = -135.0 + col / 31 * 143.5
    el = 72.0 - row / 47 * 144.0
    if abs(az) > 45 or abs(el) > 45:            # outside the camera's view
        return
    u, v = np.tan(np.deg2rad(az)), np.tan(np.deg2rad(el))
    x = int((u + 1) / 2 * view.shape[1])
    y = int((1 - v) / 2 * view.shape[0])
    cv2.circle(view, (x, y), 14, (0, 255, 255), 2)
    cv2.line(view, (x - 20, y), (x + 20, y), (0, 255, 255), 1)
    cv2.line(view, (x, y - 20), (x, y + 20), (0, 255, 255), 1)


def main():
    brain = FlyBrain()              # the real connectome (needs data/cache)
    # brain = FlyBrain(fixture=True)  # tiny offline brain, same API
    cap = cv2.VideoCapture(CAMERA_INDEX)
    print(f"following the {TARGET} point; press q to quit")
    last_command = ""
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view, like looking in a mirror

        # 1) camera image -> front face of the fly's cubemap
        cubemap = np.zeros((256, 384, 3), np.uint8)
        cubemap[:128, :128] = cv2.resize(frame, (128, 128))

        # 2) one 20 ms tick of the whole connectome brain
        control, spikes = brain.step(cubemap, brain.step_count * brain.dt)

        # 3) servo decision from the brain's eye
        command, row, col = eye_command(brain)

        # 4) report + draw
        if command != last_command:
            print(f"move: {command}")
            last_command = command
        if row is not None:
            draw_crosshair(frame, row, col)
        cv2.putText(frame, command, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame,
                    f"spikes/tick {len(spikes)}  fwd {control.forward_rate:.3f}"
                    f"  turn {control.turn_rate:+.3f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("flybrain tracking (q quits)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
