"""2track.py - the SAME webcam tracking task as tracking.py, but done the way
a real fly does it.

Read this side by side with tracking.py. The task is identical: follow a
colored point with two virtual actuators and print what the actuators do.
The difference is the CONTROL SCHEME. There is no x/y stick anywhere.

Run:    conda run -n mcp python 2track.py
Quit:   press q in the video window

TEACHING NOTES - what is different from tracking.py, and why
-----------------------------------------------------------
A real fly has no joystick in its head. Verified biology (see REPORT.md
section 5a for the full citations):

  * STEERING: the right-minus-left activity difference of two descending
    neuron types (DNa02, DNg13) correlates with turning. They are
    INDEPENDENT channels, not the two ends of one axis: DNa02 shortens
    strides on the inside of a turn, DNg13 lengthens strides on the
    outside. Either can be recruited alone.
  * THROTTLE: wingbeat amplitude is set by a POPULATION CODE in DNg02,
    ~15 near-identical cell pairs per side. Activating more cells = more
    amplitude (about 2 degrees per cell pair). The fly "speeds up" by
    recruiting more of the population, not by pushing a slider.
  * FAST STABILIZATION: the brain is too slow for millisecond flight
    corrections. Halteres (gyroscope organs) close a LOCAL feedback loop
    in the thorax that retimes wing muscles every stroke. The brain sets
    goals; the loop below the brain does the fast math.

So the natural control scheme has a HIERARCHY:

    eye signal (slow, in the brain)
        -> set GOALS: desired turn (from R-L steering cells) and desired
           effort (from a recruited fraction of the throttle population)
        -> LOCAL proportional controller (fast, below the brain)
        -> actuator commands, one per side, updated every frame

This file implements exactly that, with the real connectome brain doing the
SEEING (the same FlyBrain as tracking.py) and two small biologically-shaped
controllers doing the ACTING.

The printed commands, and what they mean:

    L-throttle: 0..8 cells   R-throttle: 0..8 cells
        the DNg02 population recruited on each side (8 = max here)
    L-steer +xx   R-steer -xx
        the DNa02/DNg13-style steering activity on each side
        (positive = "shorten stride on my side" = turn toward me)
    turn: right (rate 14.3 deg/s)
        what the local loop predicts the body will do after integration
"""

import cv2
import numpy as np

from flybrain.brain import FlyBrain

CAMERA_INDEX = 0          # try 1, 2, ... if the wrong camera opens
TARGET = "green"          # "green", "red" or "blue": the point to follow

# Color-opponency weights, identical to tracking.py (and to the brain's own
# green-opponency visual encoder).
WEIGHTS = {
    "green": (-0.5, 1.0, -0.5),
    "red": (1.0, -0.5, -0.5),
    "blue": (-0.5, -0.5, 1.0),
}

# Atlas geometry, identical to tracking.py: 48 rows x 64 cols of photoreceptor
# columns; azimuth 0 sits at col ~33.8, elevation 0 at row 23.5.
ROW_CENTER = 23.5
COL_CENTER = 33.8

# --- the three biological controllers, all tiny -----------------------------

# DNg02-style throttle: a population of 8 cells per side in this demo
# (the real fly has at least 15 pairs; 8 keeps the HUD readable).
N_CELLS = 8
DEGS_PER_CELL = 2.0     # Namiki et al.: ~2 degrees of stroke per recruited cell

# Local (haltere-style) proportional controller gains. These are NOT measured
# fly constants; they shape the demo's dynamics only.
K_STEER = 0.9           # how strongly steering cells push the predicted rate
K_CENTER = 0.05         # slow pull toward keeping the target near the fovea
RATE_LIMIT = 60.0       # deg/s cap on the predicted turn rate


def color_score(eye, target):
    """Per-photoreceptor evidence that it sees the target color.
    eye: (6006, 3) RGB in 0..1 (the brain's last retinal sample)."""
    wr, wg, wb = WEIGHTS[target]
    r, g, b = eye[:, 0], eye[:, 1], eye[:, 2]
    return np.clip(wr * r + wg * g + wb * b - 0.15, 0, None)


def eye_position(brain, target):
    """Where does the fly's EYE see the target? (same math as tracking.py)

    Returns (row, col, strength): the weighted average atlas position of the
    photoreceptors that see the color, and how much of it there is."""
    score = color_score(brain.previous_rgb, target)
    if score.sum() < 0.5:
        return None, None, 0.0
    w = score / score.sum()
    row = float((brain.visual_pixels[:, 0] * w).sum())
    col = float((brain.visual_pixels[:, 1] * w).sum())
    return row, col, float(score.sum())


def split_eyes(brain, target):
    """Compare the target signal in the LEFT vs RIGHT half of the visual field.

    A fly steers on the RIGHT-MINUS-LEFT difference of visual activity across
    its two eyes / hemispheres. We mimic that: compute the color score summed
    over left-atlas columns (0..31) and right columns (32..63) separately.
    This is the signal that would drive the steering descending neurons."""
    score = color_score(brain.previous_rgb, target)
    left = float(score[brain.visual_pixels[:, 1] < 32].sum())
    right = float(score[brain.visual_pixels[:, 1] >= 32].sum())
    return left, right


class FlyNaturalController:
    """Goal layer (brain-like) + local layer (haltere-like), no x/y anywhere."""

    def __init__(self):
        # state of the local fast loop
        self.rate = 0.0        # predicted body turn rate, deg/s (+ = right)
        self.effort = 0.0      # recruited fraction of the throttle population

    def step(self, left_score, right_score, target_col, dt):
        """One control update. All units are population activities, not axes.

        left_score / right_score : total color evidence per visual hemisphere
        target_col               : atlas column of the target (None if lost)
        dt                       : seconds since the last update
        """

        # ---- 1. GOAL: steering goal from the R-L difference ----------------
        # Like DNa02/DNg13: the side with MORE evidence about the target is
        # the side the fly turns TOWARD. The goal is a difference, not a
        # position command.
        steer_goal = (right_score - left_score)

        # ---- 2. GOAL: throttle goal = a recruited fraction of the population
        # Total visual evidence plays the role of 'how much work is needed':
        # a strong, close target recruits more DNg02 cells (more effort),
        # like a real fly increasing stroke amplitude for a demanding task.
        effort_goal = np.clip((left_score + right_score) / 20.0, 0.0, 1.0)

        # ---- 3. LOCAL LOOP: first-order dynamics toward the goals ----------
        # This plays the haltere-loop role: fast, local, proportional. It
        # chases the brain's goals with its own time constants, and it keeps
        # running its own smoothing even when the brain's goal jitters.
        steer_cmd = np.clip(steer_goal * K_STEER, -RATE_LIMIT, RATE_LIMIT)
        self.rate += (steer_cmd - self.rate) * np.clip(dt * 6.0, 0, 1)
        self.effort += (effort_goal - self.effort) * np.clip(dt * 4.0, 0, 1)

        # ---- 4. ACTUATOR COMMANDS: one per side, population-coded ----------
        # Steering cells: side that must work harder gets the higher value
        # (DNa02-style 'shorten stride on my side' pattern).
        l_steer = np.clip(-self.rate * 0.2, -10, 10)
        r_steer = np.clip(+self.rate * 0.2, -10, 10)
        # Throttle population: how many of the 8 cells per side are recruited.
        l_cells = int(round(self.effort * N_CELLS))
        r_cells = int(round(self.effort * N_CELLS))
        # If we are turning, the OUTER side recruits more cells (more thrust
        # on the outside of a turn is how flies widen/steepen a turn).
        if self.rate > 2:      # turning right -> left side is the outer side
            l_cells = min(N_CELLS, l_cells + 1)
        elif self.rate < -2:   # turning left  -> right side is the outer side
            r_cells = min(N_CELLS, r_cells + 1)

        return dict(l_cells=l_cells, r_cells=r_cells,
                    l_steer=l_steer, r_steer=r_steer,
                    rate=self.rate, effort=self.effort)


def draw_fly_view(view, row, col):
    """Draw the fly's estimate of the target position (yellow crosshair)."""
    if row is None or col is None:
        return
    if col >= 32:
        az = -8.5 + (col - 32) / 31 * 143.5
    else:
        az = -135.0 + col / 31 * 143.5
    el = 72.0 - row / 47 * 144.0
    if abs(az) > 45 or abs(el) > 45:
        return
    u, v = np.tan(np.deg2rad(az)), np.tan(np.deg2rad(el))
    x = int((u + 1) / 2 * view.shape[1])
    y = int((1 - v) / 2 * view.shape[0])
    cv2.circle(view, (x, y), 14, (0, 255, 255), 2)
    cv2.line(view, (x - 20, y), (x + 20, y), (0, 255, 255), 1)
    cv2.line(view, (x, y - 20), (x, y + 20), (0, 255, 255), 1)


def main():
    brain = FlyBrain()                  # real connectome, same as tracking.py
    # brain = FlyBrain(fixture=True)    # offline tiny brain, same API
    cap = cv2.VideoCapture(CAMERA_INDEX)
    ctrl = FlyNaturalController()
    print(f"following the {TARGET} point, fly-style (no x/y stick); q quits")
    print("     L-throttle R-throttle | L-steer  R-steer | predicted turn")
    last_line = ""
    t_last = cv2.getTickCount()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        # --- the brain sees: one 20 ms tick, camera on the front face ------
        cubemap = np.zeros((256, 384, 3), np.uint8)
        cubemap[:128, :128] = cv2.resize(frame, (128, 128))
        control, spikes = brain.step(cubemap, brain.step_count * brain.dt)
        dt = (cv2.getTickCount() - t_last) / cv2.getTickFrequency()
        t_last = cv2.getTickCount()

        # --- brain-level signals: hemispheric evidence + foveal position ---
        row, col, _ = eye_position(brain, TARGET)
        left_score, right_score = split_eyes(brain, TARGET)

        # --- fly-style control update (goals -> local loop -> actuators) ---
        cmds = ctrl.step(left_score, right_score, col, dt)

        # --- report ---------------------------------------------------------
        line = (f"L-throttle {cmds['l_cells']}/{N_CELLS}  "
                f"R-throttle {cmds['r_cells']}/{N_CELLS} | "
                f"L-steer {cmds['l_steer']:+5.1f}  R-steer {cmds['r_steer']:+5.1f} | "
                f"turn: {cmds['rate']:+6.1f} deg/s")
        if line != last_line:
            print(line)
            last_line = line
        draw_fly_view(frame, row, col)
        cv2.putText(frame, f"L{cmds['l_cells']} R{cmds['r_cells']}"
                    f"  rate {cmds['rate']:+.0f} deg/s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame,
                    f"spikes/tick {len(spikes)}  fwd {control.forward_rate:.3f}"
                    f"  turn {control.turn_rate:+.3f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("2track: fly-natural control (q quits)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
