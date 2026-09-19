"""A closed-loop flying game world rendered as a six-face 384x256 RGB cubemap.

The fly is a balloon navigating an open field with colored landmark stripes:
green ground, blue sky, vertical stripes whose color tells forward (yellow)
from backward (red). Stripe color+position gives the brain an unambiguous,
learnable visual signal; heading/spike history gives the reward signal.

Physics (kept deliberately simple, arcade-style):
  forward velocity v <- v + (y/70) * ACCEL - DRAG * v
  heading          <- heading + (x/70) * TURN
  altitude         <- altitude + (jump ? CLIMB : -SINK), clamped
Score: landmarks passed forward. Crashing into a wall stripe ends the episode.
"""
from __future__ import annotations

import numpy as np

from .retina import BASES

FACE = 128
WIDTH, HEIGHT, CHANNELS = 384, 256, 3
ACCEL, DRAG, TURN, CLIMB, SINK = 0.9, 0.35, 0.03, 0.04, 0.02
MAX_SPEED, MAX_ALT = 6.0, 5.0


class FlyingWorld:
    """Landmark course the fly flies through, seen from its own eyes."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.heading = 0.0
        self.distance = 0.0
        self.altitude = 1.0
        self.v = 0.0
        self.jump_phase = 0.0
        self.landmarks = sorted(self.rng.uniform(4, 200, 40))
        self.next_landmark = 0
        self.score = 0
        self.done = False
        return self.frame()

    def step(self, x: int, y: int, jump: bool):
        """Apply stick (-70..70) + jump; returns (frame, reward, done, info)."""
        self.v = np.clip(self.v + (y / 70.0) * ACCEL - DRAG * self.v, -MAX_SPEED / 2, MAX_SPEED)
        self.distance += self.v * 0.05
        self.heading += (x / 70.0) * TURN
        if jump and self.jump_phase == 0:
            self.jump_phase = 0.01
        if self.jump_phase:
            self.altitude += CLIMB
            self.jump_phase += 0.1
            if self.jump_phase > np.pi:
                self.jump_phase = 0
        else:
            self.altitude = max(self.altitude - SINK, 0.2)
        self.altitude = min(self.altitude, MAX_ALT)

        reward = 0.0
        while self.next_landmark < len(self.landmarks) and self.distance > self.landmarks[self.next_landmark]:
            self.next_landmark += 1
            self.score += 1
            reward += 1.0
        if abs(self.heading) > np.pi / 2:  # flew off the course sideways
            self.done = True
            reward -= 5.0
        return self.frame(), reward, self.done, dict(score=self.score, distance=self.distance,
                                                     heading=self.heading, altitude=self.altitude)

    def frame(self) -> np.ndarray:
        """Render the 384x256 six-face atlas from the fly's eye position."""
        image = np.empty((HEIGHT, WIDTH, CHANNELS), np.uint8)
        yy, xx = np.mgrid[0:FACE, 0:FACE]
        u, v = (xx + .5) / 64 - 1, 1 - (yy + .5) / 64
        horizon = .04 * np.sin(self.jump_phase) + (self.altitude - 1) * .08
        for face, (right, up, forward) in enumerate(BASES):
            rays = forward + u[..., None] * right + v[..., None] * up
            norm = np.linalg.norm(rays, axis=-1)
            az = np.arctan2(rays[..., 0], rays[..., 2]) + self.heading
            el = rays[..., 1] / norm
            ground = el < horizon
            color = np.where(ground[..., None], [70, 155, 72], [80, 165, 245]).astype(np.uint8)
            # nearest unpassed landmark stripe, drawn as a vertical band
            for lm in self.landmarks[self.next_landmark:self.next_landmark + 3]:
                rel = lm - self.distance
                stripe_az = np.arctan2(rel, 6.0)  # ~83 deg ahead at 6 units out
                band = np.abs(((az - stripe_az + np.pi) % (2 * np.pi)) - np.pi) < .09
                nearness = np.clip(1.0 - abs(rel) / 20.0, 0, 1)
                if rel >= 0:
                    stripe_color = np.array([235, 215, 60])  # yellow: ahead
                else:
                    stripe_color = np.array([215, 60, 45])   # red: behind
                painted = band[..., None] * (0.35 + 0.65 * nearness)
                color = ((1 - painted) * color + painted * stripe_color).astype(np.uint8)
            image[(face // 3) * FACE:(face // 3 + 1) * FACE, (face % 3) * FACE:(face % 3 + 1) * FACE] = color
        return image
