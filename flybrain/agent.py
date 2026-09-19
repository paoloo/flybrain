"""Backends + closed-loop experiment runner (the BrainBackend seam).

Follows the plug-in architecture of Frankweb33/flybrain-robot-bridge:
  world frame -> VisionBackend -> BrainBackend.step -> (x, y, jump) -> world
Any backend implementing `reset`/`act` can fly the game. Two ship here:

- MaleCNSPilot: the real 166,700-neuron MaleCNS LIF brain (ornata/fly dynamics).
- MockPilot: 8-population leaky rate model driven by retinal motion
  (flybrain-robot-bridge MockBrain adapted to the cubemap), fast, no download.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .brain import FlyBrain
from .retina import PREVIEW_HEIGHT, PREVIEW_WIDTH


class VisionBackend(ABC):
    """Frame -> whatever the brain backend consumes."""

    @abstractmethod
    def encode(self, frame: np.ndarray) -> object: ...


class LuminanceVision(VisionBackend):
    """Cheap motion encoder for rate models: left/right eye motion + looming."""

    def __init__(self):
        self.previous = None

    def encode(self, frame: np.ndarray):
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.previous is None or self.previous.shape != gray.shape:
            self.previous = gray
        flow = cv2.calcOpticalFlowFarneback(self.previous, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        self.previous = gray
        h, w = flow.shape[:2]
        mid = w // 2
        left_motion = float(np.hypot(flow[:, :mid, 0], flow[:, :mid, 1]).mean())
        right_motion = float(np.hypot(flow[:, mid:, 0], flow[:, mid:, 1]).mean())
        # looming: outward radial expansion around the center of the forward face
        yy, xx = np.mgrid[0:h, 0:mid]
        r = np.hypot(xx - mid / 2, yy - h / 2) + 1e-6
        fx, fy = flow[:, :mid, 0], flow[:, :mid, 1]
        radial = (fx * (xx - mid / 2) + fy * (yy - h / 2)) / r
        looming = float(np.clip(radial.mean() * 0.1, 0, 1))
        return dict(left_motion=left_motion, right_motion=right_motion, looming=looming)


class BrainBackend(ABC):
    """Plug-in point for any fly-neuron simulator."""

    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def act(self, frame: np.ndarray, dt: float) -> tuple[int, int, bool]:
        """Return game controls (stick x, stick y in -70..70, jump)."""

    @abstractmethod
    def telemetry(self) -> dict: ...

    @property
    @abstractmethod
    def label(self) -> str: ...


class MaleCNSPilot(BrainBackend):
    """Real connectome brain: retina -> LIF spikes -> descending-neuron decode."""

    def __init__(self, cache: Path | None = None, seed: int = 64):
        self.brain = FlyBrain(cache, seed=seed)

    def reset(self):
        self.brain.__init__(seed=self.brain.seed)  # deterministic fresh state

    def act(self, frame, dt):
        control, spikes = self.brain.step(frame, self.brain.step_count * dt)
        self.spikes_last = int(len(spikes))
        return control.x, control.y, control.jump

    def telemetry(self):
        rates = self.brain.pool_rates()
        return dict(label=self.label, tick=self.brain.step_count,
                    spikes_last=getattr(self, "spikes_last", 0),
                    mean_luminance=self.brain.mean_luminance,
                    temporal_energy=self.brain.temporal_energy, **rates)

    @property
    def label(self):
        return self.brain.label


class MockPilot(BrainBackend):
    """8-population leaky rate model on retinal motion; for quick experiments."""

    TAU = 0.12

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.vision = LuminanceVision()
        self.reset()

    def reset(self):
        self.state = dict(left=0.0, right=0.0, looming=0.0,
                          forward=0.0, turn=0.0, t=0)

    def act(self, frame, dt):
        sensory = self.vision.encode(frame)
        decay = np.exp(-dt / self.TAU)
        self.state["left"] = self.state["left"] * decay + sensory["left_motion"] * 0.1
        self.state["right"] = self.state["right"] * decay + sensory["right_motion"] * 0.1
        self.state["looming"] = max(self.state["looming"] * decay, sensory["looming"])
        # contralateral motion drives steering, ambient flow drives forward
        turn = (self.state["left"] - self.state["right"]) * 900.0
        forward_drive = (self.state["left"] + self.state["right"]) * 450.0 - 30.0
        x = int(np.clip(turn, -70, 70))
        y = int(np.clip(forward_drive, -70, 70))
        jump = self.state["looming"] > 0.55
        self.state["forward"] = y
        self.state["turn"] = x
        self.state["t"] += 1
        return x, y, jump

    def telemetry(self):
        return dict(label=self.label, tick=self.state["t"], spikes_last=0, **{
            k: round(v, 3) if isinstance(v, float) else v for k, v in self.state.items()})

    @property
    def label(self):
        return "MockBrain — 8-population leaky rate model (no download)"


class Experiment:
    """Closed-loop runner with reward logging, CSV telemetry and eye previews."""

    def __init__(self, world, backend: BrainBackend, out_dir: Path | None = None,
                 eye_previews: bool = False):
        self.world = world
        self.backend = backend
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.eye_previews = eye_previews
        self.log: list[dict] = []

    def run_episode(self, max_steps: int = 1500, dt: float = 0.02) -> dict:
        """Run one episode; world runs at its own pace, brain ticks per frame.

        With the real MaleCNS brain 50 neural ticks happen per rendered frame
        batch would be too slow, so we run the brain once per frame (20 ms)
        and advance physics with the returned control, matching fly64's 10 Hz
        vision / 50 Hz dynamics split loosely: 5 neural ticks per frame.
        """
        frame = self.world.reset()
        self.backend.reset()
        total_reward, done, step = 0.0, False, 0
        self.log.clear()
        while not done and step < max_steps:
            x, y, jump = self.backend.act(frame, dt)
            for _ in range(4):  # physics advances faster than vision, like fly64
                frame, reward, done, info = self.world.step(x, y, jump)
                total_reward += reward
                if done:
                    break
            entry = dict(step=step, x=x, y=y, jump=jump, reward=total_reward,
                         score=info["score"], distance=info["distance"],
                         heading=info["heading"], altitude=info["altitude"])
            entry.update(self.backend.telemetry())
            self.log.append(entry)
            if self.eye_previews and self.out_dir and step % 25 == 0:
                self._save_preview(frame, step)
            step += 1
        result = dict(steps=step, reward=total_reward, score=info["score"],
                      distance=info["distance"], done=done,
                      label=self.backend.label)
        if self.out_dir:
            self._save_csv()
        return result

    def _save_preview(self, frame, step):
        import cv2
        try:
            eye = self.backend.brain.eye_view(frame)  # MaleCNSPilot
        except AttributeError:
            eye = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
        cv2.imwrite(str(self.out_dir / f"eye_step{step:05d}.png"),
                    cv2.cvtColor(eye, cv2.COLOR_RGB2BGR))

    def _save_csv(self):
        import csv
        keys = sorted(self.log[0].keys())
        with (self.out_dir / "telemetry.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.log)
