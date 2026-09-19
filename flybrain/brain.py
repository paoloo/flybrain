"""Connectome-derived leaky integrate-and-fire fly brain with engineered I/O maps.

Dynamics and motor decoding follow ornata/fly (fly64/model.py). The wiring is
MaleCNS v1.0 measured connectome data; the dynamics, visual projection and
motor mappings are engineered approximations, not measured fly physiology.

Every 20 ms:  v <- exp(-dt/tau) * v + synaptic_gain * W @ spikes + tonic + noise + retina
Spike when v >= 1, reset to 0. Retinal current is injected only into R1-R8.

Rolling 13-tick (~260 ms) descending-neuron window decodes game controls:
  forward  = mean rate of DNg100 pool
  steering = right-minus-left mean rate of DNa02/DNg13 pools
  jump     = burst in DNp01/DNp10 pool, 0.8 s cooldown
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from .retina import SphericalRetina

DT = 0.020          # seconds per neural tick (50 Hz)
TAU_M = 0.100       # membrane time constant
THRESHOLD = 1.0


@dataclass
class Control:
    """Game controls decoded from neural activity, stick range +-70."""
    x: int
    y: int
    jump: bool
    forward_rate: float
    turn_rate: float
    jump_rate: float


class FlyBrain:
    """166,700-neuron MaleCNS LIF approximation you can feed RGB cubemaps."""

    def __init__(self, cache: Path | None = None, fixture: bool = False, seed: int = 64):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.fixture = fixture
        cache = cache or Path("data/cache")
        if fixture:
            self._load_fixture()
            self.label = "FIXTURE — synthetic 4,096-cell graph (no download needed)"
        else:
            if not (cache / "manifest.json").exists():
                raise FileNotFoundError(
                    f"{cache}/manifest.json missing; run: python -m flybrain.data --prepare --cache {cache}")
            self._load_cache(cache)
            self.label = "MaleCNS v1.0 — measured wiring, modeled dynamics"
        self.v = np.zeros(self.n, dtype=np.float32)
        # CSC event propagation traverses every outgoing edge of each spiking
        # cell. Zero-spike columns contribute exactly zero; no graph pruning.
        self.w = self.w.tocsc()
        self.spikes = np.zeros(self.n, dtype=np.float32)
        self.activity = np.zeros(self.n, dtype=np.float32)
        self.retina = SphericalRetina(self.visual_pixels)
        self.previous_rgb = np.zeros((len(self.visual), 3), dtype=np.float32)
        self.history: deque[np.ndarray] = deque(maxlen=13)
        self.motor_nodes = np.concatenate((self.forward, self.turn_left, self.turn_right, self.jump_nodes))
        self.motor_splits = np.cumsum([len(self.forward), len(self.turn_left), len(self.turn_right)])
        self.filtered_x = 0.0
        self.filtered_y = 0.0
        self.last_jump = -10.0
        self.step_count = 0
        self.mean_luminance = 0.0
        self.temporal_energy = 0.0
        self.tonic_current = 0.180
        self.synaptic_gain = 1.50

    # ---- loading -----------------------------------------------------------
    def _load_fixture(self):
        self.n = 4096
        row = self.rng.integers(0, self.n, 65536, dtype=np.int32)
        col = self.rng.integers(0, self.n, 65536, dtype=np.int32)
        weight = self.rng.lognormal(-2.7, 0.5, len(row)).astype(np.float32)
        inhibitory = self.rng.random(self.n) < 0.22
        weight *= np.where(inhibitory[col], -1.0, 1.0)
        self.visual = np.arange(0, 1536, dtype=np.int32)
        flat_pixels = np.linspace(0, 48 * 64 - 1, len(self.visual)).astype(np.int32)
        self.visual_pixels = np.column_stack((flat_pixels // 64, flat_pixels % 64)).astype(np.uint8)
        self.forward = np.arange(3600, 3660, dtype=np.int32)
        self.turn_left = np.arange(3660, 3700, dtype=np.int32)
        self.turn_right = np.arange(3700, 3740, dtype=np.int32)
        self.jump_nodes = np.arange(3740, 3760, dtype=np.int32)
        relay = np.arange(1800, 2400, dtype=np.int32)
        fixture_pre = self.rng.choice(self.visual, 12000)
        fixture_post = self.rng.choice(relay, 12000)
        motor_pre = self.rng.choice(relay, 5000)
        motor_post = self.rng.choice(
            np.concatenate((self.forward, self.turn_left, self.turn_right, self.jump_nodes)), 5000)
        row = np.concatenate((row, fixture_post, motor_post))
        col = np.concatenate((col, fixture_pre, motor_pre))
        weight = np.concatenate((weight, np.full(12000, 0.18, np.float32), np.full(5000, 0.22, np.float32)))
        self.w = sparse.csr_matrix((weight, (row, col)), shape=(self.n, self.n))
        phi = self.rng.uniform(0, 2 * np.pi, self.n)
        cost = self.rng.uniform(-1, 1, self.n)
        rad = np.sqrt(1 - cost * cost)
        self.positions = np.column_stack((rad * np.cos(phi), cost * 0.65, rad * np.sin(phi))).astype(np.float32)
        self.position_measured = np.zeros(self.n, dtype=bool)
        self.regions = (np.arange(self.n) % 8).astype(np.uint8)
        self.region_names = np.array([f"fixture group {i}" for i in range(8)])

    def _load_cache(self, cache: Path):
        meta = np.load(cache / "model.npz", allow_pickle=False)
        self.n = int(meta["n"])
        self.w = sparse.load_npz(cache / "weights.npz").astype(np.float32)
        for name in ("visual", "forward", "turn_left", "turn_right", "jump_nodes", "positions", "regions"):
            setattr(self, name, meta[name])
        self.region_names = meta["region_names"]
        self.position_measured = meta["position_measured"]
        self.visual_pixels = meta["visual_pixels"]

    # ---- sensory encoding ---------------------------------------------------
    def encode_retina(self, rgb: np.ndarray) -> np.ndarray:
        """Six-face 384x256 atlas -> per-photoreceptor drive in [0, 1]."""
        frame = self.retina.sample(rgb)
        lum = frame @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        prev_lum = self.previous_rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        temporal = np.abs(lum - prev_lum)
        color = np.maximum(frame[..., 1] - 0.5 * (frame[..., 0] + frame[..., 2]), 0)
        drive = np.clip(0.45 * lum + 1.6 * temporal + 0.25 * color, 0, 1)
        self.previous_rgb = frame
        self.mean_luminance = float(lum.mean())
        self.temporal_energy = float(temporal.mean())
        return drive

    # ---- one 20 ms tick -----------------------------------------------------
    def step(self, rgb: np.ndarray, now: float | None = None) -> tuple[Control, np.ndarray]:
        now = self.step_count * self.dt if now is None else now
        sensory = self.encode_retina(rgb)
        current = np.asarray(self.w[:, np.flatnonzero(self.spikes)].sum(axis=1)).ravel()
        current *= self.synaptic_gain
        baseline = self.rng.random(self.n) < (1.2 * self.dt)
        self.v *= np.exp(-DT / TAU_M)
        self.v += current + baseline.astype(np.float32) * 0.22 + self.tonic_current
        self.v[self.visual] += sensory * 0.62
        fired = self.v >= THRESHOLD
        self.v[fired] = 0.0
        self.spikes[:] = fired
        self.activity *= 0.82
        self.activity[fired] = 1.0
        self.history.append(fired[self.motor_nodes].copy())
        self.step_count += 1
        return self._decode(now), np.flatnonzero(fired)

    @property
    def dt(self) -> float:
        return DT

    # ---- motor decoding -----------------------------------------------------
    def _decode(self, now: float) -> Control:
        recent = np.stack(tuple(self.history), axis=0).mean(axis=0)
        forward_rate, left_rate, right_rate, jump_rate = [
            float(pool.mean()) for pool in np.split(recent, self.motor_splits)]
        turn_rate = right_rate - left_rate

        raw_y = np.clip((forward_rate - 0.008) * 2000.0, 0, 70)
        raw_x = np.clip(turn_rate * 1100.0, -70, 70)
        self.filtered_y = 0.78 * self.filtered_y + 0.22 * raw_y
        self.filtered_x = 0.78 * self.filtered_x + 0.22 * raw_x
        jump = jump_rate > 0.04 and now - self.last_jump >= 0.8
        if jump:
            self.last_jump = now
        return Control(
            int(self.filtered_x) if abs(self.filtered_x) >= 8 else 0,
            int(self.filtered_y) if self.filtered_y >= 8 else 0,
            jump, forward_rate, turn_rate, jump_rate)

    # ---- introspection ------------------------------------------------------
    def pool_rates(self) -> dict[str, float]:
        """Mean spikes/cell/tick over the last ~260 ms window for each pool."""
        if not self.history:
            return {"forward": 0.0, "turn_left": 0.0, "turn_right": 0.0, "jump": 0.0}
        recent = np.stack(tuple(self.history), axis=0).mean(axis=0)
        forward, left, right, jump = [float(p.mean()) for p in np.split(recent, self.motor_splits)]
        return {"forward": forward, "turn_left": left, "turn_right": right, "jump": jump}

    def eye_view(self, rgb: np.ndarray) -> np.ndarray:
        """256x128 paired fisheye preview of what the two eyes see."""
        return self.retina.preview(rgb)
