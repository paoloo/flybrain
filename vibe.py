"""vibe.py — the fly's descending neurons drive a (simulated or real) buttplug.io toy.

The MaleCNS brain decodes flight controls from what its eyes see (see
flybrain/brain.py). Here the decoded pools map onto a full actuator set, the
way a complex toy exposes them over buttplug.io:

  forward pool   (DNg100)        -> VIBRATE     base + tip rotor motors
  turn rate      (DNa02/DNg13)   -> ROTATE      signed shaft rotation
  jump rate      (DNp01/DNp10)   -> OSCILLATE   tip wiggle amplitude
  jump burst     (event)         -> POSITION    thrust swing, 700 ms
  forward + phase                -> CONSTRICT   3 grip rings, peristaltic
  sustained activity             -> TEMPERATURE warming coil

SimDevice is the in-process toy: same actuator surface as
buttplug.ButtplugDevice, plus 50 Hz physics (motor lag, rotation integration,
thrust speed limit, ring relaxation) so a GUI can render it faithfully.
ButtplugSink sends the same state to real hardware via Intiface Central
(buttplug-py 1.0, spec v4), feature-by-feature.

Usage:
  conda run -n mcp python vibe.py                                  # simulated, flying world
  conda run -n mcp python vibe.py --server ws://127.0.0.1:12345    # real Intiface Central
  conda run -n mcp python vibe.py --gui                            # launch vibe-gui.py
"""
from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass, field

import numpy as np

from flybrain.brain import FlyBrain
from flybrain.world import FlyingWorld

# ---- signal shaping ---------------------------------------------------------

TURN_THRESHOLD = 12.0   # stick units; below this rotation stays 0
VIB_FLOOR = 0.05        # lowest intensity worth sending (motors stall below it)
VIB_CEIL = 0.85         # safety rail: never drive rotors past this
EMA = 0.25              # smoothing toward new target each tick
POSITION_MS = 700       # thrust swing duration on jump
ROT_GAIN = 6.0          # turn rate -> signed rotate speed
OSC_GAIN = 4.0          # jump pool rate -> oscillate amplitude
TEMP_AMBIENT = 34.0     # degrees C
TEMP_MAX = 40.0


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


@dataclass
class VibeState:
    """One control frame: commanded actuator targets, all 0..1 (rotate signed)."""
    vib_base: float
    vib_tip: float
    rotate: float           # -1..1 signed speed
    oscillate: float        # 0..1 tip wiggle amplitude
    position: float         # 0..1 thrust target
    constrict: list[float]  # 3 ring pressures 0..1
    temperature: float      # 0..1 heater command
    jump: bool
    forward_rate: float
    turn_rate: float
    jump_rate: float
    log: list[str] = field(default_factory=list)


class VibeMapper:
    """Brain decode -> actuator targets. Pure function of the motor pools."""

    def __init__(self) -> None:
        self.ema_vib = 0.0
        self.ema_rotate = 0.0
        self.position = 0.15
        self.phase = 0.0        # peristaltic wave phase

    def update(self, control) -> VibeState:
        raw = _clamp01(control.y / 70.0)
        vib = min(raw, VIB_CEIL)
        if vib < VIB_FLOOR:
            vib = 0.0
        self.ema_vib += EMA * (vib - self.ema_vib)

        turn = float(control.x)
        rotate = 0.0 if abs(turn) < TURN_THRESHOLD else float(
            np.clip(turn / 70.0 * ROT_GAIN, -1.0, 1.0))
        self.ema_rotate += EMA * (rotate - self.ema_rotate)

        osc = _clamp01(control.jump_rate * OSC_GAIN)

        if control.jump:
            self.position = 0.85 if self.position < 0.5 else 0.15
            self.position = round(self.position, 2)

        fwd = _clamp01(control.forward_rate)
        self.phase = (self.phase + 0.35 * 0.02) % (2 * np.pi)  # advances ~0.35 rad/s at 50 Hz
        constrict = [_clamp01(fwd * (0.4 + 0.45 * np.sin(self.phase + i * 2 * np.pi / 3)))
                     for i in range(3)]

        temp = _clamp01((self.ema_vib + osc) / 1.2)

        log = [f"JUMP -> thrust {POSITION_MS} ms"] if control.jump else []
        return VibeState(self.ema_vib, min(self.ema_vib * 1.25, VIB_CEIL),
                         self.ema_rotate, osc, self.position, constrict,
                         temp, control.jump, control.forward_rate,
                         control.turn_rate, control.jump_rate, log)


# ---- simulated device -------------------------------------------------------

class SimDevice:
    """Physics for a complex toy: 2 vib motors, rotator, oscillator,
    linear thrust, 3 constricting rings, heater. 50 Hz step."""

    def __init__(self, name: str = "SimStroker Pro X (6 actuators)") -> None:
        self.name = name
        # commands (from VibeState) ...
        self.cmd = VibeState(0, 0, 0, 0, 0.15, [0, 0, 0], 0, False, 0, 0, 0)
        # ... and physical state
        self.vib_base = 0.0     # actual motor amplitude (first-order lag)
        self.vib_tip = 0.0
        self.rotate_speed = 0.0
        self.angle = 0.0        # accumulated shaft angle, radians
        self.osc_amp = 0.0
        self.osc_phase = 0.0
        self.position = 0.15    # actual thrust 0..1
        self.rings = [0.0, 0.0, 0.0]
        self.temp = TEMP_AMBIENT
        self.jumps = 0
        self.ticks = 0

    def apply(self, state: VibeState) -> None:
        self.cmd = state
        if state.jump:
            self.jumps += 1

    def step(self, dt: float) -> None:
        c = self.cmd
        k_mot = 1.0 - np.exp(-dt / 0.08)     # rotor spin-up lag
        self.vib_base += k_mot * (c.vib_base - self.vib_base)
        self.vib_tip += k_mot * (c.vib_tip - self.vib_tip)
        self.rotate_speed += k_mot * (c.rotate - self.rotate_speed)
        self.angle += self.rotate_speed * 12.0 * dt   # rad/s at full command
        self.osc_amp += k_mot * (c.oscillate - self.osc_amp)
        self.osc_phase = (self.osc_phase + (1.0 + 9.0 * self.osc_amp) * dt) % 1.0
        # thrust: limited slew toward target
        slew = 0.8 * dt
        delta = c.position - self.position
        self.position += np.clip(delta, -slew, slew)
        # rings: fast squeeze, slower relax
        for i in range(3):
            target = c.constrict[i]
            k = 1.0 - np.exp(-dt / (0.15 if target > self.rings[i] else 0.30))
            self.rings[i] += k * (target - self.rings[i])
        # heater
        self.temp += (1 - np.exp(-dt / 5.0)) * (
            (TEMP_AMBIENT + (TEMP_MAX - TEMP_AMBIENT) * c.temperature) - self.temp)
        self.ticks += 1

    def line(self) -> str:
        bar = "#" * int(round(self.vib_tip * 20))
        arrow = "<" * int(abs(self.rotate_speed) * 10) if self.rotate_speed < 0 \
            else ">" * int(self.rotate_speed * 10)
        return (f"[{bar:<20}] base {self.vib_base:4.2f} tip {self.vib_tip:4.2f}  "
                f"rot {arrow:>10} {int(np.degrees(self.angle)) % 360:3d}°  "
                f"osc {self.osc_amp:4.2f}@{1 + 9 * self.osc_amp:4.1f} Hz  "
                f"thrust {self.position:4.2f}  rings "
                f"{''.join(f'{r:4.2f}' for r in self.rings)}  "
                f"{self.temp:4.1f}°C  jumps {self.jumps}")


# ---- real hardware ----------------------------------------------------------

class ButtplugSink:
    """Real hardware via Intiface Central. Sends each actuator class to the
    matching device features; skips classes the toy doesn't have."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.client = None
        self.device = None
        self.last_position = 0.15

    async def connect(self, scan_seconds: float = 8.0):
        from buttplug import ButtplugClient, OutputType
        client = ButtplugClient("vibe.py fly-brain controller")
        await client.connect(self.url)
        self.client = client
        client.on_device_added = lambda d: print(f"[vibe] device found: {d.name}")
        await client.start_scanning()
        await asyncio.sleep(scan_seconds)
        await client.stop_scanning()
        devices = list(client.devices.values())
        if not devices:
            raise RuntimeError("no devices found — is the toy on and paired?")
        self.device = devices[0]
        counts = {t: len(self.device.get_features_with_output(t))
                  for t in (OutputType.VIBRATE, OutputType.ROTATE,
                            OutputType.OSCILLATE, OutputType.POSITION,
                            OutputType.CONSTRICT, OutputType.TEMPERATURE)}
        print(f"[vibe] using {self.device.name}: "
              + ", ".join(f"{k.value} {v}" for k, v in counts.items() if v))
        return self.device

    async def apply(self, state: VibeState) -> None:
        from buttplug import DeviceOutputCommand, OutputType
        d = self.device
        if d is None:
            return
        vib = d.get_features_with_output(OutputType.VIBRATE)
        if vib:
            await vib[0].run_output(DeviceOutputCommand(OutputType.VIBRATE, state.vib_base))
            if len(vib) > 1:
                await vib[1].run_output(DeviceOutputCommand(OutputType.VIBRATE, state.vib_tip))
            else:
                await vib[0].run_output(DeviceOutputCommand(
                    OutputType.VIBRATE, max(state.vib_base, state.vib_tip)))
        if state.rotate:
            for f in d.get_features_with_output(OutputType.ROTATE):
                await f.run_output(DeviceOutputCommand(OutputType.ROTATE, state.rotate))
        for f in d.get_features_with_output(OutputType.OSCILLATE):
            await f.run_output(DeviceOutputCommand(OutputType.OSCILLATE, state.oscillate))
        if state.position != self.last_position:
            for f in d.get_features_with_output(OutputType.POSITION):
                await f.run_output(DeviceOutputCommand(
                    OutputType.POSITION_WITH_DURATION, state.position, POSITION_MS))
            self.last_position = state.position
        for i, f in enumerate(d.get_features_with_output(OutputType.CONSTRICT)):
            await f.run_output(DeviceOutputCommand(
                OutputType.CONSTRICT, state.constrict[i % len(state.constrict)]))
        for f in d.get_features_with_output(OutputType.TEMPERATURE):
            await f.run_output(DeviceOutputCommand(
                OutputType.TEMPERATURE,
                TEMP_AMBIENT + (TEMP_MAX - TEMP_AMBIENT) * state.temperature))

    async def stop(self) -> None:
        if self.client and self.client.connected:
            await self.client.stop_all_devices()
            await self.client.disconnect()


# ---- headless loop ----------------------------------------------------------

async def run(server: str | None, world_on: bool, seconds: float,
              fixture: bool) -> SimDevice | ButtplugSink:
    brain = FlyBrain(fixture=fixture)
    world = FlyingWorld(seed=1) if world_on else None
    mapper = VibeMapper()
    dev = SimDevice()
    sink: SimDevice | ButtplugSink = ButtplugSink(server) if server else dev
    if server:
        await sink.connect()

    frame = np.full((256, 384, 3), 128, np.uint8) if not world_on else world.frame()
    print("[vibe] ctrl-c to stop\n")
    t0 = time.perf_counter()
    try:
        while True:
            if world_on:
                frame = world.frame()
                control, _ = brain.step(frame)
                world.step(control.x, control.y, control.jump)
            else:
                control, _ = brain.step(frame)
            state = mapper.update(control)
            if isinstance(sink, SimDevice):
                sink.apply(state)
                sink.step(brain.dt)
                print("\r" + sink.line() + "   ", end="", flush=True)
            else:
                await sink.apply(state)
            if seconds and time.perf_counter() - t0 > seconds:
                break
            await asyncio.sleep(brain.dt)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        if server:
            await sink.stop()
    return sink


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", default=None,
                    help="buttplug server URL (Intiface Central default ws://127.0.0.1:12345)")
    ap.add_argument("--no-world", action="store_true", help="flat gray frames, brain idles")
    ap.add_argument("--fixture", action="store_true", help="4,096-cell fixture brain (fast)")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds (0 = until ctrl-c)")
    ap.add_argument("--gui", action="store_true", help="launch vibe-gui.py instead")
    args = ap.parse_args()
    if args.gui:
        import subprocess, sys
        sys.exit(subprocess.call([sys.executable, "vibe-gui.py"]))
    asyncio.run(run(args.server, not args.no_world, args.seconds, args.fixture))


if __name__ == "__main__":
    main()
