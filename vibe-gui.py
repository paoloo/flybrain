"""vibe-gui.py — realtime simulation of a complex buttplug.io toy driven by the fly brain.

The canvas IS the device: a rendered shaft with dual vibration motors, a
rotator (the shaft spins), an oscillating tip, a linear thruster, three
constricting grip rings and a temperature strip. All of it is physically
simulated at 50 Hz (SimDevice in vibe.py: motor lag, rotation integration,
thrust slew, ring relax, heater) and actuated autonomously by the MaleCNS
fly brain's descending-neuron pools:

  DNg100 forward   -> vibrate base + tip
  DNa02/DNg13 turn -> rotate (signed)
  DNp01/DNp10 rate -> oscillate tip
  jump burst       -> position swing (thrust)
  forward + phase  -> constrict rings, peristaltic
  activity         -> heater

Run:
  conda run -n mcp python vibe-gui.py            # full MaleCNS (~20 s load)
  conda run -n mcp python vibe-gui.py --fixture  # 4k-cell brain, instant
  conda run -n mcp python vibe-gui.py --server ws://127.0.0.1:12345  # mirror to real toy
"""
from __future__ import annotations

import argparse
import asyncio
import math
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np

from flybrain.brain import FlyBrain
from flybrain.world import FlyingWorld
from vibe import SimDevice, VibeMapper, TEMP_AMBIENT, TEMP_MAX, VIB_CEIL

TICK_MS = 20  # brain dt = 20 ms -> 50 Hz panel refresh

# --- geometry (canvas coords) ------------------------------------------------
CX = 330           # shaft centerline x
BASE_Y = 470       # base y (bottom)
TIP_LEN = 260      # shaft length at rest
SHAFT_W = 44       # half-width of shaft


class VibeGUI:
    def __init__(self, root: tk.Tk, server: str | None, fixture: bool) -> None:
        self.root = root
        root.title("fly brain → simulated toy  ·  MaleCNS closed loop")
        self.brain = FlyBrain(fixture=fixture)
        self.world = FlyingWorld(seed=1)
        self.mapper = VibeMapper()
        self.dev = SimDevice()
        self.server = server
        self.loop = None
        self.paused = False
        self._t = 0.0

        self._build()
        self._start_async()
        root.after(TICK_MS, self._tick)

    # ---- layout -------------------------------------------------------------
    def _build(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=1)

        # --- left: the toy itself ---
        toy = ttk.LabelFrame(main, text="device simulation (all actuators physical, 50 Hz)",
                             padding=4)
        toy.grid(row=0, column=0, sticky="nsew", rowspan=2)
        self.canvas = tk.Canvas(toy, width=660, height=500, bg="#14161c",
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # --- right: pools + telemetry ---
        side = ttk.Frame(main)
        side.grid(row=0, column=1, sticky="nsew")
        pools = ttk.LabelFrame(side, text="descending pools", padding=6)
        pools.pack(fill="x")
        self.pool_bars = {}
        for name in ("forward", "turn_left", "turn_right", "jump"):
            row = ttk.Frame(pools)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=name, width=10).pack(side="left")
            bar = ttk.Progressbar(row, length=130, maximum=1.0, mode="determinate")
            bar.pack(side="left", padx=4, fill="x", expand=True)
            self.pool_bars[name] = bar

        read = ttk.LabelFrame(side, text="commands (from brain)", padding=6)
        read.pack(fill="x", pady=6)
        self.readout = tk.Text(read, height=7, width=34, bg="#14161c", fg="#9fe29f",
                               font=("Menlo", 10), highlightthickness=0)
        self.readout.pack()
        self.readout.insert("1.0", "booting brain...")

        hw = ttk.LabelFrame(side, text="hardware mirror", padding=6)
        hw.pack(fill="x")
        self.hw_status = ttk.Label(hw, text=self.server or "simulated only (--server to mirror)")
        self.hw_status.pack()

        hud = ttk.Frame(main)
        hud.grid(row=1, column=1, sticky="sew")
        self.hud = ttk.Label(hud, text="", font=("Menlo", 10))
        self.hud.pack(anchor="w")
        btns = ttk.Frame(hud)
        btns.pack(anchor="w", pady=4)
        ttk.Button(btns, text="pause", command=self._toggle).pack(side="left", padx=2)
        ttk.Button(btns, text="reset world", command=self._reset).pack(side="left", padx=2)

        for c in (0, 1):
            main.columnconfigure(c, weight=1)

        # static canvas art
        self.canvas.create_text(330, 20, text=self.dev.name, fill="#8892a6",
                                font=("Menlo", 11))
        self.canvas.create_oval(CX - 12, BASE_Y - 8, CX + 12, BASE_Y + 8,
                                fill="#2a2e3a", outline="")  # base joint

    # ---- async sink ---------------------------------------------------------
    def _start_async(self) -> None:
        if not self.server:
            return
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self._run_loop, daemon=True).start()
        asyncio.run_coroutine_threadsafe(self._connect_hw(), self.loop)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _connect_hw(self) -> None:
        from vibe import ButtplugSink
        self.hw = ButtplugSink(self.server)
        await self.hw.connect()
        self.hw_status.config(text=f"mirroring to {self.hw.device.name}")

    # ---- one 20 ms tick -----------------------------------------------------
    def _tick_once(self) -> None:
        if self.paused:
            return
        frame = self.world.frame()
        control, _ = self.brain.step(frame)
        state = self.mapper.update(control)
        self.dev.apply(state)
        self.dev.step(self.brain.dt)
        self.world.step(control.x, control.y, control.jump)
        self._draw()
        self._readout(state)
        rates = self.brain.pool_rates()
        for name, bar in self.pool_bars.items():
            bar["value"] = min(rates[name], bar["maximum"])

    def _tick(self) -> None:
        self._tick_once()
        self.root.after(TICK_MS, self._tick)

    # ---- the toy render -----------------------------------------------------
    def _draw(self) -> None:
        c = self.canvas
        c.delete("all")
        t = self._t + self.dev.ticks * self.brain.dt
        self.canvas.create_text(330, 20, text=self.dev.name, fill="#8892a6",
                                font=("Menlo", 11))

        # thrust: whole shaft slides up/down; position 0..1 -> 0..90 px
        thrust = self.dev.position * 90.0
        # rotation: shaft bends as a twisted curve; show twist by rotating the
        # tip marker around the shaft axis (projected) + ridge lines
        twist = self.dev.angle
        # oscillation: tip wiggle offset
        osc = self.dev.osc_amp * 18.0 * math.sin(t * 2 * math.pi * (1 + 9 * self.dev.osc_amp))
        # vibration: high-freq jitter at tip, amplitude scaled by motor values
        jx = (self.dev.vib_tip * 2.2) * math.sin(t * 2 * math.pi * 130)
        jy = (self.dev.vib_tip * 2.2) * math.cos(t * 2 * math.pi * 117)
        jb = (self.dev.vib_base * 1.4) * math.sin(t * 2 * math.pi * 97)

        top_y = BASE_Y - TIP_LEN + thrust  # tip y position

        # shaft body: tapered polygon with base jitter
        pts = []
        n = 24
        for i in range(n + 1):
            f = i / n                       # 0 at base, 1 at tip
            y = BASE_Y - (TIP_LEN - thrust) * f
            w = SHAFT_W * (1.0 - 0.55 * f * f)
            wig = jb * (1 - f) + jx * f + osc * f
            pts.append((CX + wig - w, y))
        for i in range(n, -1, -1):
            f = i / n
            y = BASE_Y - (TIP_LEN - thrust) * f
            w = SHAFT_W * (1.0 - 0.55 * f * f)
            wig = jb * (1 - f) + jx * f + osc * f
            pts.append((CX + wig + w, y))
        # color shifts with temperature
        heat = (self.dev.temp - TEMP_AMBIENT) / (TEMP_MAX - TEMP_AMBIENT)
        col = f"#{int(200 + 55 * heat):02x}{int(90 + 40 * heat):02x}{int(140 - 60 * heat):02x}"
        c.create_polygon(*(p for pt in pts for p in pt), fill=col, outline="")

        # rotation ridges: helix lines whose phase advances with dev.angle
        for k in range(3):
            line = []
            for i in range(0, n + 1):
                f = i / n
                y = BASE_Y - (TIP_LEN - thrust) * f
                w = SHAFT_W * (1.0 - 0.55 * f * f)
                wig = jb * (1 - f) + jx * f + osc * f
                ph = twist + f * 4.0 + k * 2 * math.pi / 3
                line.append((CX + wig + w * 0.8 * math.sin(ph), y))
            c.create_line(*(p for pt in line for p in pt),
                          fill="#5a3a55", width=2)

        # constricting rings: 3 ellipses at fixed fractions, width from ring pressure
        for i, r in enumerate(self.dev.rings):
            f = 0.3 + i * 0.22
            y = BASE_Y - (TIP_LEN - thrust) * f
            w = SHAFT_W * (1.0 - 0.55 * f * f)
            wig = jb * (1 - f) + jx * f + osc * f
            squeeze = 1.0 - 0.55 * r
            c.create_oval(CX + wig - w * squeeze, y - 7, CX + wig + w * squeeze, y + 7,
                          outline="#ffd27a" if r > 0.05 else "#3a3f4e",
                          width=3 if r > 0.05 else 1)

        # head/tip: oscillating glans ellipse
        tipx = CX + jx + osc
        tipy = top_y
        c.create_oval(tipx - 16, tipy - 26, tipx + 16, tipy + 4,
                      fill=col, outline="#2a2e3a")
        # rotation indicator: little marker orbiting the tip
        mx = tipx + 20 * math.sin(twist)
        c.create_oval(mx - 4, tipy - 15, mx + 4, tipy - 7,
                      fill="#9fe29f", outline="")

        # actuator telemetry bars, drawn on canvas
        labels = [
            ("vib base", self.dev.vib_base, 20),
            ("vib tip", self.dev.vib_tip, 20),
            ("rotate", abs(self.dev.rotate_speed), 20),
            ("oscillate", self.dev.osc_amp, 20),
            ("thrust", self.dev.position, 20),
        ]
        x0 = 470
        y0 = 60
        for i, (name, val, wid) in enumerate(labels):
            y = y0 + i * 26
            c.create_text(x0 - 8, y, text=name, fill="#8892a6",
                          anchor="e", font=("Menlo", 10))
            c.create_rectangle(x0, y - 7, x0 + wid * 6, y + 7,
                               outline="#2a2e3a", fill="#1b1e27")
            c.create_rectangle(x0, y - 7, x0 + wid * 6 * min(val, 1.0), y + 7,
                               outline="", fill="#6fcf6f")
        # rings + temp
        y = y0 + 5 * 26
        c.create_text(x0 - 8, y, text="rings", fill="#8892a6",
                      anchor="e", font=("Menlo", 10))
        for i, r in enumerate(self.dev.rings):
            c.create_rectangle(x0 + i * 44, y - 7, x0 + i * 44 + 36 * min(r, 1.0), y + 7,
                               outline="", fill="#ffd27a")
        y = y0 + 6 * 26
        c.create_text(x0 - 8, y, text="temp", fill="#8892a6",
                      anchor="e", font=("Menlo", 10))
        tf = (self.dev.temp - TEMP_AMBIENT) / (TEMP_MAX - TEMP_AMBIENT)
        c.create_rectangle(x0, y - 7, x0 + 120 * min(max(tf, 0.0), 1.0), y + 7,
                           outline="", fill="#e0715f")
        c.create_text(x0 + 130, y, text=f"{self.dev.temp:4.1f}°C",
                      fill="#8892a6", anchor="w", font=("Menlo", 10))

        self._t = t

    # ---- readout / hud ------------------------------------------------------
    def _readout(self, state) -> None:
        lines = [f"vibrate   base {state.vib_base:4.2f}  tip {state.vib_tip:4.2f}",
                 f"rotate    {state.rotate:+5.2f}",
                 f"oscillate {state.oscillate:4.2f}",
                 f"position  {state.position:4.2f}",
                 f"constrict {' '.join(f'{v:4.2f}' for v in state.constrict)}",
                 f"heater    {state.temperature:4.2f}"]
        self.readout.config(state="normal")
        self.readout.delete("1.0", "end")
        self.readout.insert("1.0", "\n".join(lines))
        self.readout.config(state="disabled")
        self.hud.config(text=f"speed {self.world.v:4.2f}  heading {np.degrees(self.world.heading):6.1f}°  "
                             f"distance {self.world.distance:6.1f}  reward {self.world.score}"
                             + ("  [paused]" if self.paused else ""))

    # ---- buttons ------------------------------------------------------------
    def _toggle(self) -> None:
        self.paused = not self.paused
        if self.paused and self.loop:
            asyncio.run_coroutine_threadsafe(self._stop_hw(), self.loop)

    async def _stop_hw(self) -> None:
        if getattr(self, "hw", None):
            await self.hw.stop()

    def _reset(self) -> None:
        self.world.reset()

    def close(self) -> None:
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)


def main() -> None:
    ap = argparse.ArgumentParser(description="fly brain → simulated toy, realtime GUI")
    ap.add_argument("--server", default=None, help="Intiface Central websocket URL")
    ap.add_argument("--fixture", action="store_true", help="small fixture brain (instant load)")
    args = ap.parse_args()
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("aqua")
    except tk.TclError:
        pass
    gui = VibeGUI(root, args.server, args.fixture)
    root.protocol("WM_DELETE_WINDOW", lambda: (gui.close(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
