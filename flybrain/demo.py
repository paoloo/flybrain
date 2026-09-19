"""Interactive demo: watch the fly brain play the flying game in real time.

Usage:
  conda run -n mcp python -m flybrain.demo --brain fixture --steps 1500
  conda run -n mcp python -m flybrain.demo --brain malecns --steps 300
  conda run -n mcp python -m flybrain.demo --brain mock --steps 1500

Shows a live OpenCV window with the game cubemap (resized), the fly's two-eye
fisheye view, and decoded controls. Press q to quit. Also logs telemetry CSV
and periodic eye-view PNGs into runs/<name>_<timestamp>/.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from .agent import MaleCNSPilot, MockPilot
from .retina import PREVIEW_HEIGHT, PREVIEW_WIDTH
from .world import FlyingWorld

PANEL_W, PANEL_H = 768, 512


def _render(frame, eye, info, control, label):
    game = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (PANEL_W // 2, PANEL_H // 2))
    eye_bgr = cv2.cvtColor(eye, cv2.COLOR_RGB2BGR)
    eye_big = cv2.resize(eye_bgr, (PANEL_W // 2, PANEL_H // 4))
    info_img = np.zeros((PANEL_H // 4, PANEL_W // 2, 3), np.uint8)
    lines = [
        f"{label[:52]}",
        f"score {info['score']}  dist {info['distance']:.1f}  alt {info['altitude']:.1f}",
        f"stick x {control[0]:+4d}  y {control[1]:+4d}  jump {control[2]}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(info_img, line, (8, 24 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 255, 200), 1, cv2.LINE_AA)
    right = np.vstack([eye_big, info_img])
    panel = np.hstack([game, right])
    panel = cv2.copyMakeBorder(panel, 0, PANEL_H - panel.shape[0], 0, PANEL_W - panel.shape[1],
                               cv2.BORDER_CONSTANT, value=(20, 20, 20))
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain", choices=["malecns", "mock", "fixture"], default="fixture")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    world = FlyingWorld(seed=args.seed)
    if args.brain == "malecns":
        backend = MaleCNSPilot()
        ticks_per_frame = 1
    elif args.brain == "mock":
        backend = MockPilot()
        ticks_per_frame = 1
    else:
        from .brain import FlyBrain
        pilot = MaleCNSPilot.__new__(MaleCNSPilot)
        pilot.brain = FlyBrain(fixture=True, seed=64)
        backend = pilot
        ticks_per_frame = 1

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or Path("runs") / f"{args.brain}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = world.reset()
    backend.reset()
    log = []
    print(f"{backend.label} | logging to {out_dir}")
    try:
        for step in range(args.steps):
            t0 = time.time()
            x, y, jump = backend.act(frame, 0.02)
            eye = backend.brain.eye_view(frame) if hasattr(backend, "brain") else \
                cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
            frame, reward, done, info = world.step(x, y, jump)
            panel = _render(frame, eye, info, (x, y, jump), backend.label)
            if not args.no_window:
                cv2.imshow("flybrain demo", panel)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            log.append(dict(step=step, x=x, y=y, jump=jump, reward=reward,
                            score=info["score"], distance=info["distance"],
                            altitude=info["altitude"], heading=info["heading"]))
            if step % 100 == 0:
                cv2.imwrite(str(out_dir / f"eye_{step:05d}.png"),
                            cv2.cvtColor(eye, cv2.COLOR_RGB2BGR))
            # pace roughly to real time for the tiny brains; malecns runs flat out
            if args.brain != "malecns":
                time.sleep(max(0.0, 0.02 - (time.time() - t0)))
            if done:
                print(f"episode done at step {step}, score {info['score']}")
                frame = world.reset()
                backend.reset()
    finally:
        cv2.imwrite(str(out_dir / "final_panel.png"),
                    panel if "panel" in dir() else np.zeros((10, 10, 3), np.uint8))
        cv2.destroyAllWindows()
        with (out_dir / "telemetry.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(log[0].keys()))
            writer.writeheader()
            writer.writerows(log)
        print(f"saved {len(log)} rows to {out_dir/'telemetry.csv'}")


if __name__ == "__main__":
    main()
