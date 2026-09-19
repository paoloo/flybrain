"""Bare minimum: load the fly brain, feed it a frame, read the controls."""
import numpy as np
from flybrain.brain import FlyBrain

brain = FlyBrain()                    # loads data/cache; FlyBrain(fixture=True) skips the download
frame = np.zeros((256, 384, 3), np.uint8)   # black world (or grab a real cubemap frame)
frame[:, 192:] = (0, 120, 0)          # paint the right half green

for _ in range(50):                   # 50 ticks = 1 second of simulated time
    control, spikes = brain.step(frame, brain.step_count * brain.dt)

print(f"stick x={control.x} y={control.y} jump={control.jump}, {len(spikes)} neurons spiked")
