# flyBrain - a fly connectome that plays a flying game

This repository runs a simulation of the adult male fruit fly's central
nervous system and lets you interact with it. The wiring between the
166,700 neurons is measured data: the MaleCNS v1.0 connectome, reconstructed
from electron microscopy and published under CC-BY (Berg et al., Cell 2026,
DOI 10.1016/j.cell.2026.08.015). On top of that wiring, the simulation adds
simple engineered rules for vision, spikes, and motor output, following the
fly64 project (ornata/fly). Nothing is trained; there is no GPU and no
neural-network library. The network is a fixed dynamical system that you
feed images and read behavior from.

```
image (256x384x3 cubemap)
   -> 6,006 photoreceptors (spherical compound-eye model)
   -> 166,700 leaky integrate-and-fire neurons (measured wiring, 25.6M edges)
   -> spike counts in descending-neuron pools over the last ~260 ms
   -> controls / steering commands / your own readouts
```

Three ways in:

| Entry point | What you get |
|---|---|
| `minimal.py` | 17 lines: load the brain, feed one frame, read controls |
| `tracking.py` | webcam color tracking, output styled as servo commands |
| `2track.py` | same task, fly-natural: hemispheric steering + population-code throttle |
| `python -m flybrain.demo` | the fly playing a flying game in a live window |

The docs explain everything: GUIDE.md is the practical manual, REPORT.md
explains how it works and how it compares with a conventional neural
network (with code snippets and sources for every factual claim),
PROVENANCE.md states file-by-file what is original and what comes from
elsewhere.

## Requirements

- macOS, Linux, or Windows; a machine with ~8 GB RAM is comfortable.
  (Development happened on an Apple M4; the full brain steps in about 10 ms.)
- Python 3.11 or newer. The session used conda; any environment manager
  works. Below we write `conda run -n mcp python` - replace with your own
  environment's python if you prefer.

Python packages: numpy, scipy, pyarrow, pandas, opencv-python (only the
demos and trackers need OpenCV; the brain itself does not).

```sh
conda create -n mcp python=3.12 -y
conda run -n mcp pip install numpy scipy pyarrow pandas opencv-python
```

## Getting the brain data (~1.2 GB, one command, not in this repo)

The connectome is not distributed here. The preparation step downloads the
official MaleCNS v1.0 tables from Janelia's public storage (CC-BY), selects
the 166,700 annotated neurons, signs and normalizes the 25.6 million
connections, and caches everything in `data/cache/`:

```sh
conda run -n mcp python -m flybrain.data --prepare --cache data/cache
```

This downloads (via curl, resumable):

- annotations and neurotransmitter tables (feather)
- the 1.1 GB connection-strength table (feather)
- the optic-column assignment spreadsheet (xlsx)

It takes a few minutes on a good connection plus a few more to process.
When it finishes you will see:

```
Prepared 166,700 neurons and 25,582,938 weighted edges in data/cache
```

The cache is reusable; every script finds it by default. To work without
the download, use the synthetic offline brain instead: `FlyBrain(fixture=True)`
builds a 4,096-neuron test graph with the same API.

## Run it

```sh
# 1. the smallest possible example (works offline after data download)
conda run -n mcp python minimal.py

# 2. webcam tracking: hold a green (or red/blue) object in view
conda run -n mcp python tracking.py     # prints "move: left/right/up/down"
conda run -n mcp python 2track.py       # prints per-side throttle cells + turn rate

# 3. the flying game, live window
conda run -n mcp python -m flybrain.demo --brain malecns --steps 600
conda run -n mcp python -m flybrain.demo --brain fixture --steps 1500   # no download needed

# 4. headless run, writes telemetry.csv + eye-view PNGs to runs/
conda run -n mcp python -m flybrain.demo --brain malecns --steps 300 --no-window
```

Press `q` to close any OpenCV window. In the trackers, change `TARGET`
("green", "red", "blue") and `CAMERA_INDEX` at the top of the file.

## Using the brain from your own code

```python
from flybrain.brain import FlyBrain
import numpy as np

brain = FlyBrain()                       # or FlyBrain(fixture=True)
frame = np.zeros((256, 384, 3), np.uint8)
frame[:, :192] = (0, 120, 0)             # left half of the world green

control, spikes = brain.step(frame, brain.step_count * brain.dt)
print(control.x, control.y, control.jump)   # decoded game controls
print(len(spikes), "neurons spiked this tick")
print(brain.pool_rates())                   # descending-neuron pool rates
print(brain.eye_view(frame).shape)          # (128, 256, 3) fisheye preview
```

One call to `step` is one 20 ms tick of the whole network. The frame
contract: a `(256, 384, 3)` uint8 cube atlas (six 128x128 faces: front,
right, back, left on the top row; up, down on the bottom). Any image source
works - webcam, game, video file - as long as it arrives in that layout.

Read GUIDE.md for the full tour (tunable parameters, the game's rules,
experiment runner, troubleshooting) and REPORT.md for how it all works.

## Repository layout

```
flybrain/           the simulation package
  brain.py            FlyBrain: LIF dynamics + motor decoder
  retina.py           spherical compound-eye model (clean-room implementation)
  world.py            the flying game (cubemap renderer + arcade physics)
  agent.py            pilot backends + closed-loop experiment runner
  data.py             download + prepare the MaleCNS cache
  demo.py             interactive CLI
tracking.py         webcam tracking, game-style commands
2track.py           webcam tracking, fly-natural control scheme
minimal.py          smallest usage example
GUIDE.md            practical manual
REPORT.md           how it works + comparison with conventional NNs + sources
PROVENANCE.md       file-by-file origin and license notes
```

## License and credits

MIT for the code (see LICENSE); the third-party notice section of LICENSE
covers the data and attribution terms below.

Brain wiring: MaleCNS v1.0, Berg et al., Cell 189(18), 2026,
https://male-cns.janelia.org/, CC-BY 4.0. Model semantics (LIF parameters,
retinal encoding, motor decoding) follow ornata/fly (fly64); the
implementation here is independent - see PROVENANCE.md for the full
provenance discussion. Eye-field geometry follows the wide-field
characterization in NeuroMechFly v2 (Wang-Chen et al., Nature Methods 2024).

Flight-control biology behind 2track.py: Yang et al., Cell 2024 (steering
descending neurons), Namiki et al., Current Biology 2022 (population-code
throttle), Dickerson et al., Current Biology 2019 (haltere stabilization);
full citations in REPORT.md.
