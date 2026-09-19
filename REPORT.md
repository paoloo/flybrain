# REPORT.md - How the fly brain simulation works

This report explains what the harness in this directory does, where its data
comes from, how the 166,700 neurons are used, and how that differs from a
conventional neural network. Every factual claim points to a source listed in
"Sources" at the end. The practical "how do I run it" material lives in
GUIDE.md; this document is the "why it works" companion.

Map of the sections:

1. The big picture (the loop, no code)
2. The data (what was downloaded and how it is prepared)
3. The neurons (the LIF update, with the implementing code)
4. The eyes (cubemap -> photoreceptor drive, with code; 4a: feeding a webcam)
5. From spikes to controls (the game decoder, with code;
   5a: what real flies do instead; 5b: the natural scheme in 2track.py)
6. The game world (physics + renderer, with code)
7. Comparison with a normal neural network
8. Limitations
9. Sources

## 1. The big picture

The program runs a loop that never touches a GPU and never trains anything:

```
render a frame of the game world (384x256 RGB cubemap)
        |
        v
6,006 photoreceptor cells sample the frame on a sphere
        |
        v
166,700 leaky integrate-and-fire neurons, wired by measured connectome data
        |
        v
count spikes in a few descending-neuron pools over the last ~260 ms
        |
        v
stick position and a jump flag
        |
        v
the game world moves, a new frame is rendered, repeat at 50 steps per second
```

The wiring between the neurons is not invented and not learned. It comes from
an electron-microscopy reconstruction of an adult male fruit fly's central
nervous system, published as the MaleCNS connectome (Berg et al., 2026). What
the simulation adds on top of that wiring is a set of simple, explicitly
engineered rules: how voltage accumulates, how light is converted into
current, and how spike counts become game controls. The fly64 project this
code follows states this plainly in its technical notes: the wiring is
measured, the dynamics and mappings are approximations (ornata/fly,
docs/technical-notes.md).

## 2. The data: a measured wiring diagram

MaleCNS v1.0 is a connectome of the entire central nervous system (brain plus
nerve cord) of an adult male Drosophila, reconstructed from electron microscopy
at 8 nm resolution by the FlyEM team at Janelia with partners in Cambridge, the
MRC Laboratory of Molecular Biology, and Google Research. The published
version reports 166,691 neurons and 11,691 cell types (Berg et al., 2026,
Cell 189(18)). The dataset is distributed under CC-BY from
https://male-cns.janelia.org/download/ as flat tables: annotations, predicted
neurotransmitters, and a 1.1 GB connection-strength table.

Our copy lives in `data/cache/raw/` with SHA-256 hashes recorded in
`data/cache/manifest.json`. The preparation step (`flybrain/data.py`) applies
the same rules as fly64:

1. Keep every neuron that has a non-empty superclass annotation. That is
   166,700 rows including 94 marked "tbc" (to be confirmed). Filtering by
   proofreading status would wrongly drop many sensory cells, which is why
   this count is slightly above the paper's 166,691 (ornata/fly,
   docs/technical-notes.md).
2. Turn each synapse count into a signed weight. The sign comes from the
   presynaptic cell's predicted neurotransmitter: GABA, glutamate and
   histamine are treated as inhibitory, everything else as excitatory. The
   paper's own transmitter predictions feed this step, but the mapping itself
   is an approximation; real receptors can flip a transmitter's effect.
3. Normalize so that each neuron's outgoing weights sum to a fixed budget:
   weight = synapse count divided by the total incoming absolute weight of the
   target cell. After normalization the network has 25,582,938 directed edges.

Two cell-type lists matter for the game. Six thousand six photoreceptors
(types R1-6, R7, R8) are the eyes. A handful of descending neurons (DNg100,
DNa02, DNg13, DNp01, DNp10) are the "muscles" whose activity gets read out as
controls. Both lists come from the published annotations; nothing is guessed
at the level of individual cell types.

## 3. The neurons: leaky integrate-and-fire, not deep learning

Each of the 166,700 neurons holds one number, its membrane voltage v. Every
20 milliseconds the simulation updates all of them at once (parameters from
ornata/fly, fly64/model.py):

```
v = exp(-dt/tau) * v + 1.5 * (W @ spikes) + 0.180 + noise + retina
```

The terms, left to right:

- Passive leak. Voltage decays toward zero with a 100 ms time constant.
- Synaptic current. Every neuron that spiked on the previous tick injects
  current into its targets, in proportion to the normalized edge weights.
  Because most neurons do not spike in a given tick, this is computed as a
  sparse matrix-vector product over the spiking columns only.
- Tonic drive (0.180). A constant background current that keeps the network
  in an active regime. fly64 documents that this value and the synaptic gain
  of 1.5 were hand-tuned so that visual events are visible in the motor
  pools; they are not measured physiological constants.
- Noise. A seeded random process gives about 1.2% of neurons a small kick
  each tick. With a fixed seed (64) the noise is identical across runs, so
  experiments stay reproducible while the network never sits perfectly still.
- Retinal drive. Light-derived current, injected only into the 6,006
  photoreceptors.

When v reaches 1.0 the neuron spikes and resets to 0. That is the entire
"learning rule": there is none. Nothing is trained, no gradient exists, no
weight ever changes. The network is a fixed dynamical system being driven by
images.

The equation is implemented one-to-one in `flybrain/brain.py`, method
`FlyBrain.step` (abridged; constants are module-level `DT = 0.020`,
`TAU_M = 0.100`, `THRESHOLD = 1.0`):

```python
def step(self, rgb, now=None):
    sensory = self.encode_retina(rgb)                    # light -> drive
    current = np.asarray(
        self.w[:, np.flatnonzero(self.spikes)].sum(axis=1)
    ).ravel() * self.synaptic_gain                       # 1.5 * W @ spikes
    baseline = self.rng.random(self.n) < (1.2 * self.dt) # seeded noise
    self.v *= np.exp(-DT / TAU_M)                        # passive leak
    self.v += current + baseline.astype(np.float32) * 0.22 + self.tonic_current
    self.v[self.visual] += sensory * 0.62                # retina -> R1-R8 only
    fired = self.v >= THRESHOLD                          # spike decision
    self.v[fired] = 0.0                                  # reset
    self.spikes[:] = fired
    self.history.append(fired[self.motor_nodes].copy())  # feed the decoder
    return self._decode(now), np.flatnonzero(fired)
```

The sparsity point from section 7 is visible in one line:
`self.w[:, np.flatnonzero(self.spikes)]` touches only the columns of spiking
presynaptic neurons; the other ~155,000 neurons contribute exactly nothing
to this tick, with no approximation involved.

## 4. The eyes: from pixels to photoreceptor currents

The game renders six 128x128 faces (front, right, back, left, up, down) that
tile the sphere around the fly, an atlas of 384x256 pixels. The retina module
(flybrain/retina.py, copied from fly64) maps each photoreceptor to a direction
in space and samples the atlas with a small acceptance cone: one central
sample weighted 0.25 plus six ring samples at 0.125 each, about 2 degrees
across. The angular bounds (roughly 270 degrees horizontal, 72 degrees up and
down) follow the wide-field geometry described for the fly visual system in
NeuroMechFly v2 (Wang-Chen et al., 2024).

Three signals are extracted per photoreceptor and combined into a drive
value between 0 and 1:

- luminance, weighted 0.45,
- temporal contrast, the absolute change since the last frame, weighted 1.6,
- green opponency (green minus the average of red and blue), weighted 0.25.

RGB is used because the game has RGB; real flies see ultraviolet, so this
channel mapping is another engineered approximation, stated as such in the
fly64 notes.

In code, `flybrain/brain.py` method `encode_retina` - note that the cone
sampling already happened in `self.retina.sample`, so everything below
operates on the (6006, 3) array of per-photoreceptor RGB values:

```python
frame = self.retina.sample(rgb)            # (6006, 3) receptor samples
lum      = frame @ [0.2126, 0.7152, 0.0722]         # luminance
temporal = np.abs(lum - prev_lum)                   # frame-to-frame change
color    = np.maximum(frame[:,1] - 0.5*(frame[:,0]+frame[:,2]), 0)  # green
drive    = np.clip(0.45*lum + 1.6*temporal + 0.25*color, 0, 1)
```

`self.previous_rgb` then stores `frame`, which is the readout both webcam
trackers analyze (section 4a).

### 4a. Feeding a webcam image to the photoreceptors (theory)

The brain consumes one thing: a numpy array shaped (256, 384, 3), laid out as
a cube map. Both demo programs (tracking.py, 2track.py) feed it a webcam with
the same three steps, shown here with the actual code.

Step 1: understand the layout. The 256x384 atlas is six 128x128 faces.
flybrain/retina.py defines the face order as front, right, back, left on the
top row, up, down on the bottom row. Each atlas position corresponds to a
viewing direction on a sphere, and each of the 6,006 photoreceptors knows
which atlas position it samples (brain.visual_pixels, a (6006, 2) array of
row/column coordinates) plus which direction that is in space.

Step 2: put the camera into the atlas. A webcam sees about 60 degrees of the
world, straight ahead. That is exactly the front face, so the minimal
conversion is a resize and a paste:

```python
cubemap = np.zeros((256, 384, 3), np.uint8)      # black world everywhere else
cubemap[:128, :128] = cv2.resize(frame, (128, 128))   # camera -> front face
```

Everything outside the front face stays black, which means the fly
"believes" it is surrounded by darkness except where the camera looks. That
is honest: the webcam genuinely provides no information about the rest of
the sphere. (tracking.py flips the frame horizontally first, with
cv2.flip(frame, 1), so the view behaves like a mirror; this affects only
which way the fly's left and right are relative to you.)

Step 3: step the brain. brain.step(cubemap) runs the full pipeline: the
retina samples the atlas through each receptor's acceptance cone, converts
samples to drive, injects drive as current into photoreceptors only, and
integrates the whole network one 20 ms tick. What comes back is a decoded
control (the game decoder) and the spike list:

```python
control, spikes = brain.step(cubemap, brain.step_count * brain.dt)
```

Two useful readouts for experiments. First, brain.previous_rgb is the
(6006, 3) array of raw RGB values each photoreceptor sampled on the last
tick, before drive conversion. It is the fly's "retinal image" and it is
what both tracking programs analyze to find the colored point. Second,
brain.visual_pixels tells you where in the atlas each receptor sits, so a
weighted average over receptors doubles as a position estimate in the fly's
visual field. The weighted-average pattern used in 2track.py:

```python
score = color_score(brain.previous_rgb, TARGET)   # (6006,) evidence array
w = score / score.sum()
row = float((brain.visual_pixels[:, 0] * w).sum())   # elevation proxy
col = float((brain.visual_pixels[:, 1] * w).sum())   # azimuth proxy
```

Frame-rate note: the neural tick is defined as 20 ms, but nothing forces you
to call step() at 50 Hz wall-clock time. Calling it once per webcam frame
(usually 30 fps) simply means the brain's subjective time runs slower than
the wall clock by that ratio. For closed-loop tracking this is harmless; if
you need subjective/real sync, sleep so that step() is called every 20 ms.

Why a cubemap and not just the camera frame? Because the connectome's
photoreceptor arrays are anatomically ordered (left eye, right eye,
column by column, matching published optic-column assignments). The atlas
is the fixed stage on which that anatomy meets a viewing direction. Any
image source can feed it - a webcam, a game, a saved video - as long as it
is delivered in the 256x384 atlas layout.

## 5. From spikes to controls

The simulation keeps a rolling window of the last 13 ticks (260 ms) and
averages spike counts inside four pools of descending neurons:

| Pool | Cell types | Readout |
|---|---|---|
| forward | DNg100 | drives stick y (forward) |
| steering | DNa02 and DNg13, split by left/right soma side | right minus left drives stick x |
| jump | DNp01 and DNp10 | a burst above 0.04 spikes/cell/tick triggers a jump, with an 800 ms cooldown |

The raw rates pass through an exponential moving average (0.78/0.22), a dead
zone of 8 units, and a clamp at plus or minus 70, the same shaping fly64 uses
before handing values to the game. These mappings are engineering choices,
documented as such. The fly64 README notes only that these descending neuron
types are the groups it reads out; no measurement in the dataset pins a
specific firing rate to a specific stick deflection.

The decoder, `flybrain/brain.py` method `_decode`, in full:

```python
recent = np.stack(tuple(self.history), axis=0).mean(axis=0)  # 13-tick window
forward_rate, left_rate, right_rate, jump_rate = [
    float(pool.mean()) for pool in np.split(recent, self.motor_splits)]
turn_rate = right_rate - left_rate            # R-L, as in the real fly

raw_y = np.clip((forward_rate - 0.008) * 2000.0, 0, 70)
raw_x = np.clip(turn_rate * 1100.0, -70, 70)
self.filtered_y = 0.78 * self.filtered_y + 0.22 * raw_y   # EMA smoothing
self.filtered_x = 0.78 * self.filtered_x + 0.22 * raw_x
jump = jump_rate > 0.04 and now - self.last_jump >= 0.8    # burst + cooldown
```

`self.motor_splits` comes from `_load_cache`: the concatenated index arrays
of DNg100, left DNa02/DNg13, right DNa02/DNg13, and DNp01/DNp10, so one
`np.split` turns a flat spike window into the four pool means. The pools
themselves are introspectable at any time via `brain.pool_rates()`.

## 5a. What the real fly does instead of x and y

The stick is a video-game convenience. A fly's head contains no axis pair.
The verified biology, checked against four primary sources this session,
looks different in three ways.

First, control is a set of named neurons, each tied to a gesture, not a pair
of signed axes. Yang et al. (Cell 187(22):6290, 2024, DOI
10.1016/j.cell.2024.08.033) imaged and perturbed steering descending neurons
in walking flies: the right-minus-left activity difference of DNa02 and DNg13
correlates linearly with rotational velocity and precedes the turn by about
150 ms. Optogenetic activation shows they drive different leg gestures.
DNa02 shortens strides on the inside of a turn; DNg13 lengthens strides on
the outside. They are recruited independently because their inputs barely
overlap; DNa02 is directly targeted by central-complex heading-output
neurons, DNg13 is not.

Second, flight amplitude is a population code. Namiki et al. (Current
Biology 32(5):1189, 2022, DOI 10.1016/j.cub.2022.01.008) identified DNg02, a
population of at least 15 nearly identical cell pairs. Optogenetically
activating more of them raises wingbeat amplitude linearly, about 2 degrees
per cell pair, like a throttle built from range fractionation. During yaw
stimuli, the left and right populations act independently, each raising the
amplitude of the contralateral wing.

Third, the brain does not do the fast stabilization. Dickerson et al.
(Current Biology 29(20):3517, 2019, DOI 10.1016/j.cub.2019.08.065) showed
that halteres, the modified hindwings, act as a dual-function gyroscope and
clock: they sense body rotation mechanically, and descending visual input
modulates the halteres' own muscles, which retimes the spike timing of wing
steering muscles within each stroke cycle. A related study (Science Advances,
2022, DOI 10.1126/sciadv.abo7461) models this stabilization reflex as a PI
controller, with different steering-motor units embodying the proportional
and integral terms. The hierarchy is that the brain sets slow goals and
millisecond corrections happen locally in the thorax, without the brain.

Against that biology, our harness keeps one faithful piece and cartoons the
rest. The right-minus-left DNa02/DNg13 steering readout matches the
published correlate of turning. DNg100-as-forward and DNp01-as-jump are
fly64's engineering choices without strong published support. The fly also
never emits a position command; it modulates stroke asymmetries and lets
aerodynamics integrate them into a turn. File 2track.py in this directory
implements the natural scheme: independent left/right steering signals, a
population-code throttle, and a local proportional controller below the
brain layer.

### 5b. The natural scheme in code (2track.py), with measured results

2track.py runs the same webcam task as tracking.py, but every control
variable is a population activity, never an axis position. The code has
three layers, mirroring the hierarchy above.

Layer 1, brain-level goals. The eye signal is split into two independent
hemisphere sums, the way steering descending neurons compare left- versus
right-eye visual activity:

```python
def split_eyes(brain, target):
    score = color_score(brain.previous_rgb, target)
    left  = float(score[brain.visual_pixels[:, 1] < 32].sum())   # left eye cols
    right = float(score[brain.visual_pixels[:, 1] >= 32].sum())  # right eye cols
    return left, right
```

The steering goal is the difference right minus left; the side with more
evidence is the side the fly turns toward. There is no position command.

Layer 2, a population-code throttle. Total evidence sets how many cells of
an 8-cell-per-side DNg02-style population are recruited (the real fly has at
least 15 pairs; 8 keeps the HUD readable):

```python
effort_goal = np.clip((left_score + right_score) / 20.0, 0.0, 1.0)
l_cells = int(round(self.effort * N_CELLS))   # 0..8 cells per side
r_cells = int(round(self.effort * N_CELLS))
```

Layer 3, a local loop below the brain, playing the haltere role: a
first-order filter that chases the brain's goals with its own time
constants, so millisecond-style smoothing happens without involving the
brain layer, and a predicted body turn rate that integrates the steering
difference over time:

```python
steer_cmd  = np.clip((right_score - left_score) * K_STEER, -RATE_LIMIT, RATE_LIMIT)
self.rate += (steer_cmd - self.rate) * np.clip(dt * 6.0, 0, 1)
self.effort += (effort_goal - self.effort) * np.clip(dt * 4.0, 0, 1)
```

Turning also recruits the outer side's throttle population, echoing the
wider strokes on the outside of a turn:

```python
if self.rate > 2:      # turning right: left side is the outer side
    l_cells = min(N_CELLS, l_cells + 1)
elif self.rate < -2:
    r_cells = min(N_CELLS, r_cells + 1)
```

Measured behavior (offline runs with the fixture brain, a green blob pasted
into the front face at known positions, 15 ticks per scenario):

| Scenario | Predicted turn rate | Throttle | Effect |
|---|---|---|---|
| blob left of center | -13.0 deg/s | L5 / R6 | turns left, outer (right) side recruits more |
| blob right of center | +10.1 deg/s | L5 / R4 | turns right, outer (left) side recruits more |
| blob centered | -2.3 deg/s | L6 / R7 | near zero rate, high effort |
| empty world | 0.0 deg/s | L0 / R0 | idle |

The sign behavior is the point: the same blob produces opposite steering
depending only on which hemisphere sees it, and the response is a rate that
integrates toward a turn, not a position jump. Two geometry facts matter
when building your own scenarios. The camera's center of view maps to atlas
column about 32.3, and the two eyes have a 17-degree central overlap
(fly64's CALIBRATION bounds are left -135 to +8.5 degrees, right -8.5 to
+135), so a target near the middle lights up both hemispheres and the
difference goes toward zero, as it should.

## 6. The game world

Because the original fly64 is welded to a patched Super Mario 64 build
through a C shared-memory bridge, this harness ships its own pure-Python
world (flybrain/world.py) that produces the same frame format, so the retina
and brain run unmodified. The world is an open field with sky, ground, and
vertical landmark stripes: yellow ahead, red behind, brighter when close.
Stick y accelerates, stick x turns, jump climbs. Passing a landmark scores a
point; steering more than 90 degrees off course ends the episode. Physics is
arcade-simple on purpose; the interesting part of the loop is the brain, not
the collision model.

The full physics is six lines in `flybrain/world.py`, method `step`
(constants `ACCEL=0.9, DRAG=0.35, TURN=0.03, CLIMB=0.04, SINK=0.02`):

```python
self.v = np.clip(self.v + (y/70)*ACCEL - DRAG*self.v, -MAX_SPEED/2, MAX_SPEED)
self.distance += self.v * 0.05
self.heading += (x/70) * TURN
if jump and self.jump_phase == 0: self.jump_phase = 0.01   # start climb
...
```

The renderer (`world.frame`) loops over the six face bases imported from
`flybrain/retina.py` (the same `BASES` array fly64's native renderer uses),
projects each pixel to a ray, classifies it as ground or sky by elevation,
then paints landmark stripes by azimuth and distance. Any alternative world
(a video player, a real robot camera) only needs to produce the same
256x384x3 atlas per frame; the brain cannot tell the difference.

## 7. How this compares with a normal neural network

### Where the weights come from

In a normal network, training chooses the weights. Here the weights are
measurements: synapse counts from an electron microscopy volume, signed by
transmitter predictions, normalized by a fixed rule. No optimizer ever
touches them. The "training data" was the fly.

### Units and state

A deep network computes layer by layer, usually in floating point activations
with no memory between forward passes. Here every neuron is a small dynamical
system with memory (its voltage) and an event (its spike). Time is real: the
same input at two different moments can produce different spikes, because the
state carries history. Computation is 50 updates per second of simulated
time, not one forward pass per request.

### Sparsity

A dense network multiplies every weight every pass. Here a tick only
propagates from neurons that actually spiked. In one observed tick of our
smoke test, 11,919 of the 166,700 neurons fired, so the sparse product
touched a small slice of the 25.6 million edges. The connectome defines what
could talk; spikes decide what does.

### What is learned

Nothing, at runtime. There is no loss, no backpropagation, no reward-modified
weight. Behavior emerges from the fixed wiring being driven by vision and
noise. If you want the network to get better at the game, you would have to
add plasticity yourself; that is future work, not part of the model.

### Noise as a feature

Deterministic networks hide their randomness in initialization or dropout.
This model runs with continuous stochastic background activity, seeded for
reproducibility. The animal-like consequence is that the network is never
silent, and stimuli modulate ongoing activity rather than switching it on
from zero.

### Interpretability of parts

In a deep network, individual units mean little. Here, single identifiable
neurons have names, measured positions, known transmitters, and known
partners. The readout is five named cell types, and you can point at them in
the data. That is the main scientific payoff of using a connectome instead of
a learned graph.

### What does not transfer

The model has no receptor-level biophysics, no neuromodulation, no learning,
no body, and no claim to reproduce fly behavior quantitatively. fly64's
technical notes are explicit that the dynamics, the visual projection, and
the motor mappings are engineered approximations. The Cell paper's
contribution is the wiring and its analysis, not a simulation protocol. Treat
the network as a measured scaffold carrying guessed dynamics.

## 8. Limitations worth keeping in mind

- Inhibitory signs are inferred from transmitter identity, and fly64 notes
  that receptor-dependent and neuromodulatory effects are omitted, so the
  sign of a connection can be wrong where a receptor reverses the usual
  effect of its transmitter (ornata/fly, docs/technical-notes.md).
- The photoreceptor-to-visual-direction map is approximate: 2,628 receptors
  get directions from published optic-column assignments, 3,242 from a
  connectivity-derived estimate, and 136 from a deterministic fallback
  (counts from our cache manifest, matching fly64).
- The motor decode thresholds (0.008 forward offset, 0.04 jump burst,
  0.8 s cooldown) were tuned so that the game character moves, not measured
  in flies (ornata/fly, docs/technical-notes.md).
- The game world here is a substitute for SM64; it shares the frame format
  but not the game's visual complexity.

## 9. Sources

All verified during this session. Access levels: "full text" means the
document was read in full or in the sections cited; "metadata" means identity
was confirmed through Crossref or the publisher but the full text was not
read here.

1. Berg, S. et al. "Sexual dimorphism in the complete Drosophila male central
   nervous system connectome." Cell 189(18):5504-5526.e15, 3 Sep 2026.
   DOI 10.1016/j.cell.2026.08.015. (metadata; Crossref-verified; preprint
   10.1101/2025.10.09.680999 read via PubMed abstract)
2. MaleCNS v1.0 data distribution: https://male-cns.janelia.org/download/
   (full text of download page; file list, sizes, CC-BY license, and the
   gs://flyem-male-cns/v1.0/ storage paths our data/ module downloads from).
3. ornata/fly repository, README.md and docs/technical-notes.md
   (https://github.com/ornata/fly, full text; source of all model parameters,
   selection rules, and limitation statements used above). The file the user
   pointed at, fly64/data.py, is preserved locally at data/data.py.
4. Wang-Chen, S. et al. "NeuroMechFly v2: simulating embodied sensorimotor
   control in adult Drosophila." Nature Methods 21, 2353-2362 (2024).
   DOI 10.1038/s41592-024-02497-y. (metadata; Crossref-verified; cited by
   fly64 as the basis for the eye's angular bounds)
5. "Eye structure shapes neuron function in Drosophila motion vision."
   Nature (2025). DOI 10.1038/s41586-025-09276-5. (metadata;
   Crossref-verified; cited by fly64 for treating visual directions as
   spherical geometry)
6. flyconnectome/2025malecns supplemental repository
   (https://github.com/flyconnectome/2025malecns, README read), source of the
   optic-column assignment spreadsheet pinned by fly64.
7. Local verification: data/cache/manifest.json (SHA-256 hashes of the four
   downloaded tables, neuron and edge counts produced by our own run).
8. Yang, H.H. et al. "Fine-grained descending control of steering in walking
   Drosophila." Cell 187(22):6290-6308.e27, 2024. DOI
   10.1016/j.cell.2024.08.033. (full text via PMC; source for DNa02/DNg13
   steering, gesture specificity, and the ~150 ms lead; see section 5a)
9. Namiki, S. et al. "A population of descending neurons that regulate the
   flight motor of Drosophila." Current Biology 32(5):1189-1196.e6, 2022.
   DOI 10.1016/j.cub.2022.01.008. (full text via PMC; source for DNg02
   population code and wingbeat amplitude control; see section 5a)
10. Dickerson, B.H. et al. "Flies Regulate Wing Motion via Active Control of
    a Dual-Function Gyroscope." Current Biology 29(20):3517-3524.e3, 2019.
    DOI 10.1016/j.cub.2019.08.065. (abstract via PubMed; source for haltere
    gyroscope-and-clock function; see section 5a)
11. "Neuromuscular embodiment of feedback control elements in Drosophila
    flight." Science Advances, 2022. DOI 10.1126/sciadv.abo7461.
    (metadata; source for the PI-controller model of stabilization; see
    section 5a)
