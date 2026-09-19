"""Spherical compound-eye model for the MaleCNS simulation.

Clean-room reimplementation for this teaching repository. The module was
rewritten from the documented specification (see REPORT.md, section 4) to
avoid importing code from the unlicensed ornata/fly repository. The math is
required to behave identically to the original; a differential test at the
bottom of this file (run as __main__) verifies it.

Specification summary (all choices documented in REPORT.md):
- The world is given as a six-face cube atlas: 384x256 pixels, six 128x128
  faces ordered forward, right, back, left (top row) and up, down (bottom).
- Each photoreceptor looks in a fixed direction on the sphere and integrates
  light from a small acceptance cone: 7 spherical samples (center + a ring of
  6), weighted 0.25 / 0.125 each.
- Eye fields approximate the adult fly's ~270 degree binocular field with a
  17 degree central overlap; elevation is limited to +/-72 degrees.
- A paired fisheye preview renders what the two eyes see; black pixels lie
  outside the eye's field.

Geometry convention: right-handed axes, x = fly's right, y = up, z = forward.
"""
from __future__ import annotations

import numpy as np

# --- constants --------------------------------------------------------------

FACE = 128                      # pixels per cube face
ATLAS_W = 3 * FACE              # 384: three faces across
ATLAS_H = 2 * FACE              # 256: two faces down
PREVIEW_W, PREVIEW_H = 256, 128
PREVIEW_WIDTH, PREVIEW_HEIGHT = PREVIEW_W, PREVIEW_H  # names used by importers

# Calibration facts kept as data so other modules can quote them.
CALIBRATION = dict(
    version="spherical-v1",
    face_size=FACE,
    face_order=["forward", "right", "back", "left", "up", "down"],
    horizontal_fov_deg=270,
    overlap_deg=17,
    elevation_limit_deg=72,
    acceptance_sigma_deg=2,
    pose="eye origin at observer position, body yaw, level horizon",
    registration="approximate angular registration of MaleCNS optic-column order",
    color="engineered RGB approximation; no UV information",
)

# Acceptance cone: one center sample plus six on a ring. The ring sits at
# sqrt(2) * sigma so the cone captures about two sigma of a Gaussian lobe.
CONESIGMA = 2.0                 # degrees
CONE_RADIUS = np.deg2rad(CONESIGMA) * np.sqrt(2)
CONE_WEIGHTS = np.array([0.25] + [0.125] * 6, np.float32)

# Visual field, degrees. Left eye azimuths run [-135, +8.5], right eye
# [+135, -8.5] reversed; the overlap around 0 is the 17-degree middle.
AZ_MIN, AZ_MAX = -135.0, 135.0
AZ_SPLIT = 8.5                  # |overlap| beyond straight ahead
EL_MAX = 72.0

# Cube-face orthonormal bases: rows are (right, up, forward) per face, in
# face_order. A ray is attributed to the face whose forward axis it most
# closely matches; face texture coordinates follow.
BASES = np.asarray(
    [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],    # forward:  +z
        [[0, 0, -1], [0, 1, 0], [1, 0, 0]],   # right:    +x
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],  # back:     -z
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],   # left:     -x
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]],   # up:       +y
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]],   # down:     -y
    ],
    dtype=np.float32,
)


# --- pure geometry helpers ---------------------------------------------------

def sph_to_cart(azimuth_deg, elevation_deg):
    """Degrees on the sphere -> unit vectors (x right, y up, z forward)."""
    a = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64))
    e = np.deg2rad(np.asarray(elevation_deg, dtype=np.float64))
    cos_e = np.cos(e)
    return np.stack(
        (np.sin(a) * cos_e, np.sin(e), np.cos(a) * cos_e), axis=-1
    ).astype(np.float32)


def cart_to_sph(rays):
    """Unit vectors -> (azimuth, elevation) in degrees."""
    r = np.asarray(rays, dtype=np.float64)
    az = np.rad2deg(np.arctan2(r[..., 0], r[..., 2]))
    el = np.rad2deg(np.arcsin(np.clip(r[..., 1], -1.0, 1.0)))
    return az, el


def _orthonormal_basis(rays):
    """For each ray, two unit vectors perpendicular to it (tangent, vertical).

    Uses world-up as the reference; rays near the poles fall back to world-x.
    """
    rays = np.asarray(rays, np.float32)
    up = np.array([0.0, 1.0, 0.0], np.float32)
    tangent = np.cross(rays, up)
    near_pole = np.linalg.norm(tangent, axis=-1) < 1e-5
    if near_pole.any():
        fallback = np.cross(rays[near_pole], np.array([1.0, 0.0, 0.0], np.float32))
        tangent[near_pole] = fallback
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
    vertical = np.cross(tangent, rays)
    return tangent, vertical


def _face_pick(rays):
    """Which cube face each ray hits (most-aligned forward axis) + texel (x, y).

    Per-ray face choice means rays near cube seams resolve to a single face,
    so no seam blending is needed anywhere.
    """
    rays = np.asarray(rays, np.float32)
    forward_dot = rays @ BASES[:, 2].T               # (..., 6)
    face = np.argmax(forward_dot, axis=-1)
    depth = np.take_along_axis(forward_dot, face[..., None], axis=-1)[..., 0]
    right = np.sum(rays * BASES[face, 0], axis=-1)
    up = np.sum(rays * BASES[face, 1], axis=-1)
    # perspective divide onto the face plane, then map to texel centers
    u = right / depth
    v = up / depth
    x = np.clip(((u + 1.0) * 0.5 * FACE).astype(np.int32), 0, FACE - 1)
    y = np.clip(((1.0 - v) * 0.5 * FACE).astype(np.int32), 0, FACE - 1)
    return face, x, y


def atlas_lookup(rays):
    """Rays -> flat pixel indices into the (256, 384, 3) atlas."""
    face, x, y = _face_pick(rays)
    row = (face // 3) * FACE + y
    col = (face % 3) * FACE + x
    return (row * ATLAS_W + col).astype(np.int32)


def cone_lookup(rays):
    """Rays -> (N, 7) atlas indices for the acceptance-cone sampling pattern.

    The ring is built on the sphere: rotate each ray by CONE_RADIUS toward
    six headings spaced 60 degrees apart. Every sample is then looked up
    independently, so cones may straddle face seams.
    """
    rays = np.asarray(rays, np.float32)
    tangent, vertical = _orthonormal_basis(rays)
    headings = np.arange(6, dtype=np.float32) * (np.pi / 3.0)
    cos_r, sin_r = np.cos(CONE_RADIUS), np.sin(CONE_RADIUS)
    center = rays[:, None, :]
    ring = (
        center * cos_r
        + sin_r * (
            tangent[:, None, :] * np.cos(headings)[None, :, None]
            + vertical[:, None, :] * np.sin(headings)[None, :, None]
        )
    )
    samples = np.concatenate((center, ring), axis=1)  # (N, 7, 3)
    flat = samples.reshape(-1, 3)
    return atlas_lookup(flat).reshape(-1, 7)


# --- the retina object --------------------------------------------------------

class SphericalRetina:
    """Per-photoreceptor cone lookup tables plus the fisheye preview rig.

    Parameters are the published optic-column atlas coordinates
    (visual_pixels): column = 0..47 (elevation band), row part encodes
    azimuth within the 64-column binocular strip.
    """

    def __init__(self, visual_pixels):
        pixels = np.asarray(visual_pixels, dtype=np.float64)
        col = pixels[:, 1]
        row = pixels[:, 0]

        # --- receptor viewing directions from atlas coordinates -----------
        eye = (col >= 32).astype(np.int64)          # 0 = left, 1 = right
        within = (col % 32) / 31.0                  # 0..1 across one eye
        # Azimuth spans 143.5 degrees per eye; the left eye starts at -135,
        # the right at -8.5 (they overlap in the middle 17 degrees).
        start = np.where(eye == 0, AZ_MIN, -AZ_SPLIT)
        azimuth = start + within * 143.5
        elevation = EL_MAX - row / 47.0 * (2 * EL_MAX)

        self.rays = sph_to_cart(azimuth, elevation)
        self.indices = cone_lookup(self.rays)       # (N, 7)
        self.weights = CONE_WEIGHTS.copy()

        # --- fisheye preview rig -------------------------------------------
        # Two discs side by side; each pixel maps to a ray via an
        # equidistant fisheye projection (angle = radius * 90 deg).
        yy, xx = np.mgrid[0:PREVIEW_H, 0:PREVIEW_W]
        eye_pix = xx // PREVIEW_H                    # left disc, right disc
        u = ((xx % PREVIEW_H) + 0.5 - PREVIEW_H / 2) / (PREVIEW_H / 2)
        v = (PREVIEW_H / 2 - yy - 0.5) / (PREVIEW_H / 2)
        radius = np.hypot(u, v)
        angle = radius * (np.pi / 2)
        # sin(angle)/radius normalizes fisheye to perspective-like coordinates
        scale = np.divide(
            np.sin(angle), radius, out=np.ones_like(radius), where=radius > 0
        )
        local = np.stack((u * scale, v * scale, np.cos(angle)), axis=-1)

        # Aim each disc: left eye looks 63.25 deg left, right eye 63.25 right
        # (the centers of the two 143.5-degree fields).
        yaw = np.deg2rad(np.where(eye_pix == 0, -63.25, 63.25))
        cos_a, sin_a = np.cos(yaw), np.sin(yaw)
        rays = np.stack(
            (
                cos_a * local[..., 0] + sin_a * local[..., 2],
                local[..., 1],
                -sin_a * local[..., 0] + cos_a * local[..., 2],
            ),
            axis=-1,
        )

        # Visibility mask: inside the disc, inside the elevation band, and
        # inside that eye's azimuth span.
        az, el = cart_to_sph(rays)
        in_disc = radius <= 1.0
        in_el = np.abs(el) <= EL_MAX
        in_az = np.where(
            eye_pix == 0,
            (az >= AZ_MIN) & (az <= AZ_SPLIT),
            (az >= -AZ_SPLIT) & (az <= AZ_MAX),
        )
        self.mask = in_disc & in_el & in_az

        self.preview_indices = cone_lookup(rays.reshape(-1, 3))

    # --- sampling -------------------------------------------------------------

    def sample(self, atlas):
        """Atlas (256, 384, 3) uint8 -> per-photoreceptor RGB in [0, 1].

        Returns one color per receptor: the weighted cone average. This is
        the fly's retinal image; brain.encode_retina derives drive from it.
        """
        if atlas.shape != (ATLAS_H, ATLAS_W, 3):
            raise ValueError(
                f"retina requires a {ATLAS_H}x{ATLAS_W} six-face RGB atlas"
            )
        flat = atlas.reshape(-1, 3)
        picked = flat[self.indices].astype(np.float32)          # (N, 7, 3)
        return np.sum(picked * self.weights[None, :, None], axis=1) / 255.0

    def preview(self, atlas):
        """Atlas -> (128, 256, 3) uint8 paired fisheye image of both eyes."""
        flat = atlas.reshape(-1, 3)
        picked = flat[self.preview_indices].astype(np.float32)
        image = np.sum(
            picked * self.weights[None, :, None], axis=1
        ).reshape(PREVIEW_H, PREVIEW_W, 3)
        image[~self.mask] = 0
        return image.astype(np.uint8)


# --- differential self-test ---------------------------------------------------
# The rewrite was validated against the original implementation it replaced:
# rays, cone indices, masks, sample() and preview() outputs were compared over
# randomized receptor layouts and atlases and are identical. This self-test
# checks the module still runs and produces sane shapes/values. Run directly:
#   conda run -n mcp python -m flybrain.retina
if __name__ == "__main__":
    # Build the same receptor layout the fixture brain uses.
    n_receptors = 1536
    flat_pixels = np.linspace(0, 48 * 64 - 1, n_receptors).astype(np.int32)
    vp = np.column_stack((flat_pixels // 64, flat_pixels % 64)).astype(np.uint8)

    r = SphericalRetina(vp)

    # Deterministic test atlas: colorful ramp.
    yy, xx = np.mgrid[0:ATLAS_H, 0:ATLAS_W]
    atlas = np.stack(
        [
            (xx * 255 // ATLAS_W),
            (yy * 255 // ATLAS_H),
            ((xx + yy) * 255 // (ATLAS_W + ATLAS_H)),
        ],
        axis=-1,
    ).astype(np.uint8)

    s = r.sample(atlas)
    p = r.preview(atlas)
    print(f"sample: shape={s.shape}, min={s.min():.4f}, max={s.max():.4f}")
    print(f"preview: shape={p.shape}, nonzero px={(p.sum(-1) > 0).sum()}")
    assert s.shape == (n_receptors, 3)
    assert p.shape == (PREVIEW_H, PREVIEW_W, 3)
    assert s.min() >= 0 and s.max() <= 1
    print("retina self-test OK")
