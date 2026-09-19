"""Download + prepare MaleCNS v1.0 into a compact simulation cache.

Reuses the exact selection/normalization rules from ornata/fly (fly64/data.py):
- all 166,700 neurons with a non-empty superclass annotation,
- weight = synapse count * presynaptic sign / incoming absolute-weight sum,
- GABA/glutamate/histamine inhibitory; other/unknown excitatory (approximation).
Run:  python -m flybrain.data --prepare --cache data/cache
"""
from __future__ import annotations

import argparse
import subprocess
import zipfile
import xml.etree.ElementTree as ET
import re
import json
import hashlib
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
from scipy import sparse

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
FILES = {
    "optic-columns.xlsx": "https://raw.githubusercontent.com/flyconnectome/2025malecns/67767d2233657983993ff6c2be48e836a935863c/supplemental_data/optic-column-type-assignments-v1.0.xlsx",
    "annotations.feather": f"{BASE}/body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "transmitters.feather": f"{BASE}/body-neurotransmitters-male-cns-v1.0.feather",
    "weights.feather": f"{BASE}/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}


def optic_columns(path: Path) -> dict[int, tuple[str, int, int]]:
    """Read the optic-column assignment sheets (xlsx) without Excel."""
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    result = {}
    with zipfile.ZipFile(path) as archive:
        strings = ["".join(e.itertext()) for e in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        for sheet in (1, 2):
            root = ET.fromstring(archive.read(f"xl/worksheets/sheet{sheet}.xml"))
            for row in root.findall("s:sheetData/s:row", ns)[1:]:
                cells = {}
                for c in row:
                    v = c.find("s:v", ns)
                    if v is not None:
                        cells[re.sub(r"\d", "", c.attrib["r"])] = strings[int(v.text)] if c.get("t") == "s" else v.text
                match = re.fullmatch(r"ME_([LR])_col_(\d+)_(\d+)", cells.get("A", ""))
                if match:
                    side, h1, h2 = match.groups()
                    for col in ("B", "C", "E"):
                        try:
                            body = int(cells.get(col, -99))
                            if body > 0:
                                result[body] = (side, int(h1), int(h2))
                        except ValueError:
                            pass
    return result


def download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url} -> {destination}")
    subprocess.run(["curl", "--fail", "--location", "--continue-at", "-", "--output", str(partial), url], check=True)
    partial.replace(destination)


def _column(table, choices):
    columns = table.column_names if hasattr(table, "column_names") else table.columns
    for name in choices:
        if name in columns:
            return name
    raise ValueError(f"none of {choices} found; columns={list(columns)}")


def prepare(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    raw = cache / "raw"
    for name, url in FILES.items():
        download(url, raw / name)
    ann = feather.read_table(raw / "annotations.feather").to_pandas()
    nt = feather.read_table(raw / "transmitters.feather", columns=["body", "consensus_nt"]).to_pandas()

    body_col = _column(ann, ("body", "bodyId", "bodyid"))
    ann = ann.loc[ann["superclass"].notna() & ann["superclass"].ne("")].copy()
    ids = ann[body_col].drop_duplicates().to_numpy(np.int64, copy=True)
    ids.sort()
    ann = ann.drop_duplicates(body_col).set_index(body_col).reindex(ids)

    nt = nt.drop_duplicates("body").set_index("body").reindex(ids)
    nt_labels = nt["consensus_nt"].fillna("unclear").astype(str).str.lower()
    neuron_sign = np.where(nt_labels.str.contains("gaba|glutamate|histamine", regex=True), -1.0, 1.0).astype(np.float32)

    edge_table = feather.read_table(raw / "weights.feather", columns=["body_pre", "body_post", "weight"], memory_map=True)
    pre_parts, post_parts, weight_parts = [], [], []
    total = edge_table.num_rows
    done = 0
    for batch in edge_table.to_batches(max_chunksize=2_000_000):
        pre_id = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        post_id = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        values = batch.column(2).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        pre = np.searchsorted(ids, pre_id)
        post = np.searchsorted(ids, post_id)
        valid = (pre < len(ids)) & (post < len(ids))
        valid &= ids[np.minimum(pre, len(ids) - 1)] == pre_id
        valid &= ids[np.minimum(post, len(ids) - 1)] == post_id
        if valid.any():
            pre_i = pre[valid].astype(np.int32)
            post_i = post[valid].astype(np.int32)
            w = values[valid] * neuron_sign[pre_i]
            pre_parts.append(pre_i)
            post_parts.append(post_i)
            weight_parts.append(w)
        done += batch.num_rows
        print(f"\rmapped {done:,}/{total:,} edge rows", end="", flush=True)
    print()
    del edge_table
    pre = np.concatenate(pre_parts)
    post = np.concatenate(post_parts)
    weights = np.concatenate(weight_parts)
    del pre_parts, post_parts, weight_parts
    incoming = np.bincount(post, weights=np.abs(weights), minlength=len(ids)).astype(np.float32)
    weights /= np.maximum(incoming[post], 1.0)
    matrix = sparse.csr_matrix((weights, (post, pre)), shape=(len(ids), len(ids)), dtype=np.float32)
    sparse.save_npz(cache / "weights.npz", matrix, compressed=False)

    cell_type = ann["flywireType"].fillna(ann["type"]).fillna("").astype(str)
    visual = np.flatnonzero(cell_type.isin(["R1-6", "R7", "R8"]).to_numpy()).astype(np.int32)
    forward = np.flatnonzero(cell_type.eq("DNg100").to_numpy()).astype(np.int32)
    turn = np.flatnonzero(cell_type.isin(["DNa02", "DNg13"]).to_numpy()).astype(np.int32)
    jump = np.flatnonzero(cell_type.isin(["DNp01", "DNp10"]).to_numpy()).astype(np.int32)
    instances = ann["instance"].fillna("").astype(str).str.upper().to_numpy()
    soma_side = ann["somaSide"].fillna(ann["rootSide"]).fillna("").astype(str).str.upper().to_numpy()
    turn_left = np.array([i for i in turn if soma_side[i] == "L" or "_L" in instances[i]], np.int32)
    turn_right = np.array([i for i in turn if soma_side[i] == "R" or "_R" in instances[i]], np.int32)
    if min(len(visual), len(forward), len(turn_left), len(turn_right), len(jump)) == 0:
        raise RuntimeError("required sensory/motor annotations were not resolved")

    rng = np.random.default_rng(64)
    phi = rng.uniform(0, 2 * np.pi, len(ids))
    cost = rng.uniform(-1, 1, len(ids))
    rad = np.sqrt(1 - cost * cost)
    coords = np.column_stack((rad * np.cos(phi), cost * 0.68, rad * np.sin(phi))).astype(np.float32)
    measured = np.zeros(len(ids), dtype=bool)
    soma_loc = ann["somaLocation"].to_numpy()
    to_soma_loc = ann["tosomaLocation"].to_numpy()
    for i, (soma, to_soma) in enumerate(zip(soma_loc, to_soma_loc)):
        location = soma if isinstance(soma, (list, np.ndarray)) and len(soma) == 3 else to_soma
        if isinstance(location, (list, np.ndarray)) and len(location) == 3:
            coords[i] = location
            measured[i] = True
    center = np.median(coords[measured], axis=0)
    scale = np.maximum(np.percentile(np.abs(coords[measured] - center), 99, axis=0), 1)
    coords[measured] = (coords[measured] - center) / scale
    region_text = ann["superclass"].fillna(ann["class"]).fillna("unassigned").astype(str)
    region_names = np.array(sorted(region_text.unique()))
    regions = np.searchsorted(region_names, region_text).astype(np.uint8)

    columns = optic_columns(raw / "optic-columns.xlsx")
    known = np.array([i for i, body in enumerate(ids) if body in columns], np.int32)
    incoming_visual = abs(matrix[known][:, visual]).tocsc()
    visual_pixels = np.zeros((len(visual), 2), np.uint8)
    visual_mapping = np.zeros(len(visual), np.uint8)
    for member, neuron in enumerate(visual):
        location = columns.get(int(ids[neuron]))
        if location:
            visual_mapping[member] = 1
        else:
            a, b = incoming_visual.indptr[member:member + 2]
            if b > a:
                index = incoming_visual.indices[a + np.argmax(incoming_visual.data[a:b])]
                location = columns[int(ids[known[index]])]
                visual_mapping[member] = 2
        if location:
            side, h1, h2 = location
            visual_pixels[member] = (np.clip(round((h2 - 1) * 47 / 38), 0, 47),
                                     (0 if side == "L" else 32) + np.clip(round((h1 - 1) * 31 / 35), 0, 31))
    for side, x0 in (("L", 0), ("R", 32)):
        members = np.flatnonzero(np.array([soma_side[i] == side for i in visual]))
        ordered = members[np.argsort(ids[visual[members]], kind="stable")]
        for rank, member in enumerate(ordered):
            if not visual_mapping[member]:
                visual_pixels[member] = (rank * 48 // max(len(ordered), 1), x0 + (rank % 32))
    np.savez_compressed(
        cache / "model.npz", n=np.int64(len(ids)), ids=ids, visual=visual,
        forward=forward, turn_left=turn_left, turn_right=turn_right,
        jump_nodes=jump, positions=coords, position_measured=measured,
        visual_pixels=visual_pixels, visual_mapping=visual_mapping,
        regions=regions, region_names=region_names.astype(str),
    )
    manifest = dict(schema=2, neurons=len(ids), edges=matrix.nnz,
                    selection="all non-empty superclass annotations, including tbc",
                    normalization="raw synapse counts / incoming absolute sum",
                    inhibitory="GABA, glutamate, histamine; other/unknown +1 approximation",
                    visual_mapping_counts=np.bincount(visual_mapping, minlength=3).tolist(),
                    measured_positions=int(measured.sum()), sources=FILES,
                    sha256={name: hashlib.file_digest(open(raw / name, "rb"), "sha256").hexdigest() for name in FILES})
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(ids):,} neurons and {matrix.nnz:,} weighted edges in {cache}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    args = parser.parse_args()
    if args.prepare:
        prepare(args.cache)


if __name__ == "__main__":
    main()
