"""Best-track archive access: positions of the *other* named storms in a frame's field.

The images are a fixed 70x70 degree box, **109-179 E, -15-55 N**. That was verified against the
sample ids and the labels: for every sample checked, x = (lon - 109)/70 and y = (55 - lat)/70
reproduce the label to 4 decimals, where lon/lat are the last two fields of the sample id. So a
best-track position maps straight to normalised pixel coordinates with no calibration.

The point of this module is the supervision that single-centre labels throw away: on 46% of the
805 frozen training frames another *named* storm sits inside the same box, and its position is
known 6-hourly from the archive that the labels themselves were derived from. `others_in_view`
returns those positions; `aux_heatmap` turns them into an 80x80 target for a second emission
channel.

Used by `train_d_aux.py` (multi-task auxiliary head) and by
`probe_besttrack_distractors.py`. Read-only.

    from td_center.reproduce.besttrack import load, others_in_view, aux_heatmap
"""
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("TD_DATA_ROOT",
                           os.environ.get("TD_DATA_ROOT", str(Path(__file__).resolve().parents[3] / "data"))))
BST = DATA / "RSMC_Best_Track_Data" / "bst_all.txt"

WEST, EAST = 109.0, 179.0
SOUTH, NORTH = -15.0, 55.0
SPAN = 70.0

REC = re.compile(r"^(\d{2})(\d{2})(\d{2})(\d{2})\s+\d{3}\s+(\d)\s+(\d{3})\s+(\d{4})")
HEAD = re.compile(r"^66666\s+(\d{4})\s")


def to_xy(lat, lon):
    """Normalised (x, y) in the 70x70 box; x grows east, y grows south (image convention)."""
    return (lon - WEST) / SPAN, (NORTH - lat) / SPAN


def in_box(lat, lon):
    return WEST <= lon <= EAST and SOUTH <= lat <= NORTH


@lru_cache(maxsize=1)
def load():
    """{timestamp 'YYYYMMDDHH': [(number, name, lat, lon, grade)]}."""
    by_time = {}
    text = BST.read_text(encoding="cp932", errors="replace")
    cur, name = None, ""
    for line in text.splitlines():
        h = HEAD.match(line)
        if h:
            cur = h.group(1)
            name = line[34:52].strip()
            continue
        m = REC.match(line)
        if m and cur:
            yy, mm, dd, hh = (int(g) for g in m.groups()[:4])
            year = 1900 + yy if yy > 50 else 2000 + yy
            ts = f"{year:04d}{mm:02d}{dd:02d}{hh:02d}"
            by_time.setdefault(ts, []).append(
                (cur, name, int(m.group(6)) / 10.0, int(m.group(7)) / 10.0, int(m.group(5))))
    return by_time


def split_id(sid):
    """NWP1601_NEPARTAK_3_2016070212_13_8.3_145.1 -> (ts, lat, lon)."""
    parts = sid.split("_")
    return parts[3], float(parts[-2]), float(parts[-1])


def others_in_view(sid, min_sep_deg=5.0, max_n=2):
    """Named storms other than the labelled one, inside the box, at least `min_sep_deg` away.

    `min_sep_deg` keeps the auxiliary target from overlapping the labelled centre; without it a
    second storm 2 deg away would make the two channels contradict each other.
    """
    ts, lat, lon = split_id(sid)
    out = []
    for num, name, la, lo, grade in load().get(ts, []):
        if abs(la - lat) < 1.5 and abs(lo - lon) < 1.5:
            continue
        if not in_box(la, lo):
            continue
        sep = float(np.hypot(la - lat, lo - lon))
        if sep < min_sep_deg:
            continue
        out.append((sep, name, la, lo))
    out.sort()
    return [(name, la, lo) for _sep, name, la, lo in out[:max_n]]


def aux_heatmap(others, size=80, sigma=2.0):
    """Sum of Gaussians at the other storms' positions, in the same 80x80 grid as the target."""
    import torch

    h = torch.zeros(size, size, dtype=torch.float32)
    if not others:
        return h
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    for _name, lat, lon in others:
        x, y = to_xy(lat, lon)
        cx, cy = x * size, y * size
        h = torch.maximum(h, torch.exp(-((xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2)
                                       / (2 * sigma ** 2)))
    return h


if __name__ == "__main__":
    # self-check: does the fixed box reproduce the labels, and how much aux supervision is there
    import sys

    repo = Path(__file__).resolve().parents[3]
    ids = [s for s in (repo / "splits/single_center_v1/train.txt")
           .read_text(encoding="utf-8").split() if s]
    worst, worst_sid = 0.0, ""
    n_aux = 0
    seps = []
    for sid in ids:
        x, y = map(float, (DATA / "single_frame/ir/train/labels/TD_center/yolo"
                           / f"{sid}.txt").read_text(encoding="utf-8").split())
        ts, lat, lon = split_id(sid)
        px, py = to_xy(lat, lon)
        err = max(abs(px - x), abs(py - y))
        if err > worst:
            worst, worst_sid = err, sid
        oth = others_in_view(sid)
        if oth:
            n_aux += 1
            seps.append(float(np.hypot(oth[0][1] - lat, oth[0][2] - lon)))
    print(f"frames checked: {len(ids)}")
    print(f"worst |box mapping - label| = {worst:.4f} ({worst_sid})")
    print(f"frames with an auxiliary centre: {n_aux} ({n_aux / len(ids) * 100:.1f}%)")
    if seps:
        print(f"nearest other storm: median {np.median(seps):.1f} deg, "
              f"min {min(seps):.1f}, max {max(seps):.1f}")
    sys.exit(0)
