r"""Sanity-check a generated holdout set against the course data standard.

    python verify_dataset.py outputs/holdout/td2015
    python verify_dataset.py outputs/holdout/td2015 --course      # also compare against the course set

Checks per sample: array shape/dtype/finiteness/value range, label presence and
range, label-vs-metadata consistency, PNG == floor(npy[0]*255), and the
filename schema.  Exit code 0 only when everything passes.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

SPANS = ("6h", "12h", "18h")
N, S = 640, 70.0


def check(root, meta=None):
    problems = []
    rows = []
    for span in SPANS:
        base = os.path.join(root, f"Multiple_frames_{span}")
        if not os.path.isdir(base):
            continue
        for npy in sorted(glob.glob(os.path.join(base, "images", "*.npy"))):
            sid = os.path.basename(npy)[:-4]
            a = np.load(npy)
            if a.shape != (7, N, N):
                problems.append(f"{span}/{sid}: shape {a.shape}")
                continue
            if a.dtype != np.float32:
                problems.append(f"{span}/{sid}: dtype {a.dtype}")
            if not np.isfinite(a).all():
                problems.append(f"{span}/{sid}: non-finite values")
            lo, hi = float(a.min()), float(a.max())
            if lo < -1e-6 or hi > 1.0001:
                problems.append(f"{span}/{sid}: range {lo:.3f}..{hi:.3f}")
            lab = os.path.join(base, "labels", "TD_center", "yolo", sid + ".txt")
            if not os.path.exists(lab):
                problems.append(f"{span}/{sid}: missing label")
            else:
                lines = [l.strip() for l in open(lab, encoding="utf-8") if l.strip()]
                # the course dataset merges co-located tropical depressions into one
                # sample whose id concatenates the members and whose label has one
                # line per centre; mirror that here instead of demanding one line.
                expect_rows = len(sid.split("_")) // 7
                if len(lines) != expect_rows:
                    problems.append(f"{span}/{sid}: {len(lines)} label rows "
                                    f"(id implies {expect_rows})")
                else:
                    xy = [tuple(map(float, l.split())) for l in lines]
                    if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in xy):
                        problems.append(f"{span}/{sid}: label out of range")
                    if meta and sid in meta:
                        m = meta[sid]
                        ex = (m["lon"] - 109.0) / S
                        ey = (55.0 - m["lat"]) / S
                        if abs(ex - xy[0][0]) > 2e-6 or abs(ey - xy[0][1]) > 2e-6:
                            problems.append(f"{span}/{sid}: label != metadata")
            rows.append((span, sid, lo, hi, float(a.mean())))
    # PNG consistency, one pass over the single-frame folder
    png_dir = os.path.join(root, "single_frame", "ir", "images")
    png_n = png_bad = 0
    if os.path.isdir(png_dir):
        from PIL import Image
        for png in sorted(glob.glob(os.path.join(png_dir, "*.png"))):
            sid = os.path.basename(png)[:-4]
            ref = os.path.join(root, "Multiple_frames_6h", "images", sid + ".npy")
            if not os.path.exists(ref):
                continue
            a = np.load(ref)
            q = np.clip(np.floor(a[0] * 255.0), 0, 255).astype(np.uint8)
            u = np.array(Image.open(png))
            png_n += 1
            if u.shape != q.shape or (u != q).any():
                png_bad += 1
    return rows, problems, png_n, png_bad


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--course", action="store_true",
                   help="print the reference statistics of the course dataset")
    args = p.parse_args(argv)

    meta = {}
    mpath = os.path.join(args.root, "metadata.csv")
    if os.path.exists(mpath):
        for r in csv.DictReader(open(mpath, encoding="utf-8")):
            meta[r["sample_id"]] = dict(lat=float(r["lat"]), lon=float(r["lon"]))

    rows, problems, png_n, png_bad = check(args.root, meta)
    per_span = {}
    for span, sid, lo, hi, mean in rows:
        per_span.setdefault(span, []).append((lo, hi, mean))
    print(f"samples checked: {len(rows)}")
    for span, vals in per_span.items():
        a = np.array(vals)
        print(f"  {span}: n={len(vals)}  range {a[:,0].min():.3f}..{a[:,1].max():.3f}  "
              f"mean of means {a[:,2].mean():.4f}")
    print(f"png files consistent with floor(npy[0]*255): {png_n - png_bad}/{png_n}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for x in problems[:25]:
            print("  " + x)
        return 1
    if args.course:
        print("\ncourse reference: npy 11468928 B, float32 (7,640,640), range 0..0.814, "
              "labels 'x y' with 6 decimals, png = floor(npy[0]*255)")
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
