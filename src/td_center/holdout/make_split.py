"""Present a generated holdout set as a ``val`` split for the existing repo code.

The repo resolves ``splits/single_center_v1/<split>.txt`` relative to the checkout
and ``DATA_ROOT/Multiple_frames_<span>/<split>/`` relative to ``TD_DATA_ROOT``.
This script builds the second half of that contract -- a directory whose ``val``
split *is* the holdout -- using junctions, so no data is copied:

    python make_split.py --data td2015 --out holdout_env/td2015

It prints the environment variables and the exact commands to run afterwards
(with an explicit independent manifest; the frozen course lists remain untouched).

Nothing here touches the course dataset or the frozen splits.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import os
import subprocess
import sys

SPANS = ("6h", "12h", "18h")
SPLIT = "val"


def junction(link: str, target: str):
    """Create an NTFS junction (no admin rights needed), or a symlink as fallback."""
    os.makedirs(os.path.dirname(link), exist_ok=True)
    if os.path.exists(link):
        return
    if sys.platform == "win32":
        subprocess.run(["cmd", "/c", "mklink", "/J", link, target],
                       check=True, capture_output=True)
    else:
        os.symlink(target, link, target_is_directory=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="directory produced by build_dataset.py")
    p.add_argument("--out", required=True, help="environment directory to create")
    p.add_argument("--split", default=SPLIT)
    p.add_argument("--besttrack", default=os.path.join(os.environ.get("TD_DATA_ROOT", "data"), "RSMC_Best_Track_Data"),
                   help="best-track directory the aided mask reads (Tb/others_in_view)")
    args = p.parse_args(argv)

    data = os.path.abspath(args.data)
    out = os.path.abspath(args.out)
    ids = []
    # eval.txt is the final manifest (single-centre AND first TD stint); fall back
    # to the single-centre list for sets that were built before that rule existed
    single = os.path.join(data, "splits", "eval.txt")
    if not os.path.exists(single):
        single = os.path.join(data, "splits", "single_center.txt")
    if os.path.exists(single):
        ids = [l.strip() for l in open(single, encoding="utf-8") if l.strip()]
        total = sum(1 for _ in csv.DictReader(open(os.path.join(data, "metadata.csv"),
                                                 encoding="utf-8")))
        print(f"{os.path.basename(single)}: {len(ids)} of {total} samples "
              f"({total - len(ids)} excluded: multi-centre and/or later TD stint)")
    else:
        with open(os.path.join(data, "metadata.csv"), encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ids.append(row["sample_id"])
    if not ids:
        raise SystemExit(f"no samples in {data}/metadata.csv")

    for span in SPANS:
        src = os.path.join(data, f"Multiple_frames_{span}")
        if not os.path.isdir(src):
            continue
        for sub in ("images", "labels"):
            s, d = os.path.join(src, sub), os.path.join(out, f"Multiple_frames_{span}",
                                                        args.split, sub)
            if os.path.isdir(s):
                junction(d, s)

    src = os.path.join(data, "single_frame", "ir")
    for sub in ("images", "labels"):
        s, d = os.path.join(src, sub), os.path.join(out, "single_frame", "ir",
                                                    args.split, sub)
        if os.path.isdir(s):
            junction(d, s)

    # the aided mask reads the RSMC best track from the dataset root; copy the
    # single text file rather than linking (junctions onto that folder are denied)
    bst = os.path.join(args.besttrack, "bst_all.txt")
    if os.path.isfile(bst):
        dst = os.path.join(out, "RSMC_Best_Track_Data")
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(bst, os.path.join(dst, "bst_all.txt"))

    manifest = os.path.join(out, "splits", args.split + ".txt")
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    with open(manifest, "w", encoding="utf-8") as f:
        f.write("\n".join(ids) + "\n")

    print(f"{len(ids)} samples -> {out}\\{args.split}")
    print(f"manifest: {manifest}")
    print()
    print("Use python -m td_center.reproduce predict_manifest with this mirror and the explicit independent manifest.")
    print("Then use python -m td_center.holdout eval_holdout; see docs/REPRODUCTION.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
