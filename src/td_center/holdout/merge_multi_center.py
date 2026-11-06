r"""Reproduce the teacher's multi-centre grouping on a generated holdout set.

The course dataset is built **one sample per 6-hourly timestamp**, not one per
storm: when two or more tropical depressions are inside the 70x70 degree field at
the same analysis time, the teacher emits a *single* image whose `TD_center/yolo`
label carries one line per centre, and whose sample id is the concatenation of the
member ids (`NWP1608_DIANMU_..._NWP1609_MINDULLE_...`).  The project then freezes
only the samples with exactly one label line (`splits/single_center_v1/`).

This script applies the same rule to a generated set:

* groups ``metadata.csv`` by timestamp;
* a timestamp with >= 2 members becomes one merged sample (concatenated id, N
  label lines, one shared image -- the frames are identical because it is the same
  field at the same time) and is marked ``included = false``;
* a timestamp with exactly one member is left alone and marked ``included = true``;
* writes ``splits/single_center.txt``, ``splits/multi_center.txt`` and
  ``splits/single_center_manifest.csv`` next to the data.

    python merge_multi_center.py td2015
    python merge_multi_center.py td2026
"""
from __future__ import annotations

import argparse
import collections
import csv
import os

SPANS = ("6h", "12h", "18h")


def paths(root, sid):
    """(image, label) pairs of one sample id, in every modality the set ships."""
    out = []
    for span in SPANS:
        out.append((
            os.path.join(root, f"Multiple_frames_{span}", "images", sid + ".npy"),
            os.path.join(root, f"Multiple_frames_{span}", "labels", "TD_center", "yolo",
                         sid + ".txt"),
        ))
    out.append((
        os.path.join(root, "single_frame", "ir", "images", sid + ".png"),
        os.path.join(root, "single_frame", "ir", "labels", "TD_center", "yolo", sid + ".txt"),
    ))
    return out


def read_lines(path):
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]


def merge(root):
    meta_path = os.path.join(root, "metadata.csv")
    rows = list(csv.DictReader(open(meta_path, encoding="utf-8")))
    if rows and "num_centers" in rows[0]:
        raise SystemExit(f"{meta_path} already carries num_centers; refusing to merge twice")

    by_time = collections.OrderedDict()
    for r in rows:
        by_time.setdefault(r["time"], []).append(r)

    out_rows, n_merged = [], 0
    for time, group in by_time.items():
        group.sort(key=lambda r: r["tc_code"])

        if len(group) == 1:
            r = dict(group[0])
            r.update(num_centers=1, included="true", reason="single_center")
            out_rows.append(r)
            continue

        members = [r["sample_id"] for r in group]
        merged_id = "_".join(members)
        label_lines = []
        for r in group:
            label_lines.extend(read_lines(paths(root, r["sample_id"])[0][1]))

        donor = paths(root, members[0])
        target = paths(root, merged_id)
        for (src_img, _src_txt), (dst_img, dst_txt) in zip(donor, target):
            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
            os.makedirs(os.path.dirname(dst_txt), exist_ok=True)
            if os.path.exists(src_img):
                os.replace(src_img, dst_img)
            with open(dst_txt, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines) + "\n")

        for m in members:                      # remove every member's leftover files
            for img, txt in paths(root, m):
                for leftover in (img, txt):
                    if os.path.exists(leftover):
                        os.remove(leftover)

        out_rows.append(dict(
            group[0],
            sample_id=merged_id,
            tc_code="+".join(r["tc_code"] for r in group),
            storm_name="+".join(r["storm_name"] for r in group),
            num_centers=len(group), included="false", reason="multi_center",
        ))
        n_merged += 1
        print(f"  merged {time}: {len(group)} centres <- "
              + ", ".join(r["tc_code"] for r in group))

    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    single = [r["sample_id"] for r in out_rows if r["included"] == "true"]
    multi = [r["sample_id"] for r in out_rows if r["included"] != "true"]
    os.makedirs(os.path.join(root, "splits"), exist_ok=True)
    with open(os.path.join(root, "splits", "single_center.txt"), "w", encoding="utf-8") as f:
        f.write("".join(line + "\n" for line in single))
    with open(os.path.join(root, "splits", "multi_center.txt"), "w", encoding="utf-8") as f:
        f.write("".join(line + "\n" for line in multi))
    with open(os.path.join(root, "splits", "single_center_manifest.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "num_centers", "included", "reason"])
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r[k] for k in ("sample_id", "num_centers", "included", "reason")})

    print(f"{os.path.basename(root)}: {len(out_rows)} samples = {len(single)} single-centre "
          f"+ {len(multi)} multi-centre timestamps")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("data")
    args = p.parse_args(argv)
    return merge(os.path.abspath(args.data))


if __name__ == "__main__":
    raise SystemExit(main())
