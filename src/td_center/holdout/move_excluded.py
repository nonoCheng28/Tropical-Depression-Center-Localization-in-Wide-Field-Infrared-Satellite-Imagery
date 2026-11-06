r"""Move samples listed in a split manifest out of a generated set's main directories.

Two kinds of samples are excluded from the delivered sets:

* ``splits/multi_center.txt`` -- the same 6-hourly timestamp carries two or more
  tropical depressions, so the course dataset's single-centre protocol drops it
  (the merged image is kept under ``excluded_multi_center/``);
* ``splits/later_stint.txt`` -- only a storm's **first** TD stint is sampled by
  the course dataset (546 later-stint rows, 0 sampled, over 2016-2025), so later
  stints are dropped as well (``excluded_later_stint/``).

Both move files with their relative layout intact, so nothing is deleted and the
dataset's ``Multiple_frames_*/images`` ends up holding exactly the evaluation set.

    python move_excluded.py td2015
    python move_excluded.py td2015 --list splits/later_stint.txt --dest excluded_later_stint
    python move_excluded.py td2026 --list splits/later_stint.txt --dest excluded_later_stint
"""
from __future__ import annotations

import argparse
import os
import shutil

SPANS = ("6h", "12h", "18h")


def paths(root, sid):
    out = []
    for span in SPANS:
        out.append((os.path.join(root, f"Multiple_frames_{span}", "images", sid + ".npy"),
                    os.path.join(root, f"Multiple_frames_{span}", "labels", "TD_center",
                                 "yolo", sid + ".txt")))
    out.append((os.path.join(root, "single_frame", "ir", "images", sid + ".png"),
                os.path.join(root, "single_frame", "ir", "labels", "TD_center", "yolo",
                             sid + ".txt")))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("--list", default=os.path.join("splits", "multi_center.txt"),
                   dest="list_rel", help="manifest of ids to move (relative to the dataset)")
    p.add_argument("--dest", default="excluded_multi_center")
    args = p.parse_args(argv)
    root = os.path.abspath(args.data)

    list_path = os.path.join(root, args.list_rel)
    if not os.path.exists(list_path):
        raise SystemExit(f"{list_path} missing")
    ids = [l.strip() for l in open(list_path, encoding="utf-8") if l.strip()]
    if not ids:
        print("manifest is empty; nothing to move")
        return 0

    dest_root = os.path.join(root, args.dest)
    moved_samples = moved_files = 0
    for sid in ids:
        touched = 0
        for src, txt in paths(root, sid):
            for path in (src, txt):
                if not os.path.exists(path):
                    continue
                rel = os.path.relpath(path, root)
                dst = os.path.join(dest_root, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(path, dst)
                touched += 1
        if touched:
            moved_samples += 1
            moved_files += touched
            print(f"  excluded {sid}")
    print(f"moved {moved_files} files for {moved_samples} samples -> {dest_root}")
    n_main = len(os.listdir(os.path.join(root, "Multiple_frames_6h", "images")))
    print(f"Multiple_frames_6h/images now holds {n_main} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
