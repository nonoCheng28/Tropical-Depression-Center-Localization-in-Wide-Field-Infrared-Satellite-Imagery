"""Build an independent holdout set for the TD-centre task.

Recreates the course dataset's standard from public raw data:

    Himawari-8/9 AHI band 13 (HSD, NOAA open AWS mirror)
    640 x 640 px, 109..179 E / 15 S..55 N, 0.109375 deg/px
    7-frame sequences at 1h / 2h / 3h spacing (6h / 12h / 18h spans), T first
    values scaled exactly like the course data (bright = cold cloud top)
    labels from the RSMC Tokyo best track (JMA), YOLO normalised centre

Sample selection is the course dataset's own rule, reverse engineered from
its 1076 file names (see docs/TEST_SET_2015.md): every 6-hourly best-track
row whose grade is 2 (tropical depression) and whose centre lies inside the
box, with the whole 18-hour sequence available.

Typical use (the archive is ~24 MB per hourly frame, so plan for the volume):

    python build_dataset.py --year 2015 --out outputs/holdout/td2015 --jobs 3
    python build_dataset.py --year 2015 --out outputs/holdout/td2015 --limit 5 --spans 6h
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import time

import numpy as np

import td_center.holdout.ahipipe as ap
SPANS = {"6h": 1, "12h": 2, "18h": 3}
BST_DEFAULT = os.path.join(os.environ.get("TD_DATA_ROOT", "data"), "RSMC_Best_Track_Data", "bst_all.txt")


# ---------------------------------------------------------------- best track
def parse_best_track(path):
    """[(storm_id, name, [rows])] with rows = dict(valid_time, grade, lat, lon)."""
    storms = []
    cur = None
    for line in open(path, encoding="utf-8", errors="replace").read().splitlines():
        if line.startswith("66666"):
            if cur:
                storms.append(cur)
            cur = {"sid": "NWP" + line[6:10], "name": line[35:55].strip(), "rows": []}
        elif cur is not None and line.strip():
            rest = line[8:]
            lat = int(rest[7:10]) / 10.0
            lon = int(rest[11:15]) / 10.0
            cur["rows"].append({
                "valid_time": "20" + line[:8],
                "grade": rest[5],
                "lat": lat,
                "lon": lon,
            })
    if cur:
        storms.append(cur)
    return storms


def in_box(lat, lon):
    return ap.BOX_LAT0 <= lat <= ap.BOX_LAT1 and ap.BOX_LON0 <= lon <= ap.BOX_LON1


def take_per_storm(rows, per_storm):
    """Keep at most ``per_storm`` 6-hourly rows of every storm (deterministic)."""
    if not per_storm:
        return rows
    seen = {}
    out = []
    for row in rows:
        n = seen.get(row["sid"], 0)
        if n < per_storm:
            seen[row["sid"]] = n + 1
            out.append(row)
    return out


def select_samples(storms, year, min_time=None):
    """Course rule: grade 2 (TD) rows inside the box, 6h apart."""
    wanted = []
    for storm in storms:
        if not storm["sid"].startswith(f"NWP{year % 100:02d}"):
            continue
        for row in storm["rows"]:
            if row["grade"] != "2" or not in_box(row["lat"], row["lon"]):
                continue
            if min_time and row["valid_time"] < min_time:
                continue
            wanted.append(dict(sid=storm["sid"], name=storm["name"], **row))
    wanted.sort(key=lambda r: (r["valid_time"], r["sid"]))
    return wanted


# ---------------------------------------------------------------- io helpers
def sample_id(row):
    t = dt.datetime.strptime(row["valid_time"], "%Y%m%d%H")
    return (
        f"{row['sid']}_{row['name']}_{int(row['grade'])}_"
        f"{t.strftime('%Y%m%d%H')}_13_{row['lat']:.1f}_{row['lon']:.1f}"
    )


def label_xy(lat, lon):
    """Course convention: lon = 109 + 70x, lat = 55 - 70y."""
    return (lon - ap.BOX_LON0) / ap.SPAN, (ap.BOX_LAT1 - lat) / ap.SPAN


def write_label(path, x, y):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{x:.6f} {y:.6f}\n")


def write_png(path, frame01):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    u8 = np.clip(np.floor(frame01 * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(path)


# ---------------------------------------------------------------- main build
def _stamp(row, hours_back=0):
    t = dt.datetime.strptime(row["valid_time"], "%Y%m%d%H")
    return (t - dt.timedelta(hours=hours_back)).strftime("%Y%m%d%H")


def write_metadata(out, manifest):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "metadata.csv")
    with open(path + ".part", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]) if manifest else ["sample_id"])
        w.writeheader()
        w.writerows(manifest)
    os.replace(path + ".part", path)


def build(year, out, cache, spans=("6h", "12h", "18h"), limit=None, jobs=3,
          bst_path=BST_DEFAULT, min_time=None, per_storm=None, manifest_path=None):
    os.makedirs(cache, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    storms = parse_best_track(bst_path)
    rows = take_per_storm(select_samples(storms, year, min_time=min_time), per_storm)
    if manifest_path:
        wanted = set(open(manifest_path, encoding="utf-8").read().split())
        rows = [r for r in rows if sample_id(r) in wanted]
        missing = wanted - {sample_id(r) for r in rows}
        if missing: raise ValueError(f"Best track does not match frozen holdout IDs: {sorted(missing)}")
    if limit:
        rows = rows[:limit]
    if not rows:
        print(f"no candidate samples for {year}", flush=True)
        return 1

    horizon = max(SPANS[s] for s in spans) * 6      # hours of hourly data needed
    groups = {}
    for row in rows:                                # keep sample order inside a storm
        groups.setdefault(row["sid"], []).append(row)
    total_frames = len({_stamp(r, k) for r in rows for k in range(horizon + 1)})
    print(f"{len(rows)} samples / {len(groups)} storms / {total_frames} hourly frames "
          f"(<= {total_frames * 24.3e6 / 1e9:.1f} GB)", flush=True)

    from concurrent.futures import ThreadPoolExecutor

    # resume: keep whatever an earlier run already wrote
    manifest = []
    meta_path = os.path.join(out, "metadata.csv")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                manifest.append(dict(r))
        print(f"resuming with {len(manifest)} samples already in metadata.csv", flush=True)
    done_ids = {m["sample_id"] for m in manifest}

    skipped, missing = [], []
    t_start = time.time()
    done_frames = 0

    for sid, grow in groups.items():
        need = sorted({_stamp(r, k) for r in grow for k in range(horizon + 1)})

        def grab(stamp):
            try:
                ap.tb_grid(dt.datetime.strptime(stamp, "%Y%m%d%H"), cache)
                return stamp, None
            except Exception as exc:
                return stamp, repr(exc)[:160]

        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            got = dict(pool.map(grab, need))
        done_frames += len(need)
        have = {k for k, v in got.items() if v is None}
        for k, v in got.items():
            if v is not None:
                missing.append((k, v))
        elapsed = (time.time() - t_start) / 60.0
        print(f"[{sid}] frames {len(have)}/{len(need)}  "
              f"overall {done_frames}/{total_frames} frames  {elapsed:.1f} min", flush=True)

        built = 0
        for row in grow:
            sample = sample_id(row)
            t0 = dt.datetime.strptime(row["valid_time"], "%Y%m%d%H")
            per_span = {}
            for span in spans:
                step = SPANS[span]
                stamps = [(t0 - dt.timedelta(hours=step * k)).strftime("%Y%m%d%H")
                          for k in range(7)]
                if all(s in have for s in stamps):
                    per_span[span] = stamps
                else:
                    break
            if len(per_span) != len(spans):
                skipped.append(sample)
                continue
            if sample in done_ids:
                continue

            x, y = label_xy(row["lat"], row["lon"])
            for span, stamps in per_span.items():
                frames = np.stack([ap.sample_field(
                    dt.datetime.strptime(s, "%Y%m%d%H"), cache) for s in stamps])
                # course disk order: [T, T-d, ..., T-6d] with T = stamps[0]
                frames = np.nan_to_num(frames.astype(np.float32), nan=0.0)
                d = os.path.join(out, f"Multiple_frames_{span}")
                os.makedirs(os.path.join(d, "images"), exist_ok=True)
                np.save(os.path.join(d, "images", sample + ".npy"), frames)
                write_label(os.path.join(d, "labels", "TD_center", "yolo",
                                         sample + ".txt"), x, y)
            write_png(os.path.join(out, "single_frame", "ir", "images", sample + ".png"),
                      np.nan_to_num(ap.sample_field(t0, cache), nan=0.0))
            write_label(os.path.join(out, "single_frame", "ir", "labels", "TD_center",
                                     "yolo", sample + ".txt"), x, y)
            manifest.append(dict(sample_id=sample, tc_code=row["sid"],
                                 storm_name=row["name"], time=row["valid_time"],
                                 grade=int(row["grade"]), lat=row["lat"], lon=row["lon"],
                                 x=x, y=y, spans="|".join(sorted(per_span))))
            built += 1
            print(f"  built {sample}", flush=True)
        write_metadata(out, manifest)
        print(f"[{sid}] wrote {built}/{len(grow)} samples", flush=True)

    with open(os.path.join(out, "build_log.json"), "w", encoding="utf-8") as f:
        json.dump(dict(year=year, spans=list(spans), samples=len(manifest),
                       skipped=skipped, missing_frames=missing,
                       params=dict(TB_ZERO=ap.TB_ZERO, TB_SPAN=ap.TB_SPAN,
                                   BLUR_SIGMA=ap.BLUR_SIGMA, LINE_SHIFT=ap.LINE_SHIFT,
                                   COL_SHIFT=ap.COL_SHIFT)), f, indent=2)
    print(f"done: {len(manifest)} samples, {len(skipped)} skipped, "
          f"{len(missing)} frames unavailable", flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", help="Frozen independent-year sample list; avoids changing selection with live archives")
    p.add_argument("--year", type=int, default=2015)
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default=None)
    p.add_argument("--spans", default="6h,12h,18h")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min-time", default=None, help="YYYYMMDDHH lower bound")
    p.add_argument("--bst", default=BST_DEFAULT)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--per-storm", type=int, default=None,
                   help="keep at most N 6-hourly rows per storm (cheap subset)")
    p.add_argument("--dry-run", action="store_true", help="only list the plan")
    args = p.parse_args(argv)
    cache = args.cache or os.path.join(args.out, "_cache")
    spans = tuple(s.strip() for s in args.spans.split(",") if s.strip())
    if args.dry_run:
        storms = parse_best_track(args.bst)
        rows = take_per_storm(select_samples(storms, args.year, min_time=args.min_time),
                              args.per_storm)
        if args.limit:
            rows = rows[:args.limit]
        horizon = max(SPANS[s] for s in spans) * 6
        need = {(dt.datetime.strptime(r["valid_time"], "%Y%m%d%H")
                 - dt.timedelta(hours=k)).strftime("%Y%m%d%H")
                for r in rows for k in range(horizon + 1)}
        print(f"year {args.year}: {len(rows)} samples, {len(need)} hourly frames, "
              f"~{len(need)*24.3e6/1e9:.1f} GB, horizon {horizon}h")
        for r in rows:
            print("  ", sample_id(r), r["lat"], r["lon"])
        return 0
    return build(args.year, args.out, cache, spans=spans,
                 limit=args.limit, jobs=args.jobs, bst_path=args.bst,
                 min_time=args.min_time, per_storm=args.per_storm, manifest_path=args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
