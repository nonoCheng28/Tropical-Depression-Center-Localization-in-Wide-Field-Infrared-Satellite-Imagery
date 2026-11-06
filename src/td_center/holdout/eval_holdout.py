r"""Evaluate the delivered aided pipeline on an independent holdout set.

Reuses `deliver_aided.run` unchanged -- the same heatmaps, the same Gaussian mask
at other named systems, the same 8 peaks / 3 deg, the same Viterbi kappa=0.003 --
so the numbers are directly comparable to the published LOYO / val / test rows.

    python eval_holdout.py --data ..\holdout_env\td2026 ^
        --artifacts ..\artifacts_2026 --label td2026 ^
        --out ..\results\td2026_result.json

What it adds over `deliver_aided.py`:
  * per-frame errors are kept, so the aided-vs-ablation difference gets a
    storm-level paired bootstrap (the project's house style: the effective n is
    storms, not frames);
  * it writes one JSON per holdout set instead of overwriting artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WT = REPO
DEG = 70.0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="mirror root (TD_DATA_ROOT)")
    p.add_argument("--artifacts", required=True, help="artifact dir holding ens3_val/")
    p.add_argument("--label", required=True, help="name written into the result json")
    p.add_argument("--out", required=True)
    p.add_argument("--ids", default=None, help="manifest; defaults to <data>/splits/val.txt")
    p.add_argument("--boot", type=int, default=20000)
    p.add_argument("--placebo-seeds", type=int, nargs="+", default=list(range(10)))
    args = p.parse_args(argv)

    os.environ["TD_DATA_ROOT"] = os.path.abspath(args.data)
    os.environ["TD_ARTIFACT_DIR"] = os.path.abspath(args.artifacts)


    import torch
    from td_center.reproduce.deliver_aided import run

    ids_path = args.ids or os.path.join(args.data, "splits", "val.txt")
    ids = [l.strip() for l in open(ids_path, encoding="utf-8") if l.strip()]
    heat = torch.load(os.path.join(args.artifacts, "ens3_val", "heatmaps_val_fp32.pt"),
                      map_location="cpu", weights_only=False)
    label_dir = os.path.join(args.data, "single_frame", "ir", "val", "labels",
                             "TD_center", "yolo")

    base = run(heat, ids, __import__("pathlib").Path(label_dir), 0.0)
    aided = run(heat, ids, __import__("pathlib").Path(label_dir), 1.5, sigma=3.0)
    # A single placebo draw is very noisy: the random blobs only matter when they
    # happen to land near the peaks, so one draw can look like a no-op and another
    # can destroy a good trajectory.  Average a fixed set of seeds and report the
    # spread -- that is the honest control.
    placebo_runs = [run(heat, ids, __import__("pathlib").Path(label_dir), 1.5,
                        "placebo", seed, sigma=3.0) for seed in args.placebo_seeds]
    placebo = placebo_runs[0]

    rng = np.random.default_rng(0)
    storms = sorted({s.split("_")[0] for s in ids})
    by_storm = {s: [i for i, sid in enumerate(ids) if sid.split("_")[0] == s] for s in storms}
    deltas, md_a, md_b = [], [], []
    for _ in range(args.boot):
        pick = rng.choice(storms, size=len(storms), replace=True)
        idx = np.concatenate([by_storm[s] for s in pick])
        d = base["err"][idx].mean() - aided["err"][idx].mean()
        deltas.append(d)
        md_b.append(base["err"][idx].mean())
        md_a.append(aided["err"][idx].mean())
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])

    def row(r):
        return {k: r[k] for k in ("MD", "DR3", "RMSE", "worst", "MAE_comp", "RMSE_comp",
                                  "storms", "storms_full", "storms_over10", "n")}

    out = {
        "holdout": args.label,
        "frames": len(ids),
        "storms": len(storms),
        "kappa": 0.003,
        "rows": {"ablation_alpha0": row(base), "aided_alpha1.5_sigma3": row(aided),
                 "placebo_alpha1.5_sigma3_mean": {
                     k: (float(np.mean([r[k] for r in placebo_runs]))
                         if isinstance(placebo_runs[0][k], (int, float)) else None)
                     for k in ("MD", "DR3", "RMSE", "worst", "MAE_comp", "RMSE_comp")}},
        "placebo_seeds": args.placebo_seeds,
        "placebo_MD_per_seed": [r["MD"] for r in placebo_runs],
        "placebo_DR3_per_seed": [r["DR3"] for r in placebo_runs],
        "difference_md": float(base["MD"] - aided["MD"]),
        "bootstrap_storm": {
            "iterations": args.boot,
            "mean_delta": float(deltas.mean()),
            "ci95": [float(lo), float(hi)],
            "p_delta_le_0": float((deltas <= 0).mean()),
            "mean_ablation": float(np.mean(md_b)),
            "mean_aided": float(np.mean(md_a)),
        },
        "storm_md": {"ablation": base["storm_md"], "aided": aided["storm_md"],
                     "placebo": placebo["storm_md"]},
        "per_frame": {"ids": ids, "ablation": base["err"].tolist(),
                      "aided": aided["err"].tolist(), "placebo": placebo["err"].tolist()},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"{args.label}: {len(ids)} frames / {len(storms)} storms")
    print(f"  {'row':<18} {'MD':>7} {'DR@3':>6} {'RMSE':>6} {'worst':>6} {'s<=3':>6} {'s>10':>6}")
    for name, r in out["rows"].items():
        if r.get("MD") is None:
            continue
        full = f"{r['storms_full']:>3d}/{r['storms']:<2d}" if "storms_full" in r else "   -"
        over = f"{r['storms_over10']:>3d}/{r['storms']:<2d}" if "storms_over10" in r else "   -"
        print(f"  {name:<18} {r['MD']:>7.3f} {r['DR3']:>6.3f} {r['RMSE']:>6.2f} "
              f"{r['worst']:>6.1f} {full} {over}")
    print(f"  placebo per seed MD: {[round(x,2) for x in out['placebo_MD_per_seed']]}")
    print(f"  ablation - aided = {out['difference_md']:+.3f} deg (positive = aided better)  "
          f"storm bootstrap {deltas.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  "
          f"P(>=0)={1 - (deltas <= 0).mean():.3f}")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
