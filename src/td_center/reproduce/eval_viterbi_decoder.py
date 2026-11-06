"""MAP trajectory decoding: Viterbi over candidates with a motion prior.

Cost of a trajectory:  sum_t [ -support_t(i_t) + kappa * ||p_t - p_{t-1}||^2 / sigma^2 ]

with sigma fixed at 1 deg (the measured 6 h displacement of the true centre) and a
single free weight `kappa`. kappa = 0 is per-frame argmax; large kappa increasingly favours
minimum-displacement trajectories; the interesting range is in between.

`kappa` is chosen **leave-one-year-out**: for held-out year Y the weight is picked
on the other seven years only, so the reported per-fold numbers are out of fold
with respect to selection.

    python experiments/d_improve/eval_viterbi_decoder.py --support scores
"""
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import torch.nn.functional as F

from td_center.reproduce.common import ARTIFACT_DIR, REPO

YEARS = list(range(2016, 2024))
DEG = 70.0
SIGMA2 = 1.0
KAPPAS = (0.0, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 1e6)


def load_fold(year, tag, support):
    art = ARTIFACT_DIR / f"{tag}loyo{year}"
    ext = torch.load(art / "refiner_eval_ext_global.pt", map_location="cpu", weights_only=False)
    heat = torch.load(art / f"heatmaps_loyo{year}_fp32.pt", map_location="cpu",
                      weights_only=False)
    starts = ext["sample_offsets"].tolist()
    ids = ext["sample_ids"]
    storms = defaultdict(list)
    for i, st in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else len(ids)
        sid = ids[st]
        h = heat[sid].float()
        hn = (h - h.min()) / (h.max() - h.min() + 1e-6)
        xy = ext["xy"][st:stop].numpy()
        g = torch.tensor(xy, dtype=torch.float32)[None] * 2 - 1
        hv = F.grid_sample(hn[None, None], g[:, :, None, :], mode="bilinear",
                           align_corners=False)[0, 0, :, 0].numpy()
        storms[sid.split("_")[0]].append({"sid": sid, "time": sid.split("_")[3],
                                          "xy": xy, "dist": ext["dist"][st:stop].numpy(),
                                          "heat": hv})
    for seq in storms.values():
        seq.sort(key=lambda r: r["time"])
    if support == "scores":
        import glob
        dirs = sorted(glob.glob(str(REPO / "runs/d_improve" / f"loyo_eval_{year}_*" / "scores.pt")))
        if not dirs:
            raise SystemExit(f"no cached refiner scores for {year}")
        cached = torch.load(dirs[-1], map_location="cpu", weights_only=False)
        by_sid = {r["sample_id"]: r["scores"] for r in cached["rows"]}
        for seq in storms.values():
            for r in seq:
                s = np.asarray(by_sid[r["sid"]], dtype=np.float64)
                pad = len(r["xy"]) - len(s)
                if pad > 0:
                    s = np.concatenate([s, np.full(pad, s.min() - 1.0)])
                rng = s.max() - s.min()
                r["sup"] = (s - s.min()) / rng if rng > 1e-9 else np.zeros_like(s)
    else:
        for seq in storms.values():
            for r in seq:
                s = r["heat"]
                rng = s.max() - s.min()
                r["sup"] = (s - s.min()) / rng if rng > 1e-9 else np.zeros_like(s)
    return storms


def viterbi(seq, kappa):
    n = len(seq)
    pts = [r["xy"] * DEG for r in seq]
    sup = [r["sup"] for r in seq]
    cost = -sup[0].copy()
    back = []
    for t in range(1, n):
        d = np.linalg.norm(pts[t][:, None, :] - pts[t - 1][None, :, :], axis=-1) ** 2
        total = cost[None, :] + kappa * d / SIGMA2
        arg = np.argmin(total, axis=1)
        back.append(arg)
        cost = total[np.arange(len(arg)), arg] - sup[t]
    idx = [int(np.argmin(cost))]
    for t in range(n - 1, 0, -1):
        idx.append(int(back[t - 1][idx[-1]]))
    return idx[::-1]


def evaluate(data, years, kappa):
    vals = []
    for year in years:
        for seq in data[year].values():
            picks = viterbi(seq, kappa)
            vals.extend(float(seq[t]["dist"][p]) for t, p in enumerate(picks))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support", default="scores", choices=("heat", "scores"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--years", type=int, nargs="*", default=None)
    args = ap.parse_args()
    years = args.years or YEARS

    data = {}
    for year in years:
        data[year] = load_fold(year, args.tag, args.support)
        print(f"loaded {year}", flush=True)

    table = {(k, y): evaluate(data, [y], k) for k in KAPPAS for y in years}
    per_year, mds, drs = {}, [], []
    for year in years:
        others = [y for y in years if y != year]
        best = min(KAPPAS, key=lambda k: np.mean([table[(k, y)] for y in others]))
        vals = []
        for seq in data[year].values():
            picks = viterbi(seq, best)
            vals.extend(float(seq[t]["dist"][p]) for t, p in enumerate(picks))
        a = np.asarray(vals)
        per_year[str(year)] = {"kappa": best, "MD": float(a.mean()),
                               "DR3": float((a <= 3).mean()), "n": int(len(a))}
        mds.append(float(a.mean())); drs.append(float((a <= 3).mean()))
        print(f"heldout {year}: kappa={best:<8g} MD={a.mean():6.3f} DR={(a <= 3).mean():.3f}",
              flush=True)

    print("\n=== fixed-kappa reference (all years, optimistic) ===")
    ref = {}
    for k in KAPPAS:
        a = np.asarray([v for y in years for v in
                        [float(sq[t]["dist"][p]) for sq in data[y].values()
                         for t, p in enumerate(viterbi(sq, k))]])
        ref[str(k)] = {"MD": float(a.mean()), "DR3": float((a <= 3).mean())}
        print(f"  kappa={k:<8g} MD={a.mean():7.3f} DR={(a<=3).mean():6.3f}")
    print(f"\nnested macro MD={np.mean(mds):.4f} DR={np.mean(drs):.4f}")

    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    run = REPO / "runs/d_improve" / f"viterbi_{args.support}_{stamp}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "result.json").write_text(json.dumps(
        {"per_year": per_year, "nested": {"macro_MD": float(np.mean(mds)),
                                          "macro_DR3": float(np.mean(drs))},
         "fixed_kappa": ref, "support": args.support, "test_used": False},
        indent=2), encoding="utf-8")
    print("wrote", run)


if __name__ == "__main__":
    main()
