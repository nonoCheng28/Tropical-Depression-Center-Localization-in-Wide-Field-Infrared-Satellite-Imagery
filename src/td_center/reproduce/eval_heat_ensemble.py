"""Ensemble two emissions of different character before decoding.

The frozen D heatmap is soft and decodes well but has a poor per-frame argmax; the
v2 focal heatmap has a much better argmax but a near-binary support that the
trajectory decoder cannot exploit (on the three folds finished so far, v2 improves
argmax MD by 2.7 deg but v2+Viterbi only improves the frozen+Viterbi MD by 0.5 deg).
Averaging the two normalised emissions should keep the v2 localisation and recover
the softness the decoder needs.

    python experiments/d_improve/eval_heat_ensemble.py --years 2019 2021 2022
"""
import argparse
import glob
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from td_center.reproduce.common import DATA_ROOT, REPO
from td_center.reproduce.eval_viterbi_decoder import viterbi as viterbi1

DEG = 70.0
KAPPAS = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0)


def norm(t):
    return (t - t.min()) / (t.max() - t.min() + 1e-6)


def load_pairs(year, sources):
    dicts = []
    for src in sources:
        path = (REPO / "outputs/artifacts" / src.format(year=year) /
                f"heatmaps_loyo{year}_fp32.pt")
        dicts.append(torch.load(path, map_location="cpu", weights_only=False))
    ids = [l.strip() for l in (REPO / "splits/loyo" / f"{year}_eval.txt")
           .read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {}
    for sid in ids:
        acc = None
        for d in dicts:
            h = norm(d[sid].float())
            acc = h if acc is None else acc + h
        out[sid] = acc / len(dicts)
    return out



def viterbi2(rows, kappa):
    """Second-order (constant-velocity) MAP trajectory: penalise acceleration."""
    n = len(rows)
    pts = [r["xy"] * DEG for r in rows]
    sup = [r["sup"] for r in rows]
    if n == 1:
        return [0]
    if n == 2:
        return viterbi1(rows, kappa)
    d1 = np.linalg.norm(pts[1][:, None, :] - pts[0][None, :, :], axis=-1) ** 2
    cost = -sup[0][None, :] - sup[1][:, None] + kappa * d1
    back = []
    for t in range(2, n):
        acc = (pts[t][None, :, None, :] - 2 * pts[t - 1][None, None, :, :]
               + pts[t - 2][:, None, None, :])
        pen = np.linalg.norm(acc, axis=-1) ** 2
        total = cost[:, None, :] + kappa * pen
        best_h = np.argmin(total, axis=0)
        best_val = np.take_along_axis(total, best_h[None], axis=0)[0]
        back.append(best_h)
        cost = best_val - sup[t][:, None]
    i_last, i_prev = np.unravel_index(int(np.argmin(cost)), cost.shape)
    idx = [int(i_last), int(i_prev)]
    for t in range(n - 1, 1, -1):
        idx.append(int(back[t - 2][idx[-2], idx[-1]]))
    return idx[::-1]

def kappa_median(rows, kappas):
    """Per-frame median over the trajectories decoded at every kappa.

    The Viterbi decoder is bistable: a small change in kappa flips whole tracks to a
    different cloud feature, so any single constant is a coin flip between them.
    Averaging over the grid removes the choice instead of making it.
    """
    pts = [r["xy"] * DEG for r in rows]
    picks = [viterbi1(rows, k) for k in kappas]
    out = []
    for t in range(len(rows)):
        ps = np.array([pts[t][p[t]] for p in picks])
        out.append(np.median(ps, axis=0))
    return out
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["loyo{year}", "v2loyo{year}"])
    ap.add_argument("--years", type=int, nargs="+", default=[2019, 2021, 2022])
    ap.add_argument("--write-tag", default=None)
    ap.add_argument("--order", type=int, default=1, choices=(1, 2, 3))
    args = ap.parse_args()

    from td_center.candidates import local_peak_candidates

    per_year = {}
    for year in args.years:
        heat = load_pairs(year, args.sources)
        if args.write_tag:
            dst = REPO / "outputs/artifacts" / f"{args.write_tag}loyo{year}"
            dst.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.half() for k, v in heat.items()},
                       dst / f"heatmaps_loyo{year}.pt")
            torch.save(heat, dst / f"heatmaps_loyo{year}_fp32.pt")
            print(f"  wrote {dst}", flush=True)
        ids = sorted(heat)
        storms = defaultdict(list)
        for sid in ids:
            storms[sid.split("_")[0]].append(sid)
        for v in storms.values():
            v.sort(key=lambda s: s.split("_")[3])
        rows_by_storm = {}
        for storm, seq in storms.items():
            rows = []
            for sid in seq:
                h = heat[sid]
                peaks = local_peak_candidates(h.numpy(), k=8, separation_deg=3.0)
                xy = np.array([[(p["col"] + 0.5) / 80, (p["row"] + 0.5) / 80]
                               for p in peaks])
                label = (DATA_ROOT / "single_frame/ir/train/labels/TD_center/yolo"
                         / f"{sid}.txt")
                x, y = map(float, label.read_text(encoding="utf-8").split())
                g = torch.tensor(xy, dtype=torch.float32)[None] * 2 - 1
                hv = F.grid_sample(h[None, None], g[:, :, None, :], mode="bilinear",
                                   align_corners=False)[0, 0, :, 0].numpy()
                rng = hv.max() - hv.min()
                rows.append({"sid": sid, "xy": xy, "target": np.array([x, y]),
                             "dist": np.linalg.norm((xy - np.array([x, y])) * DEG, axis=1),
                             "sup": (hv - hv.min()) / rng if rng > 1e-9 else np.zeros_like(hv)})
            rows_by_storm[storm] = rows
        table, table_dr = {}, {}
        if args.order == 3:
            vals = []
            for rows in rows_by_storm.values():
                pos = kappa_median(rows, KAPPAS)
                for t, p in enumerate(pos):
                    vals.append(float(np.linalg.norm(p - rows[t]["target"] * DEG)))
            a = np.asarray(vals)
            table[0.0] = float(a.mean())
            table_dr[0.0] = float((a <= 3).mean())
            per_year[year] = {"kappa": "median", "MD": float(a.mean()),
                              "DR3": float((a <= 3).mean()), "n": int(len(a)),
                              "table": {str(0.0): float(a.mean())},
                              "table_dr": {str(0.0): float((a <= 3).mean())}}
            print(f"{year}: kappa-median MD={a.mean():6.3f} DR={(a <= 3).mean():.3f}", flush=True)
            continue
        for k in KAPPAS:
            vals = []
            for rows in rows_by_storm.values():
                picks = (viterbi1(rows, k) if args.order == 1 else viterbi2(rows, k))
                vals.extend(float(rows[t]["dist"][p]) for t, p in enumerate(picks))
            a = np.asarray(vals)
            table[k] = float(a.mean())
            table_dr[k] = float((a <= 3).mean())
        best_k = min(KAPPAS, key=table.get)
        vals = []
        for rows in rows_by_storm.values():
            picks = viterbi1(rows, best_k) if args.order == 1 else viterbi2(rows, best_k)
            vals.extend(float(rows[t]["dist"][p]) for t, p in enumerate(picks))
        a = np.asarray(vals)
        per_year[year] = {"kappa": best_k, "MD": float(a.mean()),
                          "DR3": float((a <= 3).mean()), "n": int(len(a)),
                          "table": {str(k): v for k, v in table.items()},
                          "table_dr": {str(k): v for k, v in table_dr.items()}}
        print(f"{year}: best kappa={best_k:<8g} MD={a.mean():6.3f} DR={(a <= 3).mean():.3f}",
              flush=True)

    mds = [v["MD"] for v in per_year.values()]
    drs = [v["DR3"] for v in per_year.values()]
    print(f"\nensemble ({'+'.join(args.sources)}) macro MD={np.mean(mds):.4f} "
          f"DR={np.mean(drs):.4f} over {len(per_year)} folds  [per-fold best kappa]")
    # nested selection: for each held-out year choose kappa on the other years only
    if len(per_year) > 1:
        nested = {}
        for year, v in per_year.items():
            others = [w for w in per_year.values() if w is not v]
            k = min(KAPPAS, key=lambda kk: float(np.mean([w["table"][str(kk)] for w in others])))
            nested[year] = {"kappa": k, "MD": v["table"][str(k)],
                            "DR3": v["table_dr"][str(k)],
                            "per_fold_best_was": v["kappa"]}
        nmd = [w["MD"] for w in nested.values()]
        ndr = [w["DR3"] for w in nested.values()]
        print(f"ensemble nested kappa: macro MD={np.mean(nmd):.4f} DR={np.mean(ndr):.4f}")
        for year, w in nested.items():
            print(f"  {year}: nested kappa={w['kappa']:<8g} MD={w['MD']:6.3f} "
                  f"(per-fold best was {w['per_fold_best_was']})")
        per_year = {y: {**v, "nested": nested[y]} for y, v in per_year.items()}
    import json
    from datetime import datetime, timedelta, timezone
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    run = REPO / "runs/d_improve" / f"heat_ensemble_{stamp}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "result.json").write_text(json.dumps(
        {"sources": args.sources, "per_year": {str(k): v for k, v in per_year.items()},
         "macro_MD": float(np.mean(mds)), "macro_DR3": float(np.mean(drs))}, indent=2),
        encoding="utf-8")
    print("wrote", run)


if __name__ == "__main__":
    main()
