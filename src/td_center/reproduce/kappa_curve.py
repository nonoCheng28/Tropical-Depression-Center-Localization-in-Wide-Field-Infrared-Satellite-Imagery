"""kappa curve and paired significance, on the delivered emission (val + LOYO only).

Purpose: decide whether the LOYO preference for kappa = 0.01 over the frozen 0.003 is a
real difference or a within-noise wobble. Rows are computed once per fold; only the
transition cost changes between kappa values.

Outputs per-kappa MD and, for the 0.01 vs 0.003 comparison, a paired bootstrap over storm
instances (unit = one storm in one fold, or one storm in val).

    python paper/kappa_curve.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
PAPER = REPO / "outputs" / "paper"
PAPER.mkdir(parents=True, exist_ok=True)


from td_center.reproduce.common import ARTIFACT_DIR, DATA_ROOT  # noqa: E402
import td_center.reproduce.deliver_aided as DA # noqa: E402
from td_center.reproduce.eval_viterbi_decoder import viterbi  # noqa: E402
from td_center.candidates import local_peak_candidates  # noqa: E402

DEG = 70.0
ALPHA, SIGMA = 1.5, 3.0
KAPPAS = (0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.1)


def masked_rows(heat, ids, label_dir, alpha=ALPHA, sigma=SIGMA):
    by = defaultdict(list)
    for sid in ids:
        by[sid.split("_")[0]].append(sid)
    rows = []
    for storm, seq in by.items():
        seq.sort(key=lambda s: s.split("_")[3])
        for sid in seq:
            x, y = map(float, (label_dir / f"{sid}.txt").read_text(encoding="utf-8").split())
            h = heat[sid].float().clone()
            for _n, la, lo in DA.others_in_view(sid, min_sep_deg=5.0, max_n=2):
                px, py = DA.to_xy(la, lo)
                h = torch.clamp(h - alpha * DA.gaussian_at(h.shape[0], (px, py), sigma), 0.0, 1.0)
            peaks = local_peak_candidates(h.numpy(), k=8, separation_deg=3.0)
            xy = np.array([[(p["col"] + 0.5) / h.shape[1], (p["row"] + 0.5) / h.shape[0]]
                           for p in peaks])
            g = torch.tensor(xy * 2 - 1, dtype=torch.float32)[None, :, None, :]
            hv = F.grid_sample(h[None, None], g, mode="bilinear", align_corners=False)[0, 0, :, 0]
            span = float(hv.max() - hv.min())
            sup = ((hv - hv.min()) / span if span > 1e-9 else torch.zeros_like(hv)).numpy()
            rows.append({"sid": sid, "storm": storm, "xy": xy, "sup": sup,
                         "dist": np.linalg.norm((xy - np.array([x, y])) * DEG, axis=1)})
    return rows


def per_storm(rows, kappa):
    """Return ({storm_key: (mean_dist, n_frames)}, frame-weighted MD, frame-weighted DR@3)."""
    by = defaultdict(list)
    for r in rows:
        by[r["storm"]].append(r)
    out = {}
    frames = []
    for key, seq in by.items():
        picks = viterbi(seq, kappa)
        d = [float(r["dist"][p]) for r, p in zip(seq, picks)]
        out[key] = (float(np.mean(d)), len(d))
        frames += d
    a = np.array(frames)
    return out, float(a.mean()), float((a <= 3).mean())


def main():
    sets = {}

    val_heat = torch.load(ARTIFACT_DIR / "ens3_val" / "heatmaps_val_fp32.pt",
                          map_location="cpu", weights_only=False)
    val_ids = [s for s in (REPO / "splits/single_center_v1/val.txt").read_text(encoding="utf-8").split() if s]
    rows = masked_rows(val_heat, val_ids, DATA_ROOT / "single_frame/ir/val/labels/TD_center/yolo")
    sets["val"] = rows

    train_lab = DATA_ROOT / "single_frame/ir/train/labels/TD_center/yolo"
    loyo = []
    for year in range(2016, 2024):
        heat = torch.load(ARTIFACT_DIR / f"ens3loyo{year}" / f"heatmaps_loyo{year}_fp32.pt",
                          map_location="cpu", weights_only=False)
        ids = [s for s in (REPO / f"splits/loyo/{year}_eval.txt").read_text(encoding="utf-8").split() if s]
        got = masked_rows(heat, ids, train_lab)
        for r in got:
            r["storm"] = f"{year}_{r['storm']}"
        loyo += got
    sets["loyo"] = loyo

    # aggregation must match the headline: macro over folds (LOYO) or over the single
    # val split, with frame weighting inside each fold
    groups = {"val": [None], "loyo": [str(y) for y in range(2016, 2024)]}
    res = {}
    for split, rows in sets.items():
        res[split] = {}
        print(f"\n{split}")
        for k in KAPPAS:
            per = {}
            fold_md, fold_dr = [], []
            for g in groups[split]:
                sub = [r for r in rows if g is None or r["storm"].startswith(g + "_")]
                pk = viterbi  # noqa: F841
                by = defaultdict(list)
                for r in sub:
                    by[r["storm"]].append(r)
                frames = []
                for key, seq in by.items():
                    picks = viterbi(seq, k)
                    d = [float(r["dist"][pp]) for r, pp in zip(seq, picks)]
                    per[key] = (float(np.mean(d)), len(d))
                    frames += d
                a = np.array(frames)
                fold_md.append(float(a.mean()))
                fold_dr.append(float((a <= 3).mean()))
            md = float(np.mean(fold_md))
            res[split][str(k)] = {"MD": md, "DR3": float(np.mean(fold_dr)), "per_storm": per}
            print(f"  kappa={k:<6g} MD {md:6.3f}  DR {res[split][str(k)]['DR3']:.3f}")

    # paired storm bootstrap: 0.01 vs 0.003
    rng = np.random.default_rng(0)
    for split in sets:
        a = res[split]["0.003"]["per_storm"]
        b = res[split]["0.01"]["per_storm"]
        keys = sorted(a)
        d = np.array([b[k][0] - a[k][0] for k in keys])
        draws = [float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(10000)]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        better = int((d < 0).sum())
        print(f"\n{split}: 0.01 - 0.003 = {d.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
              f"  storms better {better}/{len(d)}  p(boot<=0)={float(np.mean(np.array(draws) <= 0)):.3f}")
        res[split]["delta_0.01_vs_0.003"] = {"mean": float(d.mean()), "ci": [float(lo), float(hi)],
                                             "storms_better": better, "n": len(d),
                                             "p_le_0": float(np.mean(np.array(draws) <= 0))}

    out = {"alpha": ALPHA, "sigma": SIGMA, "kappas": list(KAPPAS),
           "MD": {s: {k: res[s][k]["MD"] for k in map(str, KAPPAS)} for s in res},
           "DR3": {s: {k: res[s][k]["DR3"] for k in map(str, KAPPAS)} for s in res},
           "delta": {s: res[s]["delta_0.01_vs_0.003"] for s in res}}
    (PAPER / "kappa_curve.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote", PAPER / "kappa_curve.json")


if __name__ == "__main__":
    main()
