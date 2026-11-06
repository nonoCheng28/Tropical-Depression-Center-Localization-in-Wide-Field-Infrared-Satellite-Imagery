"""Decoder ablation and placebo, on the delivered emission, for val and LOYO only.

Reproduces the delivered frame loop (mask alpha=1.5, sigma=3 cells, 8 peaks / 3 deg,
per-frame min-max support) once per fold, then scores the same candidates under several
decoder settings:

    kappa = 0        no transition cost at all, i.e. mathematically identical to per-frame
                     argmax over the 8 candidates
    0.003            delivered value (must reproduce val 2.117 / LOYO 5.264)
    0.01, 0.1        sensitivity

Also runs the placebo (same geometry, random directions) at the delivered amplitude over
several seeds, because a single seed is bimodal on small panels.

Test is not read here: the 2025 split stays frozen.

    python paper/ablate_decoder.py --seeds 0 1 2 3 4
"""
import argparse
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


def masked_rows(heat, ids, label_dir, alpha, sigma, mode="real", seed=0):
    rnd = np.random.default_rng(seed)
    by = defaultdict(list)
    for sid in ids:
        by[sid.split("_")[0]].append(sid)
    rows = []
    for storm, seq in by.items():
        seq.sort(key=lambda s: s.split("_")[3])
        for sid in seq:
            x, y = map(float, (label_dir / f"{sid}.txt").read_text(encoding="utf-8").split())
            h = heat[sid].float().clone()
            if alpha > 0:
                for _n, la, lo in DA.others_in_view(sid, min_sep_deg=5.0, max_n=2):
                    if mode == "real":
                        px, py = DA.to_xy(la, lo)
                    else:
                        _ts, tlat, tlon = DA.split_id(sid)
                        sep = float(np.hypot(la - tlat, lo - tlon))
                        th = rnd.uniform(0, 2 * np.pi)
                        px, py = DA.to_xy(tlat + sep * np.sin(th), tlon + sep * np.cos(th))
                    h = torch.clamp(h - alpha * DA.gaussian_at(h.shape[0], (px, py), sigma), 0.0, 1.0)
            peaks = local_peak_candidates(h.numpy(), k=8, separation_deg=3.0)
            xy = np.array([[(p["col"] + 0.5) / h.shape[1], (p["row"] + 0.5) / h.shape[0]]
                           for p in peaks])
            g = torch.tensor(xy * 2 - 1, dtype=torch.float32)[None, :, None, :]
            hv = F.grid_sample(h[None, None], g, mode="bilinear", align_corners=False)[0, 0, :, 0]
            span = float(hv.max() - hv.min())
            sup = ((hv - hv.min()) / span if span > 1e-9 else torch.zeros_like(hv)).numpy()
            rows.append({"sid": sid, "storm": storm, "xy": xy, "sup": sup,
                         "target": np.array([x, y]),
                         "dist": np.linalg.norm((xy - np.array([x, y])) * DEG, axis=1)})
    return rows


def score(rows, kappa):
    by = defaultdict(list)
    for r in rows:
        by[r["storm"]].append(r)
    errs = []
    for seq in by.values():
        picks = viterbi(seq, kappa) if kappa > 0 else [int(np.argmax(r["sup"])) for r in seq]
        errs += [float(r["dist"][p]) for r, p in zip(seq, picks)]
    d = np.array(errs)
    return float(d.mean()), float((d <= 3).mean()), int(d.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--kappas", type=float, nargs="+", default=[0.003, 0.01, 0.1])
    ap.add_argument("--json", default=str(PAPER / "decoder_ablation.json"))
    args = ap.parse_args()

    out = {"alpha": ALPHA, "sigma": SIGMA, "seeds": args.seeds, "kappas": args.kappas}

    val_heat = torch.load(ARTIFACT_DIR / "ens3_val" / "heatmaps_val_fp32.pt",
                          map_location="cpu", weights_only=False)
    val_ids = [s for s in (REPO / "splits/single_center_v1/val.txt").read_text(encoding="utf-8").split() if s]
    val_lab = DATA_ROOT / "single_frame/ir/val/labels/TD_center/yolo"

    rows_val = masked_rows(val_heat, val_ids, val_lab, ALPHA, SIGMA)
    out["val_argmax"] = score(rows_val, 0.0)
    out["val_kappa"] = {k: score(rows_val, k) for k in args.kappas}
    print(f"val argmax {out['val_argmax'][0]:.3f} / {out['val_argmax'][1]:.3f}")
    for k in args.kappas:
        print(f"val kappa={k:<6g} {out['val_kappa'][k][0]:.3f} / {out['val_kappa'][k][1]:.3f}", flush=True)

    placebo_val = []
    for sd in args.seeds:
        r = masked_rows(val_heat, val_ids, val_lab, ALPHA, SIGMA, "placebo", sd)
        placebo_val.append(score(r, 0.003)[0])
    out["val_placebo_seeds"] = placebo_val

    train_lab = DATA_ROOT / "single_frame/ir/train/labels/TD_center/yolo"
    folds = defaultdict(list)
    loyo_placebo_folds = []
    for year in range(2016, 2024):
        art = ARTIFACT_DIR / f"ens3loyo{year}" / f"heatmaps_loyo{year}_fp32.pt"
        heat = torch.load(art, map_location="cpu", weights_only=False)
        ids = [s for s in (REPO / f"splits/loyo/{year}_eval.txt").read_text(encoding="utf-8").split() if s]
        rows = masked_rows(heat, ids, train_lab, ALPHA, SIGMA)
        folds["argmax"].append(score(rows, 0.0))
        for k in args.kappas:
            folds[k].append(score(rows, k))
        md = [score(masked_rows(heat, ids, train_lab, ALPHA, SIGMA, "placebo", sd), 0.003)[0]
              for sd in args.seeds]
        loyo_placebo_folds.append(float(np.mean(md)))
        print(f"{year} argmax {folds['argmax'][-1][0]:6.3f} "
              + " ".join(f"k{k:g} {folds[k][-1][0]:6.3f}" for k in args.kappas), flush=True)

    out["loyo_argmax"] = tuple(float(np.mean([x[i] for x in folds["argmax"]])) for i in (0, 1))
    out["loyo_kappa"] = {k: (float(np.mean([x[0] for x in folds[k]])),
                             float(np.mean([x[1] for x in folds[k]]))) for k in args.kappas}
    out["loyo_per_fold"] = {str(k): folds[k] for k in folds}
    out["loyo_placebo_folds"] = loyo_placebo_folds
    out["loyo_placebo"] = float(np.mean(loyo_placebo_folds))

    print("\nLOYO")
    print(f"  argmax        {out['loyo_argmax'][0]:6.3f} / {out['loyo_argmax'][1]:.3f}")
    for k in args.kappas:
        print(f"  kappa={k:<6g} {out['loyo_kappa'][k][0]:6.3f} / {out['loyo_kappa'][k][1]:.3f}")
    print("\nplacebo (alpha 1.5, sigma 3, seeds %s)" % args.seeds)
    print("  val  per seed:", " ".join(f"{x:.2f}" for x in placebo_val),
          f"-> mean {np.mean(placebo_val):.3f}")
    print("  LOYO per fold:", " ".join(f"{x:.2f}" for x in loyo_placebo_folds),
          f"-> macro {out['loyo_placebo']:.3f}")

    Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", args.json)


if __name__ == "__main__":
    main()
