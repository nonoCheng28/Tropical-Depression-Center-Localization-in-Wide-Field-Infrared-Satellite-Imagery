"""The delivered pipeline, end to end, in one command.

Delivered configuration (decided by the project owner on 2026-11-03, round 4):

    heatmaps (three-emission ensemble, per-frame min-max normalised, equal weights)
      -> subtract a Gaussian at the analysed best-track positions of OTHER named storms in the
         same 70x70 field (109-179E, -15-55N), at least 5 deg from the labelled centre
      -> 8 local peaks per frame at 3 deg separation
      -> whole-trajectory Viterbi, kappa = 0.003

`--alpha 0` is the infrared-only ablation (identical to the round-2/3 pipeline), and `--placebo`
puts the same blobs in random directions. Both exist so that the delivered number can always be
read next to the ablation and the control.

Every run also reports the auxiliary columns of `results/loyo/FINAL.md`, with the same definitions
`eval_final_decoder.py` uses: `RMSE = sqrt(mean(d^2))`, `worst` = largest frame error, and the
storm-level counts (a storm is "full" when its own mean is <= 3 deg, "over 10" when > 10 deg).
`alpha = 0` must reproduce RMSE 9.50 / worst 37.1 / 11 of 18 / 4 of 18 on test: that is the check
that the aided row below is measured with the same ruler as the ablation row.

    python experiments/d_improve/deliver_aided.py --loyo
    python experiments/d_improve/deliver_aided.py --split val
    python experiments/d_improve/deliver_aided.py --split test --allow-test

Test is frozen: `--split test` refuses to run unless the read has been recorded in
`runs_d/test_final_archive/test_used_round4_aided.json` and the flag is passed.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from td_center.reproduce.besttrack import others_in_view, split_id, to_xy
from td_center.reproduce.common import ARTIFACT_DIR, DATA_ROOT, REPO
from td_center.reproduce.eval_viterbi_decoder import viterbi

from td_center.candidates import local_peak_candidates  # noqa: E402

DEG = 70.0
KAPPA = 0.003
LEDGER = REPO / "runs_d/test_final_archive/test_used_round5_fair_arms.json"


def gaussian_at(size, xy, sigma=2.0):
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    cx, cy = xy[0] * size, xy[1] * size
    g = torch.exp(-((xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2) / (2 * sigma ** 2))
    m = float(g.max())
    return g / m if m > 0 else torch.zeros_like(g)


def run(heat, ids, label_dir, alpha=1.0, mode="real", seed=0, sigma=2.0):
    """Decode one split and return every metric for the chosen trajectory.

    Frame metrics: MD and DR@3 are the headline, RMSE = sqrt(mean(d^2)) and worst = max(d) are
    the auxiliary ones, all in degrees. `MAE_comp` / `RMSE_comp` are the *official* project
    auxiliary metrics (`td_center.metrics.center_metrics`), computed over the two coordinate
    components rather than the radial distance, i.e. sqrt(mean(dx^2, dy^2)). They are not
    interchangeable with `RMSE`: the published ablation row's 9.50 is the radial version.
    Storm metrics use the per-storm mean of the decoded frames: `storms_full` counts storms
    with that mean <= 3 deg, `storms_over10` counts storms with it > 10 deg.
    `err` / `err_comp` are the raw per-frame arrays, for pooled LOYO summaries.
    """
    rnd = np.random.default_rng(seed)
    storms = defaultdict(list)
    for sid in ids:
        storms[sid.split("_")[0]].append(sid)
    chosen, coord, target = {}, {}, {}
    for storm, seq in storms.items():
        seq.sort(key=lambda s: s.split("_")[3])
        rows = []
        for sid in seq:
            x, y = map(float, (label_dir / f"{sid}.txt").read_text(encoding="utf-8").split())
            h = heat[sid].float().clone()
            if alpha > 0:
                for _n, la, lo in others_in_view(sid, min_sep_deg=5.0, max_n=2):
                    if mode == "real":
                        px, py = to_xy(la, lo)
                    else:
                        _ts, tlat, tlon = split_id(sid)
                        sep = float(np.hypot(la - tlat, lo - tlon))
                        th = rnd.uniform(0, 2 * np.pi)
                        px, py = to_xy(tlat + sep * np.sin(th), tlon + sep * np.cos(th))
                    h = torch.clamp(h - alpha * gaussian_at(h.shape[0], (px, py), sigma), 0.0, 1.0)
            peaks = local_peak_candidates(h.numpy(), k=8, separation_deg=3.0)
            xy = np.array([[(p["col"] + 0.5) / h.shape[1], (p["row"] + 0.5) / h.shape[0]]
                           for p in peaks])
            g = torch.tensor(xy * 2 - 1, dtype=torch.float32)[None, :, None, :]
            hv = F.grid_sample(h[None, None], g, mode="bilinear", align_corners=False)[0, 0, :, 0]
            span = hv.max() - hv.min()
            sup = ((hv - hv.min()) / span if span > 1e-9 else torch.zeros_like(hv)).numpy()
            rows.append({"sid": sid, "xy": xy, "sup": sup, "target": np.array([x, y]),
                         "dist": np.linalg.norm((xy - np.array([x, y])) * DEG, axis=1)})
        picks = viterbi(rows, KAPPA)
        for r, p in zip(rows, picks):
            chosen[r["sid"]] = float(r["dist"][p])
            coord[r["sid"]] = r["xy"][p]
            target[r["sid"]] = r["target"]
    a = np.array([chosen[sid] for sid in ids])
    comp = (np.array([coord[sid] for sid in ids])
            - np.array([target[sid] for sid in ids])) * DEG
    storm_md = {s: float(np.mean([chosen[sid] for sid in seq])) for s, seq in storms.items()}
    return {"MD": float(a.mean()), "DR3": float((a <= 3).mean()), "n": int(a.size),
            "RMSE": float(np.sqrt((a ** 2).mean())), "worst": float(a.max()),
            "MAE_comp": float(np.abs(comp).mean()),
            "RMSE_comp": float(np.sqrt((comp ** 2).mean())),
            "storms": len(storms),
            "storms_full": int(sum(v <= 3 for v in storm_md.values())),
            "storms_over10": int(sum(v > 10 for v in storm_md.values())),
            "storm_md": storm_md, "err": a, "err_comp": comp}


def clean(stats, alpha, mode, sigma=2.0):
    out = {k: v for k, v in stats.items() if k not in ("err", "err_comp")}
    out.update(alpha=alpha, sigma=sigma, mode=mode)
    return out


def report(tag, rows):
    print(f"\n{tag}")
    print(f"  {'alpha':>6} {'sigma':>6} {'mode':>8} {'MD':>7} {'DR@3':>6} {'RMSE':>6} {'worst':>6} "
          f"{'MAEc':>6} {'RMSEc':>6} {'storms<=3':>10} {'storms>10':>10} {'n':>4}")
    for r in rows:
        full = f"{r['storms_full']}/{r['storms']}" if r["storms"] else "-"
        bad = f"{r['storms_over10']}/{r['storms']}" if r["storms"] else "-"
        print(f"  {r['alpha']:>6g} {r.get('sigma', 2.0):>6g} {r['mode']:>8} {r['MD']:>7.3f} "
              f"{r['DR3']:>6.3f} "
              f"{r['RMSE']:>6.2f} {r['worst']:>6.1f} {r['MAE_comp']:>6.2f}"
              f" {r['RMSE_comp']:>6.2f} {full:>10} {bad:>10} {r['n']:>4d}")


def dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", path)


# The published ablation row (results/loyo/FINAL.md), column by column. `alpha = 0` in any run
# must reproduce it: that is what makes the aided row below comparable to it.
EXPECT_DETAIL = {
    "val": {"MD": 6.701, "DR3": 0.734, "RMSE": 12.52, "worst": 32.5, "full": 14, "over": 4},
    "test": {"MD": 4.618, "DR3": 0.810, "RMSE": 9.50, "worst": 37.1, "full": 11, "over": 4},
}


def check_ablation(split, rows):
    base = next((r for r in rows if abs(r["alpha"]) < 1e-12 and r["mode"] == "real"), None)
    if base is None:
        return
    e = EXPECT_DETAIL[split]
    ok = (abs(base["MD"] - e["MD"]) < 5e-3 and abs(base["DR3"] - e["DR3"]) < 5e-3
          and abs(base["RMSE"] - e["RMSE"]) < 0.05 and abs(base["worst"] - e["worst"]) < 0.05
          and base["storms_full"] == e["full"] and base["storms_over10"] == e["over"])
    print(f"\n  ablation row check (alpha 0): "
          f"{'reproduces the published columns exactly' if ok else 'MISMATCH'}")
    if not ok:
        print(f"    got MD {base['MD']:.3f} DR {base['DR3']:.3f} RMSE {base['RMSE']:.2f}"
              f" worst {base['worst']:.1f} storms {base['storms_full']}/{base['storms']}"
              f" over10 {base['storms_over10']}/{base['storms']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loyo", action="store_true", help="eight folds, macro average")
    ap.add_argument("--split", choices=("val", "test"), default=None)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--years", type=int, nargs="+", default=None, help="LOYO fold years to decode (default all 8)")
    ap.add_argument("--dump-tag", default=None, help="suffix for the detail json, so a sweep does not overwrite the canonical file")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[2.0],
                    help="mask Gaussian width in heatmap cells; 2 cells is about 1.75 deg and is "
                         "the delivered value. Only used when alpha > 0.")
    ap.add_argument("--placebo", action="store_true", help="also run the placebo control at 1.0")
    ap.add_argument("--artifact", default=None,
                    help="heatmap artifact directory name under experiments/d_improve/artifacts; "
                         "defaults to the delivered ens3_{split}. Any other arm uses the identical "
                         "decoder code path (same constants), only the emission differs.")
    ap.add_argument("--artifact-pattern", default="ens3loyo{year}",
                    help="artifact directory pattern for --loyo; default is the delivered "
                         "ens3loyo{year}, e.g. 'loyo{year}' for the single-member frozen D")
    args = ap.parse_args()

    if args.split == "test":
        if not args.allow_test or not LEDGER.exists():
            raise SystemExit(f"test is frozen: pass --allow-test and record the read in {LEDGER}")

    if args.loyo:
        rows, per_fold = [], {}
        for sigma in args.sigmas:
            for alpha in args.alphas:
                folds, allerr, allcomp = [], [], []
                for year in (args.years or range(2016, 2024)):
                    art = ARTIFACT_DIR / args.artifact_pattern.format(year=year)
                    heat = torch.load(art / f"heatmaps_loyo{year}_fp32.pt", map_location="cpu",
                                      weights_only=False)
                    ids = [s for s in (REPO / f"splits/loyo/{year}_eval.txt")
                           .read_text(encoding="utf-8").split() if s]
                    st = run(heat, ids,
                             DATA_ROOT / "single_frame/ir/train/labels/TD_center/yolo", alpha,
                             sigma=sigma)
                    folds.append(st)
                    allerr.append(st["err"])
                    allcomp.append(st["err_comp"])
                    per_fold[f"{year}_a{alpha:g}_s{sigma:g}"] = {
                        k: v for k, v in st.items() if k not in ("err", "err_comp", "storm_md")}
                err = np.concatenate(allerr)
                comp = np.concatenate(allcomp)
                row = {"alpha": alpha, "sigma": sigma, "mode": "real",
                       "MD": float(np.mean([f["MD"] for f in folds])),
                       "DR3": float(np.mean([f["DR3"] for f in folds])),
                       "n": int(err.size),
                       "RMSE": float(np.sqrt((err ** 2).mean())), "worst": float(err.max()),
                       "MAE_comp": float(np.abs(comp).mean()),
                       "RMSE_comp": float(np.sqrt((comp ** 2).mean())),
                       "storms": 0, "storms_full": 0, "storms_over10": 0, "storm_md": {}}
                rows.append(row)
                print(f"  fold macro at alpha={alpha:g} sigma={sigma:g}: MD {row['MD']:.3f}"
                      f" DR {row['DR3']:.3f} | pooled RMSE {row['RMSE']:.2f}"
                      f" worst {row['worst']:.1f} (informational)", flush=True)
        if args.placebo:
            folds, allerr, allcomp = [], [], []
            for year in (args.years or range(2016, 2024)):
                heat = torch.load(ARTIFACT_DIR / args.artifact_pattern.format(year=year) / f"heatmaps_loyo{year}_fp32.pt",
                                  map_location="cpu", weights_only=False)
                ids = [s for s in (REPO / f"splits/loyo/{year}_eval.txt")
                       .read_text(encoding="utf-8").split() if s]
                st = run(heat, ids,
                             DATA_ROOT / "single_frame/ir/train/labels/TD_center/yolo",
                             1.0, "placebo", sigma=args.sigmas[0])
                folds.append(st)
                allerr.append(st["err"])
                allcomp.append(st["err_comp"])
            err = np.concatenate(allerr)
            comp = np.concatenate(allcomp)
            rows.append(clean({"MD": float(np.mean([f["MD"] for f in folds])),
                               "DR3": float(np.mean([f["DR3"] for f in folds])), "n": len(err),
                               "RMSE": float(np.sqrt((err ** 2).mean())), "worst": float(err.max()),
                               "MAE_comp": float(np.abs(comp).mean()),
                               "RMSE_comp": float(np.sqrt((comp ** 2).mean())),
                               "storms": 0, "storms_full": 0, "storms_over10": 0,
                               "storm_md": {}}, 1.0, "placebo", args.sigmas[0]))
        report("LOYO x8 (macro over folds; RMSE/worst pooled over all frames)", rows)
        loyo_detail = "aided_detail_loyo.json"
        if args.dump_tag:
            loyo_detail = loyo_detail.replace(".json", f"_{args.dump_tag}.json")
        dump(ARTIFACT_DIR / loyo_detail,
             {"kappa": KAPPA, "folds": per_fold, "rows": rows,
              "note": "storm columns are not defined for a macro over folds"})
        print("\n  delivered reference: aided 5.561 / 0.762 ; ablation 7.519 / 0.658")
        return

    split = args.split or "val"
    artifact = args.artifact or f"ens3_{split}"
    heat = torch.load(ARTIFACT_DIR / artifact / f"heatmaps_{split}_fp32.pt",
                      map_location="cpu", weights_only=False)
    ids = [s for s in (REPO / f"splits/single_center_v1/{split}.txt")
           .read_text(encoding="utf-8").split() if s]
    label_dir = DATA_ROOT / "single_frame/ir" / split / "labels/TD_center/yolo"
    rows = []
    for sigma in args.sigmas:
        for alpha in args.alphas:
            rows.append(clean(run(heat, ids, label_dir, alpha, sigma=sigma), alpha, "real", sigma))
    if args.placebo:
        rows.append(clean(run(heat, ids, label_dir, 1.0, "placebo", sigma=args.sigmas[0]),
                              1.0, "placebo", args.sigmas[0]))
    report(f"{split} ({len(ids)} frames, {rows[0]['storms']} storms) [{artifact}]", rows)
    if args.artifact is None:
        check_ablation(split, rows)
    detail = (f"aided_detail_{split}.json" if args.artifact is None
              else f"aided_detail_{artifact}_{split}.json")
    if args.dump_tag:
        detail = detail.replace(".json", f"_{args.dump_tag}.json")
    dump(ARTIFACT_DIR / detail,
         {"split": split, "artifact": artifact, "kappa": KAPPA, "frames": len(ids), "rows": rows,
          "metric_definitions": "RMSE = sqrt(mean(d^2)) over frames, radial; worst = max(d); "
                                "MAE_comp and RMSE_comp are the official td_center.center_metrics "
                                "values over the two coordinate components; storms<=3 = storms "
                                "whose own mean error is <= 3 deg; storms>10 = mean > 10 deg",
          "note": "alpha 0 is the infrared-only ablation and must reproduce the published "
                  "RMSE / worst / storm counts of that row"})
    if args.artifact is None:
        ref = {"val": "delivered 4.624 / 0.787 ; ablation 6.701 / 0.734",
               "test": "delivered 3.085 / 0.876 ; ablation 4.618 / 0.810"}[split]
        print(f"\n  delivered reference: {ref}")


if __name__ == "__main__":
    main()
