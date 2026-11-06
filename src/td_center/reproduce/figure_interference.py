"""Figure 4: LEEPI case - a stronger storm dominates the field, the mask removes it."""
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import font_manager

REPO = Path(__file__).resolve().parents[3]


from td_center.reproduce.common import ARTIFACT_DIR, DATA_ROOT  # noqa: E402
import td_center.reproduce.deliver_aided as DA # noqa: E402
from td_center.reproduce.besttrack import others_in_view, to_xy  # noqa: E402
from td_center.reproduce.eval_viterbi_decoder import viterbi  # noqa: E402
from td_center.candidates import local_peak_candidates  # noqa: E402

for name in ("DejaVu Sans", "Arial"):
    if name in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = name
        break

OUT = REPO / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
STORM = "NWP2412"
ALPHA, SIGMA, KAPPA = 1.5, 3.0, 0.003
WEST, SOUTH, DEG = 109.0, -15.0, 70.0
LAT0 = 55.0


def norm(h):
    h = h.float()
    span = float(h.max() - h.min())
    return (h - h.min()) / (span + 1e-6)


def rows_for(heat, ids, label_dir, alpha):
    by = defaultdict(list)
    for sid in ids:
        by[sid.split("_")[0]].append(sid)
    out = []
    for storm, seq in by.items():
        seq.sort(key=lambda s: s.split("_")[3])
        for sid in seq:
            x, y = map(float, (label_dir / f"{sid}.txt").read_text(encoding="utf-8").split())
            h = heat[sid].float().clone()
            if alpha > 0:
                for _n, la, lo in others_in_view(sid, min_sep_deg=5.0, max_n=2):
                    px, py = to_xy(la, lo)
                    h = torch.clamp(h - alpha * DA.gaussian_at(h.shape[0], (px, py), SIGMA), 0.0, 1.0)
            peaks = local_peak_candidates(h.numpy(), k=8, separation_deg=3.0)
            xy = np.array([[(p["col"] + 0.5) / h.shape[1], (p["row"] + 0.5) / h.shape[0]]
                           for p in peaks])
            g = torch.tensor(xy * 2 - 1, dtype=torch.float32)[None, :, None, :]
            hv = F.grid_sample(h[None, None], g, mode="bilinear", align_corners=False)[0, 0, :, 0]
            span = float(hv.max() - hv.min())
            sup = ((hv - hv.min()) / span if span > 1e-9 else torch.zeros_like(hv)).numpy()
            out.append({"sid": sid, "xy": xy, "sup": sup, "h": h,
                        "target": np.array([x, y]),
                        "dist": np.linalg.norm((xy - np.array([x, y])) * DEG, axis=1)})
    return out


def main():
    heat = torch.load(ARTIFACT_DIR / "ens3_val" / "heatmaps_val_fp32.pt",
                      map_location="cpu", weights_only=False)
    ids = [s for s in (REPO / "splits/single_center_v1/val.txt").read_text(encoding="utf-8").split()
           if s.startswith(STORM)]
    label_dir = DATA_ROOT / "single_frame/ir/val/labels/TD_center/yolo"

    before = rows_for(heat, ids, label_dir, 0.0)
    after = rows_for(heat, ids, label_dir, ALPHA)
    picks_b = viterbi(before, KAPPA)
    picks_a = viterbi(after, KAPPA)

    err_b = [float(r["dist"][p]) for r, p in zip(before, picks_b)]
    err_a = [float(r["dist"][p]) for r, p in zip(after, picks_a)]
    print("before:", np.round(err_b, 1), "mean", np.mean(err_b).round(2))
    print("after :", np.round(err_a, 1), "mean", np.mean(err_a).round(2))

    # frame the mask repairs: largest unmasked argmax error among frames whose masked
    # argmax lands within 3 deg
    d_b = [float(r["dist"][int(np.argmax(r["sup"]))]) for r in before]
    d_a = [float(r["dist"][int(np.argmax(r["sup"]))]) for r in after]
    cand = [i for i in range(len(before)) if d_a[i] <= 3.0]
    k = max(cand, key=lambda i: d_b[i] - d_a[i])
    row_b, row_a = before[k], after[k]
    sid = row_b["sid"]
    print("panel frame:", sid, "unmasked err",
          round(float(row_b["dist"][int(np.argmax(row_b["sup"]))]), 1),
          "masked err", round(float(row_a["dist"][int(np.argmax(row_a["sup"]))]), 1))

    frame = np.load(DATA_ROOT / f"Multiple_frames_6h/val/images/{sid}.npy")
    ir = frame[-1] if frame.ndim == 3 else frame
    ir = ir[::2, ::2]

    fig, axes = plt.subplots(1, 4, figsize=(16 / 2.54, 5.1 / 2.54), dpi=300)
    fig.subplots_adjust(left=0.006, right=0.994, top=0.90, bottom=0.13, wspace=0.20)
    ext = [WEST, WEST + DEG, SOUTH, LAT0]

    ax = axes[0]
    ax.imshow(ir, cmap="gray", extent=ext, origin="upper")
    tx, ty = row_b["target"]
    # hollow star (dark halo + white ring) so the cloud structure underneath stays visible
    ax.plot(WEST + tx * DEG, LAT0 - ty * DEG, marker="*", ms=14, mfc="none", mec="#111111",
            mew=1.8, zorder=5)
    ax.plot(WEST + tx * DEG, LAT0 - ty * DEG, marker="*", ms=14, mfc="none", mec="white",
            mew=0.9, zorder=6)
    for _n, la, lo in others_in_view(sid, min_sep_deg=5.0, max_n=2):
        ax.plot(lo, la, marker="o", ms=8, mfc="none", mec="#00527a", mew=2.2, zorder=5)
        ax.plot(lo, la, marker="o", ms=8, mfc="none", mec="#8fd3ff", mew=1.0, zorder=6)
    ax.set_title("(a) infrared frame", fontsize=6.6, pad=3)

    for ax, row, tag, sub in ((axes[1], row_b, "(b) heatmap, no mask", "before"),
                              (axes[2], row_a, "(c) heatmap, masked", "after")):
        h = norm(row["h"]).numpy()
        ax.imshow(h, cmap="magma", extent=ext, origin="upper", vmin=0, vmax=1)
        r, c = np.unravel_index(int(h.argmax()), h.shape)
        pk = (WEST + (c + 0.5) / h.shape[1] * DEG, LAT0 - (r + 0.5) / h.shape[0] * DEG)
        ax.plot(WEST + row["target"][0] * DEG, LAT0 - row["target"][1] * DEG,
                marker="*", ms=10, mfc="white", mec="black", mew=0.6, zorder=5)
        ax.plot(*pk, marker="x", ms=7, mec="white", mew=1.2, zorder=6)
        ax.text(0.03, 0.05, f"err {float(row['dist'][int(np.argmax(row['sup']))]):.1f}°",
                transform=ax.transAxes, fontsize=6.0, color="white",
                bbox=dict(boxstyle="round,pad=0.16", fc="black", ec="none", alpha=0.45))
        ax.set_title(tag, fontsize=6.6, pad=3)

    # (d) trajectories over the 14 frames
    ax = axes[3]
    tb = [d for d in before]
    truth = np.array([[WEST + r["target"][0] * DEG, LAT0 - r["target"][1] * DEG] for r in tb])
    pb = np.array([[WEST + r["xy"][p][0] * DEG, LAT0 - r["xy"][p][1] * DEG]
                   for r, p in zip(before, picks_b)])
    pa = np.array([[WEST + r["xy"][p][0] * DEG, LAT0 - r["xy"][p][1] * DEG]
                   for r, p in zip(after, picks_a)])
    ax.plot(truth[:, 0], truth[:, 1], "--", color="#555555", lw=0.9, label="truth")
    ax.plot(pb[:, 0], pb[:, 1], "-", color="#d1495b", lw=0.9, label="no mask")
    ax.plot(pa[:, 0], pa[:, 1], "-", color="#2e7d32", lw=1.1, label="masked")
    ax.scatter(pb[:, 0], pb[:, 1], s=5, color="#d1495b")
    ax.scatter(pa[:, 0], pa[:, 1], s=5, color="#2e7d32")
    ax.set_title("(d) decoded tracks (14 frames)", fontsize=6.6, pad=3)
    ax.legend(fontsize=5.4, frameon=False, loc="upper left", handlelength=1.2,
              borderaxespad=0.2)
    ax.set_xticks([120, 130, 140, 150])
    ax.set_yticks([15, 20, 25, 30])
    ax.tick_params(labelsize=5.0, length=1.4, pad=1.0, colors="#333333")
    ax.set_xlabel("°E", fontsize=5.4, labelpad=0.8)
    pass

    for ax in axes[:3]:
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes:
        for s in ax.spines.values():
            s.set_linewidth(0.6)
            s.set_color("#666666")

    fig.savefig(OUT / "fig3_interference_case.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "fig3_interference_case.pdf", facecolor="white")
    print("wrote", OUT / "fig3_interference_case.png")


if __name__ == "__main__":
    main()
