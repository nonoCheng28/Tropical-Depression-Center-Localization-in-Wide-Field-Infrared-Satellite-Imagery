"""Figure 3: four-panel heatmaps (D / v2 / v3 / ensemble) on one frame."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parents[3]

from td_center.reproduce.common import ARTIFACT_DIR, DATA_ROOT  # noqa: E402

OUT = REPO / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

for name in ("DejaVu Sans", "Arial"):
    if name in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = name
        break

SID = "NWP2415_SOULIK_0_2024091718_13_17.4_117.7"
WEST, EAST, SOUTH, NORTH = 109.0, 179.0, -15.0, 55.0
DEG = 70.0

A = ARTIFACT_DIR
maps = [
    ("D (soft)", torch.load(A / "ens3_val" / "D_fp32.pt", map_location="cpu", weights_only=False), "#fdf1d6"),
    ("v2 (sharp)", torch.load(A / "ens3_val" / "v2_fp32.pt", map_location="cpu", weights_only=False), "#e3edf9"),
    ("v3 (sharp, augmented)", torch.load(A / "ens3_val" / "v3_fp32.pt", map_location="cpu", weights_only=False), "#e3edf9"),
    ("ensemble (mean of 3)", torch.load(A / "ens3_val" / "heatmaps_val_fp32.pt", map_location="cpu", weights_only=False), "#e7f3e4"),
]


def norm(h):
    h = h.float()
    span = float(h.max() - h.min())
    return (h - h.min()) / (span + 1e-6)


x, y = map(float, (DATA_ROOT / "single_frame/ir/val/labels/TD_center/yolo" / f"{SID}.txt")
           .read_text(encoding="utf-8").split())
truth = np.array([WEST + x * DEG, NORTH - y * DEG])

fig, axes = plt.subplots(1, 4, figsize=(16 / 2.54, 5.6 / 2.54), dpi=300)
fig.subplots_adjust(left=0.055, right=0.99, top=0.845, bottom=0.30, wspace=0.07)

for ax, (title, store, _edge) in zip(axes, maps):
    h = norm(store[SID]).numpy()
    im = ax.imshow(h, extent=[WEST, EAST, SOUTH, NORTH], origin="upper",
                   cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
    r, c = np.unravel_index(int(h.argmax()), h.shape)
    peak = np.array([WEST + (c + 0.5) / h.shape[1] * DEG, NORTH - (r + 0.5) / h.shape[0] * DEG])
    err = float(np.linalg.norm(peak - truth))
    ax.plot(*truth, marker="*", ms=9, mfc="white", mec="black", mew=0.6, zorder=5)
    ax.plot(*peak, marker="x", ms=6, mec="white", mew=1.2, zorder=6)
    ax.set_title(title, fontsize=6.8, pad=3)
    ax.text(0.03, 0.05, f"err {err:.1f}°", transform=ax.transAxes, fontsize=6.2,
            color="white", ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.18", fc="black", ec="none", alpha=0.45), zorder=7)
    if ax is axes[0]:
        ax.set_xticks([110, 130, 150, 170])
        ax.set_yticks([-10, 0, 10, 20, 30, 40])
        ax.tick_params(labelsize=5.2, length=1.6, pad=1.2, colors="#333333")
        ax.set_xlabel("°E", fontsize=5.6, labelpad=1.0)
        ax.set_ylabel("°N", fontsize=5.6, labelpad=1.0)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("#666666")

cax = fig.add_axes([0.40, 0.095, 0.28, 0.030])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("normalised support", fontsize=5.8, labelpad=1.2)
cb.ax.xaxis.set_label_position("top")
cb.ax.tick_params(labelsize=5.4, length=1.5, pad=1.2)
cb.outline.set_linewidth(0.4)

fig.text(0.055, 0.94, "★ truth      × argmax", fontsize=6.2, color="#333333", va="center")
fig.savefig(OUT / "fig2_emission_maps.png", dpi=300, facecolor="white")
fig.savefig(OUT / "fig2_emission_maps.pdf", facecolor="white")
print("wrote", OUT / "fig2_emission_maps.png", "| sid", SID)
