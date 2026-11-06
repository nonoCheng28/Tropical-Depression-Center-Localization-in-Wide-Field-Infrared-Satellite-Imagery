"""Combine several emission heatmaps for validation or test, for the Viterbi decoder.

Mirrors the leave-one-year-out ensemble that pays off on the LOYO folds: the frozen
D heatmap is soft, the v2 focal heatmap is sharp, and their normalised mean decodes
better than either. This writes the combined maps in the layout
`eval_final_decoder.py` expects.

    python experiments/d_improve/combine_holdout.py --split val `
        --inputs artifacts/heatmaps_val.pt artifacts/d_v2_ens_val.pt --tag ens
"""
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

from td_center.reproduce.common import ARTIFACT_DIR, REPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("val", "test"))
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="heatmap files, relative to the artifact directory unless absolute")
    ap.add_argument("--tag", default="ens")
    args = ap.parse_args()

    paths = []
    for p in args.inputs:
        q = Path(p)
        paths.append(q if q.is_absolute() else ARTIFACT_DIR / q)
    dicts = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    keys = set(dicts[0])
    for d in dicts[1:]:
        if set(d) != keys:
            raise SystemExit("inputs cover different sample sets")
    out = {}
    for sid in sorted(keys):
        acc = None
        for d in dicts:
            h = d[sid].float()
            h = (h - h.min()) / (h.max() - h.min() + 1e-6)
            acc = h if acc is None else acc + h
        out[sid] = acc / len(dicts)
    dst = ARTIFACT_DIR / f"{args.tag}_{args.split}"
    dst.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.half() for k, v in out.items()}, dst / f"heatmaps_{args.split}.pt")
    torch.save(out, dst / f"heatmaps_{args.split}_fp32.pt")
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    meta = {"split": args.split, "inputs": [str(p) for p in paths], "n": len(out),
            "test_used": args.split == "test", "created": stamp}
    (dst / "combine_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {dst} ({len(out)} samples from {len(paths)} emissions)")


if __name__ == "__main__":
    main()
