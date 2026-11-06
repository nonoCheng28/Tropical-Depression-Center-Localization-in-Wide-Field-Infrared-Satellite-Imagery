"""Average the eight LOYO fold models into one heatmap for a split.

Every fold model was trained on seven of the eight training years, so none of them
ever saw validation 2024 or test 2025. Averaging them is therefore an honest
out-of-sample ensemble for those splits -- and a way to use all 805 training
samples at once without training a ninth model.

Probabilities (sigmoid) are averaged rather than logits, so a model that happens
to emit larger logits cannot dominate the mean.

    python experiments/d_improve/predict_d_ensemble.py --split val
    python experiments/d_improve/predict_d_ensemble.py --split test
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from td_center.reproduce.common import REPO
from td_center.reproduce.train_d_v2 import DModelV2
from td_center.model_d import DModel as DModelV1

from td_center.dataloader import DATA_ROOT, IRDataset, MANIFEST_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("train", "val", "test"))
    ap.add_argument("--tag", default="d_v2")
    ap.add_argument("--epoch-file", default="epoch44.pt")
    ap.add_argument("--epochs", type=int, default=44)
    ap.add_argument("--out", default=None)
    ap.add_argument("--arch", default="v2", choices=("v1", "v2"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        ap.error("test requires --allow-test and a registered evaluation")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    runs = {}
    for year in range(2016, 2024):
        hits = sorted(glob.glob(str(REPO / "runs/d_improve" / f"{args.tag}_{year}_*")))
        hits = [h for h in hits if (Path(h) / args.epoch_file).exists()]
        if not hits:
            raise SystemExit(f"no run for {year}")
        runs[year] = Path(hits[-1])
    print("using runs:", {y: p.name for y, p in runs.items()}, flush=True)

    models = []
    for year, run in runs.items():
        ckpt = torch.load(run / args.epoch_file, map_location="cpu", weights_only=False)
        m = (DModelV2() if args.arch == "v2" else DModelV1()).to(device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models.append(m)
        print(f"loaded {year} {run.name} epoch {ckpt['epoch']}", flush=True)

    split = "train" if args.split == "train" else args.split
    ds = IRDataset(DATA_ROOT, MANIFEST_DIR, split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    out32, out16 = {}, {}
    with torch.no_grad():
        for i, sample in enumerate(loader, 1):
            images = sample["images"].to(device)
            acc = None
            for m in models:
                p = torch.sigmoid(m(images))
                acc = p if acc is None else acc + p
            h = (acc / len(models))[0, 0].cpu()
            out32[sample["sample_id"][0]] = h
            out16[sample["sample_id"][0]] = h.half()
            if i % 20 == 0 or i == len(ds):
                print(f"{args.split} {i}/{len(ds)}", flush=True)

    out = Path(args.out) if args.out else (
        REPO / "outputs" / "artifacts" / f"{args.tag}_ens_{args.split}.pt")
    torch.save(out16, out)
    torch.save(out32, str(out).replace(".pt", "_fp32.pt"))
    meta = {"split": args.split, "runs": {str(y): str(p) for y, p in runs.items()},
            "epoch_file": args.epoch_file, "n": len(out32),
            "note": "mean sigmoid of the eight LOYO fold models", "test_used": args.split == "test"}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("saved", out, len(out32))


if __name__ == "__main__":
    main()
