"""Write one arm's validation heatmaps in the layout the shared decoder consumes.

The point is to run *any* backbone through the same mask + Viterbi decoder without
touching the decoder: this script only produces the emission, `combine_holdout.py`
applies the same per-frame min-max normalisation the delivered ensemble uses, and
`deliver_aided.py --artifact ...` decodes it with the identical constants.

Rebuilds the model from the checkpoint's own config, so the preprocessing
(PNG vs float32 current frame, 320x320 resize, 7-frame order) is the one the arm
was trained with, not one chosen here.

    python experiments/d_improve/predict_arm_heatmaps.py \
        --run runs/b_png_formal_20261104_163851_903421 --tag fair_b_png
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from td_center.reproduce.common import ARTIFACT_DIR, REPO

from td_center.heatmap import decode_argmax  # noqa: E402
from td_center.metrics import center_metrics  # noqa: E402
from td_center.training import build_datasets, build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory containing best.pt")
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--allow-test", action="store_true",
                    help="required for --split test; the round-5 registration must exist first")
    ap.add_argument("--tag", required=True, help="artifact tag, e.g. fair_b_png")
    ap.add_argument("--checkpoint", default="best.pt",
                    help="checkpoint file name inside the run; LOYO folds use last.pt because the "
                         "held-out year must not select the weights")
    ap.add_argument("--normalize", action="store_true",
                    help="apply the same per-frame min-max normalisation combine_holdout.py uses, "
                         "so the file can be decoded without a combine step")
    ap.add_argument("--out-dir", default=None, help="artifact directory name (default tag_split)")
    ap.add_argument("--out-name", default=None,
                    help="file stem (default heatmaps_<split>, LOYO folds use heatmaps_loyo<year>)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    run = Path(args.run)
    run = run if run.is_absolute() else (REPO / run)
    checkpoint = torch.load(run / args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    datasets = build_datasets(Path(config["data_root"]), config) if args.split == "val" else None
    if args.split == "test":
        if not args.allow_test:
            raise SystemExit("test is frozen: pass --allow-test (see runs_d/test_final_archive/"
                             "test_used_round5_fair_arms.json)")
        from td_center.dataloader import IRDataset as _IR, SingleFrameIRDataset as _SF, MANIFEST_DIR
        from td_center.training import CurrentFrameDataset as _CF
        root = Path(config["data_root"])
        kind = config.get("model_kind", "d")
        if kind in ("b", "tclnet"):
            dataset = (_SF(root, MANIFEST_DIR, "test")
                       if config.get("single_frame_source", "png") == "png"
                       else _CF(root, MANIFEST_DIR, "test", span=config.get("span", "6h")))
        elif kind == "c" and config.get("cnn_in_channels", 7) == 1:
            dataset = _CF(root, MANIFEST_DIR, "test", span=config.get("span", "6h"))
        else:
            dataset = _IR(root, MANIFEST_DIR, "test", span=config.get("span", "6h"))
    else:
        dataset = datasets["val"]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    heat, predictions, targets = {}, [], []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["images"].to(device)).cpu()
            for sample_id, per_sample in zip(batch["sample_id"], output):
                heat[sample_id] = per_sample[0].clone()  # [80, 80], the decoder's layout
            predictions.append(torch.stack([decode_argmax(h) for h in output]))
            targets.append(batch["center"])
    metrics = center_metrics(torch.cat(predictions), torch.cat(targets))

    if args.checkpoint == "best.pt":
        recorded = json.loads((run / "best_metrics.json").read_text(encoding="utf-8"))
    else:
        # LOYO folds keep the final epoch, so the reference is the last history row.
        history = list(csv.DictReader((run / "history.csv").open(encoding="utf-8")))
        recorded = {"mean_distance_deg": float(history[-1]["mean_distance_deg"]),
                    "epoch": int(history[-1]["epoch"])}
    drift = abs(metrics["mean_distance_deg"] - recorded["mean_distance_deg"])
    dst = ARTIFACT_DIR / (args.out_dir or f"{args.tag}_{args.split}")
    dst.mkdir(parents=True, exist_ok=True)
    stem = args.out_name or f"heatmaps_{args.split}"
    if args.normalize:
        heat = {k: (v - v.min()) / (v.max() - v.min() + 1e-6) for k, v in heat.items()}
    torch.save({k: v.half() for k, v in heat.items()}, dst / f"{stem}.pt")
    torch.save(heat, dst / f"{stem}_fp32.pt")
    meta = {"split": args.split, "run": str(run), "tag": args.tag, "n": len(heat),
            "model": config.get("model"), "model_kind": config.get("model_kind"),
            "seed": config.get("seed"), "decoder": "argmax_half_pixel",
            "argmax_MD_recomputed": metrics["mean_distance_deg"],
            "argmax_MD_recorded": None if args.split == "test" else recorded["mean_distance_deg"],
            "max_abs_drift_deg": None if args.split == "test" else drift,
            "test_used": args.split == "test"}
    (dst / "source_run.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print(f"{args.tag}: {len(heat)} frames; argmax MD {metrics['mean_distance_deg']:.4f}")
    tolerance = None if args.split == "test" else 5e-2
    if tolerance is not None and drift > tolerance:
        raise SystemExit(f"heatmaps do not reproduce the recorded argmax MD (drift {drift:.2e} "
                         f"> {tolerance:g}); refusing to continue")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
