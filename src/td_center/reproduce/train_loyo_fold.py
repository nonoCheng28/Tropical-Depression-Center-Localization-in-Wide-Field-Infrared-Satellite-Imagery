"""Train one LOYO fold of the D model (leave one year out of the frozen train list).

Implements `docs/LOYO_PROTOCOL.md` section 2:

* identical optimisation settings to `runs/d_formal_20261019_220943_751483`;
* **fixed 44 epochs, no early stopping, no internal dev** (44 = the formal run's
  val-selected epoch; it is the only budget that keeps "only the held-out year
  differs" true across folds);
* the held-out year's MD is logged every epoch **for diagnosis only** and is
  never used to pick a checkpoint;
* at the end, the epoch-44 checkpoint is used to dump fp32+half heatmaps for
  every frozen train sample.

    python src/td_center/reproduce/train_loyo_fold.py --year 2020
"""
import argparse
import csv
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from td_center.dataloader import IRDataset

REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "splits" / "loyo"
FROZEN = REPO / "splits" / "single_center_v1"
CONFIG_PATH = REPO / "configs" / "d_first_round.json"


class FoldDataset(IRDataset):
    """IRDataset bound to an explicit id list; images still come from one split dir.

    `IRDataset.__getitem__` only uses `self.base` and `self.sample_ids`, so
    overriding `__init__` reuses the exact decoding/label path while bypassing the
    train/val/test manifest check.
    """

    def __init__(self, data_root, ids, physical_split="train", span="6h"):
        if not ids:
            raise ValueError("empty id list")
        self.base = Path(data_root) / f"Multiple_frames_{span}" / physical_split
        self.sample_ids = list(ids)


def read_ids(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha256(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:  # pragma: no cover
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=44)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-root", default=str(REPO / "runs" / "d_improve"))
    ap.add_argument("--artifacts", default=str(REPO / "outputs" / "artifacts"))
    ap.add_argument("--dump-heatmaps", dest="dump_heatmaps", action="store_true", default=True)
    ap.add_argument("--no-heatmaps", dest="dump_heatmaps", action="store_false")
    ap.add_argument("--smoke", action="store_true", help="8 train / 3 eval samples, heatmaps off")
    args = ap.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    from td_center.dataloader import DATA_ROOT
    from td_center.model_d import DModel
    from td_center.training import evaluate, save_checkpoint, train_epoch, write_csv

    train_ids = read_ids(MANIFEST / f"{args.year}_train.txt")
    eval_ids = read_ids(MANIFEST / f"{args.year}_eval.txt")
    frozen = read_ids(FROZEN / "train.txt")
    assert sorted(train_ids + eval_ids) == sorted(frozen)
    assert not (set(train_ids) & set(eval_ids))
    if args.smoke:
        train_ids = train_ids[:8]
        eval_ids = eval_ids[:3]
        args.dump_heatmaps = False
        args.epochs = min(args.epochs, 1)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    config.update(
        role="core",
        experiment=f"LOYO fold {args.year}",
        protocol="docs/LOYO_PROTOCOL.md",
        protocol_sha256=sha256(REPO / "docs/LOYO_PROTOCOL.md"),
        fold_year=args.year,
        fixed_epochs=args.epochs,
        early_stopping="disabled (fixed budget, see protocol section 2)",
        train_n=len(train_ids),
        eval_n=len(eval_ids),
        micro_batch=args.micro_batch,
        accumulation_steps=16 // args.micro_batch,
        device=str(device),
        threads=args.threads,
        data_root=str(DATA_ROOT),
        span="6h",
        input="Multiple_frames_6h",
        delta_hours=1,
    )

    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    run = Path(args.out_root) / f"loyo_d_{args.year}_{stamp}"
    (run / "source").mkdir(parents=True)

    for src in sorted((REPO / "src" / "td_center").glob("*.py")):
        dst = run / "source" / "src" / "td_center" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    for src in [CONFIG_PATH, MANIFEST / f"{args.year}_train.txt",
                MANIFEST / f"{args.year}_eval.txt",
                REPO / "src/td_center/reproduce/train_loyo_fold.py",
                REPO / "docs/LOYO_PROTOCOL.md"]:
        dst = run / "source" / src.relative_to(REPO)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())

    config.update(commit=git("rev-parse", "HEAD"), git_status=git("status", "--short"))
    (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # deterministic setup identical to td_center.training.main
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    train_ds = FoldDataset(DATA_ROOT, train_ids)
    eval_ds = FoldDataset(DATA_ROOT, eval_ids)
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(train_ds, batch_size=args.micro_batch, shuffle=True,
                              generator=generator, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=args.micro_batch, shuffle=False,
                             num_workers=0)

    model = DModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    print(f"fold {args.year}: train={len(train_ds)} eval={len(eval_ds)} run={run}",
          flush=True)

    history, best, best_epoch, updates = [], math.inf, 0, 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train = train_epoch(model, train_loader, optimizer, device, 16, config["loss_alpha"])
        metrics, _ = evaluate(model, eval_loader, device, config["loss_alpha"])
        updates += train["optimizer_updates"]
        if metrics["mean_distance_deg"] < best:
            best, best_epoch = metrics["mean_distance_deg"], epoch
        history.append({"epoch": epoch, **train, **metrics, "cumulative_updates": updates,
                        "seconds": time.perf_counter() - t0})
        write_csv(run / "history.csv", history)
        state = {"epoch": epoch, "model_state_dict": model.state_dict(), "config": config,
                 "fold_year": args.year, "protocolled_epoch": args.epochs,
                 "metrics_on_heldout_year": metrics}
        save_checkpoint(run / "last.pt", state)
        if epoch == args.epochs:
            save_checkpoint(run / "epoch44.pt", state)
        print(f"epoch {epoch:>3}: train={train['train_weighted_mse']:.6f} "
              f"heldout MD={metrics['mean_distance_deg']:.4f} DR={metrics['detection_rate_at_3deg']:.3f} "
              f"({history[-1]['seconds']:.1f}s)", flush=True)

    elapsed = time.perf_counter() - started
    # diagnosis only: how sensitive is the held-out year to the epoch budget?
    curve = [(h["epoch"], round(h["mean_distance_deg"], 4)) for h in history]
    tail = [h["mean_distance_deg"] for h in history if h["epoch"] >= args.epochs - 6]
    (run / "result.json").write_text(json.dumps({
        "fold_year": args.year,
        "protocol": "LOYO_PROTOCOL.md section 2",
        "fixed_epochs": args.epochs,
        "train_n": len(train_ids),
        "eval_n": len(eval_ids),
        "heldout_metrics_at_fixed_epoch": history[-1] | {"epoch": args.epochs},
        "epoch_curve_diagnostic": curve,
        "min_epoch_curve": min(curve, key=lambda x: x[1]),
        "max_epoch_curve": max(curve, key=lambda x: x[1]),
        "tail_spread_deg": max(tail) - min(tail) if tail else None,
        "best_epoch_on_heldout_DIAGNOSTIC_ONLY": best_epoch,
        "note": "best_epoch is NOT used for selection; selection is the fixed epoch budget.",
        "elapsed_seconds": elapsed,
        "device": str(device),
    }, indent=2), encoding="utf-8")

    if args.dump_heatmaps:
        from td_center.dataloader import IRDataset

        ckpt = torch.load(run / "epoch44.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        full = IRDataset(DATA_ROOT, FROZEN, "train")
        out32, out16 = {}, {}
        with torch.no_grad():
            for i, sample in enumerate(full, 1):
                h = model(sample["images"][None].to(device))[0, 0].cpu()
                out32[sample["sample_id"]] = h
                out16[sample["sample_id"]] = h.half()
                if i % 100 == 0 or i == len(full):
                    print(f"heatmaps {i}/{len(full)}", flush=True)
        art = Path(args.artifacts)
        art.mkdir(parents=True, exist_ok=True)
        torch.save(out16, art / f"heatmaps_loyo{args.year}.pt")
        torch.save(out32, art / f"heatmaps_loyo{args.year}_fp32.pt")
        print("saved heatmaps for", len(out32))

    print("done", run, flush=True)


if __name__ == "__main__":
    main()
