"""D-model first-round training; test data is deliberately not exposed here."""
import argparse
import csv
import hashlib
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
from torch.utils.data import DataLoader, Subset

from td_center.dataloader import DATA_ROOT, MANIFEST_DIR, REPO_ROOT, IRDataset, SingleFrameIRDataset
from td_center.heatmap import decode_argmax
from td_center.losses import weighted_mse_per_sample, hard_negative_per_sample, expected_distance_per_sample
from td_center.metrics import center_metrics, centers_to_lonlat
from td_center.model_d import DModel


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_digest(path):
    # Git may change CRLF to LF on another machine; sample IDs/order stay frozen.
    canonical = "\n".join(path.read_text(encoding="utf-8-sig").splitlines()) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_output(*args):
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8")
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def split_sources(config):
    """Return {split: (manifest_path, data_directory_split, expected_count_or_None)}.

    The frozen delivered split reads `splits/single_center_v1/{train,val}.txt` from the
    `train`/`val` image folders. A leave-one-year-out fold reads `splits/loyo/{year}_train.txt`
    and `{year}_eval.txt`, and both draw images from the `train` folder, because LOYO holds out
    a training year, not the 2024 validation year.
    """
    fold = config.get("fold_year")
    if fold:
        root = Path(config.get("manifest_dir") or (REPO_ROOT / "splits" / "loyo"))
        return {"train": (root / f"{fold}_train.txt", "train", None),
                "val": (root / f"{fold}_eval.txt", "train", None)}
    return {"train": (MANIFEST_DIR / "train.txt", "train", 805),
            "val": (MANIFEST_DIR / "val.txt", "val", 94)}


def preflight(data_root, config):
    """Inspect only train/val headers and labels; never load test observations."""
    datasets, audit = {}, {}
    sources = split_sources(config)
    for split in ("train", "val"):
        manifest, data_split, count = sources[split]
        if config.get("fold_year") is None:
            if manifest_digest(manifest) != config["manifest_sha256"][split]:
                raise ValueError(f"Frozen manifest hash mismatch: {split}")
        ds = IRDataset(data_root, MANIFEST_DIR, data_split, span=config.get("span", "6h"),
                       manifest_path=manifest)
        n = len(ds)
        if (count is not None and n != count) or len(set(ds.sample_ids)) != n:
            raise ValueError(f"Invalid {split} count or duplicate IDs")
        max_geo_error = 0.
        for sid in ds.sample_ids:
            array = np.load(ds.base / "images" / f"{sid}.npy", mmap_mode="r", allow_pickle=False)
            if array.shape != (7, 640, 640) or array.dtype != np.float32:
                raise ValueError(f"Invalid array header: {sid}")
            label = ds.base / "labels" / "TD_center" / "yolo" / f"{sid}.txt"
            lines = [line.split() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(lines) != 1 or len(lines[0]) != 2:
                raise ValueError(f"Not a single-center label: {sid}")
            xy = torch.tensor([[float(v) for v in lines[0]]], dtype=torch.float64)
            actual = centers_to_lonlat(xy)[0]
            lat, lon = map(float, sid.rsplit("_", 2)[1:])
            error = max(abs(actual[0].item() - lon), abs(actual[1].item() - lat))
            max_geo_error = max(max_geo_error, error)
            if error > 1e-3:
                raise ValueError(f"Label/geography mismatch: {sid}, error={error}")
        datasets[split] = ds
        audit[split] = {"count": n, "manifest": str(manifest),
                        "max_label_filename_error_deg": max_geo_error}
        print(f"Preflight {split}: {n} headers/labels checked; coordinate error <= {max_geo_error:.8f} deg", flush=True)
    train_storms = {s.split("_")[0] for s in datasets["train"].sample_ids}
    val_storms = {s.split("_")[0] for s in datasets["val"].sample_ids}
    if train_storms & val_storms:
        raise ValueError("Train/validation storm overlap")
    return datasets, audit


def train_epoch(model, loader, optimizer, device, effective_batch=16, alpha=100., hard_negative=None, expected_distance=None):
    """Sum per-sample losses, then divide gradients by actual window size.

    The final incomplete window is retained (805 = 50*16 + 5), never dropped
    or incorrectly divided by 16. This also works with micro-batches >1.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, seen, window, updates = 0., 0, 0, 0
    base_total, hard_total, distance_total = 0., 0., 0.
    for index, batch in enumerate(loader):
        prediction = model(batch["images"].to(device))
        losses = weighted_mse_per_sample(prediction, batch["heatmap"].to(device), alpha)
        base_total += losses.detach().sum().item()
        if hard_negative:
            hard = hard_negative_per_sample(prediction, batch['heatmap'].to(device), batch['center'].to(device),
                                            hard_negative['top_k'], hard_negative['radius_deg'])
            hard_total += hard.detach().sum().item()
            losses = losses + hard_negative['weight'] * hard
        if expected_distance:
            auxiliary = expected_distance_per_sample(prediction, batch['center'].to(device), expected_distance['temperature'])
            distance_total += auxiliary.detach().sum().item()
            losses = losses + expected_distance['weight'] * auxiliary
        if not torch.isfinite(losses).all():
            raise FloatingPointError("Nonfinite training loss")
        losses.sum().backward()
        n = len(losses)
        total += losses.detach().sum().item()
        seen += n
        window += n
        if window == effective_batch or index == len(loader) - 1:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(window)
                    if not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("Nonfinite gradient")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            window = 0
    result = {"train_weighted_mse": base_total / seen, "train_n": seen, "optimizer_updates": updates}
    if hard_negative:
        result.update(train_total_loss=total / seen, train_hard_negative=hard_total / seen)
    if expected_distance:
        result.update(train_total_loss=total / seen, train_expected_distance_deg=distance_total / seen,
                      train_weighted_distance=expected_distance['weight'] * distance_total / seen)
    return result


@torch.no_grad()
def evaluate(model, loader, device, alpha=100.):
    model.eval()
    predictions, targets, rows = [], [], []
    loss_sum, mse_sum = 0., 0.
    for batch in loader:
        output = model(batch["images"].to(device))
        target = batch["heatmap"].to(device)
        losses = weighted_mse_per_sample(output, target, alpha)
        if not torch.isfinite(output).all() or not torch.isfinite(losses).all():
            raise FloatingPointError("Nonfinite validation prediction/loss")
        pred = torch.stack([decode_argmax(h) for h in output]).cpu()
        truth = batch["center"]
        predictions.append(pred)
        targets.append(truth)
        loss_sum += losses.sum().item()
        mse_sum += (output - target).square().flatten(1).mean(1).sum().item()
        geo_pred, geo_truth = centers_to_lonlat(pred), centers_to_lonlat(truth)
        for i, sid in enumerate(batch["sample_id"]):
            distance = (geo_pred[i] - geo_truth[i]).norm().item()
            rows.append({"sample_id": sid, "pred_x": pred[i, 0].item(), "pred_y": pred[i, 1].item(),
                         "true_x": truth[i, 0].item(), "true_y": truth[i, 1].item(),
                         "pred_lon": geo_pred[i, 0].item(), "pred_lat": geo_pred[i, 1].item(),
                         "true_lon": geo_truth[i, 0].item(), "true_lat": geo_truth[i, 1].item(),
                         "distance_deg": distance, "detected_at_3deg": distance <= 3.})
    metrics = center_metrics(torch.cat(predictions), torch.cat(targets))
    metrics.update(val_weighted_mse=loss_sum / metrics["n"], val_mse=mse_sum / metrics["n"])
    return metrics, rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_samples(dataset, indices, path, rows=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup = {row["sample_id"]: row for row in rows or []}
    fig, axes = plt.subplots(1, len(indices), figsize=(5 * len(indices), 5), squeeze=False)
    for ax, index in zip(axes[0], indices):
        sample = dataset[index]
        image = sample["images"]
        image = image[0] if image.ndim == 3 else image[-1, 0]
        ax.imshow(image, cmap="gray", vmin=0, vmax=1, extent=(109, 179, -15, 55))
        lonlat = centers_to_lonlat(sample["center"][None])[0]
        ax.scatter(*lonlat.tolist(), c="cyan", marker="+", label="Label")
        row = lookup.get(sample["sample_id"])
        if row:
            ax.scatter(row["pred_lon"], row["pred_lat"], facecolors="none", edgecolors="red", label="Prediction")
        ax.set_title(sample["sample_id"].split("_")[1] + (f" / {row['distance_deg']:.2f} deg" if row else ""))
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_checkpoint(path, value):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


class CurrentFrameDataset(IRDataset):
    """Float32 current frame (disk index 0, time T) taken from the shared temporal arrays.

    Interface-control arm for B: the pixels and the rescaling are exactly those the
    first input of C/D sees, so only the number of frames differs.
    """

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        return {**sample, "images": sample["images"][-1]}


def build_datasets(data_root, config):
    """Return train/val datasets for the configured model kind.

    `preflight` always audits the frozen temporal arrays and labels; this function
    chooses what the model actually reads.  A/B/C/D share the same manifests, the
    same 320x320 resize, the same 80x80 sigma=2 target and the same loss.
    """
    kind = config.get("model_kind", "d")
    span = config.get("span", "6h")
    sources = split_sources(config)
    out = {}
    for split in ("train", "val"):
        manifest, data_split, count = sources[split]
        if kind in ("b", "tclnet"):
            if config.get("single_frame_source", "png") == "png":
                ds = SingleFrameIRDataset(data_root, MANIFEST_DIR, data_split,
                                          manifest_path=manifest)
            else:
                ds = CurrentFrameDataset(data_root, MANIFEST_DIR, data_split, span=span,
                                         manifest_path=manifest)
        elif kind == "c" and config.get("cnn_in_channels", 7) == 1:
            ds = CurrentFrameDataset(data_root, MANIFEST_DIR, data_split, span=span,
                                     manifest_path=manifest)
        else:
            ds = IRDataset(data_root, MANIFEST_DIR, data_split, span=span, manifest_path=manifest)
        n = len(ds)
        if (count is not None and n != count) or len(set(ds.sample_ids)) != n:
            raise ValueError(f"Invalid {split} count or duplicate IDs for model {kind}")
        out[split] = ds
    if kind in ("b", "tclnet") and config.get("single_frame_source", "png") == "png":
        missing = [sid for ds in out.values() for sid in ds.sample_ids
                   if not (ds.base / "images" / f"{sid}.png").exists()]
        if missing:
            raise ValueError(f"Missing single-frame PNGs: {missing[:5]}")
    return out


def build_model(config):
    """One factory for all three model kinds, so the training loop is literally shared."""
    kind = config.get("model_kind", "d")
    if kind == "d":
        return DModel(spatial_context=config.get("spatial_context", False))
    if kind == "b":
        from td_center.model_b import BModel
        return BModel()
    if kind == "c":
        from td_center.model_c import CModel
        return CModel(variant=config.get("cnn_variant", "baseline"),
                      in_channels=config.get("cnn_in_channels", 7))
    if kind == "tclnet":
        from td_center.model_tclnet import TCLNetModel
        return TCLNetModel()
    raise ValueError(f"Unknown model_kind: {kind}")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--micro-batch", type=int, choices=(1, 2, 4, 8, 16), default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--span", choices=("6h", "12h", "18h"), default="6h",
                        help="history span; 6h=1h, 12h=2h, 18h=3h sampling")
    parser.add_argument("--model", choices=("d", "b", "c", "tclnet"), default="d",
                        help="d=ConvLSTM temporal (default, unchanged); b=single-frame ResNet-18; "
                             "c=seven-frame channel stacking. All three use this same training loop, "
                             "loss, optimizer, schedule, seeds, decoder and frozen manifests.")
    parser.add_argument("--cnn-variant", choices=("baseline", "dilated", "wide", "wide2"),
                        default="baseline", help="C encoder width (td_center.model_c.CNN_VARIANTS)")
    parser.add_argument("--frames", type=int, choices=(1, 7), default=7,
                        help="C only: 7 = channel stacking (default); 1 = the same encoder and head "
                             "fed the float32 current frame, i.e. a capacity-matched single-frame "
                             "control that isolates frame count from architecture")
    parser.add_argument("--single-frame-source", choices=("png", "npy"), default="png",
                        help="B input: the delivered single-frame PNG interface (default), or the "
                             "float32 current frame of the temporal array (interface control)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the frozen seed (42) to measure seed-to-seed spread; "
                             "the protocol is otherwise unchanged")
    parser.add_argument("--fold-year", type=int, default=None,
                        help="Leave-one-year-out fold: train on splits/loyo/{year}_train.txt and "
                             "evaluate on {year}_eval.txt (both read the train image folder), no "
                             "early stopping and no checkpoint selection on the held-out year")
    parser.add_argument("--manifest-dir", type=Path, default=None,
                        help="manifest directory for --fold-year (default splits/loyo)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override the epoch budget (LOYO folds use 44, matching "
                             "train_d_v2.py --epochs 44)")
    parser.add_argument("--smoke", action="store_true", help="1 epoch, 17 train / 3 val; diagnostic only")
    parser.add_argument('--hard-negative', action='store_true', help='Fixed ablation: top64 beyond 7 degrees, weight 0.1')
    parser.add_argument("--evaluate", type=Path, help="Re-evaluate a local checkpoint on its recorded validation selection")
    parser.add_argument("--spatial-context", action="store_true", help="Fixed pooled spatial attention before original head")
    parser.add_argument('--expected-distance', action='store_true', help='A ablation: original MSE + 0.001 * expected distance in degrees, temperature 0.1')
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable in this PyTorch environment")
    config_path = REPO_ROOT / "configs" / "d_first_round.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    config["span"] = args.span
    config["input"] = f"Multiple_frames_{args.span}"
    config["delta_hours"] = {"6h": 1, "12h": 2, "18h": 3}[args.span]
    config["model_kind"] = args.model
    if args.fold_year:
        config["fold_year"] = args.fold_year
        config["manifest_dir"] = str((args.manifest_dir or (REPO_ROOT / "splits" / "loyo")).resolve())
        config["max_epochs"] = args.epochs or 44
        # The held-out year is the evaluation set, so it must not pick the checkpoint either:
        # run the full fixed budget and keep the final weights.
        config["patience"] = config["max_epochs"] + 1
        config["checkpoint_selection"] = "final_epoch_no_selection"
        print(f"LOYO fold {args.fold_year}: {config['max_epochs']} epochs, no early stopping, "
              f"final-epoch weights are the only ones used downstream", flush=True)
    elif args.epochs:
        config["max_epochs"] = args.epochs
    if args.seed is not None:
        config["seed"] = args.seed
        config["seed_override"] = True
    if args.model == "b":
        config["role"] = "baseline"
        config["single_frame_source"] = args.single_frame_source
        config["input"] = f"single_frame_ir_{args.single_frame_source}"
        config["model"] = "BModel ResNet-18 (torchvision, weights=None)"
    elif args.model == "c":
        config["cnn_variant"] = args.cnn_variant
        config["cnn_in_channels"] = args.frames
        config["input"] = (f"Multiple_frames_{args.span}_7channel_stack" if args.frames == 7
                           else f"Multiple_frames_{args.span}_current_frame_only")
        config["model"] = (f"CModel {'seven-frame channel stacking' if args.frames == 7 else 'single-frame control'}"
                           f" ({args.cnn_variant}, in_channels={args.frames})")
    elif args.model == "tclnet":
        config["role"] = "baseline"
        config["single_frame_source"] = args.single_frame_source
        config["input"] = f"single_frame_ir_{args.single_frame_source}"
        config["model"] = "TCLNet (Tan, IGARSS 2021) published topology, our protocol"
        config["external_baseline"] = ("github.com/chao-tan/TCLNet, MIT; input replicated x3 for "
                                       "the 3-channel stem; supervised with our 80x80 sigma=2 "
                                       "gaussian; not a reproduction of the paper's number")
    if args.expected_distance:
        if args.hard_negative or args.spatial_context:
            parser.error('Expected-distance A must be an independent loss ablation')
        config.update(expected_distance={'weight': .001, 'temperature': .1, 'units': 'planar degrees',
                      'definition': 'sum softmax(raw_heatmap/temperature) * distance(grid_center, target)'},
                      role='advanced', loss='weighted_mse_plus_expected_distance', experiment='A: weighted MSE plus expected distance',
                      baseline_run='d_formal_20261019_220943_751483')
    if args.hard_negative:
        config.update(hard_negative={'top_k': 64, 'radius_deg': 7., 'weight': .1,
                                     'definition': 'mean squared positive residual on top-k eligible pixels'},
                      experiment='D hard-negative single-variable ablation',
                      baseline_run='d_formal_20261019_220943_751483')
    if args.spatial_context:
        if args.hard_negative:
            parser.error("Spatial context must use the original weighted MSE only")
        config.update(spatial_context=True, role='advanced',
                      context_spec={'pool_size': 8, 'attention_channels': 16, 'initialization': 'zero_output_residual'},
                      experiment='D spatial-context single-variable ablation',
                      baseline_run='d_formal_20261019_220943_751483')
    checkpoint = None
    if args.evaluate:
        checkpoint = torch.load(args.evaluate, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    _, audit = preflight(args.data_root, config)
    datasets = build_datasets(args.data_root, config)
    smoke = config.get("smoke", False) if checkpoint else args.smoke
    selections = {}
    for split, n in (("train", 17), ("val", 3)):
        selections[split] = (random.Random(config["seed"]).sample(range(len(datasets[split])), n)
                             if smoke else list(range(len(datasets[split]))))
    if checkpoint:
        selections = checkpoint["selection_indices"]
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(Subset(datasets["train"], selections["train"]), batch_size=args.micro_batch,
                              shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(Subset(datasets["val"], selections["val"]), batch_size=args.micro_batch, shuffle=False, num_workers=0)
    model = build_model(config).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics, rows = evaluate(model, val_loader, device, config["loss_alpha"])
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out = args.evaluate.parent / f"reevaluation_{args.evaluate.stem}_{stamp}"
        out.mkdir()
        write_json(out / "metrics.json", metrics)
        write_csv(out / "predictions.csv", rows)
        print(json.dumps(metrics, indent=2), flush=True)
        return
    config.update(smoke=smoke, max_epochs=1 if smoke else config["max_epochs"],
                  micro_batch=args.micro_batch, accumulation_steps=16 // args.micro_batch,
                  data_root=str(args.data_root.resolve()), device=str(device), threads=args.threads)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    kind = config.get("model_kind", "d")
    if kind == "d":
        prefix = (f"d_{'expdist_' if config.get('expected_distance') else 'context_' if config.get('spatial_context') else 'hardneg_' if config.get('hard_negative') else ''}")
    elif kind == "b":
        prefix = "b_png_" if config.get("single_frame_source", "png") == "png" else "b_npy_"
    elif kind == "tclnet":
        prefix = ("tclnet_png_" if config.get("single_frame_source", "png") == "png"
                  else "tclnet_npy_")
    else:
        frames_tag = "1frame_" if config.get("cnn_in_channels", 7) == 1 else ""
        prefix = f"c_{frames_tag}{config.get('cnn_variant', 'baseline')}_"
    seed_tag = f"_s{config['seed']}" if config.get("seed_override") else ""
    fold_tag = f"loyo{config['fold_year']}_" if config.get("fold_year") else ""
    run = REPO_ROOT / "runs" / f"{prefix}{fold_tag}{'smoke' if smoke else 'formal'}{seed_tag}_{stamp}"
    run.mkdir(parents=True)
    print(f"Run: {run}\nDevice: {device}; train={len(train_loader.dataset)} val={len(val_loader.dataset)}", flush=True)
    write_json(run / "config.json", config)
    write_json(run / "data_audit.json", audit)
    write_json(run / "sample_ids.json", {s: [datasets[s].sample_ids[i] for i in selections[s]] for s in selections})
    source_paths = sorted((REPO_ROOT / "src" / "td_center").rglob("*.py")) + [config_path, REPO_ROOT / "README.md", REPO_ROOT / "pyproject.toml", REPO_ROOT / "scripts" / "train.py", MANIFEST_DIR / "train.txt", MANIFEST_DIR / "val.txt"]
    for source in source_paths:
        destination = run / "source" / source.relative_to(REPO_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    write_json(run / "environment.json", {"python": platform.python_version(), "platform": platform.platform(),
               "torch": str(torch.__version__), "numpy": np.__version__, "cuda": torch.version.cuda,
               "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
               "commit": git_output("rev-parse", "HEAD"), "git_status": git_output("status", "--short"),
               "source_sha256": {str(p.relative_to(REPO_ROOT)): sha256(p) for p in source_paths}})
    dependencies = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    (run / "pip_freeze.txt").write_text(dependencies.stdout if dependencies.returncode == 0 else dependencies.stderr, encoding="utf-8")
    plot_samples(datasets["train"], random.Random(42).sample(selections["train"], 3), run / "train_labels.png")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    history, best, best_epoch, stale, updates = [], math.inf, 0, 0, 0
    started = time.perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        epoch_started = time.perf_counter()
        train = train_epoch(model, train_loader, optimizer, device, 16, config["loss_alpha"], config.get('hard_negative'), config.get('expected_distance'))
        metrics, rows = evaluate(model, val_loader, device, config["loss_alpha"])
        updates += train["optimizer_updates"]
        improved = metrics["mean_distance_deg"] < best
        if improved:
            best, best_epoch, stale = metrics["mean_distance_deg"], epoch, 0
        else:
            stale += 1
        history.append({"epoch": epoch, **train, **metrics, "cumulative_updates": updates,
                        "seconds": time.perf_counter() - epoch_started})
        write_csv(run / "history.csv", history)
        state = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "config": config, "selection_indices": selections, "metrics": metrics,
                 "best_mean_distance_deg": best, "best_epoch": best_epoch, "stale_epochs": stale,
                 "optimizer_updates": updates, "torch_rng_state": torch.get_rng_state(),
                 "loader_rng_state": generator.get_state()}
        save_checkpoint(run / "last.pt", state)
        if improved:
            save_checkpoint(run / "best.pt", state)
            write_json(run / "best_metrics.json", metrics)
            write_csv(run / "best_val_predictions.csv", rows)
        print(f"Epoch {epoch}: train loss={train['train_weighted_mse']:.6f}; val MD={metrics['mean_distance_deg']:.4f} deg; DR={metrics['detection_rate_at_3deg']:.3f}; {history[-1]['seconds']:.1f}s", flush=True)
        if stale >= config["patience"]:
            break
    with (run / "best_val_predictions.csv").open(encoding="utf-8") as stream:
        best_rows = list(csv.DictReader(stream))
    for row in best_rows:
        for key in ("pred_lon", "pred_lat", "distance_deg"):
            row[key] = float(row[key])
    worst = sorted(best_rows, key=lambda r: r["distance_deg"], reverse=True)[:3]
    plot_samples(datasets["val"], [datasets["val"].sample_ids.index(r["sample_id"]) for r in worst], run / "best_val_worst.png", worst)
    write_json(run / "summary.json", {"role": config["role"], "smoke": smoke, "completed_epochs": len(history),
               "model_kind": kind, "cnn_variant": config.get("cnn_variant"),
               "fold_year": config.get("fold_year"),
               "checkpoint_selection": config.get("checkpoint_selection", "best_val_md"),
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "best_epoch": best_epoch, "best_mean_distance_deg": best, "test_evaluated": False,
               "elapsed_seconds": time.perf_counter() - started,
               "stop_reason": "early_stopping" if stale >= config["patience"] else "epoch_limit"})
    print(f"Completed: {run}", flush=True)
    if config.get('expected_distance') and not smoke:
        from td_center.distance_comparison import compare_saved
        compare_saved(REPO_ROOT / 'runs' / config['baseline_run'], run)
