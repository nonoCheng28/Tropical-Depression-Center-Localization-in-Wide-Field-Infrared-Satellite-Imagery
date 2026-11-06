"""D variant with a wider receptive field and a detection loss (screening trainer).

Why: the frozen D encoder is three convolutions (16, 32, 64 channels, no
normalisation). At the 80x80 heatmap its receptive field is about 11 deg, so each
output cell decides "is this a centre" from an 11 deg window of a 70 deg field and
cannot compare one cloud system with another. Measured consequence on the eight
LOYO folds: the heatmap has eight comparable peaks spread ~20 deg apart, they are
ranked wrongly 56% of the time, and candidate 0 moves 13.3 deg between two 6 h
frames while the true centre moves 1.0 deg. The weighted-MSE objective (alpha=100
against a background of 6400 cells) also tolerates those secondary peaks.

This variant changes exactly two things, keeping the published optimisation
settings (AdamW 1e-3 / wd 1e-4, effective batch 16, seed 42, 44 epochs, no early
stopping, deterministic):

  * a six-layer GroupNorm encoder with dilations 1,2,4,2 (receptive field ~27 deg)
    plus a global average-pooled context branch added back to every cell;
  * a CenterNet penalty-reduced focal loss on the same 80x80 Gaussian target,
    which suppresses secondary peaks instead of averaging them in.

The ConvLSTM is unchanged so the temporal claim of the original model still holds.

    python experiments/d_improve/train_d_v2.py --year 2019 --loss focal --arch v2
    python experiments/d_improve/train_d_v2.py --year 2019 --loss wmse  --arch v2
"""
import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from td_center.reproduce.train_loyo_fold import FoldDataset, FROZEN, MANIFEST, REPO, git, read_ids

from td_center.ConvLSTM import ConvLSTM
from td_center.dataloader import DATA_ROOT


def cbr(cin, cout, stride=1, dilation=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=dilation,
                                   dilation=dilation, bias=False),
                         nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


class EncoderV2(nn.Module):
    """320x320 -> 80x80x64, receptive field ~27 deg, plus global context."""

    def __init__(self, widths=(32, 48, 64, 64, 96, 64), global_ctx=True, dropout=0.0,
                 in_channels=1):
        super().__init__()
        w = widths
        self.body = nn.Sequential(
            cbr(in_channels, w[0], stride=2),  # 160
            cbr(w[0], w[1], stride=2),       # 80
            cbr(w[1], w[2], dilation=1),
            cbr(w[2], w[3], dilation=2),
            cbr(w[3], w[4], dilation=4),
            cbr(w[4], w[5], dilation=2),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.global_ctx = global_ctx
        if global_ctx:
            self.gpool = nn.Sequential(nn.Linear(w[5], w[5]), nn.ReLU(inplace=True),
                                       nn.Linear(w[5], w[5]))

    def forward(self, x):
        f = self.body(x)
        if self.global_ctx:
            g = self.gpool(f.mean(dim=(2, 3)))
            f = f + g[:, :, None, None]
        return self.drop(f)


class DModelV2(nn.Module):
    def __init__(self, global_ctx=True, dropout=0.0, in_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.encoder = EncoderV2(global_ctx=global_ctx, dropout=dropout,
                                 in_channels=in_channels)
        self.temporal = ConvLSTM(input_channels=64, hidden_channels=64, kernel_size=3)
        self.head = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(32, 1, 1))

    def forward(self, images):
        b, t, c, h, w = images.shape
        if (t, c, h, w) != (7, self.in_channels, 320, 320):
            raise ValueError(f"expects [B,7,{self.in_channels},320,320]")
        feat = self.encoder(images.reshape(b * t, c, h, w))
        _, fc, fh, fw = feat.shape
        seq = feat.reshape(b, t, fc, fh, fw)
        last = self.temporal(seq)
        return self.head(last)


def focal_loss(logits, target):
    """CenterNet penalty-reduced focal loss; one positive cell per sample."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    peak = target.amax(dim=(1, 2, 3), keepdim=True)
    pos = (target >= peak - 1e-6).float()
    pos_loss = -((1 - p) ** 2) * torch.log(p) * pos
    neg_loss = -((1 - target) ** 4) * (p ** 2) * torch.log(1 - p) * (1 - pos)
    npos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / npos


def wmse_loss(logits, target, alpha=100.0):
    weight = 1.0 + alpha * target
    return (weight * (logits - target) ** 2).mean()


class MultiSpanDataset(torch.utils.data.Dataset):
    """Training dataset with temporal-span diversity and label-safe augmentation.

    The frozen dataset has three temporal samplings of the same 6-hourly samples
    (`Multiple_frames_6h/12h/18h`, frames every 1/2/3 h). Drawing the span at
    random per item triples the number of temporal contexts seen for the same 129
    training storms without touching the label set, and mirror/shift/intensity
    jitter add the rest. The held-out year is always evaluated on the 6 h span, so
    the comparison with the frozen protocol stays on identical inputs.

    Mirrors `td_center.dataloader.IRDataset`: chronological flip, 320x320
    antialiased resize, `make_gaussian_heatmap(sigma=2.0, 80x80)`.
    """

    def __init__(self, data_root, ids, spans=("6h", "12h", "18h"), augment=True,
                 max_shift=16, sigma=2.0, temporal_p=0.2, split="train",
                 rotate_deg=0.0, scale_jitter=0.0):
        from pathlib import Path as _P
        self.root = _P(data_root)
        self.ids = list(ids)
        self.spans = tuple(spans)
        self.augment = augment
        self.max_shift = max_shift
        self.sigma = sigma
        self.temporal_p = temporal_p
        self.split = split
        self.rotate_deg = float(rotate_deg)
        self.scale_jitter = float(scale_jitter)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sid = self.ids[index]
        span = self.spans[int(torch.randint(len(self.spans), (1,)))]
        arr = np.load(self.root / f"Multiple_frames_{span}" / self.split / "images" / f"{sid}.npy",
                      allow_pickle=False)
        if arr.shape != (7, 640, 640):
            raise ValueError((sid, arr.shape))
        images = torch.from_numpy(arr).float().flip(0).unsqueeze(1)
        images = F.interpolate(images, size=(320, 320), mode="bilinear",
                               align_corners=False, antialias=True)
        label = (self.root / "single_frame" / "ir" / self.split / "labels" / "TD_center"
                 / "yolo" / f"{sid}.txt")
        x, y = map(float, label.read_text(encoding="utf-8").split())
        center = torch.tensor([x, y], dtype=torch.float32)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                images = images.flip(-1)
                center[0] = 1.0 - center[0]
            dx = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)))
            dy = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)))
            new_x = float(center[0]) + dx / 320.0
            new_y = float(center[1]) + dy / 320.0
            if not (0.0 <= new_x <= 1.0):
                dx, new_x = -dx, float(center[0]) - dx / 320.0
            if not (0.0 <= new_y <= 1.0):
                dy, new_y = -dy, float(center[1]) - dy / 320.0
            center[0] = min(max(new_x, 0.0), 1.0)
            center[1] = min(max(new_y, 0.0), 1.0)
            theta = torch.tensor([[[1.0, 0.0, -2.0 * dx / 320.0],
                                   [0.0, 1.0, -2.0 * dy / 320.0]]])
            theta = theta.repeat(images.shape[0], 1, 1)
            grid = F.affine_grid(theta, images.shape, align_corners=False)
            images = F.grid_sample(images, grid, mode="bilinear",
                                   padding_mode="border", align_corners=False)
            gain = 1.0 + (torch.rand(1).item() * 0.4 - 0.2)
            bias = torch.rand(1).item() * 0.2 - 0.1
            images = (images * gain + bias).clamp(0.0, 1.0)
            if torch.rand(1).item() < self.temporal_p:
                t = int(torch.randint(0, 7, (1,)))
                src = min(max(t + (1 if torch.rand(1).item() < 0.5 else -1), 0), 6)
                images[t] = images[src]
            if self.rotate_deg > 0.0 or self.scale_jitter > 0.0:
                # rotate/zoom the whole stack about the field centre; the label is
                # transformed by the same s*R(theta) and the target is rebuilt from it
                th = (torch.rand(1).item() * 2 - 1) * self.rotate_deg * np.pi / 180.0
                sc = 1.0 + (torch.rand(1).item() * 2 - 1) * self.scale_jitter
                c, s_ = float(np.cos(th)), float(np.sin(th))
                u, v = 2.0 * float(center[0]) - 1.0, 2.0 * float(center[1]) - 1.0
                u2 = sc * (u * c - v * s_)
                v2 = sc * (u * s_ + v * c)
                if -0.98 <= u2 <= 0.98 and -0.98 <= v2 <= 0.98:
                    theta = torch.tensor([[[c / sc, s_ / sc, 0.0],
                                           [-s_ / sc, c / sc, 0.0]]])
                    theta = theta.repeat(images.shape[0], 1, 1)
                    grid = F.affine_grid(theta, images.shape, align_corners=False)
                    images = F.grid_sample(images, grid, mode="bilinear",
                                           padding_mode="border", align_corners=False)
                    center = torch.tensor([(u2 + 1.0) / 2.0, (v2 + 1.0) / 2.0],
                                          dtype=torch.float32)

        from td_center.heatmap import make_gaussian_heatmap
        return {"images": images, "center": center,
                "heatmap": make_gaussian_heatmap(center, size=(80, 80), sigma=self.sigma),
                "sample_id": sid}


class AugFoldDataset(torch.utils.data.Dataset):
    """FoldDataset plus label-safe augmentation (mirror, shift, intensity jitter).

    The frozen protocol forbids augmentation for the *baseline* comparison, and
    that comparison is kept. This variant exists because the v2 screening runs
    showed the model fitting the 753 training frames almost perfectly (focal loss
    0.07) while the held-out year stayed at 11-13 deg: the binding constraint is
    141 training storms, not capacity. Mirroring and a shift of at most 5 deg are
    physically admissible for a full-disk IR field, and the Gaussian target is
    regenerated from the transformed centre so image and label stay consistent.
    """

    def __init__(self, base, augment=True, max_shift=16, sigma=2.0):
        self.base = base
        self.augment = augment
        self.max_shift = max_shift
        self.sigma = sigma

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        images = sample["images"]
        center = sample["center"].clone()
        if self.augment:
            if torch.rand(1).item() < 0.5:
                images = images.flip(-1)
                center[0] = 1.0 - center[0]
            dx = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)))
            dy = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)))
            new_x = float(center[0]) + dx / 320.0
            new_y = float(center[1]) + dy / 320.0
            if not (0.0 <= new_x <= 1.0):
                dx = -dx
                new_x = float(center[0]) + dx / 320.0
            if not (0.0 <= new_y <= 1.0):
                dy = -dy
                new_y = float(center[1]) + dy / 320.0
            center[0] = min(max(new_x, 0.0), 1.0)
            center[1] = min(max(new_y, 0.0), 1.0)
            theta = torch.tensor([[[1.0, 0.0, -2.0 * dx / 320.0],
                                   [0.0, 1.0, -2.0 * dy / 320.0]]])
            theta = theta.repeat(images.shape[0], 1, 1)
            grid = F.affine_grid(theta, images.shape, align_corners=False)
            images = F.grid_sample(images, grid, mode="bilinear",
                                   padding_mode="border", align_corners=False)
            gain = 1.0 + (torch.rand(1).item() * 0.4 - 0.2)
            bias = torch.rand(1).item() * 0.2 - 0.1
            images = (images * gain + bias).clamp(0.0, 1.0)
        from td_center.heatmap import make_gaussian_heatmap
        return {"images": images, "center": center,
                "heatmap": make_gaussian_heatmap(center, size=(80, 80), sigma=self.sigma),
                "sample_id": sample["sample_id"]}


@torch.no_grad()
def metrics(model, loader, device):
    model.eval()
    ds, hits = [], 0
    for batch in loader:
        images = batch["images"].to(device)
        pred = model(images)
        b = pred.shape[0]
        flat = pred.reshape(b, -1).argmax(1)
        rows = flat // pred.shape[-1]
        cols = flat % pred.shape[-1]
        pred_xy = torch.stack([(cols + 0.5) / pred.shape[-1],
                               (rows + 0.5) / pred.shape[-2]], dim=1)
        d = torch.linalg.vector_norm((pred_xy.cpu() - batch["center"]) * 70.0, dim=1)
        ds.extend(d.tolist())
        hits += int((d <= 3).sum())
    a = np.asarray(ds)
    return {"MD": float(a.mean()), "DR3": float((a <= 3).mean()), "hits": hits, "n": len(a)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", required=True,
                    help="held-out training year, or 'all' to train on the whole frozen list")
    ap.add_argument("--arch", choices=("v2",), default="v2")
    ap.add_argument("--loss", choices=("focal", "wmse"), default="focal")
    ap.add_argument("--epochs", type=int, default=44)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dump-heatmaps", action="store_true", default=True)
    ap.add_argument("--no-heatmaps", dest="dump_heatmaps", action="store_false")
    ap.add_argument("--max-shift", type=int, default=16, help="max augmentation shift in 320-px units")
    ap.add_argument("--augment", action="store_true",
                    help="mirror + shift(deg) + intensity jitter; the frozen baseline keeps none")
    ap.add_argument("--spans", default="6h",
                    help="comma list of temporal spans used for training augmentation, e.g. 6h,12h,18h")
    ap.add_argument("--dropout", type=float, default=0.0, help="Dropout2d on the encoder output")
    ap.add_argument("--rotate-deg", type=float, default=0.0,
                    help="max random rotation (deg) about the field centre, 0 disables")
    ap.add_argument("--scale-jitter", type=float, default=0.0,
                    help="max relative zoom, e.g. 0.1 for 0.9-1.1, 0 disables")
    ap.add_argument("--tag", default="d_v2")
    args = ap.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    frozen = read_ids(FROZEN / "train.txt")
    global_run = str(args.year).lower() == "all"
    if global_run:
        train_ids, eval_ids = frozen, []
    else:
        train_ids = read_ids(MANIFEST / f"{args.year}_train.txt")
        eval_ids = read_ids(MANIFEST / f"{args.year}_eval.txt")
        assert sorted(train_ids + eval_ids) == sorted(frozen)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    run = REPO / "runs" / "d_improve" / f"{args.tag}_{args.year}_{stamp}"
    (run / "source").mkdir(parents=True)
    for name in ("train_d_v2.py",):
        (run / "source" / name).write_bytes((REPO / "experiments/d_improve" / name).read_bytes())
    cfg = vars(args).copy()
    cfg.update(commit=git("rev-parse", "HEAD"), git_status=git("status", "--short"),
               device=str(device), train_n=len(train_ids), eval_n=len(eval_ids),
               early_stopping="disabled (fixed budget)", test_used=False)
    (run / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    spans = tuple(s.strip() for s in args.spans.split(",") if s.strip())
    if args.augment or len(spans) > 1:
        train_ds = MultiSpanDataset(DATA_ROOT, train_ids, spans=spans,
                                    augment=args.augment, max_shift=args.max_shift,
                                    rotate_deg=args.rotate_deg,
                                    scale_jitter=args.scale_jitter)
    else:
        train_ds = FoldDataset(DATA_ROOT, train_ids)
    eval_ds = FoldDataset(DATA_ROOT, eval_ids) if eval_ids else None
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.micro_batch, shuffle=True,
                              generator=g, num_workers=0)
    eval_loader = (DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=0)
                   if eval_ds is not None else None)

    model = DModelV2(dropout=args.dropout).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"fold {args.year}: train={len(train_ds)} "
          f"eval={0 if eval_ds is None else len(eval_ds)} params={n_par/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, seen = time.perf_counter(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            images = batch["images"].to(device)
            target = batch["heatmap"].to(device)
            logits = model(images)
            loss = (focal_loss(logits, target) if args.loss == "focal"
                    else wmse_loss(logits, target, args.alpha))
            (loss / args.accum).backward()
            if step % args.accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
            total += float(loss.detach()) * len(images); seen += len(images)
        m = (metrics(model, eval_loader, device) if eval_loader is not None else
             {"MD": float("nan"), "DR3": float("nan"), "hits": 0, "n": 0})
        history.append({"epoch": epoch, "loss": total / max(seen, 1), **m,
                        "seconds": time.perf_counter() - t0})
        print(f"epoch {epoch:>3}: loss={total/max(seen,1):.6f} heldout MD={m['MD']:.4f} "
              f"DR={m['DR3']:.3f} ({history[-1]['seconds']:.1f}s)", flush=True)
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "config": cfg,
                    "metrics": m}, run / f"epoch{epoch}.pt")

    mds = [h["MD"] for h in history]
    finite = [v for v in mds if v == v]
    (run / "result.json").write_text(json.dumps({
        "fold_year": args.year, "fixed_epochs": args.epochs, "loss": args.loss,
        "final": history[-1],
        "min_heldout_MD_diagnostic": float(min(finite)) if finite else None,
        "best_epoch_diagnostic_only": (int(np.argmin(mds) + 1) if finite else None),
        "curve": [(h["epoch"], round(h["MD"], 4)) for h in history],
        "parameters": n_par,
    }, indent=2), encoding="utf-8")

    if args.dump_heatmaps:
        from td_center.dataloader import IRDataset
        model.eval()
        splits = (["val", "test"] if global_run else ["train"])
        for split in splits:
            full = IRDataset(DATA_ROOT, FROZEN, split)
            out32, out16 = {}, {}
            with torch.no_grad():
                for i, sample in enumerate(full, 1):
                    h = model(sample["images"][None].to(device))[0, 0].cpu()
                    out32[sample["sample_id"]] = h
                    out16[sample["sample_id"]] = h.half()
                    if i % 50 == 0 or i == len(full):
                        print(f"heatmaps {split} {i}/{len(full)}", flush=True)
            art = REPO / "outputs" / "artifacts" / f"{args.tag}{args.year}"
            art.mkdir(parents=True, exist_ok=True)
            torch.save(out16, art / f"heatmaps_{split}.pt")
            torch.save(out32, art / f"heatmaps_{split}_fp32.pt")
            print("saved heatmaps to", art, split)
    print("done", run, flush=True)


if __name__ == "__main__":
    main()
