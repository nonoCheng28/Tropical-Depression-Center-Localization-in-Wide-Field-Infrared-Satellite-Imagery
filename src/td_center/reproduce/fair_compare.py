"""Fair A/B/C/D comparison on validation 2024 only. Test is never read.

Every arm is trained by `scripts/train_d.py` with one flag changed (`--model`),
so the loss, optimiser, schedule, seed, decoder, manifest and metric code are
literally the same code path. This script only reads finished run directories.

    python experiments/d_improve/fair_compare.py \
        --arm b_png=runs/b_png_formal_... \
        --arm c_baseline=runs/c_baseline_formal_...

`--arm d` (default) points at the frozen first-round D run.  Per-storm paired
bootstrap resamples the 19 validation storms, not the 94 frames, because frames
inside a storm are not independent (LEEPI alone is 14 of 94 frames).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
D_RUN = "d_formal_20261019_220943_751483"
PROTOCOL_KEYS = ("loss", "loss_alpha", "learning_rate", "weight_decay", "max_epochs", "patience",
                 "scheduler", "heatmap_size", "sigma", "input_size", "augmentation",
                 "decoder", "metrics", "effective_batch_size", "optimizer")


def read_predictions(run):
    rows = list(csv.DictReader((run / "best_val_predictions.csv").open(encoding="utf-8")))
    for row in rows:
        for key in ("pred_lon", "true_lon", "pred_lat", "true_lat", "distance_deg"):
            row[key] = float(row[key])
    return rows


def frame_arrays(rows):
    lon = np.array([r["pred_lon"] - r["true_lon"] for r in rows])
    lat = np.array([r["pred_lat"] - r["true_lat"] for r in rows])
    d = np.array([r["distance_deg"] for r in rows])
    return lon, lat, d


def summary(rows):
    lon, lat, d = frame_arrays(rows)
    storm = defaultdict(list)
    for row, value in zip(rows, d):
        storm[storm_key(row["sample_id"])].append(value)
    per_storm = {k: float(np.mean(v)) for k, v in storm.items()}
    worst = sorted(zip(d, [r["sample_id"] for r in rows]), reverse=True)[:3]
    return {
        "n": int(d.size), "storms": len(storm),
        "MD": float(d.mean()),
        "MAE": float(np.abs(np.stack([lon, lat])).mean()),
        "RMSE": float(np.sqrt((np.stack([lon, lat]) ** 2).mean())),
        "DR3": float((d <= 3).mean()),
        "median": float(np.median(d)), "P90": float(np.percentile(d, 90)),
        "storms_full": int(sum(v <= 3 for v in per_storm.values())),
        "storms_over10": int(sum(v > 10 for v in per_storm.values())),
        "per_storm": per_storm, "worst": [(float(v), s) for v, s in worst],
    }


def storm_key(sample_id):
    parts = sample_id.split("_")
    return f"{parts[0]}_{parts[1]}"


def bootstrap_md(rows_a, rows_b, storms, draws=10000, seed=0):
    """Paired storm bootstrap of MD(A) - MD(B); storms are resampled, not frames."""
    index = defaultdict(lambda: defaultdict(list))
    for tag, rows in (("a", rows_a), ("b", rows_b)):
        for row in rows:
            index[tag][storm_key(row["sample_id"])].append(row["distance_deg"])
    usable = [s for s in storms if s in index["a"] and s in index["b"]]
    a = {s: np.array(index["a"][s]) for s in usable}
    b = {s: np.array(index["b"][s]) for s in usable}
    rng = np.random.default_rng(seed)
    deltas = np.empty(draws)
    for i in range(draws):
        pick = rng.integers(0, len(usable), len(usable))
        chosen = [usable[j] for j in pick]
        deltas[i] = (np.concatenate([a[s] for s in chosen]).mean()
                     - np.concatenate([b[s] for s in chosen]).mean())
    point = float(np.concatenate(list(a.values())).mean() - np.concatenate(list(b.values())).mean())
    share = min((deltas > 0).mean(), (deltas < 0).mean())
    return {"delta": point, "lo": float(np.percentile(deltas, 2.5)),
            "hi": float(np.percentile(deltas, 97.5)), "p_two_sided": float(min(1.0, 2 * share)),
            "storms": len(usable)}


def paired_storm_stats(rows_a, rows_b):
    """Per-storm paired differences: robust to one storm dominating the pooled mean."""
    per = {}
    for tag, rows in (("a", rows_a), ("b", rows_b)):
        agg = defaultdict(list)
        for row in rows:
            agg[storm_key(row["sample_id"])].append(row["distance_deg"])
        per[tag] = {k: float(np.mean(v)) for k, v in agg.items()}
    storms = sorted(set(per["a"]) & set(per["b"]))
    deltas = np.array([per["a"][s] - per["b"][s] for s in storms])
    better = int((deltas < 0).sum())
    worse = int((deltas > 0).sum())
    n = better + worse
    from math import comb
    k = min(better, worse)
    sign_p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"storms": len(storms), "a_better": better, "b_better": worse,
            "mean_delta": float(deltas.mean()), "median_delta": float(np.median(deltas)),
            "sign_p": float(sign_p)}


def protocol_diff(base, other):
    out = {}
    for key in PROTOCOL_KEYS:
        if base.get(key) != other.get(key):
            out[key] = {"reference": base.get(key), "arm": other.get(key)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="LABEL=RUNDIR")
    ap.add_argument("--json", type=Path, default=REPO / "results" / "fair_compare_val.json")
    ap.add_argument("--md", type=Path, default=REPO / "results" / "fair_compare_val.md")
    args = ap.parse_args()

    arms = {}
    for item in args.arm:
        label, _, path = item.partition("=")
        arms[label] = (REPO / path).resolve() if not Path(path).is_absolute() else Path(path)
    missing = [label for label, path in arms.items() if not (path / "best_val_predictions.csv").exists()]
    for label in missing:
        arms.pop(label)
    if not arms:
        raise SystemExit("no completed arms found")

    reference = json.loads((arms["d_convlstm"] / "config.json").read_text(encoding="utf-8-sig")) \
        if "d_convlstm" in arms else {}
    table, payload = {}, {}
    for label, run in arms.items():
        rows = read_predictions(run)
        stats = summary(rows)
        config = json.loads((run / "config.json").read_text(encoding="utf-8-sig"))
        run_summary = json.loads((run / "summary.json").read_text(encoding="utf-8")) \
            if (run / "summary.json").exists() else {}
        parameters = run_summary.get("parameter_count")
        if parameters is None and (run / "best.pt").exists():
            import torch
            state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
            parameters = sum(v.numel() for v in state["model_state_dict"].values())
            del state
        stats.update(
            run=run.name,
            seed=config.get("seed"),
            model=config.get("model"),
            model_kind=config.get("model_kind", "d"),
            cnn_variant=config.get("cnn_variant"),
            cnn_in_channels=config.get("cnn_in_channels", 7),
            parameters=parameters,
            best_epoch=run_summary.get("best_epoch"),
            completed_epochs=run_summary.get("completed_epochs"),
            stop_reason=run_summary.get("stop_reason"),
            minutes=round((run_summary.get("elapsed_seconds") or 0) / 60, 1),
            protocol_diff=protocol_diff(reference, config) if reference else {})
        table[label] = stats

    common = set.intersection(*[set(t["per_storm"]) for t in table.values()])
    common = sorted(common)
    pairs = {}
    labels = list(table)
    for i, first in enumerate(labels):
        for second in labels[i + 1:]:
            rows_first, rows_second = read_predictions(arms[first]), read_predictions(arms[second])
            value = bootstrap_md(rows_first, rows_second, common)
            value.update(paired_storm_stats(rows_first, rows_second))
            pairs[f"{first} - {second}"] = value

    order = ["seed", "MD", "MAE", "RMSE", "DR3", "median", "P90", "storms_full", "storms_over10",
             "parameters", "best_epoch", "completed_epochs", "minutes"]
    lines = ["| arm | " + " | ".join(order) + " |", "|---" + "|---:" * len(order) + "|"]
    for label, stats in table.items():
        cells = []
        for key in order:
            value = stats.get(key)
            if key == "DR3":
                cells.append(f"{100 * value:.2f}%")
            elif key == "parameters":
                cells.append(f"{value:,}" if value else "-")
            elif isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    markdown = "\n".join(lines)

    groups = {}
    for label, stats in table.items():
        groups.setdefault((stats["model"] or label, stats["parameters"]), []).append(label)
    group_rows = []
    for (signature, params), labels in groups.items():
        mds = [table[label]["MD"] for label in labels]
        drs = [table[label]["DR3"] for label in labels]
        group_rows.append((float(np.mean(mds)), signature, params, labels,
                           float(np.mean(mds)), min(mds), max(mds),
                           float(np.mean(drs)), min(drs), max(drs)))
    group_rows.sort()
    group_lines = ["", "| model | params | seeds | MD mean | MD range | DR3 mean | DR3 range |",
                   "|---|---:|---:|---:|---|---:|---|"]
    for _, signature, params, labels, md_mean, md_lo, md_hi, dr_mean, dr_lo, dr_hi in group_rows:
        group_lines.append(
            f"| {signature} | {params:,} | {len(labels)} | {md_mean:.4f} | "
            f"{md_lo:.4f}–{md_hi:.4f} | {100 * dr_mean:.2f}% | {100 * dr_lo:.2f}–{100 * dr_hi:.2f}% |")
    groups_payload = {signature: {"params": params, "runs": labels, "MD_mean": md_mean,
                                  "MD_range": [md_lo, md_hi], "DR3_mean": dr_mean}
                      for _, signature, params, labels, md_mean, md_lo, md_hi, dr_mean, _, _ in group_rows}

    worst_lines = ["", "| arm | worst frame | 2nd | 3rd |", "|---|---|---|---|"]
    for label, stats in table.items():
        worst_lines.append(f"| {label} | " + " | ".join(
            f"{v:.2f}° {s.split('_')[1]}" for v, s in stats["worst"]) + " |")

    pair_lines = ["", "| comparison | ΔMD (deg) | 95% storm-bootstrap CI | p bootstrap "
                      "| storms better (first/second) | median storm Δ | p sign |",
                  "|---|---:|---|---:|---:|---:|---:|"]
    for name, value in pairs.items():
        pair_lines.append(f"| {name} | {value['delta']:+.4f} | "
                          f"[{value['lo']:+.4f}, {value['hi']:+.4f}] | {value['p_two_sided']:.3f} | "
                          f"{value.get('a_better', '-')}/{value.get('b_better', '-')} | "
                          f"{value.get('median_delta', float('nan')):+.4f} | "
                          f"{value.get('sign_p', float('nan')):.3f} |")

    per_storm_lines = ["", "| storm | frames | " + " | ".join(labels) + " |",
                       "|---|---:" + "|---:" * len(labels) + "|"]
    counts = defaultdict(int)
    for row in read_predictions(arms[labels[0]]):
        counts[storm_key(row["sample_id"])] += 1
    for storm in sorted(common, key=lambda s: -table[labels[0]]["per_storm"][s]):
        per_storm_lines.append(f"| {storm} | {counts[storm]} | " + " | ".join(
            f"{table[label]['per_storm'][storm]:.2f}" for label in labels) + " |")

    text = ("# Fair A/B/C/D comparison (validation 2024, 94 frames, 19 storms)\n\n"
            "Test 2025 is not read. Every arm is the same training entry point with one flag "
            "changed; see the protocol diff below.\n\n## Per run\n\n" + markdown + "\n"
            "\n## Per model (seed-to-seed spread)\n" + "\n".join(group_lines)
            + "\n".join(worst_lines) + "\n" + "\n".join(pair_lines) + "\n"
            + "\n## Per-storm mean distance\n" + "\n".join(per_storm_lines) + "\n")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(text, encoding="utf-8")

    for label, stats in table.items():
        payload[label] = {k: v for k, v in stats.items() if k != "per_storm"}
    payload["_pairs"] = pairs
    payload["_groups"] = groups_payload
    payload["_per_storm"] = {label: stats["per_storm"] for label, stats in table.items()}
    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
                         encoding="utf-8")
    print(markdown)
    print("\n".join(worst_lines[2:]))
    print("\n".join(pair_lines[2:]))
    for label, stats in table.items():
        if stats["protocol_diff"]:
            print(f"protocol diff {label}: {json.dumps(stats['protocol_diff'], ensure_ascii=False)}")
    print(f"\nwrote {args.md} and {args.json}")


if __name__ == "__main__":
    main()
