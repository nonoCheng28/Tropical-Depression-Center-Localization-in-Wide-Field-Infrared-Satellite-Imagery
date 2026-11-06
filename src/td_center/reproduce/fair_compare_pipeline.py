"""Compare emissions through the *same* mask + Viterbi decoder on validation 2024.

Every arm is produced by `predict_arm_heatmaps.py` -> `combine_holdout.py` (->
`deliver_aided.py`), so the decoder constants (8 peaks, 3 deg, sigma 2 cells,
alpha 1, kappa 0.003) are identical across arms and only the emission differs.
This script reads the `aided_detail_*.json` files those runs already wrote.

    python experiments/d_improve/fair_compare_pipeline.py \
        --arm d_single=aided_detail_pipe_fair_d_single_val_val.json \
        --arm b_png=aided_detail_bpipe_val_val.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO / "outputs" / "artifacts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="LABEL=DETAIL_FILE")
    ap.add_argument("--sigma", type=float, default=2.0, help="mask width to report; 2.0 is the delivered value")
    ap.add_argument("--alpha", type=float, default=1.0, help="mask amplitude to report as the aided row")
    ap.add_argument("--json", type=Path, default=REPO / "results" / "fair_compare_pipeline_val.json")
    ap.add_argument("--md", type=Path, default=REPO / "results" / "fair_compare_pipeline_val.md")
    args = ap.parse_args()

    arms = {}
    for item in args.arm:
        label, _, name = item.partition("=")
        payload = json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))
        rows = {(r["mode"], round(r["alpha"], 3), round(r.get("sigma", 2.0), 3)): r for r in payload["rows"]}
        arms[label] = {"artifact": payload.get("artifact"), "rows": rows,
                       "storm_md": {(r["mode"], round(r["alpha"], 3), round(r.get("sigma", 2.0), 3)): r["storm_md"]
                                    for r in payload["rows"]}}

    lines = [f"| arm | emission | IR-only (alpha 0) | +mask (alpha {args.alpha:g}, sigma "
             f"{args.sigma:g}) | DR@3 (aided) | storms <=3 / >10 |",
             "|---|---|---:|---:|---:|---|"]
    for label, arm in arms.items():
        base = arm["rows"].get(("real", 0.0, args.sigma)) or arm["rows"].get(("real", 0.0, 2.0))
        aided = arm["rows"].get(("real", args.alpha, args.sigma))
        base_md = f"{base['MD']:.3f}" if base else "-"
        lines.append(f"| {label} | {arm['artifact']} | {base_md} | {aided['MD']:.3f} | "
                     f"{100 * aided['DR3']:.1f}% | {aided['storms_full']}/{aided['storms']} / "
                     f"{aided['storms_over10']}/{aided['storms']} |")
    table = "\n".join(lines)

    storms = sorted(set.intersection(*[set(arm["storm_md"][("real", args.alpha, args.sigma)]) for arm in arms.values()]))
    storms.sort(key=lambda s: -arms[list(arms)[0]]["storm_md"][("real", args.alpha, args.sigma)][s])
    storm_lines = ["", "| storm | " + " | ".join(arms) + " |",
                   "|---" + "|---:" * len(arms) + "|"]
    for storm in storms:
        storm_lines.append(f"| {storm} | " + " | ".join(
            f"{arm['storm_md'][('real', args.alpha, args.sigma)][storm]:.2f}" for arm in arms.values()) + " |")

    text = ("# Same decoder, different emissions (validation 2024, alpha = best-track mask)\n\n"
            "Test 2025 is not read. All arms go through `deliver_aided.py` unchanged.\n\n"
            + table + "\n\n## Per-storm mean distance at alpha = 1\n" + "\n".join(storm_lines) + "\n")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(text, encoding="utf-8")
    payload = {label: {f"alpha{alpha:g}_sigma{sigma:g}": {k: v for k, v in row.items() if k != "storm_md"}
                       for (mode, alpha, sigma), row in arm["rows"].items() if mode == "real"}
               for label, arm in arms.items()}
    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(table)
    print("\n".join(storm_lines))


if __name__ == "__main__":
    main()
