"""Cross-year (leave-one-year-out) comparison of emissions through the same pipeline.

Reads the `aided_detail_loyo_*.json` files written by `deliver_aided.py --loyo`. Each
arm is a (emission, alpha, sigma) triple; the decoder constants are identical, so the
comparison isolates the emission. Selection uses these eight folds, never the 2024
validation year.

    python experiments/d_improve/fair_compare_loyo.py \
        --arm "ens3_delivered=aided_detail_loyo_alphsigma.json:_a1_s2"
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO / "outputs" / "artifacts"


def load(spec):
    name, _, suffix = spec.partition(":")
    payload = json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))
    folds = {int(k.split("_")[0]): v["MD"] for k, v in payload["folds"].items() if k.endswith(suffix)}
    row = next(r for r in payload["rows"]
               if f"_a{r['alpha']:g}_s{r.get('sigma', 2.0):g}" == suffix)
    return {"folds": folds, "MD": row["MD"], "DR3": row["DR3"], "alpha": row["alpha"],
            "sigma": row.get("sigma", 2.0), "n": row["n"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="LABEL=FILE:SUFFIX")
    ap.add_argument("--json", type=Path, default=REPO / "results" / "fair_compare" / "fair_compare_loyo.json")
    ap.add_argument("--md", type=Path, default=REPO / "results" / "fair_compare" / "fair_compare_loyo.md")
    args = ap.parse_args()

    arms = {}
    for item in args.arm:
        label, _, spec = item.partition("=")
        arms[label] = load(spec)

    lines = ["| arm | alpha | sigma | LOYO macro MD | DR@3 | frames |", "|---|---:|---:|---:|---:|---:|"]
    for label, arm in arms.items():
        lines.append(f"| {label} | {arm['alpha']:g} | {arm['sigma']:g} | {arm['MD']:.3f} | "
                     f"{100 * arm['DR3']:.1f}% | {arm['n']} |")

    years = sorted(set.intersection(*[set(a["folds"]) for a in arms.values()]))
    storm = ["", "| year | " + " | ".join(arms) + " |", "|---" + "|---:" * len(arms) + "|"]
    for year in years:
        storm.append(f"| {year} | " + " | ".join(f"{a['folds'][year]:.3f}" for a in arms.values()) + " |")
    frames = ["", "| year | " + " | ".join(arms) + " | best |",
              "|---" + "|---:" * len(arms) + "|---|"]
    for year in years:
        best = min(arms, key=lambda k: arms[k]["folds"][year])
        frames.append(f"| {year} | " + " | ".join(f"{a['folds'][year]:.3f}" for a in arms.values())
                      + f"| {best} |")

    text = ("# Cross-year (8-fold LOYO) comparison through the same mask + Viterbi pipeline\n\n"
            "Every arm runs `deliver_aided.py --loyo` with identical decoder constants; only the "
            "emission, the mask amplitude and the mask width change. The 2024 validation year is "
            "not used to choose anything here.\n\n" + "\n".join(lines) + "\n\n## Per fold\n"
            + "\n".join(storm) + "\n" + "\n".join(frames) + "\n")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(text, encoding="utf-8")
    args.json.write_text(json.dumps(arms, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))
    print("\n".join(storm))
    print("\n".join(frames))


if __name__ == "__main__":
    main()
