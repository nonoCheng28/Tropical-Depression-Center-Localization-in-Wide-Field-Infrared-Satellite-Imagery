"""Build the leave-one-year-out manifests used by LOYO_PROTOCOL.md.

Reads the frozen `splits/single_center_v1/train.txt` and writes, for each held-out
year, the training list and the evaluation list. The frozen manifests are never
modified; val/test never appear here.

    python experiments/d_improve/build_loyo_manifests.py
"""
import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FROZEN = REPO / "splits" / "single_center_v1"
OUT = REPO / "outputs" / "manifests" / "loyo"


def year_of(sample_id: str) -> int:
    return 2000 + int(sample_id[3:5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2016, 2024)))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    source = FROZEN / "train.txt"
    ids = [line.strip() for line in source.read_text(encoding="utf-8").splitlines()
           if line.strip()]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate ids in train.txt")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frozen_val = set((FROZEN / "val.txt").read_text(encoding="utf-8").split())
    frozen_test = set((FROZEN / "test.txt").read_text(encoding="utf-8").split())
    if frozen_val & set(ids) or frozen_test & set(ids):
        raise SystemExit("val/test leakage into train list")

    print(f"{'year':>5} {'eval':>5} {'train':>6} {'storms_eval':>12}")
    for year in args.years:
        eval_ids = [s for s in ids if year_of(s) == year]
        train_ids = [s for s in ids if year_of(s) != year]
        if not eval_ids:
            raise SystemExit(f"no samples for {year}")
        storms = sorted({s.split("_")[0] for s in eval_ids})
        (out_dir / f"{year}_train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
        (out_dir / f"{year}_eval.txt").write_text("\n".join(eval_ids) + "\n", encoding="utf-8")
        print(f"{year:>5} {len(eval_ids):>5} {len(train_ids):>6} {len(storms):>12}")

    # A guard: folds must partition the frozen train list.
    for year in args.years:
        a = set((out_dir / f"{year}_train.txt").read_text(encoding="utf-8").split())
        b = set((out_dir / f"{year}_eval.txt").read_text(encoding="utf-8").split())
        assert not (a & b), year
        assert a | b == set(ids), year
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
