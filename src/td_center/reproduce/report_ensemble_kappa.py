"""Report the LOYO macro at a fixed kappa from finished heat-ensemble runs.

`eval_heat_ensemble.py` prints the macro at each fold's *best* kappa and stores the
whole kappa grid in its result.json. The adopted constant is 0.003, so this reads a
result.json and prints MD/DR@3 at 0.003 (plus the per-fold table), which is what the
8/8 non-degradation rule has to be judged on.

    python experiments/d_improve/report_ensemble_kappa.py --path runs/d_improve/v5ens_result.json
    python experiments/d_improve/report_ensemble_kappa.py --path A.json --path B.json --baseline 7.519
"""
import argparse
import json
from pathlib import Path

from td_center.reproduce.common import REPO

KAPPA = 0.003


def describe(path, kappa):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    per = data["per_year"]
    years = sorted(per, key=int)
    mds, drs = [], []
    print(f"\n{path}")
    print(f"  sources: {'+'.join(data['sources'])}")
    print(f"  {'year':>6} {'MD@%.3g' % kappa:>10} {'DR@3':>7} {'bestK':>9} {'MD@best':>8}")
    for y in years:
        v = per[y]
        md = v["table"][str(kappa)]
        dr = v["table_dr"][str(kappa)]
        mds.append(md)
        drs.append(dr)
        print(f"  {y:>6} {md:10.3f} {dr:7.3f} {v['kappa']:>9g} {v['MD']:8.3f}")
    n = len(mds)
    macro = sum(mds) / n
    print(f"  MACRO  {macro:10.3f} {sum(drs) / n:7.3f}   over {n} folds at kappa={kappa:g}")
    return macro, sum(drs) / n, dict(zip(years, mds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", action="append", required=True,
                    help="a heat_ensemble result.json; repeatable")
    ap.add_argument("--kappa", type=float, default=KAPPA)
    ap.add_argument("--baseline", type=float, default=None,
                    help="macro MD to compare against, e.g. 7.519")
    args = ap.parse_args()

    table = {}
    for path in args.path:
        macro, _dr, per = describe(path, args.kappa)
        table[Path(path).stem] = per
        if args.baseline is not None:
            better = sum(1 for v in per.values() if v < args.baseline)
            print(f"  folds better than {args.baseline}: {better}/{len(per)}")
            print(f"  delta vs {args.baseline}: {macro - args.baseline:+.3f}")

    if len(table) > 1:
        names = list(table)
        common = sorted(set.intersection(*[set(t) for t in table.values()]), key=int)
        print("\nper-fold comparison")
        print("  year " + " ".join(f"{n:>12}" for n in names))
        for y in common:
            print(f"  {y:>4} " + " ".join(f"{table[n][y]:12.3f}" for n in names))


if __name__ == "__main__":
    main()
