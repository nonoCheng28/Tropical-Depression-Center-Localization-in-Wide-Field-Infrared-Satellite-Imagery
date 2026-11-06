import json
from pathlib import Path

import numpy as np

d = json.loads(((Path(__file__).resolve().parents[3] / "outputs" / "paper") / "decoder_ablation.json").read_text(encoding="utf-8"))
folds = d["loyo_per_fold"]
years = [str(y) for y in range(2016, 2024)]
kappas = ["0.003", "0.01", "0.1"]

md = {k: np.array([folds[k][years.index(y)][0] for y in years]) for k in kappas}
print("pooled:", {k: round(float(md[k].mean()), 3) for k in kappas})

for cand in (["0.003", "0.01"], kappas):
    picks, chosen = [], []
    for i in range(8):
        others = [j for j in range(8) if j != i]
        means = {k: md[k][others].mean() for k in cand}
        best = min(means, key=means.get)
        chosen.append(best)
        picks.append(md[best][i])
    print(f"nested over {cand}: picks={chosen} -> {np.mean(picks):.3f}")

# how often each kappa wins per fold
wins = {k: 0 for k in kappas}
for i in range(8):
    best = min(kappas, key=lambda k: md[k][i])
    wins[best] += 1
print("per-fold winner:", wins)
