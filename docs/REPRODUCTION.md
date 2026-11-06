# Paper measurements → source and commands

All commands run from this checkout after `pip install -e .`. Examples use placeholders for checkpoint directories; there are no hidden author-specific paths. `OUT` below means a path you choose under `outputs/`. Do not replace the frozen manifests to evaluate independent years.

## 1. Train the formal controls and D

```sh
python scripts/train.py --model d --device cuda --micro-batch 1 --threads 1
python scripts/train.py --model b --single-frame-source png --device cuda --micro-batch 16 --threads 1
python scripts/train.py --model c --cnn-variant wide2 --frames 7 --device cuda --micro-batch 16 --threads 1
python scripts/train.py --model c --frames 1 --device cuda --micro-batch 16 --threads 1
python scripts/train.py --model tclnet --single-frame-source png --device cuda --micro-batch 16 --threads 1
```

These save best/last weights, configuration, epoch history, and validation predictions in `runs/`. Select the minimum validation-MD checkpoint. For a seven-frame model, effective batch 16 can use micro-batch 1 with accumulation; the controls' archived formal runs used micro-batch 16.

For each year 2016 through 2023, train the same control on its frozen fold, e.g.:

```sh
python scripts/train.py --model c --cnn-variant wide2 --frames 7 --fold-year 2016 --epochs 44 --micro-batch 16 --threads 1 --device cuda
python -m td_center.reproduce train_loyo_fold --year 2016 --epochs 44 --micro-batch 1
python -m td_center.reproduce train_d_v2 --year 2016 --epochs 44 --tag d_v2
python -m td_center.reproduce train_d_v2 --year 2016 --epochs 44 --tag d_v3 --augment --spans 6h,12h,18h
```

Repeat **exactly eight years**. For heatmap export, omit `--batch-size` to use the checkpoint's recorded `micro_batch`; a different batch size can change argmax decisions on nearly tied heatmaps. The exporter checks drift against the recorded evaluation and writes the actual inference batch size. Control LOYO uses `last.pt`; D/v2/v3 supplemental trainers also preserve `epoch44.pt`. None use the held-out year's minimum epoch. Leave optimizer/seed/loss defaults unchanged. v2: GroupNorm/dilated encoder + global context + focal loss, no augmentation. v3: same network/loss, mirror/translation/intensity and random 6h/12h/18h training span; no rotation, scale jitter, or dropout in the adopted configuration.

## 2. Generate emissions without retraining

```sh
python -m td_center.reproduce predict_arm_heatmaps --run runs/FORMAL_RUN --tag arm --normalize
python -m td_center.reproduce predict_arm_heatmaps --run runs/FOLD_RUN --checkpoint last.pt --tag arm2016 --normalize --out-dir arm2016 --out-name heatmaps_loyo2016
python -m td_center.reproduce predict_d_ensemble --tag d_v2 --split val --arch v2
python -m td_center.reproduce predict_d_ensemble --tag d_v3 --split val --arch v2
```

The eight-fold predictors average **sigmoid probabilities**, not raw logits. D is one formal checkpoint; v2 and v3 each contain eight fold checkpoints (17 total). To use explicit paths or an independent sample manifest, edit `configs/checkpoints.example.json` and run:

```sh
python -m td_center.reproduce predict_manifest --spec configs/checkpoints.example.json --data-root DATA_ROOT --manifest splits/single_center_v1/val.txt --split val --out outputs/artifacts/ens3_val --device cuda
```

This writes each emission and the normalized mean in `heatmaps_val_fp32.pt`. For test, change the manifest/split and add `--allow-test --purpose "frozen paper comparison"`. No test read is necessary for validation/LOYO development.

For LOYO each fold uses **its own** D/v2/v3 fold emissions, never the eight-model holdout ensemble. The supplemental trainers dump raw fp32 fold logits. Preserve that historical distinction; the merger applies per-frame min–max:

```sh
python -m td_center.reproduce eval_heat_ensemble --years 2016 2017 2018 2019 2020 2021 2022 2023 --sources 'loyo{year}' 'd_v2loyo{year}' 'd_v3loyo{year}' --write-tag ens3
```

The merger also prints exploratory κ values. Use only the frozen κ=0.003 report for the delivered row; do not pick its optimistic per-fold optimum. On Windows use double quotes around patterns if preferred.

## 3. Main table and independent-emission ablations

```sh
python -m td_center.reproduce evaluate --heatmaps outputs/artifacts/ens3_val/heatmaps_val_fp32.pt --data-root DATA_ROOT --split val --out outputs/reports/ens3_val.json
python -m td_center.reproduce evaluate --heatmaps outputs/artifacts/ens3_val/heatmaps_val_fp32.pt --data-root DATA_ROOT --split val --raw --out outputs/reports/ens3_raw_val.json
python -m td_center.reproduce evaluate --loyo-pattern 'outputs/artifacts/ens3loyo{year}/heatmaps_loyo{year}_fp32.pt' --data-root DATA_ROOT --out outputs/reports/ens3_loyo.json
python -m td_center.reproduce evaluate --heatmaps outputs/artifacts/ens3_test/heatmaps_test_fp32.pt --data-root DATA_ROOT --split test --allow-test --purpose "fixed paper test" --out outputs/reports/ens3_test.json
```

Repeat with each arm's **normalized** heatmaps. Add `--candidate-coverage` to obtain the unmasked top-8 coverage column. Non-raw evaluations also report each storm's share of total absolute error; use the documented older `--sigma 2` setting to reproduce the LEEPI 71% contribution diagnostic. For raw fold caches pass `--normalize`; do not apply sigmoid retroactively to the stored fold logits. `--loyo-pattern` requires all eight manifests and reports unweighted year-macro MD/DR3. A held-out fold's cache may contain all training samples, but only that fold's frozen evaluation IDs enter its score.

Current reference cells from completed archived computations (MD / DR@3): ens3 raw val 7.213 / 70.2%; C7-wide2 LOYO 7.896 / 59.2%; TCL-Net test 8.882 / 46.3%; v2 LOYO/test 10.869 / 61.9% and 4.718 / 79.3%; v3 LOYO/test 8.231 / 62.7% and 7.821 / 66.9%. These are reference checks, not outputs bundled in this source release. The completed C1 and TCL-Net LOYO rows are 11.322 / 44.3% and 10.084 / 44.4%, respectively; each uses all eight years and 805 held-out evaluation samples.

## 4. Decoder, mask, placebo and component ablations

```sh
python -m td_center.reproduce decoder_ablation --seeds 0 1 2 3 4
python -m td_center.reproduce kappa_curve
python -m td_center.reproduce nested_kappa
```

These read validation and the eight ens3 LOYO caches only. They reproduce the κ sensitivity table, storm-instance bootstrap of 0.01 versus 0.003, and nested seven-fold selection. Outputs are under `outputs/paper`.

For the mask parameter table use `deliver_aided --loyo --alphas 0.5 1 1.5 2 --sigmas 1 2 3 --dump-tag mask_grid`. Default flags in this historical module are the **older** α=1/σ=2 protocol; always supply the paper's explicit values.

For independent component removals use `evaluate` with the same emission and IDs:

| Component | Flags | Reference test MD |
|---|---|---:|
| Complete | defaults α=1.5/σ=3/κ=.003 | 3.085 |
| Remove mask only | `--alpha 0` | 4.618 |
| Remove ensemble only | input normalized D map | 3.704 |
| Remove Viterbi only | `--argmax` (mask retained) | 5.867 |

**Raw `--raw` is not the Viterbi-only ablation.** The raw unmasked test argmax is 6.761°, which must not replace 5.867°. Every test invocation requires its own explicit purpose and output registration. Recomputing from caches is evaluation, not permission to tune against test.

Paired storm bootstrap and exact sign test are implemented in `fair_compare` and `fair_compare_pipeline`. Call their `--help` for explicit per-arm run/report arguments; for example `fair_compare --arm d_convlstm=runs/YOUR_D_RUN --arm b_png=runs/YOUR_B_RUN`. They resample whole storms, not individual frames. Radial RMSE and coordinate RMSE remain separately named.

## 5. Figures and descriptive quantities

```sh
python -m td_center.reproduce figure_emissions
python -m td_center.reproduce figure_interference
python -m td_center.reproduce besttrack
```

The two figure commands regenerate the SOULIK emission maps and LEEPI interference example using the fixed samples in their source. They write vector PDF and PNG under `outputs/figures`. The manually edited framework diagram is an illustration, not a numerical result, and is intentionally absent from this code-only release. `besttrack` computes other-storm incidence and label/geographic alignment; model parameter counts can be obtained with `sum(p.numel() for p in DModel().parameters())`.

## 6. Independent 2015 and 2026 panels

See [DATA.md](DATA.md). Build public imagery using the **frozen** independent-year `eval.txt` supplied here, not an ever-expanding current-year selection. Use `holdout make_split` to expose the generated data as a physical `val` folder, then `predict_manifest` with the independent manifest. Run `holdout eval_holdout --data MIRROR_ROOT --artifacts EMISSION_ROOT --label td2015 --out outputs/reports/td2015.json --ids splits/independent_2015/eval.txt`. Here `EMISSION_ROOT/ens3_val/heatmaps_val_fp32.pt` holds predictions for that independent set. Repeat for 2026 with its own cache directory. The evaluator fixes α=1.5, σ=3, κ=.003 and reports per-frame errors, per-storm errors, placebo seeds and 20,000 storm-bootstrap draws.

Expected independent counts are 67 frames/14 storms (2015) and 23 frames/4 storms (2026). Reference aided MD/DR3: 2.361°/92.5% and 1.500°/87.0%. The 2026 snapshot is partial-year and must not be described as an annual evaluation.
