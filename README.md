# Tropical Depression Center Localization in Wide-Field Infrared Satellite Imagery

Source-only reproduction release for the manuscript's single-frame controls, seven-frame temporal emissions, three-emission ensemble, interference mask, and offline Viterbi trajectory decoder. **Role: core comparisons and advanced ablations.**

This repository contains the implementations and frozen sample manifests needed to reproduce the reported measurements. It does not contain satellite observations, best-track archives, trained weights, heatmap caches, manuscript files, presentation assets, or historical exploratory models. You must supply the original course data and regenerate or supply compatible checkpoints. Publishing this source is not a claim that all experiments have been rerun in a clean environment.

See [the result-to-command map](docs/REPRODUCTION.md), [data requirements](docs/DATA.md), and [implementation limitations](docs/LIMITATIONS.md).

## Install

Use Python 3.10 or newer and a PyTorch build suitable for your hardware. The reference experiments used Windows, Python 3.14.7, PyTorch 2.14.0 with CUDA 12.6, and an RTX 4060 Laptop GPU. These describe the observed environment, not a cross-platform bitwise-reproducibility guarantee.

```sh
python -m venv .venv
# Activate the environment using your shell's normal command.
python -m pip install -e .
python -m unittest discover -s tests -v
```

Set `TD_DATA_ROOT` to the course dataset root before running commands. `TD_ARTIFACT_DIR` optionally changes the heatmap-cache directory; its default is `outputs/artifacts`. The source package is installed from `src/td_center` and never imports another local checkout.

PowerShell:

```powershell
$env:TD_DATA_ROOT = 'YOUR_DATASET_ROOT'
python scripts/train.py --model d --device cuda --micro-batch 1 --threads 1
python -m td_center.reproduce --help
python -m td_center.holdout --help
```

POSIX shell: `export TD_DATA_ROOT=/path/to/dataset`.

## Frozen protocol

- Train: 2016–2023, 805 single-center samples; validation: 2024, 94; test: 2025, 121. Preserve `splits/single_center_v1/` verbatim. The original full splits contain 856/98/121 samples; 55 multi-center samples are excluded only within their original partitions.
- Seven-frame arrays: float32 `(7,640,640)`, stored newest first. The loader reverses time, resizes to 320×320, and **does not divide arrays by 255**. PNG controls are divided by 255 once.
- Formal baseline: AdamW, LR 1e-3, weight decay 1e-4, effective batch 16, seed 42, ≤50 epochs, patience 10, no scheduler or augmentation. Best checkpoint uses validation Mean Distance.
- Supplemental LOYO: eight training-year folds, **fixed 44 epochs and final weights**, no checkpoint selection on the held-out year. The 44-epoch budget originates from the formal D model's validation-selected epoch and is disclosed as such. v2/v3 are explicit advanced variants; v3 adds the documented multi-span augmentation.
- Delivered decoder: independently normalize each emission per frame, equal-weight average; Gaussian suppression at other named storms, α=1.5 and σ=3 heatmap cells; 8 local peaks separated by 3°; offline Viterbi κ=0.003. Best-track positions are external inference-time inputs. This is not an online forecasting system.
- MD is radial mean distance in planar degrees; DR@3 is a fraction. `MAE_comp` and `RMSE_comp` are coordinate-component metrics; radial RMSE is a different auxiliary quantity.

## Layout

| Path | Purpose |
|---|---|
| `src/td_center/` | Shared loaders, models, losses, metrics, formal training |
| `src/td_center/reproduce/` | LOYO/v2/v3 training, emissions, decoder, ablations, statistics, data-figure scripts |
| `src/td_center/holdout/` | Public HSD acquisition/reprojection and independent-year evaluation |
| `splits/` | Frozen course and independent-year sample identifiers |
| `configs/` | Training and decoder settings; checkpoint-spec example |
| `tests/` | Metrics, accumulation, candidates, manifest isolation, decoder recurrence |
| `docs/` | Reproduction commands, exclusions, provenance and validation |

Generated outputs go into ignored `runs/` and `outputs/` directories. Test evaluation requires explicit opt-in and a recorded purpose; do not select a configuration from test outcomes. The optional historical `deliver_aided` entry retains its archived-ledger guard; use the portable `evaluate` entry for a new registered evaluation.

## Attribution and reuse

The TCL-Net implementation is an architecture reproduction under this project's data and evaluation protocol, **not a reproduction of its original paper's numerical results**; its source header preserves the original reference and scope. HSD layout and navigation follow JMA documentation, and observations must be obtained from their original distributors. No new blanket license for third-party data or code is asserted by this repository.
