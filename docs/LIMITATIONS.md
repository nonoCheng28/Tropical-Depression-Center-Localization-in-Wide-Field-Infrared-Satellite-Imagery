# Reproduction boundaries

1. This is a source extraction, not a redistribution of course observations or 17 model checkpoints. End-to-end numerical reproduction requires those inputs or the full training procedure. No GPU training was restarted to create this release.
2. The supplemental C7-wide2, C1 and TCL-Net experiments each contain all eight held-out training years (805 evaluation samples total), fixed 44-epoch final checkpoints, and complete decoding. Reports use year-macro aggregation, never a partial-fold average.
3. Supplemental LOYO uses a fixed 44-epoch budget taken from the original D model's validation-selected epoch. It is not the first-round early-stopping protocol, and has weak dependence on 2024 through that one budget scalar.
4. The original fold-cache workflow min–max normalizes **raw fold logits**; v2/v3 holdout emissions average eight sigmoid predictions before normalization. Those are different calibration paths. This extraction preserves the implementation needed for the existing reported numbers rather than silently substituting a new transformation. Compare predictions with the documented path, and disclose this distinction in scientific comparisons.
5. Historical decoder CLI defaults still expose the older α=1, σ=2 round for archive comparability. All current reproduction examples explicitly use α=1.5, σ=3 or the portable `evaluate` defaults. The independent evaluator now sets those parameters explicitly instead of depending on an external worktree's patched defaults.
6. The best-track mask uses external positions at inference time. Full-sequence Viterbi is offline. Neither is evidence of operational real-time forecasting or Day-0 warning.
7. The independent years use 67 and 23 frames respectively, not earlier superseded larger panels. Frozen IDs and source-availability exclusions must be retained. The course-data reconstruction itself was reverse engineered.
8. The manuscript's descriptive wording that attributes performance to “detecting vs selecting” is an interpretation; the code reproduces candidate coverage, errors and decoder ablations, not a causal proof of that interpretation.
9. GPU/library versions and deterministic-kernel support can affect numerical equality. The CPU algorithm tests and archived-cache parity checks validate packaging/numerics, not a fresh 44-epoch retraining equivalence.

## Extraction changes

The original shared model/loss/metric equations and decoder recurrence are preserved. Changes are restricted to installable imports, relative/configurable paths, explicit independent-evaluation defaults, a manifest-based acquisition filter, test opt-in/registration, and portable prediction/evaluation entry points. Exploratory refiner/auxiliary/sequence-loss trainers, all checkpoints and caches, local queues, manuscript builders, logs, and obsolete figure variants are excluded.

`SOURCE_PROVENANCE.json` identifies copied source files and original hashes; `SOURCE_MANIFEST.json` records the final release-source hashes. Neither contains user-specific absolute paths.

## Checkpoint evaluation batch size

`predict_arm_heatmaps` now defaults to the checkpoint configuration's `micro_batch` (fallback 8 for older configs), rather than a universal batch size 8. An explicit `--batch-size` still overrides it. On TCL-Net fold 2022, batch 8 gave raw argmax MD 17.3990°, while its recorded batch 16 reproduced 17.0331°; this was an export/check failure, not a reason to change checkpoints or the decoder. The exporter records `inference_batch_size` and retains its drift guard.
