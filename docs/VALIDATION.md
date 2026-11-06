# Validation performed for this source release

- Python source syntax checked across the release package.
- 10 unit tests passed: frozen course/independent manifests, storm/year separation, Gaussian mask, exhaustive-path Viterbi agreement, candidates, metrics and effective-batch accumulation.
- Command help/import checks passed for formal training, fold training, v2/v3 training, arm/ensemble/manifest prediction, decoder evaluation and ablation.
- A CPU C1 smoke run completed one epoch with 17 training / 3 validation examples after checking all 805 training and 94 validation headers/labels. It wrote its own ignored run directory in this new checkout.
- The migrated decoder reproduced the archived ensemble validation result: MD 2.1172017034648523, DR@3 0.9042553191489362, n=94.
- The migrated eight-fold evaluator reproduced the archived normalized v2 result: macro MD 10.868633484109012, macro DR@3 0.6188893232841278, n=805.
- No satellite observations, checkpoints or heatmap tensors are tracked. No full GPU retraining or independent imagery download was performed for packaging.

The tests ran with this checkout's source package explicitly selected in a verification harness, and verified its __file__ belonged to this checkout. A local editable wheel built successfully. Installing into a separate inherited environment hit host write permissions; a successful clean-environment dependency installation is therefore not claimed. Standard installation is documented in README.md.

## Queue completion update (2026-11-06)

All 24 C7-wide2/C1/TCL-Net supplemental folds completed, with the two newly completed summaries verified against eight fold counts and fixed-final-checkpoint configurations. C1 year-macro MD/DR3: 11.32203329631229 / 0.44330043536168484. TCL-Net: 10.084374268228622 / 0.4436714707138413. Each summary covers 805 samples.

The updated public-package arm exporter was rerun locally on the archived TCL-Net 2022 final checkpoint: 88 held-out samples, inference batch 16 read from config, recomputed argmax MD 17.0331°, reproducing the epoch-44 record. No retraining or test-driven configuration changes were used. The raw checkpoints and validation cache remain untracked.
