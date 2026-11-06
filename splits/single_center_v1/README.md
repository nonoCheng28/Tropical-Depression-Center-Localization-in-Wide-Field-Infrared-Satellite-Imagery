# Single-center v1 split manifests

This directory freezes the first-stage single-center subset while leaving the
teacher-provided dataset unchanged.

- `train.txt`, `val.txt`, `test.txt`: one extension-free sample ID per line.
- `excluded_multi_center.txt`: excluded split, sample ID, and center count.
- `metadata.csv`: audit record for every original sample.
- `summary.json`: selection rule and counts.

From the repository root, audit reconstructed manifests in a separate output directory from the original dataset with:

```powershell
python scripts/data/build_single_center_manifest.py `
  --dataset-root "C:\path\to\dataset" `
  --output-dir "outputs\manifest_audit"
```

All A/B/C/D experiments must read the same manifest. A loader must fail rather
than silently selecting the first label if a listed sample does not contain
exactly one non-empty `TD_center/yolo` label line.
