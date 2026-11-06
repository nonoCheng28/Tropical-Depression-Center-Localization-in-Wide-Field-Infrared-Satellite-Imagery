# Required inputs and frozen selection

The course observations and checkpoints are **not distributed** here. Obtain the course dataset through its original authorized route. No public download location or redistribution permission for that dataset is asserted.

Set `TD_DATA_ROOT` to a directory with:

```text
Multiple_frames_6h/{train,val,test}/images/*.npy
Multiple_frames_6h/{train,val,test}/labels/TD_center/yolo/*.txt
Multiple_frames_12h/...                  # required by v3 training
Multiple_frames_18h/...
single_frame/ir/{train,val,test}/images/*.png
single_frame/ir/{train,val,test}/labels/TD_center/yolo/*.txt
RSMC_Best_Track_Data/bst_all.txt
```

Seven-frame arrays are float32, shape `(7,640,640)`, newest first. Input values were already scaled by the provider. Labels are single normalized x/y coordinates. PNG controls use one quantization/divide-by-255 step, unlike the float32 temporal arrays.

`splits/single_center_v1` contains 805/94/121 IDs, plus the excluded 55 multi-center IDs and selection metadata. `splits/loyo` partitions only those 805 training IDs into eight held-out years. Do not rebuild these from image order or pooled years. Test is never used for checkpoint or parameter selection.

## Independently acquired years

HSD acquisition/reprojection sources are documented in `td_center.holdout.ahipipe` and `hsd`: JMA Himawari Standard Data layout and NOAA's public Himawari-8/9 AHI band-13 object archives. The implementation uses the fixed 109–179°E / 15°S–55°N field, 640² pixels and 1h/2h/3h sampling intervals. The preprocessing was inferred from course samples, which is a limitation rather than a guarantee of the provider's original pipeline.

```sh
python -m td_center.holdout build_dataset --year 2015 --manifest splits/independent_2015/eval.txt --out outputs/holdout/td2015 --jobs 3
python -m td_center.holdout build_dataset --year 2026 --manifest splits/independent_2026/eval.txt --out outputs/holdout/td2026 --jobs 3
```

The `--bst` flag overrides the best-track path. An archive of approximately 24 MB per hourly frame is required; acquisition can use many GB. Sources can contain genuine missing frames; build logs record failures. Before evaluating, verify **all frozen IDs** were generated, not merely that a nonempty subset exists. A changing best-track release that no longer matches the frozen IDs is an error, not a reason to silently redefine the panel.

The frozen independent manifests encode the audited single-center, first-contiguous-TD-stint selection and source-availability exclusions. They reproduce the paper's specific panels without relying on later updates to the 2026 best track. Record the timestamp/checksum of the best-track file you supply. This source release does not bundle that archive.

The raw builder writes `Multiple_frames_6h/images`, etc., without a physical split subdirectory. After generation, copy the corresponding frozen independent manifest to the generated dataset's `splits/eval.txt`; then run:

```sh
python -m td_center.holdout make_split --data outputs/holdout/td2015 --out outputs/holdout_env/td2015 --besttrack DATA_ROOT/RSMC_Best_Track_Data
```

This creates only a new mirror under your selected output directory, using directory links. It does **not** change the course dataset or this repository's official train/val/test manifests. Use the mirror as the data root and pass the independent manifest explicitly to `predict_manifest` and `eval_holdout`.
