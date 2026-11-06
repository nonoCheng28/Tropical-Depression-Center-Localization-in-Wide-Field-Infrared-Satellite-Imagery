"""Build the frozen single-center sample manifests without modifying the dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SPLITS = ("train", "val", "test")
SPANS = ("6h", "12h", "18h")


def nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def expected_paths(dataset_root: Path, split: str, sample_id: str) -> list[Path]:
    paths = [
        dataset_root / "single_frame" / "ir" / split / "images" / f"{sample_id}.png",
        dataset_root
        / "single_frame"
        / "ir"
        / split
        / "labels"
        / "TD_center"
        / "yolo"
        / f"{sample_id}.txt",
    ]
    for span in SPANS:
        base = dataset_root / f"Multiple_frames_{span}" / split
        paths.extend(
            [
                base / "images" / f"{sample_id}.npy",
                base / "labels" / "TD_center" / "yolo" / f"{sample_id}.txt",
            ]
        )
    return paths


def build(dataset_root: Path, output_dir: Path) -> dict[str, object]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, object]] = []
    excluded: list[str] = []
    summary: dict[str, object] = {
        "manifest_version": "single_center_v1",
        "selection_rule": "keep samples with exactly one non-empty TD_center YOLO label line",
        "splits": {},
    }

    for split in SPLITS:
        label_dir = (
            dataset_root
            / "Multiple_frames_6h"
            / split
            / "labels"
            / "TD_center"
            / "yolo"
        )
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

        included_ids: list[str] = []
        split_counts = {"total": 0, "included": 0, "excluded": 0}

        for label_path in sorted(label_dir.glob("*.txt"), key=lambda path: path.name):
            sample_id = label_path.stem
            center_count = len(nonempty_lines(label_path))
            missing = [str(path) for path in expected_paths(dataset_root, split, sample_id) if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Sample {sample_id!r} is incomplete in split {split!r}:\n"
                    + "\n".join(missing)
                )

            included = center_count == 1
            reason = "single_center" if included else f"center_count_{center_count}"
            metadata.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "num_centers": center_count,
                    "included": str(included).lower(),
                    "reason": reason,
                }
            )
            split_counts["total"] += 1
            if included:
                included_ids.append(sample_id)
                split_counts["included"] += 1
            else:
                excluded.append(f"{split}\t{sample_id}\t{center_count}")
                split_counts["excluded"] += 1

        (output_dir / f"{split}.txt").write_text(
            "".join(f"{sample_id}\n" for sample_id in included_ids), encoding="utf-8"
        )
        summary["splits"][split] = split_counts

    with (output_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=("sample_id", "split", "num_centers", "included", "reason")
        )
        writer.writeheader()
        writer.writerows(metadata)

    (output_dir / "excluded_multi_center.txt").write_text(
        "split\tsample_id\tnum_centers\n" + "".join(f"{row}\n" for row in excluded),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("splits/single_center_v1"),
    )
    args = parser.parse_args()
    summary = build(args.dataset_root.resolve(), args.output_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
