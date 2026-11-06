"""Common paths and metadata for the D-model improvement experiments."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# Override on another machine: set TD_DATA_ROOT to the dataset directory.
DATA_ROOT = Path(os.environ.get(
    "TD_DATA_ROOT", str(REPO / "data")))
MANIFEST = REPO / "splits" / "single_center_v1"
# Frozen candidate cache from the reranker baseline; override if relocated.
CANDIDATE_CACHE = Path(os.environ.get(
    "TD_CANDIDATE_CACHE",
    str(REPO / "runs/d_reranker_formal_20261020_210708_303656/candidate_cache.pt")))
ARTIFACT_DIR = Path(os.environ.get("TD_ARTIFACT_DIR", str(REPO / "outputs/artifacts")))
RUN_ROOT = Path(os.environ.get("TD_RUN_ROOT", str(REPO / "runs/d_improve")))


def sample_path(split: str, sample_id: str, span: str = "6h") -> Path:
    return DATA_ROOT / f"Multiple_frames_{span}" / split / "images" / f"{sample_id}.npy"
