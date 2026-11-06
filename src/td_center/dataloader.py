
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from td_center.heatmap import make_gaussian_heatmap


# 原始数据的位置
DATA_ROOT = Path(
    os.environ.get("TD_DATA_ROOT", str(Path(__file__).resolve().parents[2] / "data"))
)

# 当前模块位于 src/td_center/；可编辑安装时向上两级为仓库根目录
REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO_ROOT / "splits" / "single_center_v1"

from torch.utils.data import Dataset, DataLoader


def _load_ids_from(path):
    path = Path(path)
    sample_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"样本清单为空或有重复 ID：{path}")
    return sample_ids


def _load_sample_ids(manifest_dir, split):
    if split not in ("train", "val", "test"):
        raise ValueError("split 必须为 train、val 或 test")
    return _load_ids_from(Path(manifest_dir) / f"{split}.txt")


def _load_center(label_path):
    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 1 or len(lines[0].split()) != 2:
        raise ValueError(f"应当恰好有一行 x y 标签：{label_path}")
    x, y = map(float, lines[0].split())
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError(f"中心坐标超出范围：{label_path}")
    return torch.tensor([x, y], dtype=torch.float32)


def _resize_ir(images):
    return F.interpolate(
        images, size=(320, 320), mode="bilinear",
        align_corners=False, antialias=True,
    )


class IRDataset(Dataset):
    def __init__(self, data_root, manifest_dir, split="train", span="6h", manifest_path=None):
        if span not in ("6h", "12h", "18h"):
            raise ValueError("span 必须为 6h、12h 或 18h")
        # 记录图像和标签所在目录
        self.base = Path(data_root) / f"Multiple_frames_{span}" / split

        # 这里只读名单，不把全部图像一次性加载到内存
        self.sample_ids = (_load_ids_from(manifest_path) if manifest_path is not None
                           else _load_sample_ids(manifest_dir, split))

    def __len__(self):
        # 告诉 PyTorch：共有多少个样本
        return len(self.sample_ids)

    def __getitem__(self, index):
        # 告诉 PyTorch：如何读取第 index 个样本
        sample_id = self.sample_ids[index]

        image_path = self.base / "images" / f"{sample_id}.npy"
        array = np.load(image_path, allow_pickle=False)

        if array.shape != (7, 640, 640):
            raise ValueError(f"图像形状异常：{image_path}")

        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"图像类型或数值异常：{image_path}")

        images = torch.from_numpy(array)
        images = images.flip(0)
        images = images.unsqueeze(1)
        images = _resize_ir(images)

        # 标签是 T 时刻的归一化中心坐标 x、y
        label_path = (
            self.base / "labels" / "TD_center"
            / "yolo" / f"{sample_id}.txt"
        )
        center = _load_center(label_path)

        heatmap = make_gaussian_heatmap(
            center, size=(80, 80), sigma=2.0
        )

        return {
            "images": images,
            "center": center,
            "heatmap": heatmap,
            "sample_id": sample_id,
        }


class SingleFrameIRDataset(Dataset):
    """Single current-frame PNG, sharing manifests/labels with temporal data.

    Returns images [1,320,320], center [2], heatmap [1,80,80] and sample_id.
    Uint8 PNG is divided by 255 once; no further normalization/augmentation.
    PNG is quantized relative to the float32 npy current frame, so the input
    values are aligned but are not claimed to be exactly identical.
    """

    def __init__(self, data_root, manifest_dir, split="train", manifest_path=None):
        self.base = Path(data_root) / "single_frame" / "ir" / split
        self.sample_ids = (_load_ids_from(manifest_path) if manifest_path is not None
                           else _load_sample_ids(manifest_dir, split))

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        from PIL import Image

        sample_id = self.sample_ids[index]
        image_path = self.base / "images" / f"{sample_id}.png"
        with Image.open(image_path) as image:
            if image.mode != "L" or image.size != (640, 640):
                raise ValueError(f"应当是 640×640 单通道灰度 PNG：{image_path}")
            array = np.array(image, copy=True)
        if array.dtype != np.uint8:
            raise ValueError(f"PNG 应为 uint8：{image_path}")
        images = torch.from_numpy(array).to(torch.float32).div_(255)
        images = _resize_ir(images.unsqueeze(0).unsqueeze(0))[0]
        label_path = self.base / "labels" / "TD_center" / "yolo" / f"{sample_id}.txt"
        center = _load_center(label_path)
        return {
            "images": images,
            "center": center,
            "heatmap": make_gaussian_heatmap(center, size=(80, 80), sigma=2.0),
            "sample_id": sample_id,
        }

def main():
    train_dataset = IRDataset(
        DATA_ROOT, MANIFEST_DIR, split="train"
    )
    val_dataset = IRDataset(
        DATA_ROOT, MANIFEST_DIR, split="val"
    )

    print("训练样本数：", len(train_dataset))
    print("验证样本数：", len(val_dataset))

    # 手动读取一个样本，会调用 __getitem__(0)
    sample = train_dataset[0]
    print("单个样本图像：", sample["images"].shape)
    print("单个样本中心：", sample["center"])

    # 自动读取并组合四个样本
    loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    # 先看第一批，之后立即结束
    for batch in loader:
        print("一批图像：", batch["images"].shape)
        print("一批中心：", batch["center"].shape)
        print("一批热图：", batch["heatmap"].shape)
        break


if __name__ == "__main__":
    main()
