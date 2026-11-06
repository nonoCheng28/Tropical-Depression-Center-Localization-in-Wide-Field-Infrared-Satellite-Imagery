import torch


def make_gaussian_heatmap(center, size=(80, 80), sigma=2.0):
    """
    center: 归一化中心 (x, y)，形状 [2]
    size:   热图的 (高度, 宽度)
    sigma:  高斯宽度，单位为热图像素

    返回: [1, H, W] 的 float32 张量
    """
    height, width = size
    center = torch.as_tensor(center, dtype=torch.float32)

    if center.shape != (2,):
        raise ValueError("center 必须是形状为 [2] 的 (x, y)")

    if not torch.isfinite(center).all():
        raise ValueError("中心坐标包含无效数值")

    if not ((center >= 0) & (center <= 1)).all():
        raise ValueError("中心坐标必须在 0～1 之间")

    if height <= 0 or width <= 0 or sigma <= 0:
        raise ValueError("尺寸和 sigma 必须大于 0")

    # 将归一化坐标转换成热图的像素中心坐标
    cx = center[0] * width - 0.5
    cy = center[1] * height - 0.5

    # 创建每个位置的行号和列号
    rows = torch.arange(
        height, dtype=torch.float32, device=center.device
    )
    cols = torch.arange(
        width, dtype=torch.float32, device=center.device
    )
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")

    # 每个位置到真实中心的距离平方
    distance_squared = (xx - cx) ** 2 + (yy - cy) ** 2

    # 套用二维高斯公式
    heatmap = torch.exp(-distance_squared / (2 * sigma ** 2))

    # [H, W] → [1, H, W]，增加一个输出通道
    return heatmap.unsqueeze(0)


def decode_argmax(heatmap):
    """把单张 [1, H, W] 热图解码为归一化中心 [2]。"""
    if heatmap.ndim != 3 or heatmap.shape[0] != 1:
        raise ValueError("输入必须为 [1, H, W]")

    _, height, width = heatmap.shape

    # 先展平成一列，找最大值的下标
    index = heatmap.reshape(-1).argmax().item()

    # 从一维下标还原行号、列号
    row = index // width
    col = index % width

    # 转回归一化坐标，顺序为 x、y
    return heatmap.new_tensor([
        (col + 0.5) / width,
        (row + 0.5) / height,
    ])

def main():
    import matplotlib.pyplot as plt

    from td_center.dataloader import (
        IRDataset, DATA_ROOT, MANIFEST_DIR, REPO_ROOT
    )

    dataset = IRDataset(
        DATA_ROOT, MANIFEST_DIR, split="train"
    )
    sample = dataset[0]

    center = sample["center"]
    heatmap = make_gaussian_heatmap(
        center, size=(80, 80), sigma=2.0
    )
    decoded = decode_argmax(heatmap)
    error = (decoded - center).abs()

    print("真实中心：", center)
    print("热图形状：", heatmap.shape)
    print("热图最大值：", heatmap.max().item())
    print("解码中心：", decoded)
    print("各轴绝对误差：", error)

    # argmax 只能返回网格中心，允许每轴最多半格的误差
    _, height, width = heatmap.shape
    tolerance = center.new_tensor([
        0.5 / width, 0.5 / height
    ])
    assert (error <= tolerance + 1e-6).all()
    print("检查通过：解码误差在半个热图像素以内")

    image = sample["images"][-1, 0].numpy()
    target = heatmap[0].numpy()

    fig, axes = plt.subplots(
        1, 3, figsize=(13, 4), constrained_layout=True
    )

    # 左：当前帧
    axes[0].imshow(
        image, cmap="gray", vmin=0, vmax=1,
        origin="upper", extent=(0, 1, 1, 0)
    )
    axes[0].set_title("Current frame")

    # 中：由标签生成的目标热图
    axes[1].imshow(
        target, cmap="hot", vmin=0, vmax=1,
        origin="upper", extent=(0, 1, 1, 0),
        interpolation="nearest",
    )
    axes[1].set_title("Gaussian target")

    # 右：叠加，便于检查位置
    axes[2].imshow(
        image, cmap="gray", vmin=0, vmax=1,
        origin="upper", extent=(0, 1, 1, 0)
    )
    axes[2].imshow(
        target, cmap="hot", vmin=0, vmax=1,
        origin="upper", extent=(0, 1, 1, 0),
        interpolation="nearest",
        alpha=0.65 * target,
    )
    axes[2].set_title("Overlay")

    for ax in axes:
        ax.scatter(
            center[0].item(), center[1].item(),
            c="cyan", marker="+", s=120,
            label="Label",
        )
        ax.scatter(
            decoded[0].item(), decoded[1].item(),
            edgecolors="lime", facecolors="none",
            s=80, label="Decoded",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    axes[0].legend(fontsize=8)

    output_dir = REPO_ROOT / "runs" / "heatmap_checks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sample['sample_id']}.png"

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print("图片已保存：", output_path)


if __name__ == "__main__":
    main()