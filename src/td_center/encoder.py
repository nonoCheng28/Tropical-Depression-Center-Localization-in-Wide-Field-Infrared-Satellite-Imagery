import torch
from torch import nn
from torch.utils.data import DataLoader

from td_center.dataloader import IRDataset, DATA_ROOT, MANIFEST_DIR


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1, widths=(16, 32), final_dilation=1):
        super().__init__()

        if not isinstance(in_channels, int) or in_channels < 1:
            raise ValueError("in_channels 必须为正整数")
        if len(widths) != 2 or any(not isinstance(width, int) or width < 1 for width in widths):
            raise ValueError("widths 必须包含两个正整数")
        if not isinstance(final_dilation, int) or final_dilation < 1:
            raise ValueError("final_dilation 必须为正整数")
        first_width, second_width = widths

        # 按顺序执行三个卷积层和对应的激活函数
        self.layers = nn.Sequential(
            # [N, in_channels, 320, 320] → [N, first_width, 160, 160]
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=first_width,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            # → [N, second_width, 80, 80]
            nn.Conv2d(
                in_channels=first_width,
                out_channels=second_width,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            # 保持输出为 64 通道，以共享同一 heatmap head
            nn.Conv2d(
                in_channels=second_width,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=final_dilation,
                dilation=final_dilation,
            ),
            nn.ReLU(),
        )

    def forward(self, images):
        # 定义图像进入网络之后如何计算
        return self.layers(images)


def main():
    torch.manual_seed(42)

    dataset = IRDataset(
        DATA_ROOT, MANIFEST_DIR, split="train"
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    # 取第一批数据
    batch = next(iter(loader))
    images = batch["images"]

    print("输入图像：", images.shape)

    # 拆出五个维度
    batch_size, num_frames, channels, height, width = images.shape

    # Conv2d 接收四维输入，因此暂时合并 batch 和时间维
    # [4, 7, 1, 320, 320] → [28, 1, 320, 320]
    frames = images.reshape(
        batch_size * num_frames,
        channels,
        height,
        width,
    )

    encoder = CNNEncoder()
    encoder.eval()

    # 当前只检查前向计算，不记录反向传播所需的信息
    with torch.no_grad():
        frame_features = encoder(frames)

    print("逐帧特征：", frame_features.shape)

    # 恢复 batch 和时间维度
    # [28, 64, 80, 80] → [4, 7, 64, 80, 80]
    _, feature_channels, feature_height, feature_width = (
        frame_features.shape
    )
    sequence_features = frame_features.reshape(
        batch_size,
        num_frames,
        feature_channels,
        feature_height,
        feature_width,
    )

    print("时序特征：", sequence_features.shape)

    assert sequence_features.shape == (
        batch_size, num_frames, 64, 80, 80
    )
    assert torch.isfinite(sequence_features).all()

    print("检查通过：共享 CNN 输出形状正确，数值均为有限值")


if __name__ == "__main__":
    main()
