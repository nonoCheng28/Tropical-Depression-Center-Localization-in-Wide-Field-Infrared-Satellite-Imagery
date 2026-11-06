import torch
from torch import nn


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_channels=64,
        hidden_channels=64,
        kernel_size=3,
    ):
        super().__init__()

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size 必须是正奇数")

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # 输入当前帧特征 x 和上一步隐藏状态 h
        # 一次卷积计算四组状态更新所需的数值
        self.gates = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        # Forget gate bias initialization
        with torch.no_grad():
            self.gates.bias[
                hidden_channels:2 * hidden_channels
            ].fill_(1.0)

    def forward(self, features):
        # features: [B, T, C, H, W]
        if features.ndim != 5:
            raise ValueError("输入必须为 [B, T, C, H, W]")

        batch_size, num_frames, channels, height, width = (
            features.shape
        )

        if channels != self.input_channels or num_frames == 0:
            raise ValueError("输入通道数不匹配，或时间序列为空")

        # 每次处理一批独立序列，都重新初始化状态
        # new_zeros 自动沿用输入的设备和数据类型
        h = features.new_zeros(
            batch_size, self.hidden_channels, height, width
        )
        c = features.new_zeros(
            batch_size, self.hidden_channels, height, width
        )

        for t in range(num_frames):
            # 当前时刻特征：[B, C, H, W]
            x = features[:, t]

            # 沿通道方向拼接当前输入与历史隐藏状态
            combined = torch.cat([x, h], dim=1)

            # 分为输入门、遗忘门、输出门、候选记忆
            i, f, o, g = self.gates(combined).chunk(4, dim=1)

            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            o = torch.sigmoid(o)
            g = torch.tanh(g)

            # 更新内部记忆和隐藏状态
            c = f * c + i * g
            h = o * torch.tanh(c)

        # 只返回处理完最后一帧后的隐藏状态
        return h


def main():
    from torch.utils.data import DataLoader

    from td_center.encoder import CNNEncoder
    from td_center.dataloader import IRDataset, DATA_ROOT, MANIFEST_DIR

    torch.manual_seed(42)

    dataset = IRDataset(
        DATA_ROOT, MANIFEST_DIR, split="train"
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))
    images = batch["images"]

    encoder = CNNEncoder()
    temporal = ConvLSTM(
        input_channels=64,
        hidden_channels=64,
        kernel_size=3,
    )

    batch_size, num_frames, channels, height, width = images.shape

    # 第一步：共享 CNN 逐帧提取特征
    frames = images.reshape(
        batch_size * num_frames, channels, height, width
    )
    frame_features = encoder(frames)

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

    # 为了检查，额外保留这个中间张量的梯度
    sequence_features.retain_grad()

    # 第二步：按时间顺序融合七帧特征
    last_hidden = temporal(sequence_features)

    print("七帧特征：", sequence_features.shape)
    print("最后隐藏状态：", last_hidden.shape)

    assert last_hidden.shape == (
        batch_size, 64, 80, 80
    )
    assert torch.isfinite(last_hidden).all()
    print("前向检查通过")

    # 临时构造一个标量，只检查反向传播是否连通
    # 这不是正式的训练损失
    probe = last_hidden.square().mean()
    probe.backward()

    # 检查 CNN 和 ConvLSTM 的参数都能收到有限梯度
    for model in (encoder, temporal):
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()

    # 保留时间维，其余维度求和，得到七个梯度大小
    frame_grad = sequence_features.grad.abs().sum(
        dim=(0, 2, 3, 4)
    )

    print("各帧梯度绝对值之和：", frame_grad)
    assert (frame_grad > 0).all()
    print("反向检查通过：全部七帧和 CNN 都能收到梯度")


if __name__ == "__main__":
    main()
