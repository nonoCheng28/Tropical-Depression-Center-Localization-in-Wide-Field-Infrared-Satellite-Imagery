import torch
from torch import nn

from td_center.encoder import CNNEncoder
from td_center.ConvLSTM import ConvLSTM
from td_center.head import make_heatmap_head


class DModel(nn.Module):
    def __init__(self, spatial_context=False):
        super().__init__()

        self.encoder = CNNEncoder()

        self.temporal = ConvLSTM(
            input_channels=64,
            hidden_channels=64,
            kernel_size=3,
        )

        # 两层卷积，将 64 通道特征转换成单通道热图
        self.head = make_heatmap_head()

        # Construct after the original layers; preserve their initialization and RNG.
        if spatial_context:
            from td_center.spatial_context import SpatialContext
            with torch.random.fork_rng(devices=[]):
                self.context = SpatialContext()
        else:
            self.context = nn.Identity()

    def forward(self, images):
        # images: [B, 7, 1, 320, 320]
        if images.ndim != 5:
            raise ValueError("输入必须为 [B, T, C, H, W]")

        if tuple(images.shape[1:]) != (7, 1, 320, 320):
            raise ValueError("当前模型要求七帧单通道 320×320 图像")

        batch_size, num_frames, channels, height, width = (
            images.shape
        )

        # 1. 展开七帧，交给同一个 CNN
        frames = images.reshape(
            batch_size * num_frames,
            channels,
            height,
            width,
        )
        frame_features = self.encoder(frames)

        # 2. 恢复时间维度
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

        # 3. 时序融合
        last_hidden = self.temporal(sequence_features)

        # 4. 输出预测热图
        predicted_heatmap = self.head(self.context(last_hidden))

        return predicted_heatmap

def main():
    from torch.utils.data import DataLoader

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
    target_heatmap = batch["heatmap"]

    model = DModel()
    model.train()
    model.zero_grad(set_to_none=True)

    # 前向计算：模型只能看到图像
    predicted_heatmap = model(images)

    print("输入图像：", images.shape)
    print("预测热图：", predicted_heatmap.shape)
    print("目标热图：", target_heatmap.shape)

    # 必须完全同形状，避免错误的广播计算
    assert predicted_heatmap.shape == target_heatmap.shape
    assert torch.isfinite(predicted_heatmap).all()
    assert torch.isfinite(target_heatmap).all()

    # 使用真实 Gaussian 目标计算损失
    criterion = nn.MSELoss()
    loss = criterion(predicted_heatmap, target_heatmap)

    assert torch.isfinite(loss)
    print(f"MSE loss：{loss.item():.6f}")

    # 对照：如果模型所有位置都输出 0，损失是多少？
    zero_loss = criterion(
        torch.zeros_like(target_heatmap),
        target_heatmap,
    )
    print(f"全零预测的 MSE：{zero_loss.item():.6f}")

    # 反向传播
    loss.backward()

    # 检查三个模块的梯度
    modules = [
        ("CNN", model.encoder),
        ("ConvLSTM", model.temporal),
        ("Heatmap head", model.head),
    ]

    for name, module in modules:
        total_grad = 0.0

        for parameter in module.parameters():
            assert parameter.grad is not None, (
                f"{name} 存在没有收到梯度的参数"
            )
            assert torch.isfinite(parameter.grad).all(), (
                f"{name} 的梯度出现无效值"
            )
            total_grad += parameter.grad.abs().sum().item()

        assert total_grad > 0, f"{name} 的梯度全部为零"
        print(f"{name} 梯度检查通过：{total_grad:.6f}")

    print("完整 D 模型前向、MSE 和反向传播检查通过")


if __name__ == "__main__":
    main()
