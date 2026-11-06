"""Core comparison: seven-frame IR stacking for current TD localization."""

import argparse

import torch
from torch import nn

from td_center.encoder import CNNEncoder
from td_center.head import make_heatmap_head

CNN_VARIANTS = {
    "baseline": {"widths": (16, 32), "final_dilation": 1},
    "dilated": {"widths": (16, 32), "final_dilation": 3},
    "wide": {"widths": (32, 64), "final_dilation": 1},
    "wide2": {"widths": (64, 128), "final_dilation": 1},
}


class CModel(nn.Module):
    """Fuse seven frames as channels, ordered oldest to current.

    Accepts [B, 7, 320, 320] or the shared loader's [B, 7, 1, 320, 320].
    Outputs a raw [B, 1, 80, 80] heatmap for time T. Frame reversal and
    resizing belong to the shared loader; no scaling is applied here.

    `in_channels=1` builds the capacity-matched single-frame control: the same
    encoder and head as the seven-frame arm, fed the current frame only.
    """

    def __init__(self, variant="baseline", in_channels=7):
        super().__init__()
        if variant not in CNN_VARIANTS:
            raise ValueError(f"未知 CNN variant：{variant}")
        self.variant = variant
        self.in_channels = in_channels
        self.encoder = CNNEncoder(in_channels=in_channels, **CNN_VARIANTS[variant])
        self.head = make_heatmap_head()

    def forward(self, images):
        if images.ndim == 5:
            if tuple(images.shape[1:]) != (self.in_channels, 1, 320, 320):
                raise ValueError(f"输入必须为 [B, {self.in_channels}, 1, 320, 320]")
            images = images.squeeze(2)
        elif images.ndim != 4 or tuple(images.shape[1:]) != (self.in_channels, 320, 320):
            raise ValueError(f"输入必须为 [B, {self.in_channels}, 320, 320] 或 "
                             f"[B, {self.in_channels}, 1, 320, 320]")
        return self.head(self.encoder(images))


def main():
    """Check forward/backward with synthetic input or frozen train samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="原始数据根目录；省略时使用合成数据")
    parser.add_argument("--cnn-variant", choices=tuple(CNN_VARIANTS), default="baseline")
    args = parser.parse_args()
    torch.manual_seed(42)
    if args.data_root:
        from torch.utils.data import DataLoader, Subset
        from td_center.dataloader import IRDataset, MANIFEST_DIR

        dataset = IRDataset(args.data_root, MANIFEST_DIR, split="train")
        batch = next(iter(DataLoader(Subset(dataset, [0, 1]), batch_size=2)))
        images, target = batch["images"], batch["heatmap"]
        print("Train sample IDs:", batch["sample_id"])
    else:
        from td_center.heatmap import make_gaussian_heatmap

        images = torch.rand(2, 7, 1, 320, 320)
        target = torch.stack([
            make_gaussian_heatmap([0.3, 0.4]),
            make_gaussian_heatmap([0.7, 0.6]),
        ])
        print("Synthetic input: this check does not measure model accuracy.")

    model = CModel(variant=args.cnn_variant)
    print("CNN variant:", args.cnn_variant)
    prediction = model(images)
    assert prediction.shape == target.shape == (2, 1, 80, 80)
    assert torch.isfinite(prediction).all()
    with torch.no_grad():
        stacked_prediction = model(images.squeeze(2))
        torch.testing.assert_close(prediction, stacked_prediction, rtol=0, atol=0)
    loss = nn.functional.mse_loss(prediction, target)
    assert torch.isfinite(loss)
    loss.backward()
    for name, module in [("encoder", model.encoder), ("head", model.head)]:
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(grad is not None and torch.isfinite(grad).all() for grad in gradients)
        assert sum(grad.abs().sum().item() for grad in gradients) > 0
        print(f"{name}: gradient check passed")
    channel_gradients = model.encoder.layers[0].weight.grad.abs().sum(dim=(0, 2, 3))
    assert (channel_gradients > 0).all(), "Each frame must receive a gradient"
    print("Input:", tuple(images.shape), "Output:", tuple(prediction.shape))
    print(f"MSE: {loss.item():.6f}; all seven input channels receive gradients")


if __name__ == "__main__":
    main()
