"""Baseline B: single-frame, randomly initialized ResNet-18 heatmap model."""

import argparse
import hashlib
import json
import sys
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F

from td_center.head import make_heatmap_head


class BModel(nn.Module):
    """Accept [B,1,320,320] and output a raw [B,1,80,80] heatmap.

    ResNet-18 spatial features [B,512,10,10] are projected to 64 channels,
    resized to 80x80 and decoded with the shared heatmap head. No pretrained
    weights, input scaling or label coordinates are used by this model.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18

        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
        # Exclude classification pooling/fc so all retained parameters are used.
        self.encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.projection = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.ReLU())
        self.head = make_heatmap_head()

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 320, 320):
            raise ValueError("B 模型输入必须为 [B, 1, 320, 320] 的单帧 IR")
        features = self.projection(self.encoder(images))
        features = F.interpolate(features, size=(80, 80), mode="bilinear", align_corners=False)
        return self.head(features)


def main():
    """Smoke check on synthetic data or two frozen training samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="原始数据根目录；省略时使用合成数据")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("cpu-threads 必须大于0")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(42)

    from td_center.dataloader import SingleFrameIRDataset, MANIFEST_DIR, REPO_ROOT
    from td_center.heatmap import make_gaussian_heatmap, decode_argmax
    import torchvision

    if args.data_root:
        from torch.utils.data import DataLoader, Subset

        dataset = SingleFrameIRDataset(args.data_root, MANIFEST_DIR, split="train")
        batch = next(iter(DataLoader(Subset(dataset, [0, 1]), batch_size=2, num_workers=0)))
        images, target = batch["images"], batch["heatmap"]
        sample_ids = batch["sample_id"]
        print("Train sample IDs:", sample_ids)
    else:
        images = torch.rand(2, 1, 320, 320)
        target = torch.stack([make_gaussian_heatmap([0.3, 0.4]), make_gaussian_heatmap([0.7, 0.6])])
        sample_ids = []
        print("Synthetic input: structure check only")

    model = BModel()
    model.train()
    feature_shape = {}
    hook = model.encoder.register_forward_hook(
        lambda _module, _inputs, output: feature_shape.update(shape=list(output.shape))
    )
    prediction = model(images)
    hook.remove()
    assert feature_shape["shape"] == [2, 512, 10, 10]
    assert prediction.shape == target.shape == (2, 1, 80, 80)
    assert torch.isfinite(prediction).all() and torch.isfinite(target).all()
    # MSE checks the computation chain; formal training loss is not chosen here.
    loss = F.mse_loss(prediction, target)
    assert torch.isfinite(loss)
    loss.backward()
    gradient_checks = {}
    for name, module in (("encoder", model.encoder), ("projection", model.projection), ("head", model.head)):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(grad is not None and torch.isfinite(grad).all() for grad in gradients)
        gradient_sum = sum(grad.abs().sum().item() for grad in gradients)
        assert gradient_sum > 0
        gradient_checks[name] = {"all_gradients_finite": True, "absolute_gradient_sum": gradient_sum}
        print(f"{name}: gradient check passed")

    model.eval()
    with torch.no_grad():
        eval_prediction = model(images)
        assert torch.isfinite(eval_prediction).all()
        decoded = torch.stack([decode_argmax(heatmap) for heatmap in eval_prediction])
        assert decoded.shape == (2, 2) and ((decoded >= 0) & (decoded <= 1)).all()
        # Check a different batch size and strict checkpoint compatibility.
        one = model(images[:1])
        assert one.shape == (1, 1, 80, 80)
        rebuilt = BModel().eval()
        rebuilt.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(eval_prediction, rebuilt(images), rtol=0, atol=0)
    for invalid_shape in ((1, 3, 320, 320), (1, 7, 1, 320, 320), (1, 1, 640, 640)):
        try:
            model(torch.zeros(invalid_shape))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid input accepted: {invalid_shape}")

    run = REPO_ROOT / "runs" / f"model_b_check_{datetime.now():%Y%m%d_%H%M%S_%f}"
    run.mkdir(parents=True, exist_ok=False)
    summary = {
        "role": "baseline", "model": "BModel", "backbone": "ResNet-18",
        "pretrained_weights": None, "seed": 42, "sample_ids": sample_ids,
        "data_source": "single_frame/ir/train PNG" if args.data_root else "synthetic",
        "input_shape": list(images.shape), "encoder_feature_shape": feature_shape["shape"],
        "output_shape": list(prediction.shape), "parameter_count": sum(p.numel() for p in model.parameters()),
        "check_mse": loss.item(), "gradient_checks": gradient_checks,
        "eval_decode_passed": True, "batch_one_passed": True,
        "strict_state_reload_passed": True, "invalid_inputs_rejected": True,
        "python": sys.version, "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
        "command": sys.argv, "cpu_threads": args.cpu_threads,
        "note": "No optimizer steps, training, validation/test evaluation or model accuracy claim",
        "source_sha256": {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (REPO_ROOT / "src/td_center/model_b.py",
                                       REPO_ROOT / "src/td_center/head.py",
                                       REPO_ROOT / "src/td_center/dataloader.py")},
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Input:", tuple(images.shape), "Output:", tuple(prediction.shape))
    print("Parameters:", summary["parameter_count"], "Check MSE:", f"{loss.item():.6f}")
    print("Train/eval, gradients, decoding, checkpoint reload and input checks passed")
    print("Run directory:", run)


if __name__ == "__main__":
    main()
