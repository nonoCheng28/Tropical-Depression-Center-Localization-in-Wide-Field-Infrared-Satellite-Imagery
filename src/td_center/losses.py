"""Sample-weighted loss shared by formal training and validation."""
import torch
import math


def expected_distance_per_sample(prediction, centers, temperature=.1):
    """W1 to a point target, in project planar degrees; not distance of mean.

    Raw heatmap outputs are logits. The spatial grid uses decoder half-pixels.
    Labels only enter this training objective; inference remains argmax.
    """
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError('Expected [B, 1, H, W] heatmaps')
    if centers.shape != (prediction.shape[0], 2):
        raise ValueError('Expected [B, 2] normalized xy centers')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    h, w = prediction.shape[-2:]
    centers = centers.to(prediction)
    x = (torch.arange(w, device=prediction.device, dtype=prediction.dtype) + .5) / w
    y = (torch.arange(h, device=prediction.device, dtype=prediction.dtype) + .5) / h
    distance = (((x[None, None, :] - centers[:, 0, None, None]) * 70).square()
                + ((y[None, :, None] - centers[:, 1, None, None]) * 70).square()).sqrt()
    probability = (prediction.flatten(1) / temperature).softmax(dim=1)
    return (probability * distance.flatten(1)).sum(1)


def hard_negative_per_sample(prediction, target, centers, top_k=64, radius_deg=7.):
    """Mean squared positive residual at top-k pixels >= radius from truth.

    One-sided: already-low background is not pushed upward. Geography and
    half-pixel grid match the frozen decoder. Labels are used only in loss.
    """
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError('Expected [B, 1, H, W] matching heatmaps')
    h, w = prediction.shape[-2:]
    if top_k < 1 or radius_deg <= 0:
        raise ValueError('Positive top_k and radius required')
    x = (torch.arange(w, device=prediction.device, dtype=prediction.dtype) + .5) / w
    y = (torch.arange(h, device=prediction.device, dtype=prediction.dtype) + .5) / h
    distance2 = ((x[None, None, :] - centers[:, 0, None, None]) * 70).square() + ((y[None, :, None] - centers[:, 1, None, None]) * 70).square()
    mask = distance2 >= radius_deg ** 2
    if (mask.flatten(1).sum(1) < top_k).any():
        raise ValueError('Not enough eligible background pixels')
    error = (prediction[:, 0] - target[:, 0]).clamp_min(0).square()
    return error.masked_fill(~mask, -torch.inf).flatten(1).topk(top_k, dim=1).values.mean(1)


def weighted_mse_per_sample(prediction, target, alpha=100.):
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Expected matching [B, 1, H, W] tensors")
    weight = 1. + alpha * target
    axes = (1, 2, 3)
    return (weight * (prediction - target).square()).sum(axes) / weight.sum(axes)


def softmax_ce_per_sample(prediction, target, label_smoothing=0.0):
    """Spatial softmax cross-entropy: every pixel competes with the true cell.

    ``target`` is the Gaussian heatmap; it is normalised per sample.  This loss
    directly optimises the argmax probability rather than pixelwise regression.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Expected matching [B, 1, H, W] tensors")
    if not (0 <= label_smoothing < 1):
        raise ValueError("label_smoothing must be in [0, 1)")
    b, _, h, w = prediction.shape
    logp = torch.log_softmax(prediction.flatten(1), dim=1)
    t = target.flatten(1).clamp_min(0)
    total = t.sum(1, keepdim=True)
    if (total <= 0).any():
        raise ValueError("Target heatmap has no mass")
    t = t / total
    if label_smoothing:
        t = (1 - label_smoothing) * t + label_smoothing / (h * w)
    return -(t * logp).sum(1)
