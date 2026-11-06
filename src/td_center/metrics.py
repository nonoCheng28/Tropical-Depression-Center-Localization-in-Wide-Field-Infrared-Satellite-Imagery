"""Shared planar longitude/latitude metrics (degrees), not great-circle km."""
import torch


def centers_to_lonlat(centers):
    centers = torch.as_tensor(centers, dtype=torch.float64)
    if centers.ndim != 2 or centers.shape[1] != 2 or not torch.isfinite(centers).all():
        raise ValueError("Expected finite [N, 2] normalized x,y centers")
    if not ((centers >= 0) & (centers <= 1)).all():
        raise ValueError("Centers must lie in [0, 1]")
    return centers * centers.new_tensor([70., -70.]) + centers.new_tensor([109., 55.])


def center_metrics(predicted, target):
    predicted, target = centers_to_lonlat(predicted), centers_to_lonlat(target)
    if predicted.shape != target.shape or len(target) == 0:
        raise ValueError("Predictions and targets must have matching nonempty shapes")
    error = predicted - target
    distance = torch.linalg.vector_norm(error, dim=1)
    return {
        "mean_distance_deg": distance.mean().item(),
        "mae_deg": error.abs().mean().item(),
        "rmse_deg": error.square().mean().sqrt().item(),
        "detection_rate_at_3deg": (distance <= 3.).double().mean().item(),
        "n": len(target),
    }
