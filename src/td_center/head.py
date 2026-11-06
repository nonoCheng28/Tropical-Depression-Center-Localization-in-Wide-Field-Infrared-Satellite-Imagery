"""Shared heatmap head for the stacking and temporal models."""

from torch import nn


def make_heatmap_head():
    """Map 64-channel spatial features to a single raw heatmap."""
    return nn.Sequential(
        nn.Conv2d(64, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 1, kernel_size=1),
    )
