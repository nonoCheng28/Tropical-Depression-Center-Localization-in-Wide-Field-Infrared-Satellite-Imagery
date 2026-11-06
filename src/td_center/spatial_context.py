"""Global spatial context via 64 pooled tokens; no temporal/loss changes."""
import torch
from torch import nn


class SpatialContext(nn.Module):
    def __init__(self, channels=64, attention_channels=16, pool_size=8):
        super().__init__()
        self.pool_size = pool_size
        self.scale = attention_channels ** -0.5
        self.query = nn.Conv2d(channels, attention_channels, 1)
        self.key = nn.Conv2d(channels, attention_channels, 1)
        self.value = nn.Conv2d(channels, attention_channels, 1)
        self.output = nn.Conv2d(attention_channels, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features):
        b, c, h, w = features.shape
        p = self.pool_size
        if h % p or w % p:
            raise ValueError("Feature size must be divisible by the fixed pooling grid")
        pooled = features.reshape(b, c, p, h // p, p, w // p).mean(dim=(3, 5))
        query = self.query(features).flatten(2).transpose(1, 2)
        key = self.key(pooled).flatten(2)
        value = self.value(pooled).flatten(2).transpose(1, 2)
        attention = torch.softmax(torch.bmm(query, key) * self.scale, dim=-1)
        context = torch.bmm(attention, value).transpose(1, 2)
        context = context.reshape(features.shape[0], -1, *features.shape[-2:])
        return features + self.output(context)
