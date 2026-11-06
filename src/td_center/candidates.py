"""Label-free local-maximum candidates for oracle diagnostics, not deployment."""
import numpy as np


def local_peak_candidates(heatmap, k=5, separation_deg=3.):
    h = np.asarray(heatmap, dtype=np.float64)
    if h.ndim != 2 or not np.isfinite(h).all() or k < 1 or separation_deg <= 0:
        raise ValueError('Finite 2D heatmap, positive k/separation required')
    height, width = h.shape
    padded = np.pad(h, 1, constant_values=-np.inf)
    maxima = np.ones_like(h, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            maxima &= h >= padded[dy:dy+height, dx:dx+width]
    # Collapse connected equal-valued plateaus to their first row-major pixel.
    remaining = set(map(tuple, np.argwhere(maxima)))
    representatives = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack = [first]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (y+dy, x+dx)
                    if neighbor in remaining and h[neighbor] == h[first]:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
        representatives.append(first)
    representatives.sort(key=lambda p: (-h[p], p[0], p[1]))
    selected = []
    for y, x in representatives:
        lon, lat = 109 + (x+.5)*70/width, 55 - (y+.5)*70/height
        if any(np.hypot(lon-p['lon'], lat-p['lat']) < separation_deg for p in selected):
            continue
        selected.append({'rank': len(selected)+1, 'row': int(y), 'col': int(x),
                         'lon': float(lon), 'lat': float(lat), 'response': float(h[y, x])})
        if len(selected) == k:
            break
    return selected
