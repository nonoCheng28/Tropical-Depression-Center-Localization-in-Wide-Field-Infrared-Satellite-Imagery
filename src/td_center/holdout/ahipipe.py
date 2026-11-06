"""Acquisition + reprojection pipeline for a Himawari AHI band-13 dataset.

Goal: reproduce the exact data standard of the course dataset:

    IR, band 13 of AHI (10.4 um), brightness temperature
    70 deg x 70 deg box, 109E..179E / 15S..55N, 640 x 640 px (0.109375 deg/px)
    7-frame sequences at 1h / 2h / 3h spacing (6h / 12h / 18h spans), T first
    values scaled to the same [0, ~0.81] range as the course data

Sources (public, no registration):
    IR      https://noaa-himawari8.s3.amazonaws.com/AHI-L1b-FLDK/...  (2015-2022)
            https://noaa-himawari9.s3.amazonaws.com/AHI-L1b-FLDK/...  (2022-2026)
            -> Himawari Standard Data (HSD), band 13, FLDK, R20, 5500x5500
    Labels  RSMC Tokyo best track (JMA), grades/positions at 6h synoptic times
"""
from __future__ import annotations

import bz2
import datetime as dt
import os
import ssl
import time
import urllib.request

import numpy as np

import td_center.holdout.hsd as hsd # ---------------------------------------------------------------- constants
BOX_LON0, BOX_LON1 = 109.0, 179.0
BOX_LAT0, BOX_LAT1 = -15.0, 55.0   # lat increases upward in the image
GRID_N = 640
SPAN = BOX_LON1 - BOX_LON0          # 70.0

# Normalisation fitted against the course dataset (see docs/TEST_SET_2015.md):
#   value = clip((TB_ZERO - Tb) / TB_SPAN, 0, 1)
TB_ZERO = 300.00
TB_SPAN = 140.00

# Rendering corrections fitted against the course dataset (same reference):
# the course product is slightly smoother than a point-sampled 2 km field and
# its pixel centres sit ~1 HSD pixel north-west of the nominal HSD geometry.
BLUR_SIGMA = 1.2      # in HSD pixels (~2 km)
LINE_SHIFT = -1.0     # HSD pixels
COL_SHIFT = -1.0      # HSD pixels

BUCKETS = {8: "noaa-himawari8", 9: "noaa-himawari9"}
_CTX = ssl._create_unverified_context()


def sensor_for(when: dt.datetime) -> int:
    """Himawari-8 until 2022-12-13 00:00 UTC, Himawari-9 afterwards."""
    return 8 if when < dt.datetime(2022, 12, 13) else 9


#: the 640x640 target box never leaves full-disk lines 382..3542, i.e. segments
#: 1..7 of the 10 FLDK segments (550 lines each); 8..10 are pure space.
NEEDED_SEGMENTS = tuple(range(1, 8))


def hsd_urls(when: dt.datetime, sensor: int | None = None,
             segments=None) -> list[str]:
    """HSD band-13 FLDK URLs for one observation time.

    ``segments`` selects a subset for the Himawari-9 archive (one file per
    segment); the Himawari-8 archive stores all segments in a single file.
    """
    sensor = sensor or sensor_for(when)
    bucket = BUCKETS[sensor]
    tag = f"H{sensor:02d}"
    day = when.strftime("%Y/%m/%d")
    hhmm = when.strftime("%H%M")
    stem = f"HS_{tag}_{when.strftime('%Y%m%d')}_{hhmm}_B13_FLDK_R20"
    base = f"https://s3.amazonaws.com/{bucket}/AHI-L1b-FLDK/{day}/{hhmm}/"
    if sensor == 8:
        # older archive: one file holding all 10 segments
        return [base + f"{stem}_S0101.DAT.bz2"]
    segs = segments if segments is not None else NEEDED_SEGMENTS
    return [base + f"{stem}_S{i:02d}10.DAT.bz2" for i in segs]


def _download(url: str, dst: str, attempts: int = 12,
              read_timeout: float = 45.0, deadline: float = 240.0,
              stall: float = 60.0) -> str:
    """Chunked download that survives slow links but gives up on a stall.

    ``read1`` is used on purpose: ``read(n)`` on ``HTTPResponse`` blocks until
    *n* bytes arrive, so a trickle (or a proxy that sends headers only) would
    hold a worker hostage.  With ``read1`` every 256 KiB chunk is timed, so a
    dead connection is dropped after ``stall`` seconds and the frame re-queued.
    """
    tmp = dst + ".part"
    last = None
    for attempt in range(attempts):
        got = 0
        started = time.time()
        last_data = started
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=read_timeout, context=_CTX) as r:
                expect = r.headers.get("Content-Length")
                expect = int(expect) if expect and expect.isdigit() else None
                with open(tmp, "wb") as f:
                    while True:
                        chunk = r.read1(262144)
                        now = time.time()
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                        last_data = now
                        if now - started > deadline:
                            raise TimeoutError(f"deadline after {got} bytes")
            if got < 100_000 or (expect is not None and got != expect):
                raise IOError(f"short body ({got} of {expect} bytes)")
            os.replace(tmp, dst)
            return dst
        except Exception as exc:
            last = exc
            try:
                os.remove(tmp)
            except OSError:
                pass
            time.sleep(min(1.5 * (attempt + 1), 8.0))
    raise RuntimeError(f"download failed after {attempts} attempts: {last!r}")


def fetch(when: dt.datetime, outdir: str, sensor: int | None = None) -> list[str]:
    """Download the band-13 HSD file(s) of one timestamp; returns local paths."""
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for url in hsd_urls(when, sensor):
        dst = os.path.join(outdir, url.rsplit("/", 1)[-1])
        if os.path.exists(dst) and os.path.getsize(dst) > 100_000:
            paths.append(dst)
            continue
        paths.append(_download(url, dst))
    return paths


def decode(paths: list[str]) -> tuple[np.ndarray, dict]:
    """Decode the HSD file(s) of one timestamp into a full-disk count raster."""
    if len(paths) == 1:
        counts, hdr = hsd.read_counts(paths[0])
        return counts, hdr
    rows = []
    hdr = None
    for p in sorted(paths):
        c, h = hsd.read_counts(p)
        rows.append(c)
        hdr = hdr or h
    counts = np.vstack(rows)
    # rebuild navigation offsets so the stacked raster equals the full disk
    hdr = dict(hdr)
    hdr["nlines"] = counts.shape[0]
    hdr["first_line"] = 1
    return counts, hdr


def tb_full_disk(when: dt.datetime, cachedir: str, sensor: int | None = None):
    """Brightness temperature of the full disk for one timestamp (K, float32)."""
    sensor = sensor or sensor_for(when)
    paths = fetch(when, os.path.join(cachedir, "_hsd", when.strftime("%Y%m%d%H")))
    try:
        counts, hdr = decode(paths)
        tb = hsd.counts_to_tb(counts, hdr).astype(np.float32)
    except Exception:
        # a truncated download must not be cached as if it were usable
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass
        raise
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass
    return tb, hdr


def tb_grid(when: dt.datetime, cachedir: str, sensor: int | None = None) -> np.ndarray:
    """Cached 640x640 brightness-temperature grid for one timestamp."""
    sensor = sensor or sensor_for(when)
    os.makedirs(os.path.join(cachedir, "grid"), exist_ok=True)
    path = os.path.join(cachedir, "grid", when.strftime("%Y%m%d%H") + f"_H{sensor}.npy")
    if os.path.exists(path):
        return np.load(path)
    tb, hdr = tb_full_disk(when, cachedir, sensor)
    out = reproject(tb, hdr).astype(np.float32)
    np.save(path + ".part.npy", out)
    os.replace(path + ".part.npy", path)
    return out


# ---------------------------------------------------------------- reprojection
def grid_latlon(n: int = GRID_N):
    """Pixel-centre latitudes / longitudes of the target grid."""
    lat = BOX_LAT1 - SPAN * (np.arange(n) + 0.5) / n
    lon = BOX_LON0 + SPAN * (np.arange(n) + 0.5) / n
    return (np.repeat(lat[:, None], n, axis=1),
            np.repeat(lon[None, :], n, axis=0))


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur (edge padded)."""
    if sigma <= 0:
        return img
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k = (k / k.sum()).astype(np.float32)
    p = np.pad(img, ((r, r), (0, 0)), mode="edge")
    tmp = np.zeros_like(img)
    for i, w in enumerate(k):
        tmp += w * p[i:i + img.shape[0], :]
    p2 = np.pad(tmp, ((0, 0), (r, r)), mode="edge")
    out = np.zeros_like(img)
    for i, w in enumerate(k):
        out += w * p2[:, i:i + img.shape[1]]
    return out


def reproject(tb: np.ndarray, hdr: dict, lat=None, lon=None) -> np.ndarray:
    """Resample a full-disk Tb field onto the 640x640 target grid."""
    if lat is None:
        lat, lon = grid_latlon()
    line, col = hsd.grid_linecol(lat, lon, hdr)
    line = line + LINE_SHIFT
    col = col + COL_SHIFT
    src = gaussian_blur(tb, BLUR_SIGMA)
    H, W = src.shape
    i0 = np.floor(line).astype(np.int64)
    j0 = np.floor(col).astype(np.int64)
    fl = (line - i0).astype(np.float32)
    fc = (col - j0).astype(np.float32)
    acc = np.zeros(line.shape, dtype=np.float32)
    wsum = np.zeros(line.shape, dtype=np.float32)
    for di in (0, 1):
        for dj in (0, 1):
            w = (fl if di else 1.0 - fl) * (fc if dj else 1.0 - fc)
            v = src[np.clip(i0 + di, 0, H - 1), np.clip(j0 + dj, 0, W - 1)]
            ok = np.isfinite(v)
            acc += np.where(ok, w * v, 0.0)
            wsum += np.where(ok, w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = acc / wsum
    return out.astype(np.float32)



def normalise(tb_grid: np.ndarray) -> np.ndarray:
    """Course-dataset scaling: bright = cold cloud top."""
    v = (TB_ZERO - tb_grid) / TB_SPAN
    return np.clip(v, 0.0, 1.0).astype(np.float32)


def sample_field(when: dt.datetime, cachedir: str, sensor: int | None = None) -> np.ndarray:
    """One normalised 640x640 IR frame for a timestamp."""
    return normalise(tb_grid(when, cachedir, sensor))


def frame_times(when: dt.datetime, step_hours: int, n: int = 7) -> list[dt.datetime]:
    """[T, T-step, ..., T-(n-1)*step] as in the course dataset."""
    return [when - dt.timedelta(hours=step_hours * k) for k in range(n)]
