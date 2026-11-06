"""Minimal reader for JMA Himawari Standard Data (HSD) L1b files.

Implemented from the public field layout of the
*Himawari-8/9 Himawari Standard Data User's Guide* (JMA, 2015/2019).
Only the fields needed for band calibration + navigation are decoded.
No third-party dependencies beyond numpy.

Layout reminder
---------------
The HSD file is a chain of self-describing header blocks (block number +
block length), followed by one contiguous big/little-endian uint16 raster of
``number_of_lines x number_of_columns``.  NOAA distributes the same file as a
plain bzip2 of that byte stream (the internal per-block compression of the
original HSD distribution is *not* used there), so ``bz2.decompress`` gives
the buffer below directly.
"""
from __future__ import annotations

import bz2
import datetime as dt
import struct

import numpy as np

EPOCH = dt.datetime(1858, 11, 17)

# ---------------------------------------------------------------- header dtypes
BASIC_INFO_TYPE = np.dtype([
    ("hblock_number", "u1"),
    ("blocklength", "<u2"),
    ("total_number_of_hblocks", "<u2"),
    ("byte_order", "u1"),
    ("satellite", "S16"),
    ("proc_center_name", "S16"),
    ("observation_area", "S4"),
    ("other_observation_info", "S2"),
    ("observation_timeline", "<u2"),
    ("observation_start_time", "f8"),
    ("observation_end_time", "f8"),
    ("file_creation_time", "f8"),
    ("total_header_length", "<u4"),
    ("total_data_length", "<u4"),
    ("quality_flag1", "u1"),
    ("quality_flag2", "u1"),
    ("quality_flag3", "u1"),
    ("quality_flag4", "u1"),
    ("file_format_version", "S32"),
    ("file_name", "S128"),
    ("spare", "S40"),
])

DATA_INFO_TYPE = np.dtype([
    ("hblock_number", "u1"),
    ("blocklength", "<u2"),
    ("number_of_bits_per_pixel", "<u2"),
    ("number_of_columns", "<u2"),
    ("number_of_lines", "<u2"),
    ("compression_flag_for_data", "u1"),
    ("spare", "S40"),
])

PROJ_INFO_TYPE = np.dtype([
    ("hblock_number", "u1"),
    ("blocklength", "<u2"),
    ("sub_lon", "f8"),
    ("CFAC", "<u4"),
    ("LFAC", "<u4"),
    ("COFF", "f4"),
    ("LOFF", "f4"),
    ("distance_from_earth_center", "f8"),
    ("earth_equatorial_radius", "f8"),
    ("earth_polar_radius", "f8"),
    ("req2_rpol2_req2", "f8"),
    ("rpol2_req2", "f8"),
    ("req2_rpol2", "f8"),
    ("coeff_for_sd", "f8"),
    ("resampling_types", "<i2"),
    ("resampling_size", "<i2"),
    ("spare", "S40"),
])

NAV_INFO_TYPE = np.dtype([
    ("hblock_number", "u1"),
    ("blocklength", "<u2"),
    ("navigation_info_time", "f8"),
    ("SSP_longitude", "f8"),
    ("SSP_latitude", "f8"),
    ("distance_earth_center_to_satellite", "f8"),
    ("nadir_longitude", "f8"),
    ("nadir_latitude", "f8"),
    ("sun_position", "f8", (3,)),
    ("moon_position", "f8", (3,)),
    ("spare", "S40"),
])

CAL_INFO_TYPE = np.dtype([
    ("hblock_number", "u1"),
    ("blocklength", "<u2"),
    ("band_number", "<u2"),
    ("central_wave_length", "f8"),
    ("valid_number_of_bits_per_pixel", "<u2"),
    ("count_value_error_pixels", "<u2"),
    ("count_value_outside_scan_pixels", "<u2"),
    ("gain_count2rad_conversion", "f8"),
    ("offset_count2rad_conversion", "f8"),
])

IRCAL_INFO_TYPE = np.dtype([
    ("c0_rad2tb_conversion", "f8"),
    ("c1_rad2tb_conversion", "f8"),
    ("c2_rad2tb_conversion", "f8"),
    ("c0_tb2rad_conversion", "f8"),
    ("c1_tb2rad_conversion", "f8"),
    ("c2_tb2rad_conversion", "f8"),
    ("speed_of_light", "f8"),
    ("planck_constant", "f8"),
    ("boltzmann_constant", "f8"),
    ("spare", "S40"),
])

SEGMENT_INFO_TYPE = np.dtype([
    ("hblock_number", "u1"),
    ("blocklength", "<u2"),
    ("total_number_of_segments", "u1"),
    ("segment_sequence_number", "u1"),
    ("first_line_number_of_image_segment", "u2"),
    ("spare", "S40"),
])


def _walk_blocks(buf, start=0):
    """Yield (block_number, offset, length) following the self-describing chain."""
    pos = start
    while pos + 3 <= len(buf):
        num = int(buf[pos])
        if num == 0 or num > 32:
            break
        length = struct.unpack("<H", buf[pos + 1:pos + 3])[0]
        if length <= 0:
            break
        yield num, pos, length
        pos += length
        if num == 11:
            break


def read_header(path_or_buf):
    """Decode the header chain. Returns a dict with the raw blocks and parsed values."""
    if isinstance(path_or_buf, (bytes, bytearray)):
        buf = bytes(path_or_buf)
    elif str(path_or_buf).endswith(".bz2"):
        buf = bz2.decompress(open(path_or_buf, "rb").read())
    else:
        buf = open(path_or_buf, "rb").read()

    blocks = {}
    for num, pos, length in _walk_blocks(buf):
        blocks[num] = buf[pos:pos + length]
    if len(blocks) < 7:
        raise ValueError(f"unexpected HSD header chain: {sorted(blocks)}")

    b1 = np.frombuffer(blocks[1], dtype=BASIC_INFO_TYPE, count=1)[0]
    b2 = np.frombuffer(blocks[2], dtype=DATA_INFO_TYPE, count=1)[0]
    b3 = np.frombuffer(blocks[3], dtype=PROJ_INFO_TYPE, count=1)[0]
    b4 = np.frombuffer(blocks[4], dtype=NAV_INFO_TYPE, count=1)[0]
    b5 = np.frombuffer(blocks[5][:CAL_INFO_TYPE.itemsize], dtype=CAL_INFO_TYPE, count=1)[0]
    # IR / VIS calibration coefficients follow block 5 inside the same block
    if int(b5["band_number"]) >= 7:
        cal = np.frombuffer(blocks[5][CAL_INFO_TYPE.itemsize:], dtype=IRCAL_INFO_TYPE, count=1)[0]
    else:
        cal = None
    b7 = np.frombuffer(blocks[7][:47], dtype=SEGMENT_INFO_TYPE, count=1)[0]

    header_end = int(b1["total_header_length"])
    return dict(
        buf_len=len(buf),
        header_end=header_end,
        satellite=b1["satellite"].split(b"\x00")[0].decode("ascii", "replace"),
        area=b1["observation_area"].split(b"\x00")[0].decode("ascii", "replace"),
        obs_start=EPOCH + dt.timedelta(days=float(b1["observation_start_time"])),
        obs_end=EPOCH + dt.timedelta(days=float(b1["observation_end_time"])),
        n_bits=int(b2["number_of_bits_per_pixel"]),
        ncols=int(b2["number_of_columns"]),
        nlines=int(b2["number_of_lines"]),
        compression=int(b2["compression_flag_for_data"]),
        sub_lon=float(b3["sub_lon"]),
        cfac=int(b3["CFAC"]),
        lfac=int(b3["LFAC"]),
        coff=float(b3["COFF"]),
        loff=float(b3["LOFF"]),
        sat_height=float(b3["distance_from_earth_center"]),
        req=float(b3["earth_equatorial_radius"]),
        rpol=float(b3["earth_polar_radius"]),
        ssp_lon=float(b4["SSP_longitude"]),
        ssp_lat=float(b4["SSP_latitude"]),
        band=int(b5["band_number"]),
        central_wavelength=float(b5["central_wave_length"]),
        valid_bits=int(b5["valid_number_of_bits_per_pixel"]),
        count_error=int(b5["count_value_error_pixels"]),
        count_outside=int(b5["count_value_outside_scan_pixels"]),
        gain=float(b5["gain_count2rad_conversion"]),
        offset=float(b5["offset_count2rad_conversion"]),
        ir_cal=cal,
        total_segments=int(b7["total_number_of_segments"]),
        segment_number=int(b7["segment_sequence_number"]),
        first_line=int(b7["first_line_number_of_image_segment"]),
        raw=buf,
    )


def read_counts(path):
    """Return the raw uint16 count raster (nlines, ncols) of an HSD file."""
    hdr = read_header(path)
    buf = hdr["raw"]
    n = hdr["nlines"] * hdr["ncols"]
    data = np.frombuffer(buf, dtype="<u2", count=n, offset=hdr["header_end"])
    return data.reshape(hdr["nlines"], hdr["ncols"]), hdr


def counts_to_rad(counts, hdr):
    """Count -> radiance (W m-2 sr-1 um-1); invalid pixels become NaN."""
    invalid = (counts == hdr["count_error"]) | (counts == hdr["count_outside"])
    rad = counts.astype(np.float64) * hdr["gain"] + hdr["offset"]
    return np.where(invalid, np.nan, rad)


def rad_to_tb(rad, hdr):
    """Radiance -> brightness temperature (K) via the HSD inverse Planck."""
    cal = hdr["ir_cal"]
    cwl = hdr["central_wavelength"] * 1e-6
    c_ = float(cal["speed_of_light"])
    h_ = float(cal["planck_constant"])
    k_ = float(cal["boltzmann_constant"])
    a_ = (h_ * c_) / (k_ * cwl)
    with np.errstate(divide="ignore", invalid="ignore"):
        b_ = (2 * h_ * c_ ** 2) / (rad * 1.0e6 * cwl ** 5) + 1
        te = a_ / np.log(b_)
    c0 = float(cal["c0_rad2tb_conversion"])
    c1 = float(cal["c1_rad2tb_conversion"])
    c2 = float(cal["c2_rad2tb_conversion"])
    tb = c0 + c1 * te + c2 * te ** 2
    return np.where(np.isfinite(tb), tb, np.nan)


def counts_to_tb(counts, hdr):
    """Count -> radiance -> brightness temperature (K), following the HSD guide."""
    return rad_to_tb(counts_to_rad(counts, hdr), hdr)



# ------------------------------------------------------------------ navigation
def scan_angles_to_latlon(x, y, hdr):
    """(x, y) scanning angles (rad) -> (lat, lon) in degrees.

    Solves the exact ellipsoid intersection for the line of sight from the
    satellite, using the same sign convention as :func:`latlon_to_scan_angles`.
    """
    h = hdr["sat_height"] * 1000.0
    req = hdr["req"] * 1000.0
    rpol = hdr["rpol"] * 1000.0
    lon0 = hdr["sub_lon"]

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    cx, cy, sx, sy = np.cos(x), np.cos(y), np.sin(x), np.sin(y)

    a = cy ** 2 / req ** 2 + sy ** 2 / rpol ** 2
    b = -2.0 * h * cx * cy / req ** 2
    c = h ** 2 / req ** 2 - 1.0
    disc = b ** 2 - 4.0 * a * c
    with np.errstate(invalid="ignore"):
        t = (-b - np.sqrt(disc)) / (2.0 * a)
    t = np.where(disc >= 0, t, np.nan)

    px = h - t * cx * cy
    py = t * cy * sx
    pz = -t * sy
    lat = np.rad2deg(np.arctan((req / rpol) ** 2 * pz / np.hypot(px, py)))
    lon = lon0 + np.rad2deg(np.arctan2(py, px))
    return lat, lon



def latlon_to_scan_angles(lat, lon, hdr):
    """(lat, lon) degrees -> (x, y) scanning angles (rad) (CGMS forward)."""
    h = hdr["sat_height"] * 1000.0
    req = hdr["req"] * 1000.0
    rpol = hdr["rpol"] * 1000.0
    lat = np.deg2rad(np.asarray(lat, dtype=np.float64))
    dlon = np.deg2rad(np.asarray(lon, dtype=np.float64) - hdr["sub_lon"])

    c_lat = np.arctan((rpol / req) ** 2 * np.tan(lat))
    r_l = rpol / np.sqrt(1.0 - (1.0 - (rpol / req) ** 2) * np.cos(c_lat) ** 2)
    r1 = h - r_l * np.cos(c_lat) * np.cos(dlon)
    r2 = -r_l * np.cos(c_lat) * np.sin(dlon)
    r3 = r_l * np.sin(c_lat)
    rn = np.sqrt(r1 ** 2 + r2 ** 2 + r3 ** 2)
    x = np.arctan2(-r2, r1)
    y = np.arcsin(-r3 / rn)
    return x, y


def grid_linecol(lat, lon, hdr):
    """Map (lat, lon) to 0-based (line, col) inside *this* file's raster.

    CFAC/LFAC are expressed in degrees per 2**16 (JMA HSD convention), so the
    scan angles are converted to degrees before scaling.  Lines run north->south.
    """
    x, y = latlon_to_scan_angles(lat, lon, hdr)
    x_deg = np.rad2deg(x)
    y_deg = np.rad2deg(y)
    col_1based = hdr["coff"] + x_deg * (hdr["cfac"] / 2.0 ** 16)
    line_1based = hdr["loff"] + y_deg * (hdr["lfac"] / 2.0 ** 16)
    return line_1based - hdr["first_line"], col_1based - 1.0


def linecol_to_latlon(line, col, hdr):
    """Inverse of :func:`grid_linecol` (0-based indices)."""
    x_deg = (np.asarray(col, dtype=np.float64) + 1.0 - hdr["coff"]) * 2.0 ** 16 / hdr["cfac"]
    y_deg = (np.asarray(line, dtype=np.float64) + hdr["first_line"] - hdr["loff"]) * 2.0 ** 16 / hdr["lfac"]
    return scan_angles_to_latlon(np.deg2rad(x_deg), np.deg2rad(y_deg), hdr)
