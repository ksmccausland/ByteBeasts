#!/usr/bin/env python3
"""
Download a grid of Google Static Maps satellite images (labels off).

MODES:
1) Bounding box mode:
   --lat-min --lat-max --lon-min --lon-max

2) Center + radius mode:
   --center-lat --center-lon --radius-m

Coordinates can be:
- Decimal degrees: 36.061639, -121.506222
- DMS:            36°03'41.9"N, 121°30'22.4"W

EXAMPLES

CENTER+RADIUS (DMS):
  python gmaps_sat_grid.py \
    --api-key "YOUR_KEY" \
    --center-lat '36°03'"'"'41.9"N' \
    --center-lon '121°30'"'"'22.4"W' \
    --radius-m 800 \
    --zoom 19 \
    --out ./captures

CENTER+RADIUS (decimal):
  python gmaps_sat_grid.py \
    --api-key "YOUR_KEY" \
    --center-lat 36.061639 \
    --center-lon -121.506222 \
    --radius-m 800 \
    --zoom 19 \
    --out ./captures

BBOX (decimal):
  python gmaps_sat_grid.py \
    --api-key "YOUR_KEY" \
    --lat-min 47.6000 --lat-max 47.7000 \
    --lon-min -122.4200 --lon-max -122.3000 \
    --zoom 18 \
    --out ./captures
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Iterator, Tuple

import requests


# ---------------------------
# Coordinate parsing
# ---------------------------

def dms_to_decimal(dms_str: str) -> float:
    """
    Convert DMS string like:
      36°03'41.9"N
      121°30'22.4"W
    to decimal degrees.
    """
    s = dms_str.strip().upper()

    # Normalize common variants
    s = s.replace("º", "°").replace("’", "'").replace("′", "'").replace("”", '"').replace("″", '"')

    pattern = r"""
        (?P<deg>-?\d+(?:\.\d+)?)[°\s]+
        (?P<min>\d+(?:\.\d+)?)[\'\s]+
        (?P<sec>\d+(?:\.\d+)?)[\"\s]*
        (?P<dir>[NSEW])
    """
    m = re.match(pattern, s, re.VERBOSE)
    if not m:
        raise ValueError(f"Invalid DMS format: {dms_str}")

    deg = float(m.group("deg"))
    minutes = float(m.group("min"))
    seconds = float(m.group("sec"))
    direction = m.group("dir")

    decimal = abs(deg) + minutes / 60.0 + seconds / 3600.0
    if direction in ("S", "W"):
        decimal *= -1.0
    return decimal


def parse_coord(value: str) -> float:
    """
    Parse either decimal degrees or DMS.
    """
    v = str(value).strip()

    # If it contains a direction letter, treat as DMS
    if any(ch.upper() in ("N", "S", "E", "W") for ch in v):
        return dms_to_decimal(v)

    try:
        return float(v)
    except ValueError:
        raise ValueError(
            f"Invalid coordinate: {value} (expected decimal degrees or DMS like 36°03'41.9\"N)"
        )


# ---------------------------
# Geometry helpers
# ---------------------------

@dataclass(frozen=True)
class BoundingBox:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def normalized(self) -> "BoundingBox":
        lat_min = min(self.lat_min, self.lat_max)
        lat_max = max(self.lat_min, self.lat_max)
        lon_min = min(self.lon_min, self.lon_max)
        lon_max = max(self.lon_min, self.lon_max)
        return BoundingBox(lat_min, lat_max, lon_min, lon_max)


def meters_per_pixel(lat_deg: float, zoom: int) -> float:
    # Web Mercator approx
    return 156543.03392 * math.cos(math.radians(lat_deg)) / (2 ** zoom)


def meters_to_deg_lat(meters: float) -> float:
    return meters / 111_320.0


def meters_to_deg_lon(meters: float, lat_deg: float) -> float:
    return meters / (111_320.0 * math.cos(math.radians(lat_deg)))


def bbox_from_center_radius(center_lat: float, center_lon: float, radius_m: float) -> BoundingBox:
    """
    Approx bounding box around a center with given radius (meters).
    """
    dlat = meters_to_deg_lat(radius_m)
    dlon = meters_to_deg_lon(radius_m, center_lat)
    return BoundingBox(
        lat_min=center_lat - dlat,
        lat_max=center_lat + dlat,
        lon_min=center_lon - dlon,
        lon_max=center_lon + dlon,
    ).normalized()


def frange(start: float, stop: float, step: float) -> Iterator[float]:
    if step == 0:
        raise ValueError("step cannot be 0")
    if step > 0:
        x = start
        while x <= stop + 1e-12:
            yield x
            x += step
    else:
        x = start
        while x >= stop - 1e-12:
            yield x
            x += step


# ---------------------------
# Static Maps API
# ---------------------------

def build_static_maps_url(
    api_key: str,
    center_lat: float,
    center_lon: float,
    zoom: int,
    width: int,
    height: int,
    scale: int,
) -> str:
    # Attempt to remove labels (works for most label overlays)
    style = "feature:all|element:labels|visibility:off"

    params = {
        "center": f"{center_lat:.7f},{center_lon:.7f}",
        "zoom": str(zoom),
        "size": f"{width}x{height}",
        "scale": str(scale),
        "maptype": "satellite",
        "style": style,
        "key": api_key,
    }

    qs = "&".join(f"{k}={requests.utils.quote(str(v), safe=':,|')}" for k, v in params.items())
    return f"https://maps.googleapis.com/maps/api/staticmap?{qs}"


def download_image(url: str, out_path: str, timeout: float = 30.0) -> None:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)


# ---------------------------
# Grid generator
# ---------------------------

def compute_grid_centers(
    bbox: BoundingBox,
    zoom: int,
    width: int,
    height: int,
    overlap: float,
) -> Iterator[Tuple[int, int, float, float]]:
    bbox = bbox.normalized()

    mid_lat = (bbox.lat_min + bbox.lat_max) / 2.0
    mpp = meters_per_pixel(mid_lat, zoom)

    cover_w_m = mpp * width
    cover_h_m = mpp * height

    step_w_m = cover_w_m * (1.0 - overlap)
    step_h_m = cover_h_m * (1.0 - overlap)
    if step_w_m <= 0 or step_h_m <= 0:
        raise ValueError("overlap too large; must be < 1.0")

    row = 0
    # north -> south
    for center_lat in frange(bbox.lat_max, bbox.lat_min, -meters_to_deg_lat(step_h_m)):
        dlon = meters_to_deg_lon(step_w_m, center_lat)
        col = 0
        for center_lon in frange(bbox.lon_min, bbox.lon_max, dlon):
            yield (row, col, center_lat, center_lon)
            col += 1
        row += 1


# ---------------------------
# CLI / main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Grid-download satellite images (labels off) from Google Static Maps API.")

    ap.add_argument("--api-key", required=True, help="Google Maps Static API key")

    mode = ap.add_mutually_exclusive_group(required=True)

    # BBOX mode
    mode.add_argument("--bbox", action="store_true", help="Use bounding box mode (requires lat/lon min/max args)")

    # CENTER mode
    mode.add_argument("--center", action="store_true", help="Use center+radius mode (requires center-lat/lon + radius-m)")

    # BBOX args (strings so we can parse decimal OR DMS)
    ap.add_argument("--lat-min", help="Decimal or DMS")
    ap.add_argument("--lat-max", help="Decimal or DMS")
    ap.add_argument("--lon-min", help="Decimal or DMS")
    ap.add_argument("--lon-max", help="Decimal or DMS")

    # CENTER+RADIUS args
    ap.add_argument("--center-lat", help="Decimal or DMS (e.g., 36°03'41.9\"N)")
    ap.add_argument("--center-lon", help="Decimal or DMS (e.g., 121°30'22.4\"W)")
    ap.add_argument("--radius-m", type=float, help="Radius in meters")

    # Imaging/grid params
    ap.add_argument("--zoom", type=int, required=True)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--scale", type=int, default=2, choices=[1, 2])
    ap.add_argument("--overlap", type=float, default=0.10)
    ap.add_argument("--out", default="./captures")
    ap.add_argument("--sleep", type=float, default=0.1)

    args = ap.parse_args()

    # Build bbox from chosen mode
    if args.bbox:
        missing = [k for k in ("lat_min", "lat_max", "lon_min", "lon_max") if getattr(args, k.replace("-", "_"), None) is None]
        # Above is messy due to hyphens; do explicit:
        if any(v is None for v in (args.lat_min, args.lat_max, args.lon_min, args.lon_max)):
            raise SystemExit("BBOX mode requires: --lat-min --lat-max --lon-min --lon-max")

        lat_min = parse_coord(args.lat_min)
        lat_max = parse_coord(args.lat_max)
        lon_min = parse_coord(args.lon_min)
        lon_max = parse_coord(args.lon_max)
        bbox = BoundingBox(lat_min, lat_max, lon_min, lon_max).normalized()

    else:
        if args.center_lat is None or args.center_lon is None or args.radius_m is None:
            raise SystemExit("CENTER mode requires: --center-lat --center-lon --radius-m")

        center_lat = parse_coord(args.center_lat)
        center_lon = parse_coord(args.center_lon)
        if args.radius_m <= 0:
            raise SystemExit("--radius-m must be > 0")
        bbox = bbox_from_center_radius(center_lat, center_lon, args.radius_m)

    # Sanity checks
    if not (-90 <= bbox.lat_min <= 90 and -90 <= bbox.lat_max <= 90):
        raise SystemExit("Latitude must be between -90 and 90.")
    if not (-180 <= bbox.lon_min <= 180 and -180 <= bbox.lon_max <= 180):
        raise SystemExit("Longitude must be between -180 and 180.")

    os.makedirs(args.out, exist_ok=True)

    total = 0
    for (r_i, c_i, lat, lon) in compute_grid_centers(bbox, args.zoom, args.width, args.height, args.overlap):
        url = build_static_maps_url(
            api_key=args.api_key,
            center_lat=lat,
            center_lon=lon,
            zoom=args.zoom,
            width=args.width,
            height=args.height,
            scale=args.scale,
        )
        filename = f"r{r_i:03d}_c{c_i:03d}_lat{lat:.6f}_lon{lon:.6f}.png"
        out_path = os.path.join(args.out, filename)

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            continue

        try:
            download_image(url, out_path)
            total += 1
            print(f"Saved {out_path}")
        except requests.HTTPError as e:
            print(f"HTTP error for r{r_i} c{c_i} ({lat},{lon}): {e}")
        except Exception as e:
            print(f"Error for r{r_i} c{c_i} ({lat},{lon}): {e}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"Done. Downloaded {total} images into {args.out}")


if __name__ == "__main__":
    main()
