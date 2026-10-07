"""Small, strict IONEX reader used by the GIM matching pipeline.

Supported containers: Unix-compress ``.Z``, gzip, and plain IONEX text.
Only TEC maps are read; RMS maps and DCB auxiliary records are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
from pathlib import Path
import re

import numpy as np
from unlzw3 import unlzw


_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")


@dataclass(frozen=True)
class IonexMap:
    time: datetime
    lat: np.ndarray
    lon: np.ndarray
    vtec: np.ndarray


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    lower_name = path.name.lower()
    if lower_name.endswith(".z"):
        raw = unlzw(raw)
    elif lower_name.endswith(".gz"):
        raw = gzip.decompress(raw)
    return raw.decode("ascii", errors="replace")


def _numbers(line: str) -> list[float]:
    return [float(value) for value in _NUMBER_RE.findall(line[:60])]


def _epoch(line: str) -> datetime:
    values = [int(float(value)) for value in _NUMBER_RE.findall(line[:36])[:6]]
    if len(values) != 6:
        raise ValueError(f"Invalid IONEX epoch record: {line!r}")
    return datetime(*values, tzinfo=timezone.utc)


def read_first_tec_epoch(path: str | Path) -> datetime:
    """Read only the first TEC epoch, for fast real-time inventory creation."""
    path = Path(path)
    inside_tec_map = False
    for line in _read_text(path).splitlines():
        label = line[60:].strip() if len(line) > 60 else ""
        if label == "START OF TEC MAP":
            inside_tec_map = True
        elif inside_tec_map and label == "EPOCH OF CURRENT MAP":
            return _epoch(line)
    raise ValueError(f"No TEC epoch found in {path}")


def _canonicalize_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ascending latitude and a periodic -180..180 longitude grid."""
    lat_order = np.argsort(lat)
    lat = lat[lat_order]
    data = data[lat_order, :]

    normalized_lon = ((lon + 180.0) % 360.0) - 180.0
    rounded_lon = np.round(normalized_lon, decimals=8)
    unique_lon = np.unique(rounded_lon)
    unique_lon.sort()

    if unique_lon.size < 2:
        raise ValueError("IONEX longitude grid has fewer than two unique points")

    step = float(np.median(np.diff(unique_lon)))
    expected_points = int(round(360.0 / step))
    is_global = (
        unique_lon.size == expected_points
        and np.allclose(np.diff(unique_lon), step, atol=1e-6)
    )
    if not is_global:
        raise ValueError(
            "This pipeline requires a complete regular global longitude grid; "
            f"got {unique_lon.size} unique points with median step {step:g}"
        )

    merged_columns: list[np.ndarray] = []
    for value in unique_lon:
        indices = np.flatnonzero(np.isclose(rounded_lon, value, atol=1e-7))
        if indices.size == 1:
            merged_columns.append(data[:, indices[0]])
        else:
            # -180 and +180 are the same meridian. Averaging also makes the
            # seam exactly periodic if a producer rounded the endpoints apart.
            with np.errstate(invalid="ignore"):
                merged_columns.append(np.nanmean(data[:, indices], axis=1))

    unique_data = np.stack(merged_columns, axis=1).astype(np.float32, copy=False)
    lon_out = np.concatenate([unique_lon, [unique_lon[0] + 360.0]])
    data_out = np.concatenate([unique_data, unique_data[:, :1]], axis=1)
    return lat.astype(np.float32), lon_out.astype(np.float32), data_out


def read_ionex_tec_maps(path: str | Path) -> list[IonexMap]:
    """Read and normalize every TEC map in an IONEX file."""
    path = Path(path)
    lines = _read_text(path).splitlines()

    exponent = None
    header_end = None
    for index, line in enumerate(lines):
        label = line[60:].strip() if len(line) > 60 else ""
        if label == "EXPONENT":
            values = _numbers(line)
            if not values:
                raise ValueError(f"Missing EXPONENT value in {path}")
            exponent = int(values[0])
        elif label == "END OF HEADER":
            header_end = index
            break

    if exponent is None or header_end is None:
        raise ValueError(f"Incomplete IONEX header in {path}")

    scale = 10.0**exponent
    maps: list[IonexMap] = []
    index = header_end + 1

    while index < len(lines):
        line = lines[index]
        label = line[60:].strip() if len(line) > 60 else ""
        if label != "START OF TEC MAP":
            index += 1
            continue

        index += 1
        while index < len(lines):
            label = lines[index][60:].strip() if len(lines[index]) > 60 else ""
            if label == "EPOCH OF CURRENT MAP":
                map_time = _epoch(lines[index])
                index += 1
                break
            index += 1
        else:
            raise ValueError(f"TEC map without epoch in {path}")

        latitudes: list[float] = []
        longitude_grid: np.ndarray | None = None
        rows: list[np.ndarray] = []

        while index < len(lines):
            line = lines[index]
            label = line[60:].strip() if len(line) > 60 else ""
            if label == "END OF TEC MAP":
                index += 1
                break
            if label != "LAT/LON1/LON2/DLON/H":
                index += 1
                continue

            values = _numbers(line)
            if len(values) < 5:
                raise ValueError(f"Invalid latitude row header in {path}: {line!r}")
            latitude, lon1, lon2, dlon, _height = values[:5]
            point_count = int(round((lon2 - lon1) / dlon)) + 1
            row_values: list[float] = []
            index += 1
            while index < len(lines) and len(row_values) < point_count:
                data_line = lines[index]
                try:
                    values_on_line = [float(value) for value in data_line.split()]
                except ValueError as exc:
                    raise ValueError(
                        f"IONEX row ended after {len(row_values)}/{point_count} values "
                        f"in {path}"
                    ) from exc
                if not values_on_line:
                    raise ValueError(
                        f"IONEX row ended after {len(row_values)}/{point_count} values "
                        f"in {path}"
                    )
                row_values.extend(values_on_line)
                index += 1

            row = np.asarray(row_values[:point_count], dtype=np.float32)
            row[row >= 9999.0] = np.nan
            row *= scale
            current_lon = lon1 + np.arange(point_count, dtype=np.float64) * dlon
            if longitude_grid is None:
                longitude_grid = current_lon
            elif not np.allclose(longitude_grid, current_lon, atol=1e-7):
                raise ValueError(f"Longitude grid changes within TEC map in {path}")
            latitudes.append(latitude)
            rows.append(row)
        else:
            raise ValueError(f"Unterminated TEC map in {path}")

        if not rows or longitude_grid is None:
            raise ValueError(f"Empty TEC map at {map_time.isoformat()} in {path}")

        lat = np.asarray(latitudes, dtype=np.float64)
        vtec = np.stack(rows, axis=0)
        lat, lon, vtec = _canonicalize_grid(lat, longitude_grid, vtec)
        maps.append(IonexMap(time=map_time, lat=lat, lon=lon, vtec=vtec))

    if not maps:
        raise ValueError(f"No TEC maps found in {path}")
    return maps
