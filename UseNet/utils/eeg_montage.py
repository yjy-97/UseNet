from __future__ import annotations

import json
import math
import os
import warnings
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


_STANDARD_2D: Dict[str, Tuple[float, float]] = {
    "FP1": (-0.32, 0.92), "FPZ": (0.00, 0.96), "FP2": (0.32, 0.92),
    "AF7": (-0.58, 0.78), "AF3": (-0.28, 0.76), "AFZ": (0.00, 0.80),
    "AF4": (0.28, 0.76), "AF8": (0.58, 0.78),
    "F7": (-0.78, 0.52), "F5": (-0.55, 0.50), "F3": (-0.32, 0.48),
    "F1": (-0.12, 0.46), "FZ": (0.00, 0.48), "F2": (0.12, 0.46),
    "F4": (0.32, 0.48), "F6": (0.55, 0.50), "F8": (0.78, 0.52),
    "FT7": (-0.88, 0.27), "FC5": (-0.60, 0.25), "FC3": (-0.38, 0.24),
    "FC1": (-0.15, 0.23), "FCZ": (0.00, 0.24), "FC2": (0.15, 0.23),
    "FC4": (0.38, 0.24), "FC6": (0.60, 0.25), "FT8": (0.88, 0.27),
    "T7": (-0.96, 0.00), "C5": (-0.66, 0.00), "C3": (-0.40, 0.00),
    "C1": (-0.16, 0.00), "CZ": (0.00, 0.00), "C2": (0.16, 0.00),
    "C4": (0.40, 0.00), "C6": (0.66, 0.00), "T8": (0.96, 0.00),
    "TP7": (-0.88, -0.27), "CP5": (-0.60, -0.25), "CP3": (-0.38, -0.24),
    "CP1": (-0.15, -0.23), "CPZ": (0.00, -0.24), "CP2": (0.15, -0.23),
    "CP4": (0.38, -0.24), "CP6": (0.60, -0.25), "TP8": (0.88, -0.27),
    "P7": (-0.78, -0.52), "P5": (-0.55, -0.50), "P3": (-0.32, -0.48),
    "P1": (-0.12, -0.46), "PZ": (0.00, -0.48), "P2": (0.12, -0.46),
    "P4": (0.32, -0.48), "P6": (0.55, -0.50), "P8": (0.78, -0.52),
    "PO7": (-0.58, -0.78), "PO3": (-0.28, -0.76), "POZ": (0.00, -0.80),
    "PO4": (0.28, -0.76), "PO8": (0.58, -0.78),
    "O1": (-0.32, -0.92), "OZ": (0.00, -0.96), "O2": (0.32, -0.92),
    "A1": (-1.00, -0.05), "A2": (1.00, -0.05),
    "M1": (-1.00, -0.05), "M2": (1.00, -0.05),
}

_ALIASES = {
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
    "T1": "FT7", "T2": "FT8",
}

STANDARD_19 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3",
    "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
]

STANDARD_32 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2",
    "FC6", "T7", "C3", "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6",
    "P7", "P3", "Pz", "P4", "P8", "PO3", "PO4", "O1", "Oz", "O2",
    "A1", "A2",
]


def _canonical_name(name: str) -> str:
    key = str(name).strip().upper().replace("-REF", "").replace("_REF", "")
    return _ALIASES.get(key, key)


def _project_2d_to_unit_sphere(x: float, y: float) -> Tuple[float, float, float]:
    radius = math.sqrt(x * x + y * y)
    if radius >= 0.995:
        scale = 0.995 / max(radius, 1e-8)
        x, y = x * scale, y * scale
    z = math.sqrt(max(1.0 - x * x - y * y, 1e-6))
    norm = math.sqrt(x * x + y * y + z * z)
    return x / norm, y / norm, z / norm


def coordinates_from_names(names: Sequence[str]) -> torch.Tensor:
    coords: List[Tuple[float, float, float]] = []
    unknown: List[str] = []
    for name in names:
        key = _canonical_name(name)
        if key not in _STANDARD_2D:
            unknown.append(str(name))
            coords.append((float("nan"), float("nan"), float("nan")))
        else:
            coords.append(_project_2d_to_unit_sphere(*_STANDARD_2D[key]))
    if unknown:
        raise ValueError(
            "Unknown EEG channel name(s): " + ", ".join(unknown) +
            ". Supply exact coordinates through --channel_config."
        )
    return torch.tensor(coords, dtype=torch.float32)


def fibonacci_hemisphere(count: int) -> torch.Tensor:
    """Deterministic fallback coordinates for unknown channel layouts.

    These points are *not* anatomical channel coordinates.  They only keep the
    program executable and are accompanied by a warning.  Experimental results
    should use an explicit montage configuration.
    """
    if count <= 0:
        raise ValueError("Channel count must be positive.")
    points = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(count):
        z = 0.12 + 0.88 * (1.0 - (i + 0.5) / count)
        radius = math.sqrt(max(1.0 - z * z, 0.0))
        theta = i * golden
        points.append((radius * math.cos(theta), radius * math.sin(theta), z))
    return torch.tensor(points, dtype=torch.float32)


def _normalize_coordinates(coords: torch.Tensor) -> torch.Tensor:
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Coordinates must have shape [channels, 3], got {tuple(coords.shape)}")
    if not torch.isfinite(coords).all():
        raise ValueError("Channel coordinates contain NaN or infinity.")
    return coords / coords.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _read_json_layout(path: str) -> Tuple[List[str], torch.Tensor]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    names: List[str] = []
    coords: List[Sequence[float]] = []

    if isinstance(obj, dict) and "channels" in obj:
        for item in obj["channels"]:
            if not isinstance(item, dict) or "name" not in item:
                raise ValueError("Each item in channel_config['channels'] needs a 'name'.")
            xyz = item.get("xyz", item.get("coord", item.get("coordinates")))
            if xyz is None:
                raise ValueError(f"Channel {item['name']} has no xyz coordinates.")
            names.append(str(item["name"]))
            coords.append(xyz)
    elif isinstance(obj, dict) and "channel_names" in obj:
        names = [str(v) for v in obj["channel_names"]]
        raw_coords = obj.get("coordinates", obj.get("coords"))
        if isinstance(raw_coords, dict):
            coords = [raw_coords[name] for name in names]
        elif raw_coords is not None:
            coords = raw_coords
        else:
            return names, coordinates_from_names(names)
    else:
        raise ValueError(
            "channel_config must contain either {'channels': [...]} or "
            "{'channel_names': [...], 'coordinates': ...}."
        )

    tensor = torch.tensor(coords, dtype=torch.float32)
    if len(names) != tensor.shape[0]:
        raise ValueError("Number of channel names and coordinates does not match.")
    return names, _normalize_coordinates(tensor)


def load_source_layout(
    source_channels: int,
    channel_config: str = "",
    channel_names: str = "",
) -> Tuple[List[str], torch.Tensor, bool]:
    """Return source names, coordinates and whether the layout is anatomical."""
    if channel_config:
        if not os.path.exists(channel_config):
            raise FileNotFoundError(f"Channel configuration not found: {channel_config}")
        names, coords = _read_json_layout(channel_config)
        if len(names) != source_channels:
            raise ValueError(
                f"channel_config contains {len(names)} channels, but data has "
                f"{source_channels} channels."
            )
        return names, coords, True

    if channel_names:
        names = [part.strip() for part in channel_names.split(",") if part.strip()]
        if len(names) != source_channels:
            raise ValueError(
                f"--channel_names contains {len(names)} entries, but data has "
                f"{source_channels} channels."
            )
        return names, _normalize_coordinates(coordinates_from_names(names)), True

    if source_channels == len(STANDARD_19):
        warnings.warn(
            "No channel configuration was supplied. Assuming the built-in 19-channel "
            "10-20 order. Pass --channel_config if the actual order differs.",
            RuntimeWarning,
        )
        return STANDARD_19.copy(), _normalize_coordinates(coordinates_from_names(STANDARD_19)), True

    if source_channels == len(STANDARD_32):
        warnings.warn(
            "No channel configuration was supplied. Assuming the built-in 32-channel "
            "order. Pass --channel_config if the actual order differs.",
            RuntimeWarning,
        )
        return STANDARD_32.copy(), _normalize_coordinates(coordinates_from_names(STANDARD_32)), True

    warnings.warn(
        f"No anatomical layout is available for {source_channels} channels. "
        "Using deterministic pseudo-coordinates only as a runtime fallback. "
        "Provide --channel_config before reporting scientific results.",
        RuntimeWarning,
    )
    names = [f"CH{i + 1}" for i in range(source_channels)]
    return names, fibonacci_hemisphere(source_channels), False


def load_virtual_layout(
    virtual_channels: int,
    virtual_channel_names: str = "",
) -> Tuple[List[str], torch.Tensor]:
    if virtual_channel_names:
        names = [part.strip() for part in virtual_channel_names.split(",") if part.strip()]
        if len(names) != virtual_channels:
            raise ValueError(
                f"--virtual_channel_names contains {len(names)} entries, expected "
                f"{virtual_channels}."
            )
        return names, _normalize_coordinates(coordinates_from_names(names))

    if virtual_channels == 19:
        names = STANDARD_19.copy()
        return names, _normalize_coordinates(coordinates_from_names(names))

    if virtual_channels <= len(STANDARD_32):
        if virtual_channels == 1:
            indices = [len(STANDARD_32) // 2]
        else:
            indices = [round(i * (len(STANDARD_32) - 1) / (virtual_channels - 1)) for i in range(virtual_channels)]
        names = [STANDARD_32[i] for i in indices]
        return names, _normalize_coordinates(coordinates_from_names(names))

    names = [f"VIRTUAL_{i + 1}" for i in range(virtual_channels)]
    warnings.warn(
        "More than 32 virtual channels requested without names; using pseudo-coordinates.",
        RuntimeWarning,
    )
    return names, fibonacci_hemisphere(virtual_channels)


def parse_frequency_bands(text: str, sampling_rate: float) -> List[Tuple[float, float]]:
    bands: List[Tuple[float, float]] = []
    nyquist = sampling_rate / 2.0
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        pieces = token.replace(":", "-").split("-")
        if len(pieces) != 2:
            raise ValueError(
                f"Invalid frequency band '{token}'. Use forms such as 1-4,4-8,8-13."
            )
        low, high = float(pieces[0]), float(pieces[1])
        if not 0 <= low < high:
            raise ValueError(f"Invalid frequency band '{token}'.")
        clipped_high = min(high, nyquist)
        if low < clipped_high:
            bands.append((low, clipped_high))
    if not bands:
        raise ValueError(
            f"No valid frequency bands remain below Nyquist ({nyquist:g} Hz)."
        )
    return bands
