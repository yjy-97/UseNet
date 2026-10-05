from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
except Exception as exc:  # pragma: no cover - matplotlib import failure is runtime-specific
    raise ImportError(
        "usenet_visualization.py requires matplotlib. Please install matplotlib first."
    ) from exc


def _configure_times_new_roman() -> None:
    """Use Times New Roman for all VEUS visualization figures.

    If Times New Roman is not installed on the running system, matplotlib will
    fall back to the next serif font in the list.  The code still keeps the
    figure style compatible with manuscript requirements on machines where
    Times New Roman is available.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


_configure_times_new_roman()

try:  # scipy is optional.
    from scipy.interpolate import griddata as _scipy_griddata
except Exception:  # pragma: no cover
    _scipy_griddata = None

TensorLike = Union[torch.Tensor, np.ndarray, Sequence[float]]


DEFAULT_BAND_NAMES = ["δ", "θ", "α", "β", "γ"]


_STANDARD_2D_POS: Dict[str, Tuple[float, float]] = {
    "FP1": (-0.32, 0.92), "FPZ": (0.00, 0.96), "FP2": (0.32, 0.92),
    "AF7": (-0.58, 0.80), "AF3": (-0.30, 0.76), "AFZ": (0.00, 0.78),
    "AF4": (0.30, 0.76), "AF8": (0.58, 0.80),
    "F7": (-0.78, 0.55), "F5": (-0.55, 0.50), "F3": (-0.38, 0.45),
    "F1": (-0.18, 0.42), "FZ": (0.00, 0.43), "F2": (0.18, 0.42),
    "F4": (0.38, 0.45), "F6": (0.55, 0.50), "F8": (0.78, 0.55),
    "FT9": (-0.96, 0.32), "FT7": (-0.86, 0.30), "FC5": (-0.62, 0.24),
    "FC3": (-0.42, 0.20), "FC1": (-0.20, 0.18), "FCZ": (0.00, 0.18),
    "FC2": (0.20, 0.18), "FC4": (0.42, 0.20), "FC6": (0.62, 0.24),
    "FT8": (0.86, 0.30), "FT10": (0.96, 0.32),
    "T7": (-0.95, 0.00), "T3": (-0.95, 0.00), "C5": (-0.66, 0.00),
    "C3": (-0.44, 0.00), "C1": (-0.22, 0.00), "CZ": (0.00, 0.00),
    "C2": (0.22, 0.00), "C4": (0.44, 0.00), "C6": (0.66, 0.00),
    "T8": (0.95, 0.00), "T4": (0.95, 0.00),
    "TP9": (-0.96, -0.32), "TP7": (-0.86, -0.30), "CP5": (-0.62, -0.24),
    "CP3": (-0.42, -0.20), "CP1": (-0.20, -0.18), "CPZ": (0.00, -0.18),
    "CP2": (0.20, -0.18), "CP4": (0.42, -0.20), "CP6": (0.62, -0.24),
    "TP8": (0.86, -0.30), "TP10": (0.96, -0.32),
    "P7": (-0.78, -0.55), "T5": (-0.78, -0.55), "P5": (-0.55, -0.50),
    "P3": (-0.38, -0.45), "P1": (-0.18, -0.42), "PZ": (0.00, -0.43),
    "P2": (0.18, -0.42), "P4": (0.38, -0.45), "P6": (0.55, -0.50),
    "P8": (0.78, -0.55), "T6": (0.78, -0.55),
    "PO7": (-0.58, -0.80), "PO3": (-0.30, -0.76), "POZ": (0.00, -0.78),
    "PO4": (0.30, -0.76), "PO8": (0.58, -0.80),
    "O1": (-0.32, -0.92), "OZ": (0.00, -0.96), "O2": (0.32, -0.92),
    "A1": (-1.05, 0.00), "M1": (-1.05, 0.00), "A2": (1.05, 0.00), "M2": (1.05, 0.00),
}


def _to_numpy(x: TensorLike) -> np.ndarray:
    """Convert tensor-like input to a detached float NumPy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _safe_minmax(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    v_min = np.nanmin(values)
    v_max = np.nanmax(values)
    if not np.isfinite(v_min) or not np.isfinite(v_max) or abs(v_max - v_min) < eps:
        return np.zeros_like(values, dtype=np.float32)
    return (values - v_min) / (v_max - v_min + eps)


def _standardize_name(name: str) -> str:
    name = str(name).strip().upper().replace(" ", "")
    name = name.replace("-REF", "").replace("_REF", "")
    return name


def _fallback_circle_layout(num_channels: int) -> np.ndarray:
    """Stable fallback layout for unknown channel names."""
    if num_channels <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, num_channels, endpoint=False)
    radius = 0.82
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.stack([x, y], axis=1).astype(np.float32)


def get_channel_xy(
    channel_names: Sequence[str],
    coordinates: Optional[Union[TensorLike, Dict[str, Tuple[float, float]]]] = None,
) -> np.ndarray:
    """Return 2D channel coordinates in a unit head circle.

    Args:
        channel_names: Names such as ``["Fp1", "F3", "Cz", ...]``.
        coordinates: Optional explicit coordinates.  If an array is provided, the
            first two dimensions are used.  If a dict is provided, keys are matched
            by channel names.

    Unknown channel names fall back to a deterministic circular layout, but known
    channels still keep their standard 10-20 positions.
    """
    names = list(channel_names)
    n = len(names)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if coordinates is not None:
        if isinstance(coordinates, dict):
            pos = []
            fallback = _fallback_circle_layout(n)
            for idx, name in enumerate(names):
                key = _standardize_name(name)
                value = coordinates.get(name, coordinates.get(key, None))
                if value is None:
                    pos.append(fallback[idx])
                else:
                    arr = np.asarray(value, dtype=np.float32).reshape(-1)
                    pos.append(arr[:2] if arr.size >= 2 else fallback[idx])
            pos_arr = np.asarray(pos, dtype=np.float32)
            max_abs = np.max(np.abs(pos_arr))
            if max_abs > 1.25:
                pos_arr = pos_arr / max_abs
            return pos_arr
        arr = _to_numpy(coordinates)
        if arr.ndim == 2 and arr.shape[0] == n and arr.shape[1] >= 2:
            pos_arr = arr[:, :2].astype(np.float32)
            max_abs = np.max(np.abs(pos_arr))
            if max_abs > 1.25:
                pos_arr = pos_arr / max_abs
            return pos_arr

    fallback = _fallback_circle_layout(n)
    pos = []
    for idx, name in enumerate(names):
        key = _standardize_name(name)
        pos.append(_STANDARD_2D_POS.get(key, fallback[idx]))
    return np.asarray(pos, dtype=np.float32)


def _interpolate_on_head(
    xy: np.ndarray,
    values: np.ndarray,
    grid_res: int = 160,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate sparse electrode values onto a scalp grid."""
    xy = np.asarray(xy, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if xy.shape[0] != values.shape[0]:
        raise ValueError(f"xy and values length mismatch: {xy.shape[0]} vs {values.shape[0]}")
    if xy.shape[0] == 0:
        raise ValueError("No channels are available for topomap visualization.")

    grid_axis = np.linspace(-1.05, 1.05, int(grid_res), dtype=np.float32)
    gx, gy = np.meshgrid(grid_axis, grid_axis)
    head_mask = gx ** 2 + gy ** 2 <= 1.0 ** 2

    valid = np.isfinite(values) & np.isfinite(xy).all(axis=1)
    xy_valid = xy[valid]
    val_valid = values[valid]
    if xy_valid.shape[0] == 0:
        zi = np.zeros_like(gx, dtype=np.float32)
        zi[~head_mask] = np.nan
        return gx, gy, zi
    if xy_valid.shape[0] == 1:
        zi = np.full_like(gx, float(val_valid[0]), dtype=np.float32)
        zi[~head_mask] = np.nan
        return gx, gy, zi

    if _scipy_griddata is not None and xy_valid.shape[0] >= 4:
        points = xy_valid
        targets = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)
        zi_linear = _scipy_griddata(points, val_valid, targets, method="cubic")
        zi_nearest = _scipy_griddata(points, val_valid, targets, method="nearest")
        zi = np.where(np.isfinite(zi_linear), zi_linear, zi_nearest).reshape(gx.shape)
    else:
        targets = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)
        distance = np.linalg.norm(targets[:, None, :] - xy_valid[None, :, :], axis=-1)
        weights = 1.0 / np.maximum(distance, 1e-4) ** 2
        zi = (weights @ val_valid.reshape(-1, 1)).reshape(gx.shape) / weights.sum(axis=1).reshape(gx.shape)

    zi = zi.astype(np.float32)
    zi[~head_mask] = np.nan
    return gx, gy, zi


def _draw_head_outline(ax) -> None:
    theta = np.linspace(0, 2 * np.pi, 256)
    ax.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1.2, alpha=0.9)
    ax.plot([-0.10, 0.00, 0.10], [0.99, 1.13, 0.99], color="black", linewidth=1.0, alpha=0.9)
    ear_y = np.linspace(-0.24, 0.24, 40)
    left_ear_x = -1.05 - 0.08 * np.cos(np.linspace(-np.pi, np.pi, 40))
    right_ear_x = 1.05 + 0.08 * np.cos(np.linspace(-np.pi, np.pi, 40))
    ax.plot(left_ear_x, ear_y, color="black", linewidth=1.0, alpha=0.9)
    ax.plot(right_ear_x, ear_y, color="black", linewidth=1.0, alpha=0.9)


def plot_single_topomap(
    values: TensorLike,
    channel_names: Sequence[str],
    ax=None,
    coordinates: Optional[Union[TensorLike, Dict[str, Tuple[float, float]]]] = None,
    title: str = "",
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center_zero: bool = False,
    grid_res: int = 160,
    show_channels: bool = True,
    show_names: bool = False,
    contour: bool = True,
    colorbar: bool = False,
):
    """Plot one EEG-style scalp topomap.

    Args:
        values: Channel scores shaped [C].
        channel_names: Source or virtual channel names.
        center_zero: If True, use a diverging normalization centered at 0.
        colorbar: Whether to attach a colorbar to this axis.

    Returns:
        ``(fig, ax, image)``.
    """
    _configure_times_new_roman()

    values_np = _to_numpy(values).reshape(-1).astype(np.float32)
    if len(channel_names) != values_np.shape[0]:
        channel_names = [f"Ch{i + 1}" for i in range(values_np.shape[0])]
    xy = get_channel_xy(channel_names, coordinates=coordinates)

    if vmin is None:
        vmin = float(np.nanmin(values_np)) if np.isfinite(values_np).any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(values_np)) if np.isfinite(values_np).any() else 1.0
    if abs(vmax - vmin) < 1e-8:
        vmax = vmin + 1.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=150)
    else:
        fig = ax.figure

    gx, gy, zi = _interpolate_on_head(xy, values_np, grid_res=grid_res)
    if center_zero:
        abs_max = max(abs(float(vmin)), abs(float(vmax)), 1e-6)
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    else:
        norm = Normalize(vmin=float(vmin), vmax=float(vmax))

    image = ax.imshow(
        zi,
        extent=[gx.min(), gx.max(), gy.min(), gy.max()],
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",
    )
    if contour and np.isfinite(zi).sum() > 20:
        try:
            ax.contour(gx, gy, zi, levels=6, colors="k", linewidths=0.35, alpha=0.35)
        except Exception:
            pass

    _draw_head_outline(ax)
    if show_channels:
        ax.scatter(xy[:, 0], xy[:, 1], s=10, c="black", alpha=0.75, linewidths=0.0)
    if show_names:
        for (x_i, y_i), name in zip(xy, channel_names):
            ax.text(x_i, y_i, str(name), ha="center", va="center", fontsize=6, color="black")

    ax.set_title(title, fontsize=10, pad=4)
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")

    if colorbar:
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    return fig, ax, image


def _names_from_explanation(explanation, mode: str, length: int) -> List[str]:
    if mode == "virtual":
        names = list(getattr(explanation, "virtual_channel_names", []) or [])
    else:
        names = list(getattr(explanation, "source_channel_names", []) or [])
    if len(names) != length:
        prefix = "V" if mode == "virtual" else "Ch"
        names = [f"{prefix}{idx + 1}" for idx in range(length)]
    return names


def plot_faot_topomap_panel(
    explanation,
    sample_index: int = 0,
    mode: str = "source",
    band_names: Optional[Sequence[str]] = None,
    include_overall: bool = True,
    save_path: Optional[Union[str, os.PathLike]] = None,
    cmap: str = "RdBu_r",
    show_names: bool = False,
    individual_colorbars: bool = False,
    grid_res: int = 160,
    dpi: int = 300,
    horizontal: bool = True,
):
    """Plot VEUS explanation topomaps as a manuscript-ready panel.

    The default layout is a single horizontal row, e.g.

        Overall | delta | theta | alpha | beta | gamma | colorbar

    This avoids the previous two-row layout where the shared colorbar could
    overlap with the last topomap.  All text uses Times New Roman through the
    module-level matplotlib configuration.

    Args:
        explanation: VEUSExplanation object returned by compute_usenet_explanation.
        sample_index: Which sample in the batch to visualize.
        mode: ``"source"`` plots original-electrode attribution; ``"virtual"``
            plots fixed virtual-channel attribution.
        band_names: Names for the frequency bands. Default is delta/theta/alpha/beta/gamma.
        include_overall: Add an overall topomap before band-specific maps.
        save_path: If provided, save the figure to this path.
        horizontal: If True, draw all topomaps in one horizontal row.  If False,
            keep a compact grid layout for very small display areas.

    Returns:
        Matplotlib figure.
    """
    _configure_times_new_roman()

    if mode not in {"source", "virtual"}:
        raise ValueError("mode must be 'source' or 'virtual'.")

    if mode == "source":
        overall = _to_numpy(explanation.source_score[sample_index])
        band_scores = _to_numpy(explanation.source_band_score[sample_index])  # [F,C]
        names = _names_from_explanation(explanation, mode="source", length=overall.shape[0])
    else:
        overall = _to_numpy(explanation.virtual_score[sample_index])
        band_scores = _to_numpy(explanation.virtual_band_score[sample_index])  # [F,K]
        names = _names_from_explanation(explanation, mode="virtual", length=overall.shape[0])

    num_bands = int(band_scores.shape[0])
    if band_names is None:
        if num_bands <= len(DEFAULT_BAND_NAMES):
            band_names = DEFAULT_BAND_NAMES[:num_bands]
        else:
            band_names = [f"Band {idx + 1}" for idx in range(num_bands)]
    else:
        band_names = list(band_names)
        if len(band_names) < num_bands:
            band_names = band_names + [f"Band {idx + 1}" for idx in range(len(band_names), num_bands)]

    panels: List[Tuple[str, np.ndarray]] = []
    if include_overall:
        panels.append(("Overall", overall))
    for b in range(num_bands):
        panels.append((str(band_names[b]), band_scores[b]))

    num_panels = len(panels)
    all_values = np.concatenate([panel_values.reshape(-1) for _, panel_values in panels], axis=0)
    vmin = float(np.nanmin(all_values)) if np.isfinite(all_values).any() else 0.0
    vmax = float(np.nanmax(all_values)) if np.isfinite(all_values).any() else 1.0
    if abs(vmax - vmin) < 1e-8:
        vmax = vmin + 1.0

    if horizontal:
        fig_width = max(2.45 * num_panels + 1.05, 10.0)
        fig_height = 3.25
        fig, axes = plt.subplots(1, num_panels, figsize=(fig_width, fig_height), dpi=dpi)
        axes = np.asarray(axes).reshape(-1)
        fig.subplots_adjust(
            left=0.018,
            right=0.885,
            bottom=0.055,
            top=0.765,
            wspace=0.16,
        )
    else:
        ncols = min(4, num_panels)
        nrows = int(math.ceil(num_panels / ncols))
        fig_width = max(3.0 * ncols + 0.7, 6.0)
        fig_height = 3.05 * nrows
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), dpi=dpi)
        axes = np.asarray(axes).reshape(-1)
        fig.subplots_adjust(
            left=0.035,
            right=0.890,
            bottom=0.055,
            top=0.885,
            wspace=0.22,
            hspace=0.28,
        )

    last_image = None
    for ax, (title, values) in zip(axes, panels):
        _, _, image = plot_single_topomap(
            values=values,
            channel_names=names,
            ax=ax,
            title=title,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center_zero=False,
            grid_res=grid_res,
            show_channels=True,
            show_names=show_names,
            contour=True,
            colorbar=individual_colorbars,
        )
        last_image = image

    for ax in axes[num_panels:]:
        ax.axis("off")

    if (not individual_colorbars) and last_image is not None:
        if horizontal:
            cax = fig.add_axes([0.915, 0.175, 0.012, 0.565])
        else:
            cax = fig.add_axes([0.925, 0.155, 0.014, 0.690])
        cbar = fig.colorbar(last_image, cax=cax)
        cbar.ax.tick_params(labelsize=9)
        cbar.outline.set_linewidth(0.8)

    pred = int(explanation.target_class[sample_index].detach().cpu().item())
    prob = float(explanation.probabilities[sample_index].detach().cpu().max().item())
    fig.suptitle(
        f"VEUS explanation topomaps | target/pred class: {pred} | probability: {prob:.3f}",
        fontsize=12,
        y=0.965 if horizontal else 0.985,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_channel_band_heatmap(
    explanation,
    sample_index: int = 0,
    mode: str = "source",
    band_names: Optional[Sequence[str]] = None,
    top_k: Optional[int] = 20,
    save_path: Optional[Union[str, os.PathLike]] = None,
    cmap: str = "RdBu_r",
    dpi: int = 300,
):
    """Plot channel-by-band attribution heatmap.

    This is useful when the topomap shows where the abnormal evidence is, while
    the heatmap shows which frequency bands drive each electrode/virtual region.
    """
    _configure_times_new_roman()

    if mode not in {"source", "virtual"}:
        raise ValueError("mode must be 'source' or 'virtual'.")
    if mode == "source":
        matrix = _to_numpy(explanation.source_band_score[sample_index]).T  # [C,F]
        overall = _to_numpy(explanation.source_score[sample_index])
        names = _names_from_explanation(explanation, mode="source", length=matrix.shape[0])
    else:
        matrix = _to_numpy(explanation.virtual_band_score[sample_index]).T  # [K,F]
        overall = _to_numpy(explanation.virtual_score[sample_index])
        names = _names_from_explanation(explanation, mode="virtual", length=matrix.shape[0])

    if top_k is not None and matrix.shape[0] > int(top_k):
        order = np.argsort(-overall)[: int(top_k)]
        matrix = matrix[order]
        names = [names[i] for i in order]
    else:
        order = np.argsort(-overall)
        matrix = matrix[order]
        names = [names[i] for i in order]

    num_bands = matrix.shape[1]
    if band_names is None:
        band_names = DEFAULT_BAND_NAMES[:num_bands] if num_bands <= len(DEFAULT_BAND_NAMES) else [f"B{i+1}" for i in range(num_bands)]
    else:
        band_names = list(band_names)
        if len(band_names) < num_bands:
            band_names = band_names + [f"B{i+1}" for i in range(len(band_names), num_bands)]

    fig_height = max(3.2, 0.28 * len(names) + 1.2)
    fig, ax = plt.subplots(figsize=(5.6, fig_height), dpi=dpi)
    matrix_max = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0
    if matrix_max <= 1e-8:
        matrix_max = 1.0
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=matrix_max)
    ax.set_xticks(np.arange(num_bands))
    ax.set_xticklabels(band_names)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Frequency band")
    ax.set_ylabel("Channel" if mode == "source" else "Virtual region")
    ax.set_title("Channel-band abnormal contribution")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_topk_channel_bar(
    explanation,
    sample_index: int = 0,
    mode: str = "source",
    top_k: int = 10,
    save_path: Optional[Union[str, os.PathLike]] = None,
    dpi: int = 300,
):
    """Plot Top-K original electrodes or virtual regions as a horizontal bar chart."""
    _configure_times_new_roman()

    if mode not in {"source", "virtual"}:
        raise ValueError("mode must be 'source' or 'virtual'.")
    if mode == "source":
        scores = _to_numpy(explanation.source_score[sample_index])
        names = _names_from_explanation(explanation, mode="source", length=scores.shape[0])
        ylabel = "Source channel"
    else:
        scores = _to_numpy(explanation.virtual_score[sample_index])
        names = _names_from_explanation(explanation, mode="virtual", length=scores.shape[0])
        ylabel = "Virtual region"

    k = min(int(top_k), scores.shape[0])
    order = np.argsort(scores)[-k:][::-1]
    labels = [names[i] for i in order][::-1]
    values = scores[order][::-1]

    fig_height = max(3.0, 0.32 * k + 1.2)
    fig, ax = plt.subplots(figsize=(5.6, fig_height), dpi=dpi)
    ax.barh(np.arange(k), values)
    ax.set_yticks(np.arange(k))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Normalized contribution")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, max(1.0, float(values.max()) * 1.05))
    ax.set_title(f"Top-{k} abnormal contribution")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig




def compute_band_similarity_diagnostics(
    explanation,
    sample_index: int = 0,
    mode: str = "source",
) -> Dict[str, object]:
    """Return simple diagnostics for checking whether band maps are duplicated.

    The returned correlation matrix is useful for debugging.  If most
    off-diagonal correlations are above 0.95, the band-level explanation is
    still too similar across bands.
    """
    if mode == "source":
        band_scores = _to_numpy(explanation.source_band_score[sample_index])
    elif mode == "virtual":
        band_scores = _to_numpy(explanation.virtual_band_score[sample_index])
    else:
        raise ValueError("mode must be 'source' or 'virtual'.")

    num_bands = band_scores.shape[0]
    corr = np.eye(num_bands, dtype=np.float32)
    for i in range(num_bands):
        for j in range(i + 1, num_bands):
            a = band_scores[i].reshape(-1)
            b = band_scores[j].reshape(-1)
            if np.std(a) < 1e-8 or np.std(b) < 1e-8:
                r = 0.0
            else:
                r = float(np.corrcoef(a, b)[0, 1])
            corr[i, j] = corr[j, i] = r

    stats = []
    for i in range(num_bands):
        values = band_scores[i]
        stats.append({
            "band_index": i,
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
        })
    return {"correlation": corr, "stats": stats}


def save_band_similarity_diagnostics(
    explanation,
    sample_index: int,
    mode: str,
    save_path: Union[str, os.PathLike],
) -> str:
    """Save band-map min/max/mean/std and correlation matrix as a text file."""
    diag = compute_band_similarity_diagnostics(explanation, sample_index=sample_index, mode=mode)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"VEUS band similarity diagnostics | sample={sample_index} | mode={mode}\n")
        f.write("Band statistics:\n")
        for row in diag["stats"]:
            f.write(
                f"  band {row['band_index']}: min={row['min']:.6f}, "
                f"max={row['max']:.6f}, mean={row['mean']:.6f}, std={row['std']:.6f}\n"
            )
        f.write("Correlation matrix:\n")
        corr = diag["correlation"]
        for i in range(corr.shape[0]):
            f.write("  " + " ".join(f"{corr[i, j]:.4f}" for j in range(corr.shape[1])) + "\n")
    return str(save_path)


def save_usenet_explanation_report(
    explanation,
    sample_index: int = 0,
    output_dir: Union[str, os.PathLike] = "usenet_explanation_figures",
    prefix: str = "sample",
    mode: str = "source",
    band_names: Optional[Sequence[str]] = None,
    top_k: int = 10,
    show_names: bool = False,
    dpi: int = 300,
) -> Dict[str, str]:
    """Save topomap panel, channel-band heatmap, and Top-K bar figure.

    Returns:
        A dictionary with paths of saved figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_{sample_index}_{mode}"

    paths = {
        "topomap_panel": str(output_dir / f"{stem}_topomap_panel.png"),
        "channel_band_heatmap": str(output_dir / f"{stem}_channel_band_heatmap.png"),
        "topk_bar": str(output_dir / f"{stem}_topk_bar.png"),
        "band_diagnostics": str(output_dir / f"{stem}_band_diagnostics.txt"),
    }
    figs = []
    figs.append(
        plot_faot_topomap_panel(
            explanation=explanation,
            sample_index=sample_index,
            mode=mode,
            band_names=band_names,
            save_path=paths["topomap_panel"],
            show_names=show_names,
            dpi=dpi,
        )
    )
    figs.append(
        plot_channel_band_heatmap(
            explanation=explanation,
            sample_index=sample_index,
            mode=mode,
            band_names=band_names,
            top_k=top_k,
            save_path=paths["channel_band_heatmap"],
            dpi=dpi,
        )
    )
    figs.append(
        plot_topk_channel_bar(
            explanation=explanation,
            sample_index=sample_index,
            mode=mode,
            top_k=top_k,
            save_path=paths["topk_bar"],
            dpi=dpi,
        )
    )
    save_band_similarity_diagnostics(
        explanation=explanation,
        sample_index=sample_index,
        mode=mode,
        save_path=paths["band_diagnostics"],
    )
    for fig in figs:
        plt.close(fig)
    return paths


@torch.no_grad()
def explain_and_save_topomaps(
    model,
    x: torch.Tensor,
    time_mask: Optional[torch.Tensor] = None,
    channel_mask: Optional[torch.Tensor] = None,
    target_class: Optional[Union[int, torch.Tensor]] = None,
    normal_stats: Optional[Dict[str, torch.Tensor]] = None,
    output_dir: Union[str, os.PathLike] = "usenet_explanation_figures",
    sample_index: int = 0,
    mode: str = "source",
    use_occlusion: bool = True,
    band_names: Optional[Sequence[str]] = None,
    top_k: int = 10,
    show_names: bool = False,
) -> Tuple[object, Dict[str, str]]:
    """Convenience wrapper: compute VEUS explanation and save topomap figures.

    Example:
        explanation, paths = explain_and_save_topomaps(
            model, batch_x, time_mask=padding_mask, output_dir="vis", mode="source"
        )
    """
    from utils.usenet_interpretability import compute_usenet_explanation

    with torch.enable_grad():
        explanation = compute_usenet_explanation(
            model=model,
            x=x,
            time_mask=time_mask,
            channel_mask=channel_mask,
            target_class=target_class,
            normal_stats=normal_stats,
            use_occlusion=use_occlusion,
        )
    paths = save_usenet_explanation_report(
        explanation=explanation,
        sample_index=sample_index,
        output_dir=output_dir,
        prefix="faot",
        mode=mode,
        band_names=band_names,
        top_k=top_k,
        show_names=show_names,
    )
    return explanation, paths


__all__ = [
    "DEFAULT_BAND_NAMES",
    "get_channel_xy",
    "plot_single_topomap",
    "plot_faot_topomap_panel",
    "plot_channel_band_heatmap",
    "plot_topk_channel_bar",
    "save_usenet_explanation_report",
    "explain_and_save_topomaps",
]
