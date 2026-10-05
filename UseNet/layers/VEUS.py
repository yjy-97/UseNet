from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VirtualEEGUnifiedSpatialRepresentation(nn.Module):
    def __init__(
        self,
        source_coordinates: torch.Tensor,
        virtual_coordinates: torch.Tensor,
        sampling_rate: float,
        frequency_bands: Sequence[Tuple[float, float]],
        sinkhorn_epsilon: float = 0.08,
        sinkhorn_iterations: int = 20,
        geometry_weight: float = 1.0,
        signal_weight: float = 0.25,
        quality_weight: float = 0.10,
        confidence_floor: float = 0.05,
        confidence_geometry_temperature: float = 0.35,
    ):
        super().__init__()
        if source_coordinates.ndim != 2 or source_coordinates.shape[1] != 3:
            raise ValueError("source_coordinates must have shape [C, 3].")
        if virtual_coordinates.ndim != 2 or virtual_coordinates.shape[1] != 3:
            raise ValueError("virtual_coordinates must have shape [K, 3].")
        if sinkhorn_epsilon <= 0:
            raise ValueError("sinkhorn_epsilon must be positive.")
        if sinkhorn_iterations <= 0:
            raise ValueError("sinkhorn_iterations must be positive.")

        self.register_buffer(
            "source_coordinates",
            F.normalize(source_coordinates.float(), dim=-1),
            persistent=False,
        )
        self.register_buffer(
            "virtual_coordinates",
            F.normalize(virtual_coordinates.float(), dim=-1),
            persistent=True,
        )

        self.sampling_rate = float(sampling_rate)
        self.frequency_bands = [(float(lo), float(hi)) for lo, hi in frequency_bands]
        self.num_bands = len(self.frequency_bands)
        self.virtual_channels = virtual_coordinates.shape[0]
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.geometry_weight = float(geometry_weight)
        self.signal_weight = float(signal_weight)
        self.quality_weight = float(quality_weight)
        self.confidence_floor = float(confidence_floor)
        self.confidence_geometry_temperature = float(confidence_geometry_temperature)

        self.band_region_prototypes = nn.Parameter(
            torch.zeros(self.num_bands, self.virtual_channels, 2)
        )
        nn.init.normal_(self.band_region_prototypes, mean=0.0, std=0.02)

        self.band_fusion_logits = nn.Parameter(
            torch.zeros(self.num_bands, self.virtual_channels)
        )

    @property
    def source_channels(self) -> int:
        return int(self.source_coordinates.shape[0])

    def set_source_coordinates(self, coordinates: torch.Tensor) -> None:
        """Replace runtime source coordinates without changing parameters."""
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("coordinates must have shape [C, 3].")
        self.source_coordinates = F.normalize(
            coordinates.to(device=self.source_coordinates.device, dtype=self.source_coordinates.dtype),
            dim=-1,
        )

    def _prepare_time_mask(
        self, x: torch.Tensor, time_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, time, _ = x.shape
        if time_mask is None:
            mask = torch.ones(batch, time, device=x.device, dtype=x.dtype)
        else:
            if time_mask.shape != (batch, time):
                raise ValueError(
                    f"time_mask must have shape {(batch, time)}, got {tuple(time_mask.shape)}"
                )
            mask = time_mask.to(device=x.device, dtype=x.dtype)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) * mask.unsqueeze(-1)
        return x, mask

    def infer_channel_quality(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
        time_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a detached quality score and a strict validity mask."""
        batch, time, channels = x.shape
        if channel_mask is None:
            valid = torch.ones(batch, channels, device=x.device, dtype=torch.bool)
        else:
            if channel_mask.ndim == 1:
                channel_mask = channel_mask.unsqueeze(0).expand(batch, -1)
            if channel_mask.shape != (batch, channels):
                raise ValueError(
                    f"channel_mask must have shape {(batch, channels)}, got {tuple(channel_mask.shape)}"
                )
            valid = channel_mask.to(device=x.device).bool()

        if time_mask is None:
            tm = torch.ones(batch, time, device=x.device, dtype=x.dtype)
        else:
            tm = time_mask.to(device=x.device, dtype=x.dtype)
        denom = tm.sum(dim=1, keepdim=True).clamp_min(2.0)
        mean = (x * tm.unsqueeze(-1)).sum(dim=1) / denom
        centered = (x - mean.unsqueeze(1)) * tm.unsqueeze(-1)
        var = centered.square().sum(dim=1) / denom
        valid = valid & torch.isfinite(var) & (var > 1e-8)

        fourth = centered.pow(4).sum(dim=1) / denom
        kurtosis = fourth / var.clamp_min(1e-8).square()
        flat_score = 1.0 - torch.exp(-var.clamp_min(0.0))
        tail_score = torch.exp(-F.relu(kurtosis - 10.0) / 10.0)
        quality = (flat_score * tail_score).clamp(0.02, 1.0)
        quality = quality * valid.to(quality.dtype)
        return quality.detach(), valid

    def _frequency_descriptors(
        self, x: torch.Tensor, time_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, time, channels = x.shape
        if time_mask is not None:
            x = x * time_mask.to(x.dtype).unsqueeze(-1)
        spectrum = torch.fft.rfft(x, dim=1)
        frequencies = torch.fft.rfftfreq(
            time, d=1.0 / self.sampling_rate, device=x.device
        ).to(x.dtype)
        spectral_power = spectrum.abs().square().clamp_min(1e-12)

        descriptor_list = []
        log_power_list = []
        for low, high in self.frequency_bands:
            select = (frequencies >= low) & (frequencies < high)
            if not torch.any(select):
                center = 0.5 * (low + high)
                nearest = torch.argmin((frequencies - center).abs())
                select = torch.zeros_like(frequencies, dtype=torch.bool)
                select[nearest] = True
            values = spectral_power[:, select, :]
            arithmetic = values.mean(dim=1).clamp_min(1e-12)
            geometric = torch.exp(torch.log(values).mean(dim=1))
            flatness = (geometric / arithmetic).clamp(0.0, 1.0)
            log_power = torch.log1p(arithmetic)
            descriptor_list.append(torch.stack([log_power, flatness], dim=-1))
            log_power_list.append(log_power)

        descriptors = torch.stack(descriptor_list, dim=1)
        raw_log_power = torch.stack(log_power_list, dim=1)
        return descriptors, raw_log_power

    @staticmethod
    def _masked_standardize(
        descriptors: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        weight = valid[:, None, :, None].to(descriptors.dtype)
        count = weight.sum(dim=2, keepdim=True).clamp_min(1.0)
        mean = (descriptors * weight).sum(dim=2, keepdim=True) / count
        variance = ((descriptors - mean).square() * weight).sum(dim=2, keepdim=True) / count
        normalized = (descriptors - mean) / torch.sqrt(variance + 1e-5)
        return normalized * weight

    def geometry_cost(self) -> torch.Tensor:
        return 1.0 - torch.matmul(
            self.virtual_coordinates, self.source_coordinates.transpose(0, 1)
        ).clamp(-1.0, 1.0)

    def _sinkhorn(
        self,
        cost: torch.Tensor,
        source_mass: torch.Tensor,
    ) -> torch.Tensor:
        batch, bands, targets, sources = cost.shape
        tiny = torch.finfo(cost.dtype).tiny
        source_mass = source_mass.clamp_min(tiny)
        source_mass = source_mass / source_mass.sum(dim=-1, keepdim=True).clamp_min(tiny)
        target_mass = torch.full(
            (batch, bands, targets),
            1.0 / targets,
            device=cost.device,
            dtype=cost.dtype,
        )
        source_mass = source_mass[:, None, :].expand(batch, bands, sources)

        log_kernel = -cost / self.sinkhorn_epsilon
        log_a = torch.log(target_mass.clamp_min(tiny))
        log_b = torch.log(source_mass.clamp_min(tiny))
        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)

        for _ in range(self.sinkhorn_iterations):
            log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
            log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)

        plan = torch.exp(log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2))
        return plan

    def forward(
        self,
        x: torch.Tensor,
        time_mask: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError(f"Expected EEG input [B,T,C], got {tuple(x.shape)}")
        batch, time, channels = x.shape
        if channels != self.source_channels:
            raise ValueError(
                f"Input has {channels} channels, but the current source montage has "
                f"{self.source_channels}."
            )

        x, time_mask = self._prepare_time_mask(x, time_mask)
        quality, valid = self.infer_channel_quality(x, channel_mask, time_mask)
        empty = ~valid.any(dim=-1)
        if empty.any():
            fallback = x.var(dim=1).argmax(dim=-1)
            valid = valid.clone()
            quality = quality.clone()
            valid[empty, fallback[empty]] = True
            quality[empty, fallback[empty]] = 1.0

        descriptors, source_band_log_power = self._frequency_descriptors(x, time_mask)
        descriptors = self._masked_standardize(descriptors, valid)

        geometry = self.geometry_cost().to(device=x.device, dtype=x.dtype)
        geometry = geometry[None, None, :, :]
        prototypes = self.band_region_prototypes.to(dtype=x.dtype)[None, :, :, None, :]
        signal_cost = (
            descriptors[:, :, None, :, :] - prototypes
        ).square().mean(dim=-1)
        quality_cost = (1.0 - quality)[:, None, None, :]
        total_cost = (
            self.geometry_weight * geometry
            + self.signal_weight * signal_cost
            + self.quality_weight * quality_cost
        )
        total_cost = total_cost + (~valid)[:, None, None, :].to(x.dtype) * 50.0

        source_mass = quality * valid.to(quality.dtype)
        plan = self._sinkhorn(total_cost, source_mass)
        plan = plan * valid[:, None, None, :].to(plan.dtype)
        band_mapping = plan / plan.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        band_weights = torch.softmax(self.band_fusion_logits, dim=0)
        fused_mapping = (
            band_mapping * band_weights[None, :, :, None].to(x.dtype)
        ).sum(dim=1)
        fused_mapping = fused_mapping / fused_mapping.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        virtual_signal_raw = torch.einsum("bkc,btc->btk", fused_mapping, x)

        local_support = torch.exp(
            -self.geometry_cost().to(x.dtype)
            / max(self.confidence_geometry_temperature, 1e-6)
        )
        coverage = torch.einsum(
            "bkc,bc,kc->bk", fused_mapping, quality, local_support
        ).clamp(0.0, 1.0)
        entropy = -(
            fused_mapping.clamp_min(1e-8) * torch.log(fused_mapping.clamp_min(1e-8))
        ).sum(dim=-1)
        valid_count = valid.sum(dim=-1, keepdim=True).to(x.dtype).clamp_min(2.0)
        normalized_entropy = entropy / torch.log(valid_count)
        confidence = coverage * torch.exp(-normalized_entropy)
        confidence = confidence.clamp(self.confidence_floor, 1.0)

        _, virtual_band_log_power = self._frequency_descriptors(
            virtual_signal_raw, time_mask
        )
        virtual_signal = virtual_signal_raw * confidence.unsqueeze(1)

        info = {
            "mapping": fused_mapping,
            "band_mapping": band_mapping,
            "confidence": confidence,
            "channel_quality": quality,
            "valid_channels": valid,
            "source_band_log_power": source_band_log_power,
            "virtual_band_log_power": virtual_band_log_power,
            "transport_entropy": normalized_entropy,
        }
        return virtual_signal, info

    def make_counterfactual_mask(
        self,
        valid_mask: torch.Tensor,
        minimum_drop_ratio: float,
        maximum_drop_ratio: float,
        minimum_keep_channels: int,
        regional_drop_probability: float,
    ) -> torch.Tensor:
        """Create random and spatially contiguous channel-removal views."""
        if valid_mask.ndim != 2:
            raise ValueError("valid_mask must have shape [B,C].")
        if not 0.0 <= minimum_drop_ratio <= maximum_drop_ratio < 1.0:
            raise ValueError("Drop ratios must satisfy 0 <= min <= max < 1.")
        batch, channels = valid_mask.shape
        result = valid_mask.clone().bool()
        coordinates = self.source_coordinates.to(valid_mask.device)
        pair_distance = 1.0 - coordinates @ coordinates.transpose(0, 1)

        for b in range(batch):
            indices = torch.where(valid_mask[b])[0]
            count = int(indices.numel())
            if count <= minimum_keep_channels:
                continue
            ratio = float(
                torch.empty(1, device=valid_mask.device).uniform_(
                    minimum_drop_ratio, maximum_drop_ratio
                ).item()
            )
            drop_count = min(
                max(1, int(round(count * ratio))),
                count - minimum_keep_channels,
            )
            if drop_count <= 0:
                continue

            use_regional = bool(
                torch.rand(1, device=valid_mask.device).item() < regional_drop_probability
            )
            if use_regional:
                center = indices[
                    torch.randint(count, (1,), device=valid_mask.device).item()
                ]
                distances = pair_distance[center, indices]
                drop_local = torch.topk(distances, k=drop_count, largest=False).indices
                drop_indices = indices[drop_local]
            else:
                permutation = torch.randperm(count, device=valid_mask.device)
                drop_indices = indices[permutation[:drop_count]]
            result[b, drop_indices] = False
        return result
