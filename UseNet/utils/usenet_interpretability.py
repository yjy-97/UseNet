from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


@dataclass
class VEUSExplanation:
    """Container returned by :func:`compute_usenet_explanation`.

    Shapes:
        logits: [B, Y]
        probabilities: [B, Y]
        target_class: [B]
        virtual_score: [B, K]
        virtual_band_score: [B, F, K]
        source_score: [B, C]
        source_band_score: [B, F, C]
        mapping: [B, K, C]
        band_mapping: [B, F, K, C]
        confidence: [B, K]
    """

    logits: Tensor
    probabilities: Tensor
    target_class: Tensor
    virtual_score: Tensor
    virtual_band_score: Tensor
    source_score: Tensor
    source_band_score: Tensor
    mapping: Tensor
    band_mapping: Tensor
    confidence: Tensor
    virtual_gradient_score: Tensor
    virtual_occlusion_score: Optional[Tensor]
    virtual_band_deviation: Optional[Tensor]
    source_channel_names: List[str]
    virtual_channel_names: List[str]


def _unwrap_model(model):
    """Return the actual model when DataParallel/DistributedDataParallel is used."""
    return model.module if hasattr(model, "module") else model


def _normalize_last_dim(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Min-max normalize each sample along the last dimension."""
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def _normalize_matrix_per_sample(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Min-max normalize each sample for tensors shaped [B, ...]."""
    flat = x.reshape(x.shape[0], -1)
    x_min = flat.amin(dim=-1).view(-1, *([1] * (x.ndim - 1)))
    x_max = flat.amax(dim=-1).view(-1, *([1] * (x.ndim - 1)))
    return (x - x_min) / (x_max - x_min + eps)


def _time_mask_from_argument(x: Tensor, time_mask: Optional[Tensor]) -> Optional[Tensor]:
    if time_mask is None:
        return None
    if time_mask.ndim == 2 and tuple(time_mask.shape) == tuple(x.shape[:2]):
        return time_mask.to(device=x.device, dtype=x.dtype)
    return None


def _forward_from_virtual_signal(
    model,
    virtual_signal: Tensor,
    confidence: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Run the iTransformer part after the VEUS adapter.

    Args:
        model: unwrapped UseNet model.
        virtual_signal: [B, T, K] virtual EEG signal.
        confidence: [B, K] virtual-channel confidence.

    Returns:
        logits and final virtual-token features.
    """
    enc_out = model.enc_embedding(virtual_signal, None)
    enc_out = (
        enc_out
        + model.region_embedding.to(enc_out.dtype)
        + model.confidence_embedding(confidence.unsqueeze(-1))
    )
    enc_out, _ = model.encoder(enc_out, attn_mask=None)
    features = model.dropout(model.act(enc_out))
    logits = model.projection(features.reshape(features.shape[0], -1))
    return logits, enc_out


def compute_virtual_gradient_score(
    model,
    x: Tensor,
    time_mask: Optional[Tensor] = None,
    channel_mask: Optional[Tensor] = None,
    target_class: Optional[Union[int, Tensor]] = None,
    multiply_by_input: bool = True,
) -> Tuple[Tensor, Dict[str, Tensor], Tensor, Tensor]:
    """Compute virtual-channel gradient attribution.

    The VEUS adapter is first used to generate virtual channels.  The generated
    virtual signal is then detached and used as the attribution input, so the
    score measures how much each virtual channel supports the selected class,
    rather than how the adapter parameters should change.

    Returns:
        virtual_gradient_score: [B, K]
        adapter_info: mapping, band_mapping, confidence, band powers, etc.
        logits: [B, Y]
        target_class: [B]
    """
    model = _unwrap_model(model)
    was_training = model.training
    model.eval()

    x = x.float()
    time_mask = _time_mask_from_argument(x, time_mask)

    with torch.no_grad():
        virtual_signal, adapter_info = model.adapter(
            x, time_mask=time_mask, channel_mask=channel_mask
        )
        confidence = adapter_info["confidence"]

    virtual_signal = virtual_signal.detach().requires_grad_(True)
    logits, _ = _forward_from_virtual_signal(model, virtual_signal, confidence)
    probabilities = torch.softmax(logits.detach(), dim=-1)

    if target_class is None:
        target = probabilities.argmax(dim=-1)
    elif isinstance(target_class, int):
        target = torch.full(
            (x.shape[0],), int(target_class), device=x.device, dtype=torch.long
        )
    else:
        target = target_class.to(device=x.device, dtype=torch.long).view(-1)

    score = logits.gather(dim=1, index=target.view(-1, 1)).sum()
    if virtual_signal.grad is not None:
        virtual_signal.grad.zero_()
    score.backward()

    grad = virtual_signal.grad.detach()
    if multiply_by_input:
        saliency_time = (grad * virtual_signal.detach()).abs()
    else:
        saliency_time = grad.abs()
    virtual_gradient_score = saliency_time.mean(dim=1)  # [B,K]
    virtual_gradient_score = _normalize_last_dim(virtual_gradient_score)

    if was_training:
        model.train()
    return virtual_gradient_score.detach(), adapter_info, logits.detach(), target.detach()


@torch.no_grad()
def compute_virtual_occlusion_score(
    model,
    x: Tensor,
    time_mask: Optional[Tensor] = None,
    channel_mask: Optional[Tensor] = None,
    target_class: Optional[Union[int, Tensor]] = None,
    baseline: str = "zero",
) -> Tensor:
    """Compute probability-drop score by masking each virtual channel.

    Args:
        baseline: ``"zero"`` masks a virtual channel to zero; ``"mean"`` replaces
            it with the within-sample mean virtual signal.

    Returns:
        occlusion_score: [B, K], min-max normalized per sample.
    """
    model = _unwrap_model(model)
    was_training = model.training
    model.eval()

    x = x.float()
    time_mask = _time_mask_from_argument(x, time_mask)
    virtual_signal, adapter_info = model.adapter(
        x, time_mask=time_mask, channel_mask=channel_mask
    )
    confidence = adapter_info["confidence"]
    logits, _ = _forward_from_virtual_signal(model, virtual_signal, confidence)
    probabilities = torch.softmax(logits, dim=-1)

    if target_class is None:
        target = probabilities.argmax(dim=-1)
    elif isinstance(target_class, int):
        target = torch.full(
            (x.shape[0],), int(target_class), device=x.device, dtype=torch.long
        )
    else:
        target = target_class.to(device=x.device, dtype=torch.long).view(-1)

    base_prob = probabilities.gather(dim=1, index=target.view(-1, 1)).squeeze(1)
    batch, _, virtual_channels = virtual_signal.shape
    drops = []
    mean_signal = virtual_signal.mean(dim=2, keepdim=True)

    for k in range(virtual_channels):
        masked = virtual_signal.clone()
        if baseline == "mean":
            masked[:, :, k : k + 1] = mean_signal
        else:
            masked[:, :, k] = 0.0
        masked_logits, _ = _forward_from_virtual_signal(model, masked, confidence)
        masked_prob = torch.softmax(masked_logits, dim=-1).gather(
            dim=1, index=target.view(-1, 1)
        ).squeeze(1)
        drops.append((base_prob - masked_prob).clamp_min(0.0))

    occlusion_score = torch.stack(drops, dim=-1).reshape(batch, virtual_channels)
    occlusion_score = _normalize_last_dim(occlusion_score)

    if was_training:
        model.train()
    return occlusion_score.detach()


@torch.no_grad()
def estimate_normal_band_statistics(
    model,
    dataloader: Iterable,
    normal_class: int = 0,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, Tensor]:
    """Estimate normal/reference virtual-band statistics from a dataloader.

    The dataloader is expected to return ``(batch_x, label, padding_mask)`` as in
    the original classification pipeline.  Only samples with ``label ==
    normal_class`` are used.

    Returns:
        ``{"mean": [F,K], "std": [F,K]}`` on CPU.
    """
    model = _unwrap_model(model)
    was_training = model.training
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    values = []
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch_x, label, padding_mask = batch[:3]
        keep = label == int(normal_class)
        if not keep.any():
            continue
        batch_x = batch_x[keep].float().to(device)
        padding_mask = padding_mask[keep].float().to(device)
        _, adapter_info = model.adapter(batch_x, time_mask=padding_mask)
        values.append(adapter_info["virtual_band_log_power"].detach().cpu())

    if not values:
        raise RuntimeError(
            "No normal/reference samples were found when estimating band statistics."
        )
    all_values = torch.cat(values, dim=0)  # [N,F,K]
    mean = all_values.mean(dim=0)
    std = all_values.std(dim=0, unbiased=False).clamp_min(1e-6)

    if was_training:
        model.train()
    return {"mean": mean, "std": std}


def compute_band_deviation(
    virtual_band_log_power: Tensor,
    normal_stats: Optional[Dict[str, Tensor]],
) -> Optional[Tensor]:
    """Return z-score deviation of virtual-band powers from normal statistics."""
    if normal_stats is None:
        return None
    mean = normal_stats["mean"].to(
        device=virtual_band_log_power.device, dtype=virtual_band_log_power.dtype
    )
    std = normal_stats["std"].to(
        device=virtual_band_log_power.device, dtype=virtual_band_log_power.dtype
    ).clamp_min(1e-6)
    deviation = ((virtual_band_log_power - mean[None, :, :]) / std[None, :, :]).abs()
    return _normalize_matrix_per_sample(deviation)




def compute_intrinsic_band_evidence(
    virtual_band_log_power: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Estimate band-specific evidence without normal/reference statistics.

    The previous implementation expanded one virtual-channel attribution score
    to all frequency bands, which can make the delta/theta/alpha/beta/gamma
    topomaps visually almost identical.  This function extracts a genuinely
    band-dependent term from the VEUS adapter's log-band power.

    Args:
        virtual_band_log_power: log band power shaped [B, F, K] or [B, F, C].

    Returns:
        Normalized band evidence with the same shape.  The score combines
        (1) spatial contrast within each band and (2) relative band contrast
        within each virtual region/source channel.
    """
    x = torch.nan_to_num(virtual_band_log_power.float(), nan=0.0, posinf=0.0, neginf=0.0)

    spatial_mean = x.mean(dim=-1, keepdim=True)
    spatial_std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    spatial_contrast = ((x - spatial_mean) / spatial_std).abs()

    band_mean = x.mean(dim=1, keepdim=True)
    band_std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    band_contrast = ((x - band_mean) / band_std).abs()

    evidence = 0.5 * spatial_contrast + 0.5 * band_contrast
    return _normalize_matrix_per_sample(evidence)


def backproject_virtual_to_source(
    virtual_band_score: Tensor,
    band_mapping: Tensor,
    confidence: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Back-project virtual abnormal scores to original source electrodes.

    Args:
        virtual_band_score: [B,F,K]
        band_mapping: [B,F,K,C]
        confidence: [B,K]

    Returns:
        source_band_score: [B,F,C]
        source_score: [B,C]
    """
    weighted_virtual = virtual_band_score * confidence[:, None, :]
    source_band_score = torch.einsum("bfkc,bfk->bfc", band_mapping, weighted_virtual)
    source_band_score = _normalize_matrix_per_sample(source_band_score)
    source_score = source_band_score.sum(dim=1)
    source_score = _normalize_last_dim(source_score)
    return source_band_score, source_score


def compute_usenet_explanation(
    model,
    x: Tensor,
    time_mask: Optional[Tensor] = None,
    channel_mask: Optional[Tensor] = None,
    target_class: Optional[Union[int, Tensor]] = None,
    normal_stats: Optional[Dict[str, Tensor]] = None,
    use_occlusion: bool = True,
    weights: Tuple[float, float, float] = (0.45, 0.35, 0.20),
) -> VEUSExplanation:
    """Compute virtual-region, source-channel and band-level explanations.

    Args:
        model: UseNet model or DataParallel wrapper.
        x: EEG tensor shaped [B,T,C].
        time_mask: optional temporal mask shaped [B,T].
        channel_mask: optional source-channel mask shaped [B,C].
        target_class: class to explain.  If ``None``, the predicted class is used.
        normal_stats: optional normal/reference band statistics returned by
            :func:`estimate_normal_band_statistics`.
        use_occlusion: whether to compute virtual-channel occlusion score.
        weights: ``(gradient, deviation, occlusion)`` weights used to construct
            the final virtual-band abnormal score.

    Returns:
        :class:`VEUSExplanation`.
    """
    model_unwrapped = _unwrap_model(model)
    grad_score, adapter_info, logits, target = compute_virtual_gradient_score(
        model_unwrapped,
        x=x,
        time_mask=time_mask,
        channel_mask=channel_mask,
        target_class=target_class,
    )
    probabilities = torch.softmax(logits, dim=-1)

    occlusion_score = None
    if use_occlusion:
        occlusion_score = compute_virtual_occlusion_score(
            model_unwrapped,
            x=x,
            time_mask=time_mask,
            channel_mask=channel_mask,
            target_class=target,
        )

    virtual_band_log_power = adapter_info["virtual_band_log_power"]
    source_band_log_power = adapter_info.get("source_band_log_power", None)
    virtual_band_deviation = compute_band_deviation(virtual_band_log_power, normal_stats)

    band_evidence = compute_intrinsic_band_evidence(virtual_band_log_power)  # [B,F,K]
    num_bands = adapter_info["band_mapping"].shape[1]

    grad_band = grad_score[:, None, :].expand(-1, num_bands, -1) * band_evidence
    grad_band = _normalize_matrix_per_sample(grad_band)

    if virtual_band_deviation is None:
        deviation_band = band_evidence
    else:
        deviation_band = _normalize_matrix_per_sample(
            0.7 * virtual_band_deviation + 0.3 * band_evidence
        )

    if occlusion_score is None:
        occlusion_band = torch.zeros_like(grad_band)
    else:
        occlusion_band = occlusion_score[:, None, :].expand(-1, num_bands, -1) * band_evidence
        occlusion_band = _normalize_matrix_per_sample(occlusion_band)

    wg, wd, wo = weights
    virtual_band_score = wg * grad_band + wd * deviation_band + wo * occlusion_band
    virtual_band_score = _normalize_matrix_per_sample(virtual_band_score)
    virtual_score = virtual_band_score.sum(dim=1)
    virtual_score = _normalize_last_dim(virtual_score)

    source_band_score, source_score = backproject_virtual_to_source(
        virtual_band_score=virtual_band_score,
        band_mapping=adapter_info["band_mapping"],
        confidence=adapter_info["confidence"],
    )

    if source_band_log_power is not None:
        source_band_evidence = compute_intrinsic_band_evidence(source_band_log_power)  # [B,F,C]
        source_band_score = source_band_score * (0.5 + 0.5 * source_band_evidence)
        source_band_score = _normalize_matrix_per_sample(source_band_score)
        source_score = source_band_score.sum(dim=1)
        source_score = _normalize_last_dim(source_score)

    return VEUSExplanation(
        logits=logits.detach(),
        probabilities=probabilities.detach(),
        target_class=target.detach(),
        virtual_score=virtual_score.detach(),
        virtual_band_score=virtual_band_score.detach(),
        source_score=source_score.detach(),
        source_band_score=source_band_score.detach(),
        mapping=adapter_info["mapping"].detach(),
        band_mapping=adapter_info["band_mapping"].detach(),
        confidence=adapter_info["confidence"].detach(),
        virtual_gradient_score=grad_score.detach(),
        virtual_occlusion_score=None if occlusion_score is None else occlusion_score.detach(),
        virtual_band_deviation=None if virtual_band_deviation is None else virtual_band_deviation.detach(),
        source_channel_names=list(getattr(model_unwrapped, "source_channel_names", [])),
        virtual_channel_names=list(getattr(model_unwrapped, "virtual_channel_names", [])),
    )


def topk_source_channels(
    explanation: VEUSExplanation,
    sample_index: int = 0,
    k: int = 5,
) -> List[Dict[str, Union[str, float, int]]]:
    """Return Top-k original source electrodes contributing to the decision."""
    scores = explanation.source_score[sample_index].detach().cpu()
    top_values, top_indices = torch.topk(scores, k=min(k, scores.numel()))
    names = explanation.source_channel_names or [f"Ch{idx}" for idx in range(scores.numel())]
    rows = []
    for rank, (value, idx) in enumerate(zip(top_values.tolist(), top_indices.tolist()), start=1):
        band_scores = explanation.source_band_score[sample_index, :, idx].detach().cpu()
        main_band = int(torch.argmax(band_scores).item())
        rows.append(
            {
                "rank": rank,
                "channel": names[idx] if idx < len(names) else f"Ch{idx}",
                "score": float(value),
                "main_band_index": main_band,
                "main_band_score": float(band_scores[main_band].item()),
            }
        )
    return rows


def topk_virtual_regions(
    explanation: VEUSExplanation,
    sample_index: int = 0,
    k: int = 5,
) -> List[Dict[str, Union[str, float, int]]]:
    """Return Top-k virtual regions contributing to the decision."""
    scores = explanation.virtual_score[sample_index].detach().cpu()
    top_values, top_indices = torch.topk(scores, k=min(k, scores.numel()))
    names = explanation.virtual_channel_names or [f"V{idx}" for idx in range(scores.numel())]
    rows = []
    for rank, (value, idx) in enumerate(zip(top_values.tolist(), top_indices.tolist()), start=1):
        band_scores = explanation.virtual_band_score[sample_index, :, idx].detach().cpu()
        main_band = int(torch.argmax(band_scores).item())
        confidence = explanation.confidence[sample_index, idx].detach().cpu().item()
        rows.append(
            {
                "rank": rank,
                "virtual_region": names[idx] if idx < len(names) else f"V{idx}",
                "score": float(value),
                "confidence": float(confidence),
                "main_band_index": main_band,
                "main_band_score": float(band_scores[main_band].item()),
            }
        )
    return rows


def explanation_to_dict(
    explanation: VEUSExplanation,
    sample_index: int = 0,
    top_k: int = 5,
) -> Dict[str, object]:
    """Compact serializable summary for one EEG sample."""
    probs = explanation.probabilities[sample_index].detach().cpu()
    return {
        "target_class": int(explanation.target_class[sample_index].detach().cpu().item()),
        "predicted_class": int(torch.argmax(probs).item()),
        "predicted_probability": float(probs.max().item()),
        "top_virtual_regions": topk_virtual_regions(explanation, sample_index, top_k),
        "top_source_channels": topk_source_channels(explanation, sample_index, top_k),
    }
