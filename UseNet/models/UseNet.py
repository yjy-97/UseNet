from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.VEUS_Adapter import VirtualEEGUnifiedSpatialRepresentation
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, SharedQueryAttentionLayer
from layers.Embed import DataEmbedding_inverted
from utils.eeg_montage import (
    load_source_layout,
    load_virtual_layout,
    parse_frequency_bands,
)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.output_attention = configs.output_attention
        self.source_channels = int(configs.source_enc_in)
        self.virtual_channels = int(configs.virtual_channels)

        source_names, source_coordinates, anatomical = load_source_layout(
            self.source_channels,
            getattr(configs, "channel_config", ""),
            getattr(configs, "channel_names", ""),
        )
        virtual_names, virtual_coordinates = load_virtual_layout(
            self.virtual_channels,
            getattr(configs, "virtual_channel_names", ""),
        )
        self.source_channel_names = source_names
        self.virtual_channel_names = virtual_names
        self.has_anatomical_source_layout = anatomical

        bands = parse_frequency_bands(
            getattr(configs, "frequency_bands", "1-4,4-8,8-13,13-30,30-45"),
            float(configs.sampling_rate),
        )
        self.adapter = VirtualEEGUnifiedSpatialRepresentation(
            source_coordinates=source_coordinates,
            virtual_coordinates=virtual_coordinates,
            sampling_rate=float(configs.sampling_rate),
            frequency_bands=bands,
            sinkhorn_epsilon=float(configs.sinkhorn_epsilon),
            sinkhorn_iterations=int(configs.sinkhorn_iterations),
            geometry_weight=float(configs.geometry_weight),
            signal_weight=float(configs.signal_weight),
            quality_weight=float(configs.quality_weight),
            confidence_floor=float(configs.confidence_floor),
            confidence_geometry_temperature=float(configs.confidence_geometry_temperature),
        )

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.region_embedding = nn.Parameter(
            torch.zeros(1, self.virtual_channels, configs.d_model)
        )
        nn.init.trunc_normal_(self.region_embedding, std=0.02)
        self.confidence_embedding = nn.Sequential(
            nn.Linear(1, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.d_model),
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    SharedQueryAttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                        num_queries=int(getattr(configs, "shared_query_tokens", self.virtual_channels)),
                        conditioned=bool(getattr(configs, "shared_query_conditioned", True)),
                        query_dropout=float(getattr(configs, "shared_query_dropout", 0.0)),
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(
            configs.d_model * self.virtual_channels, configs.num_class
        )

        self.counterfactual_distillation = bool(configs.counterfactual_distillation)
        self.drop_ratio_min = float(configs.channel_drop_min)
        self.drop_ratio_max = float(configs.channel_drop_max)
        self.minimum_keep_channels = int(configs.minimum_keep_channels)
        self.regional_drop_probability = float(configs.regional_drop_probability)
        self.distillation_temperature = float(configs.distillation_temperature)

    def _time_mask_from_argument(
        self, x: torch.Tensor, x_mark_enc: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if x_mark_enc is None:
            return None
        if x_mark_enc.ndim == 2 and x_mark_enc.shape[:2] == x.shape[:2]:
            return x_mark_enc.to(device=x.device, dtype=x.dtype)
        return None

    def _encode_view(
        self,
        x: torch.Tensor,
        time_mask: Optional[torch.Tensor],
        channel_mask: Optional[torch.Tensor],
        return_virtual_signal: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        virtual_signal, adapter_info = self.adapter(
            x, time_mask=time_mask, channel_mask=channel_mask
        )
        if return_virtual_signal:
            adapter_info["virtual_signal"] = virtual_signal
        enc_out = self.enc_embedding(virtual_signal, None)
        confidence = adapter_info["confidence"]
        enc_out = (
            enc_out
            + self.region_embedding.to(enc_out.dtype)
            + self.confidence_embedding(confidence.unsqueeze(-1))
        )
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        features = self.dropout(self.act(enc_out))
        logits = self.projection(features.reshape(features.shape[0], -1))
        if self.output_attention:
            adapter_info["attentions"] = attns
        return logits, adapter_info, enc_out

    @staticmethod
    def _mapping_js_loss(
        teacher_mapping: torch.Tensor,
        student_mapping: torch.Tensor,
        student_valid: torch.Tensor,
        student_confidence: torch.Tensor,
    ) -> torch.Tensor:
        common = student_valid[:, None, :].to(teacher_mapping.dtype)
        p = teacher_mapping.detach() * common
        q = student_mapping * common
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        midpoint = 0.5 * (p + q)
        kl_pm = (p * (torch.log(p.clamp_min(1e-8)) - torch.log(midpoint.clamp_min(1e-8)))).sum(dim=-1)
        kl_qm = (q * (torch.log(q.clamp_min(1e-8)) - torch.log(midpoint.clamp_min(1e-8)))).sum(dim=-1)
        js = 0.5 * (kl_pm + kl_qm)
        return (js * student_confidence.detach()).sum() / student_confidence.detach().sum().clamp_min(1e-8)

    @staticmethod
    def _band_consistency_loss(
        teacher_band_power: torch.Tensor,
        student_band_power: torch.Tensor,
        student_confidence: torch.Tensor,
    ) -> torch.Tensor:
        difference = F.smooth_l1_loss(
            student_band_power,
            teacher_band_power.detach(),
            reduction="none",
        )
        weight = student_confidence.detach()[:, None, :]
        return (difference * weight).sum() / (
            weight.sum().clamp_min(1e-8) * difference.shape[1]
        )

    def _distillation_loss(
        self, teacher_logits: torch.Tensor, student_logits: torch.Tensor
    ) -> torch.Tensor:
        temperature = self.distillation_temperature
        teacher_probability = torch.softmax(
            teacher_logits.detach() / temperature, dim=-1
        )
        student_log_probability = torch.log_softmax(
            student_logits / temperature, dim=-1
        )
        per_sample = F.kl_div(
            student_log_probability,
            teacher_probability,
            reduction="none",
        ).sum(dim=-1)
        certainty = teacher_probability.max(dim=-1).values.detach().clamp_min(0.1)
        return temperature * temperature * (per_sample * certainty).sum() / certainty.sum().clamp_min(1e-8)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
        x_dec=None,
        x_mark_dec=None,
        mask=None,
        channel_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        return_diagnostics: bool = False,
    ):
        time_mask = self._time_mask_from_argument(x_enc, x_mark_enc)
        full_logits, full_info, full_features = self._encode_view(
            x_enc, time_mask, channel_mask, return_virtual_signal=return_diagnostics
        )

        if not (self.training and return_aux and self.counterfactual_distillation):
            if return_diagnostics:
                return {
                    "logits": full_logits,
                    "full_info": full_info,
                    "full_features": full_features,
                    "source_channel_names": self.source_channel_names,
                    "virtual_channel_names": self.virtual_channel_names,
                }
            return full_logits

        base_valid = full_info["valid_channels"].detach()
        sparse_mask = self.adapter.make_counterfactual_mask(
            base_valid,
            minimum_drop_ratio=self.drop_ratio_min,
            maximum_drop_ratio=self.drop_ratio_max,
            minimum_keep_channels=self.minimum_keep_channels,
            regional_drop_probability=self.regional_drop_probability,
        )
        student_logits, student_info, student_features = self._encode_view(
            x_enc, time_mask, sparse_mask, return_virtual_signal=return_diagnostics
        )

        auxiliary_losses = {
            "kd": self._distillation_loss(full_logits, student_logits),
            "mapping": self._mapping_js_loss(
                full_info["mapping"],
                student_info["mapping"],
                sparse_mask,
                student_info["confidence"],
            ),
            "band": self._band_consistency_loss(
                full_info["virtual_band_log_power"],
                student_info["virtual_band_log_power"],
                student_info["confidence"],
            ),
        }
        result = {
            "logits": full_logits,
            "student_logits": student_logits,
            "aux_losses": auxiliary_losses,
        }
        if return_diagnostics:
            result.update(
                {
                    "full_info": full_info,
                    "student_info": student_info,
                    "full_features": full_features,
                    "student_features": student_features,
                    "student_channel_mask": sparse_mask,
                    "source_channel_names": self.source_channel_names,
                    "virtual_channel_names": self.virtual_channel_names,
                }
            )
        return result
