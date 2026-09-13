"""Protocol-query observation residual and reliability readout.

Three learned protocol queries attend to all eight frozen evidence slots.  The
selected query produces an episode-level correction profile while a local head
retains frame-specific evidence.  The module only edits observations and their
precision; the downstream physical solver remains authoritative.
"""
from __future__ import annotations

import math
import torch
from torch import nn


class PhysicsQueryReadoutV14(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        correction_limit_m: float = 0.02,
        minimum_weight: float = 0.05,
        initial_weight: float = 0.90,
    ):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or hidden_dim % heads:
            raise ValueError("Invalid query-readout dimensions")
        if not 0 < correction_limit_m < float("inf"):
            raise ValueError("Invalid correction bound")
        if not 0 < minimum_weight < initial_weight < 1:
            raise ValueError("Invalid reliability bounds")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.correction_limit_m = float(correction_limit_m)
        self.minimum_weight = float(minimum_weight)
        self.frame_projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.frame_norm = nn.LayerNorm(hidden_dim)
        self.protocol_queries = nn.Parameter(torch.empty(3, hidden_dim))
        nn.init.normal_(self.protocol_queries, std=0.02)
        self.query_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=0.0, batch_first=True
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.query_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.local_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU()
        )
        self.global_correction_head = nn.Linear(hidden_dim, 8)
        self.local_correction_head = nn.Linear(hidden_dim, 1)
        self.global_reliability_head = nn.Linear(hidden_dim, 8)
        self.local_reliability_head = nn.Linear(hidden_dim, 1)
        for head in (
            self.global_correction_head,
            self.local_correction_head,
            self.global_reliability_head,
            self.local_reliability_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        probability = (initial_weight - minimum_weight) / (1 - minimum_weight)
        self.register_buffer(
            "initial_reliability_logit",
            torch.tensor(math.log(probability / (1 - probability)), dtype=torch.float32),
        )

    def reliability_parameters(self):
        yield from self.global_reliability_head.parameters()
        yield from self.local_reliability_head.parameters()

    def forward(
        self,
        features: torch.Tensor,
        valid: torch.Tensor,
        protocol_index: torch.Tensor,
    ):
        if features.ndim != 3 or features.shape[1:] != (8, self.input_dim):
            raise ValueError("Expected B x 8 x input_dim features")
        if valid.shape != features.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("Expected matching boolean validity")
        if protocol_index.shape != (len(features),) or protocol_index.dtype != torch.long:
            raise ValueError("Expected one int64 protocol index per observation")
        if features.dtype != self.frame_projection[0].weight.dtype:
            raise ValueError("Feature dtype must match the model")
        if features.device != valid.device or features.device != protocol_index.device:
            raise ValueError("Features, validity, and protocol must share a device")
        if not bool(torch.isfinite(features).all()) or not bool(valid.all()):
            raise ValueError("V14 requires finite all-eight-frame observations")
        if bool(((protocol_index < 0) | (protocol_index > 2)).any()):
            raise ValueError("Protocol index must be in [0,2]")

        frames = self.frame_norm(self.frame_projection(features))
        queries = self.protocol_queries[None].expand(len(features), -1, -1)
        attended, _ = self.query_attention(
            queries, frames, frames, key_padding_mask=~valid, need_weights=False
        )
        queries = self.query_norm(queries + attended)
        queries = self.query_norm(queries + self.query_ffn(queries))
        selected = queries[torch.arange(len(features), device=features.device), protocol_index]
        fused = self.local_fusion(
            torch.cat((frames, selected[:, None].expand(-1, 8, -1)), dim=-1)
        )

        raw_correction = (
            self.global_correction_head(selected)
            + self.local_correction_head(fused).squeeze(-1)
        )
        correction = self.correction_limit_m * torch.tanh(raw_correction)
        reliability_logit = (
            self.initial_reliability_logit
            + self.global_reliability_head(selected)
            + self.local_reliability_head(fused).squeeze(-1)
        )
        probability = reliability_logit.sigmoid()
        weight = self.minimum_weight + (1 - self.minimum_weight) * probability
        return {
            "correction_m": torch.where(valid, correction, torch.zeros_like(correction)),
            "frame_weight": torch.where(valid, weight, torch.zeros_like(weight)),
        }


def apply_observation_update(xz, sigma_xz, active, correction_m, frame_weight):
    if xz.ndim != 3 or xz.shape[1:] != (8, 2) or sigma_xz.shape != xz.shape:
        raise ValueError("Expected matching B x 8 x 2 observations and scales")
    if active.shape != (len(xz), 3) or active.dtype != torch.bool:
        raise ValueError("Expected B x 3 active mask")
    if correction_m.shape != xz.shape[:2] or frame_weight.shape != xz.shape[:2]:
        raise ValueError("Expected B x 8 correction and reliability")
    if correction_m.dtype != xz.dtype or frame_weight.dtype != xz.dtype:
        raise ValueError("Update tensors must match observation dtype")
    if not bool(torch.isfinite(xz).all() and torch.isfinite(sigma_xz).all()
                and torch.isfinite(correction_m).all() and torch.isfinite(frame_weight).all()):
        raise ValueError("Observation update inputs must be finite")
    if bool((sigma_xz <= 0).any() or (frame_weight <= 0).any() or (frame_weight > 1).any()):
        raise ValueError("Scales must be positive and reliability in (0,1]")
    axis = active[:, 2].long()
    row = torch.arange(len(xz), device=xz.device)[:, None]
    frame = torch.arange(8, device=xz.device)[None]
    observed = xz.clone()
    sigma = sigma_xz.clone()
    observed[row, frame, axis[:, None]] += correction_m
    sigma[row, frame, axis[:, None]] /= frame_weight.sqrt()
    return observed, sigma
