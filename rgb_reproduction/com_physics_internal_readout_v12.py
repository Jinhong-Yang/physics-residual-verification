"""Small temporal readout for physics-directed observation correction.

The module consumes an eight-frame sequence after external TRAIN-only feature
normalization and returns one bounded correction on the physically active world
axis per frame.  It does not predict material properties or bypass the numeric
solver.  A zero-initialized output layer makes the initial model exactly the
RGB observation baseline.
"""
from __future__ import annotations

import torch
from torch import nn


class PhysicsInternalReadoutV12(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, correction_limit_m: float = 0.02):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or not 0 < correction_limit_m < float("inf"):
            raise ValueError("Invalid temporal readout dimensions or correction bound")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.correction_limit_m = float(correction_limit_m)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )
        self.temporal = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError("Expected B x T x input_dim features")
        if valid.shape != features.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("Expected a matching boolean validity mask")
        if features.dtype != self.input_projection[0].weight.dtype or features.device != valid.device:
            raise ValueError("Feature dtype/device and validity device must match the model")
        if not bool(torch.isfinite(features).all()) or not bool(valid.all()):
            raise ValueError("V12 currently requires finite all-eight-frame observations")
        hidden, _ = self.temporal(self.input_projection(features))
        raw = self.readout(hidden).squeeze(-1)
        correction = self.correction_limit_m * torch.tanh(raw)
        return torch.where(valid, correction, torch.zeros_like(correction))


def apply_active_correction(
    xz: torch.Tensor, active: torch.Tensor, correction_m: torch.Tensor
) -> torch.Tensor:
    if xz.ndim != 3 or xz.shape[1:] != (8, 2):
        raise ValueError("Expected B x 8 x 2 world observations")
    if active.shape != (len(xz), 3) or active.dtype != torch.bool:
        raise ValueError("Expected B x 3 active mask")
    if correction_m.shape != xz.shape[:2] or correction_m.dtype != xz.dtype:
        raise ValueError("Correction must match world observation dtype and frame shape")
    if not bool(torch.isfinite(xz).all() and torch.isfinite(correction_m).all()):
        raise ValueError("World observations and corrections must be finite")
    axis = active[:, 2].long()
    row = torch.arange(len(xz), device=xz.device)[:, None]
    frame = torch.arange(xz.shape[1], device=xz.device)[None]
    output = xz.clone()
    output[row, frame, axis[:, None]] = output[row, frame, axis[:, None]] + correction_m
    return output
