"""Differentiable reliability-weighted local Gaussian update (V2 development).

The original diagonal prior is conditioned on a single local likelihood. Callers
may relinearize, but must keep that original prior instead of accumulating the
same evidence repeatedly. Covariance is conditional on the supplied weights;
with fitted robust weights it is a local Gaussian/IRLS approximation, not an
exact Bayesian posterior. No existing experiment or checkpoint is changed.
"""
import math

import torch
from torch import nn


def _finite_double(name, value, shape, device):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}")
    if value.dtype != torch.float64 or value.device != device:
        raise ValueError(f"{name} must be float64 on the prior device")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def weighted_gaussian_update(prior_mean, prior_scale, jacobian, residual,
                             nuisance_jacobian, weights, valid, active,
                             linearization_mean):
    """Return a batched, prior-whitened local linear Gaussian update.

    J and r are *raw*, noise-normalized quantities evaluated at
    ``linearization_mean``. J differentiates with respect to coordinates
    standardized by ``prior_scale``. Nuisance columns may be dependent or padded
    with zeros. Both r and J are reprojected after applying sqrt(weights).

    No singular-value cutoff is applied to physical directions: a weak physical
    likelihood contributes continuously through I + J.T @ J. Pseudoinverse
    tolerance 1e-12 is solely for numerical rank of nuisance coordinates.
    Gradients with respect to positive weights are valid on constant nuisance
    rank regions. Zero weights are supported as omitted observations, but rank
    changes at the zero-weight boundary do not have a general unique derivative.

    All numeric inputs are float64. ``valid`` and ``active`` are boolean. The
    returned projected residual includes the original-prior intercept shift.
    """
    if not isinstance(prior_mean, torch.Tensor) or prior_mean.ndim != 2 or prior_mean.shape[1] != 3:
        raise ValueError("prior_mean must be B x 3")
    batch = prior_mean.shape[0]
    if batch < 1 or not isinstance(residual, torch.Tensor) or residual.ndim != 2:
        raise ValueError("Nonempty batch and B x T residual required")
    times = residual.shape[1]
    if times < 1 or not isinstance(nuisance_jacobian, torch.Tensor) or nuisance_jacobian.ndim != 3:
        raise ValueError("T >= 1 and B x T x Q nuisance Jacobian required")
    device = prior_mean.device
    for name, value, shape in (
        ("prior_mean", prior_mean, (batch, 3)),
        ("prior_scale", prior_scale, (batch, 3)),
        ("jacobian", jacobian, (batch, times, 3)),
        ("residual", residual, (batch, times)),
        ("nuisance_jacobian", nuisance_jacobian, (batch, times, nuisance_jacobian.shape[2])),
        ("weights", weights, (batch, times)),
        ("linearization_mean", linearization_mean, (batch, 3)),
    ):
        _finite_double(name, value, shape, device)
    for name, value, shape in (("valid", valid, (batch, times)), ("active", active, (batch, 3))):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.bool or value.device != device or tuple(value.shape) != shape:
            raise ValueError(f"{name} must be boolean with shape {shape} on the prior device")
    if not bool((prior_scale > 0).all()) or not bool((weights >= 0).all()):
        raise ValueError("Prior scales must be positive and weights nonnegative")

    effective = torch.where(valid, weights, torch.zeros_like(weights))
    # Avoid sqrt'(0) producing NaN through the inactive branch of torch.where.
    root = torch.where(effective > 0, effective.clamp_min(torch.finfo(torch.float64).tiny).sqrt(), torch.zeros_like(effective))
    permitted_j = jacobian * active[:, None, :]
    shifted = residual + (permitted_j @ ((prior_mean - linearization_mean) / prior_scale).unsqueeze(-1)).squeeze(-1)
    weighted_j = root[:, :, None] * permitted_j
    weighted_r = root * shifted
    weighted_n = root[:, :, None] * nuisance_jacobian
    if nuisance_jacobian.shape[2]:
        inverse_n = torch.linalg.pinv(weighted_n, atol=0.0, rtol=1e-12)
        projected_j = weighted_j - weighted_n @ (inverse_n @ weighted_j)
        projected_r = weighted_r - (weighted_n @ (inverse_n @ weighted_r.unsqueeze(-1))).squeeze(-1)
    else:
        projected_j, projected_r = weighted_j, weighted_r
    projected_j = projected_j * active[:, None, :]
    identity = torch.eye(3, dtype=torch.float64, device=device).expand(batch, 3, 3)
    precision = identity + projected_j.transpose(-1, -2) @ projected_j
    factor = torch.linalg.cholesky(precision)
    rhs = -(projected_j.transpose(-1, -2) @ projected_r.unsqueeze(-1))
    update = torch.cholesky_solve(rhs, factor).squeeze(-1)
    white_covariance = torch.cholesky_inverse(factor)
    covariance = prior_scale[:, :, None] * white_covariance * prior_scale[:, None, :]
    mean = prior_mean + prior_scale * update
    # Exact identity behavior when there is structurally no physical evidence.
    closed = (~active.any(-1)) | (~(effective > 0).any(-1)) | (~(permitted_j != 0).any(dim=(1, 2)))
    mean = torch.where(closed[:, None], prior_mean, mean)
    covariance = torch.where(closed[:, None, None], torch.diag_embed(prior_scale.square()), covariance)
    update = torch.where(closed[:, None], torch.zeros_like(update), update)
    return {"mean": mean, "covariance": covariance, "whitened_update": update,
            "residual_at_prior": shifted,
            "projected_residual_at_prior": projected_r,
            "projected_residual": projected_r, "projected_jacobian": projected_j}


class ReliabilityWeightNet(nn.Module):
    """Per-frame bounded weight with masked episode context; no posterior readout."""
    def __init__(self, feature_dim=None, hidden_dim=64, minimum_weight=0.05, initial_weight=0.95, *, input_dim=None):
        super().__init__()
        if feature_dim is None:
            feature_dim = input_dim
        elif input_dim is not None and feature_dim != input_dim:
            raise ValueError("feature_dim and input_dim aliases disagree")
        if feature_dim is None:
            raise ValueError("Supply feature_dim or input_dim")
        if feature_dim < 1 or hidden_dim < 1 or not 0 < minimum_weight < initial_weight < 1:
            raise ValueError("Invalid weight-network dimensions or bounds")
        self.feature_dim = feature_dim
        self.minimum_weight = float(minimum_weight)
        self.network = nn.Sequential(nn.Linear(2 * feature_dim, hidden_dim, dtype=torch.float64), nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim, dtype=torch.float64), nn.SiLU(),
                                     nn.Linear(hidden_dim, 1, dtype=torch.float64))
        probability = (initial_weight - minimum_weight) / (1 - minimum_weight)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, math.log(probability / (1 - probability)))

    def forward(self, features, valid):
        if features.ndim != 3 or features.shape[-1] != self.feature_dim or not bool(torch.isfinite(features).all()):
            raise ValueError("Expected finite B x T x F features")
        if features.dtype != self.network[0].weight.dtype or features.device != self.network[0].weight.device:
            raise ValueError("Features must match network dtype and device")
        if valid.shape != features.shape[:2] or valid.dtype != torch.bool or valid.device != features.device:
            raise ValueError("Invalid frame mask")
        masked = torch.where(valid[:, :, None], features, torch.zeros_like(features))
        context = masked.sum(1) / valid.sum(1).clamp_min(1)[:, None]
        values = self.network(torch.cat((masked, context[:, None, :].expand_as(masked)), dim=-1)).squeeze(-1)
        weights = self.minimum_weight + (1 - self.minimum_weight) * values.sigmoid()
        return torch.where(valid, weights, torch.zeros_like(weights))


def robust_weights(residual, valid=None, kind="huber", tuning=None):
    """Nonlearned IRLS weights for standardized per-frame residuals."""
    if not isinstance(residual, torch.Tensor) or not residual.is_floating_point() or not bool(torch.isfinite(residual).all()):
        raise ValueError("Residual must be a finite floating tensor")
    if kind not in ("huber", "cauchy"):
        raise ValueError("Unknown robust weight kind")
    tuning = (1.345 if kind == "huber" else 2.385) if tuning is None else float(tuning)
    if not math.isfinite(tuning) or tuning <= 0:
        raise ValueError("Tuning constant must be finite and positive")
    standardized = residual.abs() / tuning
    weights = 1 / standardized.clamp_min(1) if kind == "huber" else 1 / (1 + standardized.square())
    if valid is not None:
        if valid.dtype != torch.bool or valid.shape != residual.shape or valid.device != residual.device:
            raise ValueError("Invalid robust weight mask")
        weights = torch.where(valid, weights, torch.zeros_like(weights))
    return weights
