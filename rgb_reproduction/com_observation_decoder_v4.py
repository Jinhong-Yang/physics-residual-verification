"""Small causal COM observation decoder; frozen image features are external.

This is a supervised simulator-COM readout, not a surface-point tracker or a
language-decoder reasoning intervention. No physical target is a forward input.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


CONDITIONS = ('qwen_temporal_native', 'rgb_temporal_native',
              'qwen_spatial_native', 'qwen_temporal_lowpass')


class ConvGRUCell(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.gates = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1)
        self.candidate = nn.Conv2d(2 * channels, channels, 3, padding=1)

    def forward(self, x, h):
        reset, update = self.gates(torch.cat((x, h), 1)).sigmoid().chunk(2, 1)
        candidate = self.candidate(torch.cat((x, reset * h), 1)).tanh()
        return update * h + (1 - update) * candidate


def rgb_patch_descriptor(rgb: torch.Tensor, grid_size: int = 32) -> torch.Tensor:
    """RGB plus gradient magnitude; 4x16x16 entries, no padding channels.

    rgb: [B,T,3,H,W], float [0,1]. All pixels use the same fixed transformation.
    The fourth channel is a nonlinear Sobel magnitude, not a copied RGB channel.
    """
    b, t = rgb.shape[:2]
    im = F.interpolate(rgb.reshape(b*t, 3, *rgb.shape[-2:]),
                       (grid_size*16, grid_size*16), mode='bicubic',
                       align_corners=False, antialias=True).clamp(0, 1)
    gray = (im * im.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)
    kernels = im.new_tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                             [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]])[:, None] / 8
    gradient = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='reflect'), kernels)
    magnitude = (gradient.square().sum(1, keepdim=True) + 1e-12).sqrt()
    descriptor_image = torch.cat((2*im-1, 2*magnitude-1), 1)
    return F.unfold(descriptor_image, kernel_size=16, stride=16).transpose(1, 2).reshape(b, t, grid_size**2, 1024)


class COMObservationDecoderV4(nn.Module):
    def __init__(self, condition='qwen_temporal_native', native_size=280):
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(condition)
        if native_size < 16 or native_size % 4:
            raise ValueError('native_size must be a multiple of four, >=16')
        self.condition = condition
        self.native_size = native_size
        self.feature_norm = nn.LayerNorm(1024)
        self.feature_projection = nn.Linear(1024, 64)
        self.rgb_conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.rgb_norm1 = nn.GroupNorm(4, 16)
        self.rgb_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.rgb_norm2 = nn.GroupNorm(8, 32)
        self.fusion = nn.Conv2d(163, 64, 1)
        self.fusion_norm = nn.GroupNorm(8, 64)
        self.temporal = ConvGRUCell(64)
        self.coarse_head = nn.Conv2d(64, 1, 1)
        self.residual_conv = nn.Conv2d(80, 32, 3, padding=1)
        self.residual_norm = nn.GroupNorm(8, 32)
        self.residual_head = nn.Conv2d(32, 3, 1)

    @staticmethod
    def _resize(x, size):
        return F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)

    def _query_and_maps(self, semantic, bbox):
        # All coordinates describe pixel centers. A stride-four cell represents
        # the center of its 4x4 pixel footprint, (4j+1.5,4i+1.5).
        b, _, side, _ = semantic.shape
        c = (torch.arange(side, device=semantic.device, dtype=semantic.dtype) + .5) * self.native_size / side - .5
        yy, xx = torch.meshgrid(c, c, indexing='ij')
        mask = ((xx[None] >= bbox[:, 0, None, None]) & (xx[None] < bbox[:, 2, None, None]) &
                (yy[None] >= bbox[:, 1, None, None]) & (yy[None] < bbox[:, 3, None, None]))
        count = mask.sum((1, 2))
        query = (semantic * mask[:, None]).sum((2, 3)) / count.clamp_min(1)[:, None]
        # Empty support has an explicit bilinear cue-center fallback.
        center = (bbox[:, :2] + bbox[:, 2:] - 1) / 2
        grid = (2*(center + .5)/self.native_size - 1)[:, None, None]
        fallback = F.grid_sample(semantic, grid, mode='bilinear', padding_mode='border', align_corners=False)[:, :, 0, 0]
        query = torch.where((count > 0)[:, None], query, fallback)
        xy = torch.stack((2*(xx+.5)/self.native_size-1, 2*(yy+.5)/self.native_size-1))[None].expand(b, -1, -1, -1)
        return query, xy, mask[:, None].to(semantic.dtype), count == 0

    def forward(self, rgb, premerge, bbox_xyxy, *, return_diagnostics=False):
        if rgb.ndim != 5 or rgb.shape[2:] != (3, self.native_size, self.native_size):
            raise ValueError('RGB must be [B,T,3,native_size,native_size]')
        b, t = rgb.shape[:2]
        if t < 1 or bbox_xyxy.shape != (b, 4):
            raise ValueError('empty clip or invalid bbox shape')
        if not torch.isfinite(rgb).all() or not torch.isfinite(bbox_xyxy).all():
            raise ValueError('nonfinite observation')
        if (rgb.min() < 0) or (rgb.max() > 1):
            raise ValueError('RGB must be in [0,1]')
        if not ((bbox_xyxy[:, 2:] > bbox_xyxy[:, :2]).all()):
            raise ValueError('bbox must have positive extent')
        high = self.native_size // 2
        low = self.native_size // 4
        if self.condition.startswith('rgb_'):
            # Production always uses 32x32 patches; tiny test images use the
            # supplied feature grid to avoid expensive irrelevant test work.
            grid_size = 32 if premerge is None else math.isqrt(premerge.shape[2])
            features = rgb_patch_descriptor(rgb, grid_size)
        else:
            if premerge is None or premerge.ndim != 4 or premerge.shape[:2] != (b, t) or premerge.shape[-1] != 1024:
                raise ValueError('premerge must be [B,T,N,1024]')
            grid_size = math.isqrt(premerge.shape[2])
            if grid_size**2 != premerge.shape[2]:
                raise ValueError('premerge tokens must be a square raster grid')
            features = premerge.detach().float()
        semantic = self.feature_projection(self.feature_norm(features.float()))
        semantic = semantic.reshape(b*t, grid_size, grid_size, 64).permute(0, 3, 1, 2)
        semantic = self._resize(semantic, low).reshape(b, t, 64, low, low)
        pixels = rgb.reshape(b*t, 3, self.native_size, self.native_size)
        if self.condition.endswith('lowpass'):
            pixels = self._resize(F.interpolate(pixels, size=(low, low), mode='bilinear', align_corners=False, antialias=True), self.native_size)
        # Average pooling defines exact pixel-footprint centers. Convolutions
        # follow at stride one; this avoids half-pixel shifts of stride-2 convs.
        raw_high = F.silu(self.rgb_norm1(self.rgb_conv1(F.avg_pool2d(pixels, 2))))
        raw_low = F.silu(self.rgb_norm2(self.rgb_conv2(F.avg_pool2d(raw_high, 2))))
        raw_high = raw_high.reshape(b, t, 16, high, high)
        raw_low = raw_low.reshape(b, t, 32, low, low)
        query, xy, cue, fallback = self._query_and_maps(semantic[:, 0], bbox_xyxy.to(semantic.dtype))
        query_map = query[:, :, None, None].expand(-1, -1, low, low)
        centers = (torch.arange(high, device=rgb.device, dtype=rgb.dtype) + .5)*2 - .5
        yy, xx = torch.meshgrid(centers, centers, indexing='ij')
        center_grid = torch.stack((xx, yy))[None]
        predicted, entropies, histories = [], [], []
        h = None
        for frame in range(t):
            x = F.silu(self.fusion_norm(self.fusion(torch.cat((semantic[:, frame], raw_low[:, frame], query_map, xy, cue), 1))))
            h = self.temporal(x, x if h is None or self.condition == 'qwen_spatial_native' else h)
            up = self._resize(h, high)
            residual = self.residual_head(F.silu(self.residual_norm(self.residual_conv(torch.cat((up, raw_high[:, frame]), 1)))))
            logits = self._resize(self.coarse_head(h), high) + residual[:, :1]
            probability = logits.flatten(2).softmax(-1).reshape(b, 1, high, high)
            location = ((center_grid + residual[:, 1:].tanh()) * probability).sum((2, 3))
            predicted.append(location)
            if return_diagnostics:
                entropies.append(-(probability * probability.clamp_min(1e-12).log()).sum((1, 2, 3)))
                histories.append(probability)
        out = {'pixels': torch.stack(predicted, 1)}
        if return_diagnostics:
            out.update(entropy=torch.stack(entropies, 1), probability=torch.stack(histories, 1), cue_empty_grid_fallback=fallback)
        return out


def observation_loss(predicted, target):
    if predicted.shape != target.shape or predicted.shape[-1] != 2:
        raise ValueError('COM label shape mismatch')
    return F.smooth_l1_loss(predicted / 2, target / 2, beta=1)
