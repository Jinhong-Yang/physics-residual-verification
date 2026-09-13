"""Observation-only common translation preconditioner; frozen solver unchanged."""
import torch


def translation_consistent_batch(batch,new_xz):
    rgb=batch['xz'];valid=batch['valid'];q=batch['nuisance']
    if new_xz.shape!=rgb.shape or rgb.ndim!=3 or rgb.shape[1:]!=(8,2) or new_xz.dtype!=torch.float64:
        raise ValueError('Require float64 Bx8x2 coordinates')
    if q.shape!=(len(rgb),4) or valid.shape!=rgb.shape[:2] or valid.dtype!=torch.bool:
        raise ValueError('Invalid shared nuisance/mask shape')
    if not bool(torch.isfinite(new_xz).all() and torch.isfinite(rgb).all() and torch.isfinite(q).all()):
        raise ValueError('Nonfinite coordinates/nuisance')
    bounce=batch['active'][:,2]
    axis=bounce.to(torch.long)
    displacement=(new_xz-rgb).gather(2,axis[:,None,None].expand(-1,8,1)).squeeze(-1)
    count=valid.sum(-1)
    delta=(displacement*valid).sum(-1)/count.clamp_min(1)
    moved=q.clone();moved[:,0]=q[:,0]+delta
    moved[:,2]=torch.where(bounce,q[:,2]+delta,q[:,2])
    # Refine recomputes residual/J/Jn before using them. Stored caches unchanged.
    return dict(batch,xz=new_xz,nuisance=moved),delta
