"""Learned full Gaussian update restricted to local prior-whitened directions.

Supplementary implementation. Not wired into the frozen main experiment.
The Jacobian is supplied by observation-only physics at a current estimate.
No derivatives are propagated through the discrete rank/SVD decision.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def observable_basis(jacobian, relative_threshold=1e-3, absolute_threshold=1.):
    if jacobian.ndim!=3 or jacobian.shape[-1]!=3 or jacobian.shape[-2]<1:
        raise ValueError('Expected batch x observations x 3 whitened Jacobian')
    if not torch.isfinite(jacobian).all():raise ValueError('Nonfinite Jacobian')
    if relative_threshold<0 or absolute_threshold<0:raise ValueError('Negative threshold')
    with torch.no_grad():
        _,singular,vh=torch.linalg.svd(jacobian.detach(),full_matrices=False)
        threshold=torch.maximum(singular[:,:1]*relative_threshold,torch.full_like(singular[:,:1],absolute_threshold))
        keep=singular>threshold
        basis=vh.transpose(-1,-2)
        # Fixed three-column readout, including underdetermined n<3 cases.
        padding=3-basis.shape[-1]
        if padding:
            basis=F.pad(basis,(0,padding));singular=F.pad(singular,(0,padding));keep=F.pad(keep,(0,padding),value=False)
    return basis,keep,singular


def restricted_gaussian(prior_mean,prior_scale,delta,raw_precision,basis,keep):
    if prior_mean.ndim!=2 or prior_mean.shape[-1]!=3:raise ValueError('Expected batch x 3 means')
    b=len(prior_mean)
    if not (prior_scale.shape==delta.shape==raw_precision.shape==prior_mean.shape and basis.shape==(b,3,3) and keep.shape==(b,3)):
        raise ValueError('Inconsistent posterior shapes')
    if not all(torch.isfinite(x).all() for x in [prior_mean,prior_scale,delta,raw_precision,basis]):raise ValueError('Nonfinite posterior input')
    if not (prior_scale>0).all():raise ValueError('Prior scales must be positive')
    basis=basis.detach();keep=keep.detach().bool();active=basis*keep[:,None,:]
    gram=active.transpose(-1,-2)@active
    if not torch.allclose(gram,torch.diag_embed(keep.to(gram.dtype)),atol=2e-5,rtol=2e-5):raise ValueError('Active basis is not orthonormal')
    projected=active@(active.transpose(-1,-2)@delta[:,:,None]);projected=projected[:,:,0]
    strength=F.softplus(raw_precision)*keep
    identity=torch.eye(3,dtype=prior_mean.dtype,device=prior_mean.device).expand(b,3,3)
    precision=identity+(active*strength[:,None,:])@active.transpose(-1,-2)
    # Orthonormal-basis inverse avoids numerical cancellation in generic inversion
    # while preserving the exact prior covariance in the unresolved subspace.
    contraction=strength/(1+strength)
    white_cov=identity-(active*contraction[:,None,:])@active.transpose(-1,-2)
    # Generic Cholesky verifies SPD; extreme overconfidence must fail explicitly.
    chol=torch.linalg.cholesky(white_cov)
    mean=prior_mean+prior_scale*projected
    covariance=prior_scale[:,:,None]*white_cov*prior_scale[:,None,:]
    rank=keep.sum(-1);closed=rank==0
    mean=torch.where(closed[:,None],prior_mean,mean)
    covariance=torch.where(closed[:,None,None],torch.diag_embed(prior_scale.square()),covariance)
    return {'mean':mean,'covariance':covariance,'marginal_scale':torch.sqrt(torch.diagonal(covariance,dim1=-2,dim2=-1)),
      'prior_mean':prior_mean,'prior_scale':prior_scale,'whitened_mean_update':projected,'whitened_precision':precision,
      'whitened_covariance':white_cov,'whitened_cholesky':chol,'rank':rank,'basis':basis,'keep':keep}


def full_gaussian_nll(target,mean,covariance,reduction='mean'):
    if target.shape!=mean.shape or covariance.shape!=(*mean.shape,3):raise ValueError('Invalid Gaussian shapes')
    chol=torch.linalg.cholesky(covariance)
    residual=torch.linalg.solve_triangular(chol,(target-mean)[:,:,None],upper=False)[:,:,0]
    values=.5*(residual.square().sum(-1)+2*torch.log(torch.diagonal(chol,dim1=-2,dim2=-1)).sum(-1)+3*math.log(2*math.pi))
    if reduction=='none':return values
    if reduction=='mean':return values.mean()
    raise ValueError('Unknown reduction')


class ProjectedPosteriorHead(nn.Module):
    """Evidence head; appearance-only prior is an explicit separate input."""
    def __init__(self,evidence_dim):
        super().__init__();self.update=nn.Linear(evidence_dim,6)
        with torch.no_grad():self.update.weight.mul_(.01);self.update.bias[:3].zero_();self.update.bias[3:].fill_(-5.)

    def forward(self,evidence,prior_mean,prior_scale,jacobian):
        delta,raw_precision=self.update(evidence).chunk(2,-1)
        basis,keep,singular=observable_basis(jacobian)
        result=restricted_gaussian(prior_mean,prior_scale,delta,raw_precision,basis,keep)
        result['singular_values']=singular
        return result


class AppearancePrior(nn.Module):
    """Fit with appearance-only frozen features and proper Gaussian NLL."""
    def __init__(self,feature_dim):
        super().__init__();self.readout=nn.Linear(feature_dim,6)

    def initialize(self,mean,scale):
        if not (scale>1e-3).all():raise ValueError('Initialization scale too small')
        with torch.no_grad():
            self.readout.weight.zero_();self.readout.bias[:3].copy_(mean)
            self.readout.bias[3:].copy_(torch.log(torch.expm1(scale-1e-3)))

    def forward(self,appearance):
        mean,raw_scale=self.readout(appearance).chunk(2,-1)
        return mean,F.softplus(raw_scale)+1e-3
