"""Untrained V8 draft: RGB-motion skip plus first-only bias and causal residual.

Frozen external Qwen vision features are optional. This is a COM observation
head, not language reasoning. The state warp assumes translation only.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from com_observation_decoder_v4 import COMObservationDecoderV4, rgb_patch_descriptor


def translate_state(state, displacement_xy, native_size):
    """Current-grid x reads previous state at x-displacement, align_corners=False."""
    b,_,height,width=state.shape
    y=2*(torch.arange(height,device=state.device,dtype=state.dtype)+.5)/height-1
    x=2*(torch.arange(width,device=state.device,dtype=state.dtype)+.5)/width-1
    yy,xx=torch.meshgrid(y,x,indexing='ij')
    grid=torch.stack((xx,yy),-1)[None].expand(b,-1,-1,-1)
    grid=grid-2*displacement_xy[:,None,None,:]/native_size
    shifted=F.grid_sample(state,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
    support=F.grid_sample(torch.ones_like(state[:,:1]),grid,mode='bilinear',padding_mode='zeros',align_corners=False)
    identity=(displacement_xy==0).all(-1)
    # Avoid interpolation roundoff for the exact no-motion identity operator.
    shifted=torch.where(identity[:,None,None,None],state,shifted)
    retained=torch.where(identity,torch.ones_like(identity,dtype=state.dtype),support.mean((1,2,3)))
    return shifted,retained


class COMObservationDecoderV8(COMObservationDecoderV4):
    """Forward inputs contain observations only; both branches initialize to zero.

    a_rgb is a causally prepared RGB-only center sequence with finite fallback
    (last valid center; first failure uses initial bbox center). It is not COM
    supervision. mode='warp'/'no_warp' changes only recurrent-state transport.
    """
    def __init__(self,backbone='qwen',mode='warp',native_size=280):
        if backbone not in ('qwen','rgb') or mode not in ('warp','no_warp'):
            raise ValueError('Unknown observation backbone or transport mode')
        super().__init__(backbone+'_temporal_native',native_size)
        self.backbone=backbone;self.mode=mode
        # All 166 inputs participate: old 163 plus past dx/native,dy/native,dt.
        self.fusion=nn.Conv2d(166,64,1)
        # A spatially constant softmax logit bias has no effect; omit that weight.
        self.coarse_head=nn.Conv2d(64,1,1,bias=False)
        del self.residual_head
        self.static_head=nn.Linear(32,2)
        self.dynamic_head=nn.Linear(32,2)
        for head in (self.static_head,self.dynamic_head):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def _region(self,bbox,side):
        centers=(torch.arange(side,device=bbox.device,dtype=bbox.dtype)+.5)*self.native_size/side-.5
        yy,xx=torch.meshgrid(centers,centers,indexing='ij')
        mask=((xx[None]>=bbox[:,0,None,None])&(xx[None]<bbox[:,2,None,None])&
              (yy[None]>=bbox[:,1,None,None])&(yy[None]<bbox[:,3,None,None]))
        empty=~mask.flatten(1).any(1)
        # Upper-exclusive integer-pixel bbox center. Only sampling support is
        # clamped at the border; returned a_rgb+b0+u coordinates stay unclipped.
        center=(bbox[:,:2]+bbox[:,2:]-1)/2
        indices=((center+.5)*side/self.native_size-.5).round().long().clamp(0,side-1)
        fallback=torch.zeros_like(mask)
        fallback[torch.arange(len(mask),device=mask.device),indices[:,1],indices[:,0]]=True
        mask=torch.where(empty[:,None,None],fallback,mask)
        return mask[:,None],torch.stack((xx,yy))[None],empty

    def forward(self,rgb,premerge,bbox_xyxy,a_rgb,timestamps_s,*,return_diagnostics=False):
        if rgb.ndim!=5 or rgb.shape[2:]!=(3,self.native_size,self.native_size):
            raise ValueError('Expected native RGB B,T,3,H,W')
        b,t=rgb.shape[:2]
        if t<1 or bbox_xyxy.shape!=(b,4) or a_rgb.shape!=(b,t,2) or timestamps_s.shape!=(b,t):
            raise ValueError('Wrong observation shape')
        if not all(bool(torch.isfinite(v).all()) for v in (rgb,bbox_xyxy,a_rgb,timestamps_s)):
            raise ValueError('Nonfinite observation')
        if bool((rgb<0).any() or (rgb>1).any() or (bbox_xyxy[:,2:]<=bbox_xyxy[:,:2]).any()):
            raise ValueError('Invalid RGB range/bbox')
        if bool((torch.diff(timestamps_s,dim=1)<=0).any() or (timestamps_s<0).any()):
            raise ValueError('Require increasing nonnegative actual observation times')
        a=a_rgb.detach().to(rgb.dtype);times=timestamps_s.detach().to(rgb.dtype)
        bbox=bbox_xyxy.detach().to(rgb.dtype)
        high=self.native_size//2;low=self.native_size//4
        if self.backbone=='rgb':
            grid=32 if premerge is None else math.isqrt(premerge.shape[2])
            feature=rgb_patch_descriptor(rgb,grid)
        else:
            if premerge is None or premerge.ndim!=4 or premerge.shape[:2]!=(b,t) or premerge.shape[-1]!=1024:
                raise ValueError('Expected frozen Qwen B,T,N,1024 raster')
            grid=math.isqrt(premerge.shape[2])
            if grid*grid!=premerge.shape[2] or not bool(torch.isfinite(premerge).all()):
                raise ValueError('Invalid frozen spatial raster')
            feature=premerge.detach().float()
        semantic=self.feature_projection(self.feature_norm(feature.float()))
        semantic=semantic.reshape(b*t,grid,grid,64).permute(0,3,1,2)
        semantic=self._resize(semantic,low).reshape(b,t,64,low,low)
        raw=rgb.reshape(b*t,3,self.native_size,self.native_size)
        raw_high=F.silu(self.rgb_norm1(self.rgb_conv1(F.avg_pool2d(raw,2))))
        raw_low=F.silu(self.rgb_norm2(self.rgb_conv2(F.avg_pool2d(raw_high,2))))
        raw_high=raw_high.reshape(b,t,16,high,high);raw_low=raw_low.reshape(b,t,32,low,low)
        query,_,_,query_fallback=self._query_and_maps(semantic[:,0],bbox)
        query=query[:,:,None,None].expand(-1,-1,low,low)
        h=None;predictions=[];dynamic=[];supports=[];empty_regions=[];bias=None
        for frame in range(t):
            displacement=torch.zeros_like(a[:,0]) if frame==0 else a[:,frame]-a[:,frame-1]
            dt=torch.zeros_like(times[:,0]) if frame==0 else times[:,frame]-times[:,frame-1]
            motion=torch.cat((2*displacement/self.native_size,dt[:,None]),1)[:,:,None,None].expand(-1,-1,low,low)
            shift=a[:,frame]-a[:,0]
            moving_bbox=bbox+torch.cat((shift,shift),1)
            cue,gridxy,_=self._region(moving_bbox,low)
            xy=2*(gridxy-a[:,frame,:,None,None])/self.native_size
            x=F.silu(self.fusion_norm(self.fusion(torch.cat((semantic[:,frame],raw_low[:,frame],query,xy,cue.to(rgb.dtype),motion),1))))
            retained=torch.ones(b,device=rgb.device,dtype=rgb.dtype)
            if h is None: previous=x
            elif self.mode=='warp': previous,retained=translate_state(h,displacement,self.native_size)
            else: previous=h
            h=self.temporal(x,previous)
            up=self._resize(h,high)
            dense=F.silu(self.residual_norm(self.residual_conv(torch.cat((up,raw_high[:,frame]),1))))
            region,_,empty=self._region(moving_bbox,high)
            logits=self._resize(self.coarse_head(h),high).masked_fill(~region,-torch.inf)
            attention=logits.flatten(2).softmax(-1).reshape(b,1,high,high)
            pooled=(attention*dense).sum((2,3))
            if frame==0:
                bias=self.static_head(pooled)
                u=torch.zeros_like(bias)
            else:u=self.dynamic_head(pooled)
            # Preserve the original observation's float64 low bits. A constant
            # float32 bias must not acquire time-varying rounding on the skip.
            predictions.append(a_rgb[:,frame].detach().double()+bias.double()+u.double());dynamic.append(u)
            supports.append(retained);empty_regions.append(empty)
        out={'pixels':torch.stack(predictions,1)}
        if return_diagnostics:
            out.update(static_bias=bias,dynamic_residual=torch.stack(dynamic,1),
                warp_support_fraction=torch.stack(supports,1),empty_region_fallback=torch.stack(empty_regions,1),
                initial_query_fallback=query_fallback)
        return out
