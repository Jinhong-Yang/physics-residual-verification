"""Exact AST-extracted slide function; original module hash in INPUT_MANIFEST.json."""
import numpy as np

def slide(times,x0,v0,mu,invm,forces,switch):
    # Exact piecewise constant acceleration with a non-reversing/static stop.
    def advance(x,v,a,dt):
        duration=np.asarray(dt);moving=np.minimum(duration,v/max(-a,1e-15)) if a<0 else duration
        xx=x+v*moving+.5*a*moving*moving;vv=np.maximum(0,v+a*moving)
        return xx,vv
    t=np.asarray(times);a0=forces[0]*invm-mu*9.81;a1=forces[1]*invm-mu*9.81
    before=np.minimum(t,switch);x,v=advance(x0,v0,a0,before);xx,vv=advance(x,v,a1,np.maximum(0,t-switch))
    return xx
