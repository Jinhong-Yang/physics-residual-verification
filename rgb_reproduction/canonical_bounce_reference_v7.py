"""Independent constant-restitution Newton state oracle, without an engine.

Height is above the contact plane (not COM height). Initial height H >= 0,
initial velocity zero, positive constant gravity g, and 0 <= e < 1.
Velocity is positive upwards. Add geometry/COM/floor offsets outside this API.
At an impact the returned state is the post-impact state. Floating-point times
within eight float64 ULP-scale units of an analytic impact are interpreted as
that impact; this is a representation convention, not a contact velocity or
height threshold. At and after the finite Zeno accumulation time, height and
velocity are zero. There is no collision-count cap or 0.2 m/s cutoff.

The oracle uses a geometric-series inversion, not an iterative bounce solver.
It does not establish that this law should replace an existing data contract.
"""
from __future__ import annotations
import numpy as np


def _parameters(drop_height_m, restitution, gravity_m_s2):
    values=[]
    for name,value in [('drop_height_m',drop_height_m),('restitution',restitution),('gravity_m_s2',gravity_m_s2)]:
        array=np.asarray(value)
        if array.ndim!=0 or array.dtype.kind not in 'fiu':
            raise ValueError(f'{name} must be a finite real scalar')
        val=float(array)
        if not np.isfinite(val):raise ValueError(f'{name} must be finite')
        values.append(np.longdouble(val))
    h,e,g=values
    if h<0 or not 0<=e<1 or g<=0:
        raise ValueError('Require H>=0, 0<=e<1, and g>0')
    return h,e,g


def bounce_event_times(drop_height_m, restitution, gravity_m_s2=9.81):
    """Return (first impact, Zeno rest time); both zero for H=0.

    e=0 rests at its first impact. These times are float64 representations of
    the exact-law expressions, not a simulated timestep grid.
    """
    h,e,g=_parameters(drop_height_m,restitution,gravity_m_s2)
    first=np.sqrt(2*h/g)
    rest=first*(1+e)/(1-e)
    result=np.asarray([first,rest],dtype=np.float64)
    if not np.isfinite(result).all():raise ValueError('Event times overflow float64')
    return float(result[0]),float(result[1])


def canonical_bounce_state(times_s,drop_height_m,restitution,gravity_m_s2=9.81):
    """Return ``(height_m, velocity_m_s)`` arrays shaped like ``times_s``.

    Times can be unsorted, scalar, empty, or any array shape; they must be
    finite and nonnegative. Inputs and outputs are numeric float64, with
    extended-precision intermediates where the platform supports them.
    """
    h,e,g=_parameters(drop_height_m,restitution,gravity_m_s2)
    raw=np.asarray(times_s)
    if raw.dtype.kind not in 'fiu':raise ValueError('times_s must be real numeric values')
    times=np.asarray(raw,dtype=np.float64)
    if not np.isfinite(times).all() or (times<0).any():raise ValueError('Times must be finite and nonnegative')
    shape=times.shape;t=times.reshape(-1).astype(np.longdouble)
    heights=np.zeros_like(t);velocity=np.zeros_like(t)
    if h==0:return heights.astype(np.float64).reshape(shape),velocity.astype(np.float64).reshape(shape)
    first=np.sqrt(2*h/g);speed=g*first;rest=first*(1+e)/(1-e)
    # Impact comparison tolerance is explicitly only a floating-point scale.
    tol=8*np.finfo(np.float64).eps*np.maximum(abs(t),abs(first))
    falling=t<first-tol
    heights[falling]=h-g*t[falling]**2/2;velocity[falling]=-g*t[falling]
    if e>0:
        bouncing=(~falling)&(t<rest-tol)
        tb=t[bouncing];eps=tol[bouncing]
        elapsed=np.maximum(tb-first,0)
        first_flight=2*e*first
        lifetime=first_flight/(1-e)
        log_e=np.log(e)
        # k is the number of COMPLETE rebound flights after first impact.
        log_remaining=np.log1p(-np.minimum(elapsed/lifetime,np.nextafter(np.longdouble(1),np.longdouble(0))))
        k=np.maximum(np.floor(log_remaining/log_e),0)
        start=-lifetime*np.expm1(k*log_e)
        end=-lifetime*np.expm1((k+1)*log_e)
        # Snap upper impact boundaries forward, giving the post-impact state.
        advance=elapsed>=end-eps
        k=k+advance.astype(np.longdouble)
        start=-lifetime*np.expm1(k*log_e)
        tau=elapsed-start
        tau=np.where(abs(tau)<=eps,0,tau)
        if (tau<0).any():raise ArithmeticError('Geometric-series flight index became inconsistent')
        launch=e*speed*np.exp(k*log_e)
        heights[bouncing]=np.maximum(launch*tau-g*tau**2/2,0)
        velocity[bouncing]=launch-g*tau
    result_h=heights.astype(np.float64).reshape(shape)
    result_v=velocity.astype(np.float64).reshape(shape)
    if not np.isfinite(result_h).all() or not np.isfinite(result_v).all():raise ValueError('State overflows float64')
    return result_h,result_v
