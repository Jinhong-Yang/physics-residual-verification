"""Pure metadata/array contracts for a new canonical confirmation cohort.

No import-time IO, generator invocation, model, PyBullet, or random draws.
Physical parameter and state helpers are generation-side only.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
from pathlib import PurePosixPath
import numpy as np

VERSION='canonical_confirmation_v11'
PROTOCOLS=('multiple_forces','unforced_slide','bounce')
OBS_KEYS={'id','family','variant','appearance_variant','split','protocol','frame_paths',
    'frame_sha256','timestamps_s','camera','camera_metric_calibrated','scale_pixels_per_m',
    'force_N','force_switch_s','surface','gravity_m_s2','contact_fixture','rotation',
    'occlusion','initial_cue','path_base','dataset_version'}


def times(protocol):
    if protocol not in PROTOCOLS:raise ValueError('Unregistered protocol')
    return [0.,.16,.28,.36,.44,.52,.60,.72] if protocol=='bounce' else list(np.arange(8)/10)


def portable(name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or ':' in name or '\\' in name or '..' in p.parts:
        raise ValueError('Expected research-root relative POSIX reference')
    return name


def family_contract(records,exclusion,expected=72):
    if len(records)!=expected:raise ValueError('Wrong predeclared family count')
    families=[];seeds=[];physical=[]
    for record in records:
        if set(record)!={'family','geometry_seed','physical_seed'}:raise ValueError('Family metadata keys')
        f=record['family']
        if not isinstance(f,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',f):
            raise ValueError('Invalid family namespace')
        for key in ('geometry_seed','physical_seed'):
            if type(record[key]) is not int or record[key]<0:raise ValueError('Seed must be explicit nonnegative integer')
        families.append(f);seeds.append(record['geometry_seed']);physical.append(record['physical_seed'])
    if len(set(families))!=expected or len(set(seeds))!=expected or len(set(physical))!=expected:
        raise ValueError('Family/seed duplication')
    if set(families)&set(exclusion['family_ids']) or set(seeds)&set(exclusion['geometry_seeds']):
        raise ValueError('Previously generated or source-declared family/shape seed')
    return records


def groups(records):
    output=[]
    for f in records:
        for v in range(5):
            for protocol in PROTOCOLS:
                ident=hashlib.sha256(f"{VERSION}/{f['family']}/{v}/{protocol}".encode()).hexdigest()[:24]
                output.append({'id':ident,'family':f['family'],'variant':v,'appearance_variant':int(v==4),
                    'object':f"{f['family']}_physical_{0 if v==4 else v}",'protocol':protocol,'split':'confirmation'})
    if len({r['id'] for r in output})!=len(output):raise ValueError('ID collision')
    return output


def parameter_pairs(draws):
    """Same main3d pairing; accepts four generation-side independent draws."""
    if len(draws)!=4:raise ValueError('Exactly four independent physical draws')
    fields={'mass_kg','dynamic_friction','restitution','forces'}
    if any(set(r)!=fields for r in draws):raise ValueError('Physical draw schema')
    result=[dict(r,forces=list(r['forces'])) for r in draws]
    result[1]={**result[0],'mass_kg':result[1]['mass_kg'],'forces':list(result[0]['forces'])}
    result.append(dict(result[0],forces=list(result[0]['forces'])))
    return result


def sample_parameters(seed):
    # Called only after generation authority; identical RNG draw order/distribution.
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(4):
        draws.append({'mass_kg':float(np.exp(rng.uniform(np.log(.15),np.log(1.5)))),
            'dynamic_friction':float(np.exp(rng.uniform(np.log(.05),np.log(.35)))),
            'restitution':float(rng.uniform(.25,.9)), 'forces':[0.,float(rng.uniform(.5,1.5))]})
    return parameter_pairs(draws)


def observation(group,frame_paths,hashes,camera,cue,forces):
    p=group['protocol'];t=times(p)
    if len(frame_paths)!=len(hashes) or len(hashes)!=8:raise ValueError('Require eight frames')
    for path in frame_paths:portable(path)
    if set(camera)!={'width','height','view_matrix','projection_matrix','kind'}:
        raise ValueError('Camera metadata allowlist')
    if set(cue)!={'frame','bbox_xyxy','failure','component_area','coordinate_convention'}:
        raise ValueError('RGB cue metadata allowlist')
    if camera['width']!=280 or camera['height']!=280:raise ValueError('Camera resolution changed')
    for key in ('view_matrix','projection_matrix'):
        if len(camera[key])!=16 or not np.isfinite(camera[key]).all():raise ValueError('Invalid camera matrix')
    # Construct from an explicit allowlist: never copy a renderer/target dictionary.
    result={k:group[k] for k in ('id','family','variant','appearance_variant','split','protocol')}
    result.update(frame_paths=frame_paths,frame_sha256=hashes,timestamps_s=t,camera=camera,
        camera_metric_calibrated=True,scale_pixels_per_m=None,force_N=list(forces) if p=='multiple_forces' else None,
        force_switch_s=.25 if p=='multiple_forces' else None,surface='fixed_horizontal_plane',gravity_m_s2=9.81,
        contact_fixture='single_spherical_tip_radius_.025m_aligned_vertical' if p=='bounce' else 'flat_convex_body_settled_contact',
        rotation='fixed_identity_canonical_1d' if p=='bounce' else 'free_finite_inertia',occlusion=False,
        initial_cue=cue,path_base='research_root',dataset_version=VERSION)
    if set(result)!=OBS_KEYS:raise ValueError('Observation allowlist mismatch')
    return result


def sampled_slide_states(trace,protocol):
    a=np.asarray(trace,dtype=np.float64);indices=np.rint(np.asarray(times(protocol))/.001).astype(np.int64)
    if protocol=='bounce' or a.ndim!=2 or a.shape[1]!=13 or len(a)<=int(indices.max()) or not np.isfinite(a).all():
        raise ValueError('Wrong slide trace')
    return a[indices].copy(),indices


def index_contract(episodes,expected_families=72):
    if len(episodes)!=expected_families*15 or len({r['id'] for r in episodes})!=len(episodes):
        raise ValueError('Missing or duplicate episode')
    families={r['family'] for r in episodes}
    if len(families)!=expected_families:raise ValueError('Wrong family count')
    for family in families:
        selected=[r for r in episodes if r['family']==family]
        if {(r['variant'],r['protocol']) for r in selected}!={(v,p) for v in range(5) for p in PROTOCOLS}:
            raise ValueError('Incomplete paired family')
    for r in episodes:
        if set(r)!=OBS_KEYS or r['split']!='confirmation' or r['timestamps_s']!=times(r['protocol']):
            raise ValueError('Observation field/time leakage or changed split')
    return True
