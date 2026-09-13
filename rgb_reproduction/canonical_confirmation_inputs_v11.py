"""Pure observation/prior/input adapter contracts. No import-time IO or RNG."""
import copy
import numpy as np
from pathlib import Path,PurePosixPath

PROMPT='Describe only the visible appearance of the object.'
ROW_KEYS=('id','timestamps_s','protocol','camera','camera_metric_calibrated','scale_pixels_per_m',
    'force_N','force_switch_s','surface','gravity_m_s2','contact_fixture','rotation','occlusion')


def root_reference(root,name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or ':' in name or '\\' in name or '..' in p.parts:
        raise ValueError('Require research-root-relative POSIX path')
    result=(Path(root)/name).resolve()
    if not result.is_relative_to(Path(root).resolve()):raise ValueError('Reference escapes research root')
    return result


def tracking_row(episode):
    """Explicit key conversion; ROOT is the required tracking base, not DATA."""
    if episode['path_base']!='research_root':raise ValueError('Wrong frame path base')
    if len(episode['frame_paths'])!=8 or len(episode['timestamps_s'])!=8 or len(episode['frame_sha256'])!=8:
        raise ValueError('Eight actual frames required')
    if episode['occlusion'] or not episode['camera_metric_calibrated']:raise ValueError('Unregistered observation condition')
    row={key:copy.deepcopy(episode[key]) for key in ROW_KEYS}
    row['frames']=list(episode['frame_paths'])
    for name in row['frames']:
        p=PurePosixPath(name)
        if p.is_absolute() or ':' in name or '\\' in name or '..' in p.parts:raise ValueError('Nonportable frame')
    if row['protocol']=='multiple_forces' and (row['force_N'] is None or row['force_switch_s']!=.25):
        raise ValueError('Known force/switch missing or changed')
    return row


def first_image_index(episodes):
    images={};joins=[]
    for episode in episodes:
        tracking_row(episode)
        h=episode['frame_sha256'][0];p=episode['frame_paths'][0]
        images.setdefault(h,{'feature_id':h,'source_paths':[]})
        if p not in images[h]['source_paths']:images[h]['source_paths'].append(p)
        joins.append({'id':episode['id'],'feature_id':h})
    if len({j['id'] for j in joins})!=len(joins):raise ValueError('Duplicate episode')
    for r in images.values():r['source_paths'].sort()
    return [images[k] for k in sorted(images)],joins


def join_priors(ids,feature_ids,mean,scale,joins):
    if ids!=[j['id'] for j in joins]:raise ValueError('Prior/observation ID order changed')
    if len(set(feature_ids))!=len(feature_ids):raise ValueError('Duplicate prior feature')
    m,s=np.asarray(mean),np.asarray(scale)
    if m.shape!=s.shape or m.shape!=(len(feature_ids),3) or not np.isfinite(m).all() or not np.isfinite(s).all() or not (s>0).all():
        raise ValueError('Invalid prior output')
    lookup={h:i for i,h in enumerate(feature_ids)}
    indices=[lookup[j['feature_id']] for j in joins]
    return m[indices].copy(),s[indices].copy()


def head_batch_indices(n,batch=8):
    if n<1 or batch!=8:raise ValueError('Frozen B8 evaluation batching')
    return [list(range(i,min(i+batch,n))) for i in range(0,n,batch)]


def prior_batches(features):
    """Original predict_projection_priors.py splits unique features by 256."""
    return features.split(256)


def require_regression(receipt):
    if receipt.get('status')!='passed' or receipt.get('episodes')!=900 or receipt.get('T')!=8:
        raise ValueError('Original900/T8 regression is required')
    if receipt.get('batch_sizes')!=[8]*112+[4] or receipt.get('all900_bitexact') is not True:
        raise ValueError('Original B8x112+B4 shape/order/endpoint parity missing')
    if receipt.get('targets_opened') is not False:raise ValueError('Regression must be label-free')
    if receipt.get('prior_mean_scale_bitexact') is not True or receipt.get('numeric_all_fields_bitexact') is not True:
        raise ValueError('Actual prior and RGB/J computation parity is required')
    if receipt.get('prior_GPU_forwards')!=3 or receipt.get('prior_CPU_batch_sizes')!=[256]*9+[234]:
        raise ValueError('Original first-token/prior regression schedule is required')


def pack_inputs(episodes,records,tracks):
    """Preserve all rows. Failure rows are saved but cannot acquire READY seal."""
    if len(episodes)!=len(records) or len(records)!=len(tracks):raise ValueError('Input count mismatch')
    arrays={k:np.stack([r[k] for r in records]) for k in records[0]}
    pixels=np.array([np.asarray(t['pixels'],dtype=np.float64) if t['status']=='tracked' else np.full((8,2),np.nan) for t in tracks])
    arrays.update(ids=np.array([e['id'] for e in episodes]),family_ids=np.array([e['family'] for e in episodes]),
        split=np.array([e['split'] for e in episodes]),pixels=pixels)
    failures=[e['id'] for e,t in zip(episodes,tracks) if t['status']!='tracked']
    ready=not failures and np.isfinite(pixels).all() and bool(arrays['valid'].all())
    return arrays,{'all_rows_retained':True,'excluded':0,'tracking_failure_ids':failures,
        'ready_for_frozen_head':bool(ready),'fallback_introduced':False}
