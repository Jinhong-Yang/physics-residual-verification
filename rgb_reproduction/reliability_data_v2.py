"""Development-only observation extraction; target parsing is a separate stage."""
import hashlib
import numpy as np

SELECTOR = 'reliability_v2_20260912:'


def select_families(groups, counts=None):
    counts = counts or {'train': 48, 'validation': 12}
    family_split = {}
    for group in groups:
        family, split = group['family'], group['split']
        if family in family_split and family_split[family] != split:
            raise ValueError('Family crosses splits')
        family_split[family] = split
    selected = {}
    for split, count in counts.items():
        candidates = [f for f, s in family_split.items() if s == split]
        candidates.sort(key=lambda f: (hashlib.sha256((SELECTOR + f).encode()).hexdigest(), f))
        if len(candidates) < count:
            raise ValueError('Insufficient families')
        selected[split] = candidates[:count]
    return selected


def extract_observation(row, base, prior_mean, prior_scale, tracked=None):
    from main3d_refinement import tracking
    from numerical_observation_likelihood import observation_likelihood
    from main3d_evaluation import natural
    from main3d_numeric import slide, vertical
    mean, scale = np.asarray(prior_mean, float), np.asarray(prior_scale, float)
    if mean.shape != (3,) or scale.shape != (3,) or not np.isfinite(mean).all() or not (scale > 0).all():
        raise ValueError('Invalid prior')
    tracked = tracking(row, base) if tracked is None else tracked
    local = observation_likelihood(row, base, mean, scale, tracked)
    out = dict(prior_mean=mean, prior_scale=scale, linearization_mean=mean.copy(),
               J=np.zeros((8, 3)), Jn=np.zeros((8, 4)), residual=np.zeros(8),
               valid=np.zeros(8, bool), active=np.zeros(3, bool), times=np.zeros(8),
               xz=np.zeros((8, 2)), sigma_xz=np.ones((8, 2)), force=np.zeros(2),
               force_switch=np.array(0.), protocol=np.array(row['protocol']),
               nuisance=np.zeros(4), nuisance_dim=np.array(0),
               source_projected_J=np.zeros((8, 3)), source_rank=np.array(0),
               support_projected=np.array(False), forward_calls=np.array(local['total_forward_calls']),
               extra_physical_calls=np.array(0), likelihood_available=np.array(False),
               status=np.array(local['status']), tracking_status=np.array(tracked['status']))
    times = np.asarray(row.get('timestamps_s') or [], float)
    if len(times) > 8:
        raise ValueError('More than eight observed timestamps')
    out['times'][:len(times)] = times
    if row.get('force_N') is not None:
        out['force'] = np.asarray(row['force_N'], float)
    if row.get('force_switch_s') is not None:
        out['force_switch'] = np.array(row['force_switch_s'], float)
    if not local['likelihood_available']:
        return out
    n = len(times)
    if not 3 <= n <= 8 or tracked['status'] != 'tracked':
        raise ValueError('Available likelihood without valid tracking')
    q = np.asarray(local['nuisance_parameters'], float)
    linear = np.asarray(local['linearization_theta'], float)
    bounce = row['protocol'] == 'bounce'
    axis = 1 if bounce else 0
    active = [2] if bounce else [0, 1] if row['protocol'] == 'multiple_forces' else [1]
    xz, sigma = np.asarray(tracked['xz']), np.asarray(tracked['sigma_xz_m'])
    count = 0
    def residual_at(theta):
        nonlocal count
        count += 1
        m, mu, e = natural(theta)
        pred = (vertical(times, q[0], q[1], q[2], e, q[3]) if bounce else
                slide(times, q[0], q[1], mu, 1/m,
                      row['force_N'] if row['protocol'] == 'multiple_forces' else [0., 0.],
                      row['force_switch_s'] if row['protocol'] == 'multiple_forces' else 1.))
        return (pred - xz[:, axis]) / sigma[:, axis]
    for k in active:
        step = np.zeros(3); step[k] = scale[k] * 1e-3
        out['J'][:n, k] = (residual_at(linear + step) - residual_at(linear - step)) / 2e-3
    assert count == 2 * len(active)
    out['Jn'][:n, :len(q)] = local['nuisance_jacobian']
    out['residual'][:n] = local['unprojected_residual']
    rebuilt = out['J'][:n] - local['nuisance_jacobian'] @ np.linalg.pinv(local['nuisance_jacobian'], rcond=1e-8) @ out['J'][:n]
    if not np.allclose(rebuilt, local['jacobian'], atol=1e-8, rtol=1e-10):
        raise ValueError('Raw physical Jacobian does not reconstruct original projection')
    out['source_projected_J'][:n] = local['jacobian']
    out['valid'][:n] = True; out['active'][active] = True
    out['xz'][:n] = xz; out['sigma_xz'][:n] = sigma
    out['nuisance'][:len(q)] = q
    out.update(linearization_mean=linear, nuisance_dim=np.array(len(q)),
               source_rank=np.array(local['rank']), likelihood_available=np.array(True),
               support_projected=np.array(local.get('linearization_point_projected_to_ID_support', False)),
               extra_physical_calls=np.array(count), forward_calls=np.array(local['total_forward_calls'] + count))
    for key, value in out.items():
        if np.asarray(value).dtype.kind in 'f' and not np.isfinite(value).all():
            raise ValueError('Nonfinite ' + key)
    return out
