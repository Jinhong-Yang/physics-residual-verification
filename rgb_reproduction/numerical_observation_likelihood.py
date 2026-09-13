"""Exploratory residual adapter around the unchanged physical linearization.

Requires the caller's shared tracking record. No target values are accessed.
Extra residual evaluations are counted separately from the frozen 200-call J.
"""
import numpy as np
from main3d_evaluation import natural
from main3d_numeric import slide, vertical
from projected_observation import local_jacobian


def observation_likelihood(row, base, prior_mean, prior_scale, tracked):
    if tracked is None:
        raise ValueError('Supply the shared tracking record explicitly')
    local = local_jacobian(row, base, prior_mean, prior_scale, tracked)
    result = dict(local)
    result.update(residual_calls=0, total_forward_calls=local['calls'],
                  projected_residual=np.zeros(local['jacobian'].shape[0]),
                  likelihood_available=False)
    if 'nuisance_parameters' not in local:
        return result
    times = np.asarray(row['timestamps_s'], dtype=float)
    bounce = row['protocol'] == 'bounce'
    axis = 1 if bounce else 0
    data = np.asarray(tracked['xz'], dtype=float)[:, axis]
    sigma = np.asarray(tracked['sigma_xz_m'], dtype=float)[:, axis]
    m, mu, e = natural(local['linearization_theta'])
    nuisance = np.asarray(local['nuisance_parameters'], dtype=float)

    def residual(q):
        result['residual_calls'] += 1
        if bounce:
            predicted = vertical(times, q[0], q[1], q[2], e, q[3])
        else:
            multiple = row['protocol'] == 'multiple_forces'
            predicted = slide(times, q[0], q[1], mu, 1 / m,
                              row['force_N'] if multiple else [0, 0],
                              row['force_switch_s'] if multiple else 1.)
        value = (predicted - data) / sigma
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite reconstructed observation residual')
        return value

    raw = residual(nuisance)
    derivatives = []
    for k in range(len(nuisance)):
        step = np.zeros(len(nuisance)); step[k] = 1e-4
        derivatives.append((residual(nuisance + step) - residual(nuisance - step)) / (2e-4))
    nuisance_jacobian = np.stack(derivatives, axis=1)
    nuisance_inverse = np.linalg.pinv(nuisance_jacobian, rcond=1e-8)
    projected = raw - nuisance_jacobian @ nuisance_inverse @ raw
    result.update(projected_residual=projected, unprojected_residual=raw,
                  nuisance_jacobian=nuisance_jacobian, likelihood_available=True,
                  total_forward_calls=local['calls'] + result['residual_calls'])
    if result['residual_calls'] != 1 + 2 * len(nuisance) or result['total_forward_calls'] > 209:
        raise RuntimeError('Residual reconstruction call accounting mismatch')
    return result
