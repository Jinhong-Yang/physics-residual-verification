"""Prescribed secondary M95: three starts, then three winner updates.

This single, more generous numeric comparator was prescribed from the M67/K4
wall-time gap. No data-dependent iteration choice or additional model fitting.
It does not replace the frozen primary M67 comparison. Original prior, starts,
step algebra, trust region and line search all remain unchanged.
"""
import torch
import reliability_nonlinear_v2 as reference
import reliability_contact_multistart_v2 as multistart
import reliability_nonlinear_fast_v2 as fast


def bounce_beam(batch, trust_radius=1.):
    b = len(batch['prior_mean'])
    if not b or not bool((batch['active'] == batch['active'].new_tensor([False, False, True])).all()):
        raise ValueError('Bounce beam requires a homogeneous bounce group')
    if not bool(batch['valid'].any(-1).all()):
        raise ValueError('Bounce beam needs valid observations')
    if not bool(torch.isfinite(batch['alpha']).all() and (batch['alpha'] > 0).all()):
        raise ValueError('Need finite positive frozen alpha')
    offsets = batch['prior_mean'].new_tensor([0., -1., 1.])
    index = torch.arange(b, device=batch['prior_mean'].device).repeat(3)
    expanded = {key: value[index] for key, value in batch.items()}
    initial = expanded['prior_mean'].clone()
    initial[:, 2] += offsets.repeat_interleave(b) * expanded['prior_scale'][:, 2]
    q0 = reference._nuisance_bounds(batch, batch['nuisance'])
    weights = batch['alpha'][:, None] * batch['valid'].double()
    first = fast.one_step(expanded, initial, q0.repeat(3, 1), weights.repeat(3, 1), trust_radius)
    costs = first['cost'].reshape(3, b).transpose(0, 1)
    if not bool(torch.isfinite(costs).all()):
        raise ValueError('Nonfinite branch comparison cost')
    chosen = costs.argmin(1)
    gather = chosen * b + torch.arange(b, device=chosen.device)
    theta, nuisance = first['mean'][gather], first['nuisance'][gather]
    calls = 3 * first['forward_calls_per_observation']
    guard = first['numerical_log_guard_evaluations_per_observation'].reshape(3, b).sum(0)
    # Exactly three prescribed updates after the unchanged first-step selection.
    # Only the final selected branch receives covariance computation.
    for _ in range(3):
        winner = fast.one_step(batch, theta, nuisance, weights, trust_radius)
        theta, nuisance = winner['mean'], winner['nuisance']
        calls += winner['forward_calls_per_observation']
        guard = guard + winner['numerical_log_guard_evaluations_per_observation']
    final = fast.relinearize(batch, theta, nuisance)
    local = dict(batch, **{key: final[key] for key in ['J', 'Jn', 'residual', 'linearization_mean', 'nuisance']})
    curvature = reference._solve(local, weights)
    calls += final['forward_calls_per_observation']
    if calls != 95:
        raise ValueError('Declared bounce residual budget changed')
    guard = guard + final['numerical_log_guard_evaluations_per_observation']
    return {'mean': theta, 'covariance': curvature['covariance'], 'nuisance': nuisance,
            'weights': weights, 'local_gaussian_mean': curvature['mean'],
            'forward_calls_per_observation': torch.full((b,), calls, device=chosen.device, dtype=torch.int64),
            'numerical_log_guard_evaluations_per_observation': guard,
            'selected_start_index': chosen, 'selected_start_offset': offsets[chosen],
            'first_step_branch_costs': costs, 'winner_final_cost': winner['cost'],
            'first_step_branch_means': first['mean'].reshape(3, b, 3).transpose(0, 1),
            'forward_calls': calls * b, 'physical_kernel_calls': calls * b}


_outer = fast._copy_function(multistart.refine, {'bounce_beam':bounce_beam, 'efficient':fast._efficient_branch})


def refine(batch, trust_radius=1.):
    """M95 bounce with the same fixed-alpha four-update slide protocols."""
    result = _outer(batch, trust_radius)
    result['physical_kernel_calls'] = result['forward_calls']
    return result
