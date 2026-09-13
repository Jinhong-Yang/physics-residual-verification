"""Three-start, two-depth bounce baseline within 67 residual evaluations.

Each first-stage branch performs one step without final covariance. Its stored
observation-plus-original-prior cost selects the branch. Only the winner gets
one further step and one final curvature calculation. Original priors never
change. Contact derivatives and branch selection remain piecewise operations.
"""
import torch
import reliability_nonlinear_v2 as reference
import reliability_nonlinear_efficient_v2 as efficient


def one_step(batch, theta, nuisance, weights, trust_radius=1.):
    """Exact reference update algebra, without final covariance/linearization."""
    if not 0 < trust_radius < float('inf'):
        raise ValueError('Invalid trust radius')
    prior, scale = batch['prior_mean'], batch['prior_scale']
    active = batch['active']
    meaningful = active.any(-1) & batch['valid'].any(-1)
    lin = efficient.relinearize(batch, theta, nuisance)
    local = dict(batch, **{key: lin[key] for key in ['J', 'Jn', 'residual', 'linearization_mean', 'nuisance']})
    fit = reference._solve(local, weights)
    raw_step = (fit['mean'] - theta) / scale
    norm = torch.linalg.vector_norm(raw_step, dim=-1)
    contraction = (trust_radius / norm.clamp_min(1e-30)).clamp_max(1)
    no_weighted_evidence = ~(weights > 0).any(-1)
    contraction = torch.where(no_weighted_evidence, torch.ones_like(contraction), contraction)
    step = raw_step * contraction[:, None] * active
    proposal = theta + scale * step
    root = torch.where(weights > 0, weights.clamp_min(torch.finfo(torch.float64).tiny).sqrt(), torch.zeros_like(weights))
    residual_after_physical = lin['residual'] + (lin['J'] @ step[:, :, None]).squeeze(-1)
    weighted_n = root[:, :, None] * lin['Jn']
    dq = -(torch.linalg.pinv(weighted_n, atol=0., rtol=1e-12)
           @ (root * residual_after_physical)[:, :, None]).squeeze(-1)
    proposed_q = reference._nuisance_bounds(batch, nuisance + dq)
    def cost(point, residual):
        return .5 * (((point - prior) / scale).square().sum(-1) + (weights * residual.square()).sum(-1))
    before = cost(theta, lin['residual'])
    best_cost, best_theta, best_q = before, theta, nuisance
    accepted = torch.zeros(len(theta), dtype=theta.dtype, device=theta.device)
    calls = lin['forward_calls_per_observation']
    guard = lin['numerical_log_guard_evaluations_per_observation']
    for fraction in [1., .5, .25]:
        trial_theta = theta + fraction * (proposal - theta)
        trial_q = nuisance + fraction * (proposed_q - nuisance)
        trial_r = reference.observation_residual(batch, trial_theta, trial_q)
        calls += 1
        guard = guard + (trial_theta[:, :2].abs() > 30).any(-1).long()
        trial_cost = cost(trial_theta, trial_r)
        better = torch.isfinite(trial_cost) & (trial_cost < best_cost) & meaningful
        best_cost = torch.where(better, trial_cost, best_cost)
        best_theta = torch.where(better[:, None], trial_theta, best_theta)
        best_q = torch.where(better[:, None], trial_q, best_q)
        accepted = torch.where(better, torch.full_like(accepted, fraction), accepted)
    best_theta = torch.where(no_weighted_evidence[:, None], prior, best_theta)
    best_cost = torch.where(no_weighted_evidence, torch.zeros_like(best_cost), best_cost)
    return {'mean': best_theta, 'nuisance': best_q, 'cost': best_cost, 'cost_before': before,
            'accepted_fraction': accepted, 'forward_calls_per_observation': calls,
            'numerical_log_guard_evaluations_per_observation': guard}


def bounce_beam(batch, trust_radius=1.):
    b = len(batch['prior_mean'])
    if not b or not bool((batch['active'] == batch['active'].new_tensor([False, False, True])).all()):
        raise ValueError('Bounce beam requires a homogeneous bounce group')
    if not bool(batch['valid'].any(-1).all()):
        raise ValueError('Bounce beam needs valid observations')
    if not bool(torch.isfinite(batch['alpha']).all() and (batch['alpha'] > 0).all()):
        raise ValueError('Need finite positive frozen alpha')
    # Center first gives deterministic center-preferred ties; side offsets are symmetric.
    offsets = batch['prior_mean'].new_tensor([0., -1., 1.])
    index = torch.arange(b, device=batch['prior_mean'].device).repeat(3)
    expanded = {key: value[index] for key, value in batch.items()}
    initial = expanded['prior_mean'].clone()
    initial[:, 2] += offsets.repeat_interleave(b) * expanded['prior_scale'][:, 2]
    q0 = reference._nuisance_bounds(batch, batch['nuisance'])
    weights = batch['alpha'][:, None] * batch['valid'].double()
    first = one_step(expanded, initial, q0.repeat(3, 1), weights.repeat(3, 1), trust_radius)
    costs = first['cost'].reshape(3, b).transpose(0, 1)
    if not bool(torch.isfinite(costs).all()):
        raise ValueError('Nonfinite branch comparison cost')
    chosen = costs.argmin(1)
    gather = chosen * b + torch.arange(b, device=chosen.device)
    winner = one_step(batch, first['mean'][gather], first['nuisance'][gather], weights, trust_radius)
    # Only the winner gets final covariance. The three first-stage calls above
    # have no hidden final relinearization and no final covariance calculation.
    final = efficient.relinearize(batch, winner['mean'], winner['nuisance'])
    local = dict(batch, **{key: final[key] for key in ['J', 'Jn', 'residual', 'linearization_mean', 'nuisance']})
    curvature = reference._solve(local, weights)
    calls = 3 * first['forward_calls_per_observation'] + winner['forward_calls_per_observation'] + final['forward_calls_per_observation']
    if calls != 67:
        raise ValueError('Declared bounce residual budget changed')
    guard = (first['numerical_log_guard_evaluations_per_observation'].reshape(3, b).sum(0)
             + winner['numerical_log_guard_evaluations_per_observation']
             + final['numerical_log_guard_evaluations_per_observation'])
    return {'mean': winner['mean'], 'covariance': curvature['covariance'], 'nuisance': winner['nuisance'],
            'weights': weights, 'local_gaussian_mean': curvature['mean'],
            'forward_calls_per_observation': torch.full((b,), calls, device=chosen.device, dtype=torch.int64),
            'numerical_log_guard_evaluations_per_observation': guard,
            'selected_start_index': chosen, 'selected_start_offset': offsets[chosen],
            'first_step_branch_costs': costs, 'winner_final_cost': winner['cost'],
            'first_step_branch_means': first['mean'].reshape(3, b, 3).transpose(0, 1),
            'forward_calls': calls * b, 'physical_kernel_calls': 2 * calls * b}


def refine(batch, trust_radius=1.):
    """Bounce beam plus unchanged alpha-unit4 for force and free sliding."""
    prior = batch['prior_mean']; b = len(prior); device = prior.device
    meaningful = batch['active'].any(-1) & batch['valid'].any(-1)
    bounce = meaningful & batch['active'][:, 2]
    other = meaningful & ~bounce
    result = {'mean': prior, 'covariance': torch.diag_embed(batch['prior_scale'].square()),
              'nuisance': batch['nuisance'], 'weights': torch.zeros_like(batch['times']),
              'local_gaussian_mean': prior,
              'forward_calls_per_observation': torch.zeros(b, dtype=torch.int64, device=device),
              'numerical_log_guard_evaluations_per_observation': torch.zeros(b, dtype=torch.int64, device=device),
              'selected_start_index': torch.full((b,), -1, dtype=torch.int64, device=device),
              'selected_start_offset': torch.zeros(b, dtype=prior.dtype, device=device),
              'first_step_branch_costs': torch.zeros(b, 3, dtype=prior.dtype, device=device),
              'winner_final_cost': torch.zeros(b, dtype=prior.dtype, device=device),
              'first_step_branch_means': torch.zeros(b, 3, 3, dtype=prior.dtype, device=device)}
    for mask, use_beam in [(other, False), (bounce, True)]:
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        if not len(indices):
            continue
        local = {key: value[indices] for key, value in batch.items()}
        if use_beam:
            group = bounce_beam(local, trust_radius)
        else:
            group = efficient.refine(local, weights_callback=lambda local, unit, iteration:
                local['alpha'][:, None] * local['valid'].double(), iterations=4, trust_radius=trust_radius)
        for key in result:
            if key in group:
                result[key] = result[key].index_copy(0, indices, group[key])
    result['forward_calls'] = int(result['forward_calls_per_observation'].sum())
    # The unchanged residual routine evaluates both physical branches per call.
    result['physical_kernel_calls'] = 2 * result['forward_calls']
    result['covariance_scope'] = ('conditional winner local Gaussian/IRLS curvature; excludes start selection, '
        'scale fitting, contact nonlinearity, nuisance bounds and trust-region uncertainty')
    return result
