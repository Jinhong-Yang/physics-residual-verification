"""Prospective bounded nonlinear development solver with an unchanged prior.

This is not a replacement for any locked experiment. Relinearization optimizes
the same original-prior objective; it never promotes a posterior to a new prior.
Robust weights, line-search branch choices and contact events make derivatives
piecewise. The final covariance is conditional local Gaussian/IRLS curvature,
not a posterior accounting for fitted weights, bounds or trust-region truncation.
"""
import torch

from reliability_update_v2 import robust_weights, weighted_gaussian_update


def slide(times, x0, v0, mu, inverse_mass, forces, switch):
    """Batched counterpart of main3d_numeric.slide, with B x T times."""
    def advance(x, v, acceleration, duration):
        stop = v / (-acceleration).clamp_min(1e-15)
        moving = torch.where(acceleration < 0, torch.minimum(duration, stop), duration)
        return x + v * moving + .5 * acceleration * moving.square(), (v + acceleration * moving).clamp_min(0)
    a0 = (forces[:, 0] * inverse_mass - mu * 9.81)[:, None]
    a1 = (forces[:, 1] * inverse_mass - mu * 9.81)[:, None]
    x, velocity = advance(x0[:, None], v0[:, None], a0, torch.minimum(times, switch[:, None]))
    return advance(x, velocity, a1, (times - switch[:, None]).clamp_min(0))[0]


def vertical(times, z0, v0, floor, restitution, gravity):
    """Batched counterpart of main3d_numeric.vertical, including 20-bounce cap."""
    if bool((gravity <= 0).any()):
        raise ValueError('Gravity must be positive')
    disc = (v0.square() + 2 * gravity * (z0 - floor).clamp_min(0)).clamp_min(0)
    impact_speed = torch.where(disc > 0, disc.clamp_min(torch.finfo(disc.dtype).tiny).sqrt(), torch.zeros_like(disc))
    hit = (v0 + impact_speed) / gravity
    before = times <= hit[:, None]
    remaining = times - hit[:, None]
    velocity = restitution * impact_speed
    out = torch.where(before, z0[:, None] + v0[:, None] * times - .5 * gravity[:, None] * times.square(), floor[:, None])
    active = ~before
    for _ in range(20):
        period = 2 * velocity / gravity
        inside = active & (remaining <= period[:, None])
        position = floor[:, None] + velocity[:, None] * remaining - .5 * gravity[:, None] * remaining.square()
        out = torch.where(inside, position, out)
        active = active & ~inside
        remaining = torch.where(active, remaining - period[:, None], remaining)
        velocity = velocity * restitution
        active = active & (velocity >= .01)[:, None]
    return out


def observation_residual(batch, theta, nuisance):
    """Raw noise-normalized residual; only permitted observation axis is used.

    Broad log guard [-30,30] prevents overflow, is never an ID support clamp and
    is reported by relinearize/refine. Inactive rows have exact zero residual.
    """
    bouncing = batch['active'][:, 2]
    meaningful = batch['active'].any(-1) & batch['valid'].any(-1)
    mass = theta[:, 0].clamp(-30, 30).exp()
    mu = theta[:, 1].clamp(-30, 30).exp()
    e = theta[:, 2].sigmoid()
    # The unused branch still runs; give it benign, physical nuisance values.
    gravity = torch.where(bouncing, nuisance[:, 3], torch.full_like(nuisance[:, 3], 9.81))
    moving = slide(batch['times'], nuisance[:, 0], nuisance[:, 1], mu, 1 / mass, batch['force'], batch['force_switch'])
    falling = vertical(batch['times'], nuisance[:, 0], nuisance[:, 1], nuisance[:, 2], e, gravity)
    axis = bouncing.long()[:, None, None].expand(-1, batch['times'].shape[1], 1)
    observed = batch['xz'].gather(2, axis).squeeze(-1)
    sigma = batch['sigma_xz'].gather(2, axis).squeeze(-1)
    if not bool((sigma > 0).all()):
        raise ValueError('Observation standard deviations must be positive')
    predicted = torch.where(bouncing[:, None], falling, moving)
    return torch.where(batch['valid'] & meaningful[:, None], (predicted - observed) / sigma, torch.zeros_like(predicted))


def _validate(batch, theta, nuisance):
    if theta.ndim != 2 or theta.shape[1] != 3 or nuisance.shape != (len(theta), 4):
        raise ValueError('Expected theta B x 3 and nuisance B x 4')
    if theta.dtype != torch.float64 or nuisance.dtype != torch.float64:
        raise ValueError('Nonlinear solver requires float64')
    b = len(theta)
    if batch['times'].ndim != 2 or batch['times'].shape[0] != b:
        raise ValueError('Expected B x T times')
    t = batch['times'].shape[1]
    for key, shape in [('prior_mean', (b, 3)), ('prior_scale', (b, 3)), ('times', (b, t)),
                       ('xz', (b, t, 2)), ('sigma_xz', (b, t, 2)), ('force', (b, 2)), ('force_switch', (b,))]:
        value = batch[key]
        if value.shape != shape or value.dtype != torch.float64 or value.device != theta.device or not bool(torch.isfinite(value).all()):
            raise ValueError('Invalid float64 observation input: ' + key)
    for key, shape in [('valid', (b, t)), ('active', (b, 3))]:
        if batch[key].shape != shape or batch[key].dtype != torch.bool or batch[key].device != theta.device:
            raise ValueError('Invalid structural mask: ' + key)
    if not bool(torch.isfinite(theta).all() & torch.isfinite(nuisance).all()) or not bool((batch['prior_scale'] > 0).all()):
        raise ValueError('Nonfinite iterate or nonpositive prior scale')
    if bool((batch['active'][:, 2] & batch['active'][:, :2].any(-1)).any()):
        raise ValueError('A row cannot mix slide and bounce protocols')


def relinearize(batch, theta, nuisance):
    """Central finite differences retain the computation graph.

    Physical increments are 1e-3 * original prior_scale; nuisance increments
    are 1e-4. Calls count every B-row observation evaluation actually executed,
    including unused structural columns in a mixed batch (15 evaluations/row).
    Each observation evaluation computes both slide and vertical branches;
    ``physical_kernel_calls`` exposes that additional execution explicitly.
    """
    _validate(batch, theta, nuisance)
    guard = torch.zeros(len(theta), dtype=torch.int64, device=theta.device)
    calls = 0
    def evaluate(point, q):
        nonlocal calls, guard
        calls += 1
        guard = guard + (point[:, :2].abs() > 30).any(-1).long()
        return observation_residual(batch, point, q)
    residual = evaluate(theta, nuisance)
    derivatives = []
    for k in range(3):
        direction = torch.zeros_like(theta)
        direction[:, k] = batch['prior_scale'][:, k] * 1e-3
        derivatives.append((evaluate(theta + direction, nuisance) - evaluate(theta - direction, nuisance)) / 2e-3)
    jacobian = torch.stack(derivatives, -1) * batch['active'][:, None, :]
    nuisance_active = torch.stack([batch['active'].any(-1), batch['active'].any(-1), batch['active'][:, 2], batch['active'][:, 2]], -1)
    derivatives = []
    for k in range(4):
        direction = torch.zeros_like(nuisance)
        direction[:, k] = 1e-4
        derivatives.append((evaluate(theta, nuisance + direction) - evaluate(theta, nuisance - direction)) / 2e-4)
    nuisance_jacobian = torch.stack(derivatives, -1) * nuisance_active[:, None, :]
    return {'residual': residual, 'J': jacobian, 'Jn': nuisance_jacobian,
            'linearization_mean': theta, 'nuisance': nuisance,
            'forward_calls_per_observation': calls, 'forward_calls': calls * len(theta),
            'physical_kernel_calls': 2 * calls * len(theta),
            'numerical_log_guard_evaluations_per_observation': guard}


def _solve(batch, weights):
    return weighted_gaussian_update(batch['prior_mean'], batch['prior_scale'], batch['J'], batch['residual'],
                                    batch['Jn'], weights, batch['valid'], batch['active'], batch['linearization_mean'])


def _weights(local, callback, kind, iteration):
    unit = _solve(local, local['valid'].double())
    if callback is not None:
        weights = callback(local, unit, iteration).to(dtype=torch.float64)
    elif kind == 'unit':
        weights = local['valid'].double()
    else:
        innovation = unit['projected_residual'] + (unit['projected_jacobian'] @ unit['whitened_update'][:, :, None]).squeeze(-1)
        weights = robust_weights(innovation, local['valid'], kind=kind)
    if weights.shape != local['valid'].shape or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError('Invalid reliability weights')
    return torch.where(local['valid'], weights, torch.zeros_like(weights))


def _nuisance_bounds(batch, proposed):
    bouncing = batch['active'][:, 2, None]
    lo_slide = proposed.new_tensor([-5., 0., 0., 0.])
    hi_slide = proposed.new_tensor([5., 10., 0., 0.])
    lo_bounce = proposed.new_tensor([-.5, -2., -.4, 5.])
    hi_bounce = proposed.new_tensor([2., 2., .5, 15.])
    low, high = torch.where(bouncing, lo_bounce, lo_slide), torch.where(bouncing, hi_bounce, hi_slide)
    return torch.maximum(low, torch.minimum(high, proposed))


def refine(batch, weights_callback=None, kind='unit', iterations=3, trust_radius=1.0):
    """Perform 1--3 bounded same-original-prior MAP/IRLS iterations.

    Callback API: ``weights_callback(local_batch, unit_result, iteration)``.
    The callback receives raw current J/Jn/residual and a unit-weight original-
    prior fit. Nonlearned Huber/Cauchy weights use that fit's innovation.
    Each iteration line-searches [1,.5,.25,0] under fixed current weights plus
    the original prior cost. Across iterations changing weights change the
    objective, so only within-iteration monotonicity is claimed.
    """
    if iterations not in [1, 2, 3] or not 0 < trust_radius < float('inf') or kind not in ['unit', 'huber', 'cauchy']:
        raise ValueError('Invalid bounded refinement controls')
    prior, scale = batch['prior_mean'], batch['prior_scale']
    active = batch['active']
    meaningful = active.any(-1) & batch['valid'].any(-1)
    theta = torch.where(active & meaningful[:, None], batch['linearization_mean'], prior)
    q = torch.where(meaningful[:, None], _nuisance_bounds(batch, batch['nuisance']), batch['nuisance'])
    _validate(batch, theta, q)
    calls = 0
    guard = torch.zeros(len(theta), dtype=torch.int64, device=theta.device)
    diagnostics = []
    current_weights = None
    def cost(point, residual, weights):
        return .5 * (((point - prior) / scale).square().sum(-1) + (weights * residual.square()).sum(-1))
    for iteration in range(iterations):
        lin = relinearize(batch, theta, q)
        calls += lin['forward_calls_per_observation']
        guard = guard + lin['numerical_log_guard_evaluations_per_observation']
        local = dict(batch, **{k: lin[k] for k in ['J', 'Jn', 'residual', 'linearization_mean', 'nuisance']})
        current_weights = _weights(local, weights_callback, kind, iteration)
        candidate_fit = _solve(local, current_weights)
        raw_step = (candidate_fit['mean'] - theta) / scale
        norm = torch.linalg.vector_norm(raw_step, dim=-1)
        contraction = (trust_radius / norm.clamp_min(1e-30)).clamp_max(1)
        no_weighted_evidence = ~(current_weights > 0).any(-1)
        # Exact prior fallback is a closed observation case, not a trust-limited
        # evidence update. It can exceed the trust radius when initialization was
        # support projected; report that exception explicitly below.
        contraction = torch.where(no_weighted_evidence, torch.ones_like(contraction), contraction)
        step = raw_step * contraction[:, None] * active
        proposal = theta + scale * step
        root = torch.where(current_weights > 0, current_weights.clamp_min(torch.finfo(torch.float64).tiny).sqrt(), torch.zeros_like(current_weights))
        residual_after_physical = lin['residual'] + (lin['J'] @ step[:, :, None]).squeeze(-1)
        weighted_n = root[:, :, None] * lin['Jn']
        dq = -(torch.linalg.pinv(weighted_n, atol=0., rtol=1e-12) @ (root * residual_after_physical)[:, :, None]).squeeze(-1)
        proposed_q = _nuisance_bounds(batch, q + dq)
        before = cost(theta, lin['residual'], current_weights)
        best_cost, best_theta, best_q = before, theta, q
        accepted = torch.zeros(len(theta), dtype=theta.dtype, device=theta.device)
        for fraction in [1., .5, .25]:
            trial_theta = theta + fraction * (proposal - theta)
            trial_q = q + fraction * (proposed_q - q)
            trial_r = observation_residual(batch, trial_theta, trial_q)
            calls += 1
            guard = guard + (trial_theta[:, :2].abs() > 30).any(-1).long()
            trial_cost = cost(trial_theta, trial_r, current_weights)
            better = torch.isfinite(trial_cost) & (trial_cost < best_cost) & meaningful
            best_cost = torch.where(better, trial_cost, best_cost)
            best_theta = torch.where(better[:, None], trial_theta, best_theta)
            best_q = torch.where(better[:, None], trial_q, best_q)
            accepted = torch.where(better, torch.full_like(accepted, fraction), accepted)
        best_theta = torch.where(no_weighted_evidence[:, None], prior, best_theta)
        best_cost = torch.where(no_weighted_evidence, torch.zeros_like(best_cost), best_cost)
        diagnostics.append({'iteration': iteration, 'cost_before': before, 'cost_after': best_cost,
                            'accepted_fraction': accepted, 'proposed_whitened_step_norm': norm,
                            'trust_region_applied': (norm > trust_radius) & ~no_weighted_evidence,
                            'exact_prior_fallback': no_weighted_evidence,
                            'accepted_whitened_step_norm': torch.linalg.vector_norm((best_theta - theta) / scale, dim=-1),
                            'nuisance_bounds_applied': ((q + dq) != proposed_q).any(-1)})
        theta, q = best_theta, best_q
    # Final curvature at returned iterate under the final iteration's weights.
    # This does NOT move theta to the untruncated local Gaussian mean.
    final = relinearize(batch, theta, q)
    calls += final['forward_calls_per_observation']
    guard = guard + final['numerical_log_guard_evaluations_per_observation']
    final_local = dict(batch, **{k: final[k] for k in ['J', 'Jn', 'residual', 'linearization_mean', 'nuisance']})
    approximation = _solve(final_local, current_weights)
    theta = torch.where(meaningful[:, None], theta, prior)
    return {'mean': theta, 'covariance': approximation['covariance'], 'nuisance': q,
            'whitened_update': (theta - prior) / scale, 'weights': current_weights,
            'residual': final['residual'], 'J': final['J'], 'Jn': final['Jn'],
            'local_gaussian_mean': approximation['mean'],
            'covariance_scope': 'conditional_final_linear_Gaussian_IRLS_approximation; excludes weight uncertainty, nuisance bounds, contact nonlinearity and trust/line-search truncation; local_gaussian_mean can differ from returned mean',
            'original_prior_reused': True, 'iterations': iterations, 'diagnostics': diagnostics,
            'forward_calls_per_observation': calls, 'forward_calls': calls * len(theta),
            'physical_kernel_calls': 2 * calls * len(theta),
            'numerical_log_guard_evaluations_per_observation': guard,
            'final_numerical_log_guard': (theta[:, :2].abs() > 30).any(-1)}
