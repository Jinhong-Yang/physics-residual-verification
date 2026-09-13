"""Development-only features and analytic actions for reliability V2.

No label is used by make_features. Action supervision is a separate function.
These actions are analytic surrogates, not a new independent-engine result.
"""
import math
import numpy as np
import torch


def slog(value):
    return value.sign() * torch.log1p(value.abs())


def make_features(batch, unit_result):
    r, j, n, valid = batch['residual'], batch['J'], batch['Jn'], batch['valid']
    time = batch['times']
    count = valid.sum(-1).clamp_min(1)
    innovation = unit_result['projected_residual'] + torch.einsum(
        'bti,bi->bt', unit_result['projected_jacobian'], unit_result['whitened_update'])
    gradient = torch.einsum('bti,bt->bi', j, r)
    information = j.transpose(-1, -2) @ j
    indices = torch.triu_indices(3, 3, device=j.device)
    global_values = torch.cat([
        batch['prior_mean'], batch['prior_scale'].log(),
        (batch['linearization_mean'] - batch['prior_mean']) / batch['prior_scale'],
        slog(gradient), slog(information[:, indices[0], indices[1]]),
        torch.sqrt((innovation.square() * valid).sum(-1) / count)[:, None].log1p(),
        batch['force'], batch['force_switch'][:, None], batch['active'].to(j.dtype)], -1)
    delta_t = torch.cat([torch.zeros_like(time[:, :1]), time[:, 1:] - time[:, :-1]], -1)
    local_values = torch.cat([
        slog(r)[..., None], slog(innovation)[..., None], slog(j), slog(n),
        time[..., None], delta_t[..., None],
        (time - batch['force_switch'][:, None])[..., None],
        (time >= batch['force_switch'][:, None]).to(j.dtype)[..., None],
        batch['xz'], batch['sigma_xz'].clamp_min(1e-12).log(),
        global_values[:, None].expand(-1, time.shape[1], -1)], -1)
    features = torch.where(valid[..., None], local_values, torch.zeros_like(local_values))
    if not torch.isfinite(features).all():
        raise ValueError('Nonfinite observation-only features')
    return features.float(), innovation


def torch_slide(times, mass, mu, forces, switch, velocity):
    def advance(x, v, acceleration, duration):
        stop = v / (-acceleration).clamp_min(1e-12)
        moving = torch.where(acceleration < 0, torch.minimum(duration, stop), duration)
        return x + v * moving + .5 * acceleration * moving.square(), (v + acceleration * moving).clamp_min(0)
    m = mass[:, None]; friction = mu[:, None]
    a0, a1 = forces[0] / m - friction * 9.81, forces[1] / m - friction * 9.81
    before = times.clamp_max(switch)[None]
    after = (times - switch).clamp_min(0)[None]
    x, v = advance(torch.zeros_like(m), torch.full_like(m, velocity), a0, before)
    return advance(x, v, a1, after)[0]


def torch_bounce(times, restitution, height=.85):
    gravity = 9.81
    impact = math.sqrt(2 * height / gravity)
    t = times[None].expand(len(restitution), -1)
    remaining = (t - impact).clamp_min(0)
    velocity = restitution[:, None] * math.sqrt(2 * gravity * height)
    result = torch.where(t <= impact, height - .5 * gravity * t.square(), torch.zeros_like(t))
    active = t > impact
    for _ in range(20):
        period = 2 * velocity / gravity
        inside = active & (remaining <= period)
        position = velocity * remaining - .5 * gravity * remaining.square()
        result = torch.where(inside, position, result)
        active = active & ~inside
        remaining = torch.where(active, remaining - period, remaining)
        velocity = velocity * restitution[:, None]
        active = active & (velocity >= .01)
    return result


def analytic_actions(mean, active, training=False):
    # Broad numerical guard only; any activated guard is reported by scoring.
    mass, mu = mean[:, 0].clamp(-30, 30).exp(), mean[:, 1].clamp(-30, 30).exp()
    restitution = mean[:, 2].sigmoid()
    times = torch.linspace(.125, 1., 8, dtype=mean.dtype, device=mean.device)
    if training:
        forced = torch_slide(times, mass, mu, (.6, .9), .4, .3)
        free = torch_slide(times, mass, mu, (0., 0.), .4, .8)
        bounce = torch_bounce(times, restitution, .65)
    else:
        forced = torch_slide(times, mass, mu, (.45, 1.1), .35, .25)
        free = torch_slide(times, mass, mu, (0., 0.), .35, 1.)
        bounce = torch_bounce(times, restitution, .85)
    return torch.where(active[:, 2, None], bounce,
                       torch.where(active[:, 0, None], forced, free))


def existing_rollout_loss(mean, target):
    # Same original train_pilot analytic surrogate to isolate the update change.
    def response(value):
        mass = value[:, 0].clamp(-5, 3).exp()
        mu = value[:, 1].clamp(-6, 2).exp()
        e = value[:, 2].sigmoid()
        return torch.stack([(.75 / mass - mu * 9.81) / 10,
                            (1.25 / mass - mu * 9.81) / 10, e.square()], -1)
    return (response(mean) - response(target)).square().mean()


def gaussian_scores(mean, covariance, target):
    error = target - mean
    chol = torch.linalg.cholesky(covariance)
    standardized = torch.linalg.solve_triangular(chol, error[..., None], upper=False).squeeze(-1)
    nll = (.5 * standardized.square().sum(-1) + chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
           + 1.5 * math.log(2 * math.pi)) / 3
    marginal_scale = covariance.diagonal(dim1=-2, dim2=-1).sqrt()
    z = error / marginal_scale
    phi = torch.exp(-.5 * z.square()) / math.sqrt(2 * math.pi)
    cdf = .5 * (1 + torch.erf(z / math.sqrt(2)))
    crps = (marginal_scale * (z * (2 * cdf - 1) + 2 * phi - 1 / math.sqrt(math.pi))).mean(-1)
    # chi-square(3) 90th percentile, fixed before evaluation.
    joint90 = standardized.square().sum(-1) <= 6.251388631170325
    return {'nll': nll, 'crps': crps, 'joint90': joint90,
            'marginal90': z.abs() <= 1.6448536269514722,
            'width90': 2 * 1.6448536269514722 * marginal_scale}


def corrupt_residual(batch, seed):
    """Synthetic post-tracking measurement stress, not image corruption.

    Fixed linearization/J/nuisance are shared. No corruption indicator goes to
    the model; observation xz and residual both reflect the added measurement.
    """
    device = batch['residual'].device
    generator = torch.Generator(device=device).manual_seed(seed)
    count, frames = batch['residual'].shape
    noise = torch.randn(count, frames, dtype=batch['residual'].dtype,
                        device=device, generator=generator)
    where = torch.randint(frames, (count,), generator=generator, device=device)
    signs = torch.randint(2, (count,), generator=generator, device=device) * 2 - 1
    noise[torch.arange(count, device=device), where] += signs * 8.
    noise = noise * batch['valid']
    out = dict(batch)
    out['residual'] = batch['residual'] - noise
    out['xz'] = batch['xz'].clone()
    axis = batch['active'][:, 2].long()
    index = torch.arange(count, device=device)
    out['xz'][index, :, axis] += noise * batch['sigma_xz'][index, :, axis]
    return out


def load_inputs(directory, device='cpu'):
    with np.load(directory / 'inputs.npz', allow_pickle=False) as payload:
        metadata = {key: payload[key].copy() for key in ['ids', 'family_ids', 'split', 'protocol']}
        keys = ['prior_mean', 'prior_scale', 'linearization_mean', 'J', 'residual', 'Jn', 'times',
                'xz', 'sigma_xz', 'force', 'force_switch', 'valid', 'active']
        arrays = {key: torch.from_numpy(payload[key].copy()).to(device=device,
                  dtype=torch.bool if key in ['valid', 'active'] else torch.float64) for key in keys}
    with np.load(directory / 'labels.npz', allow_pickle=False) as payload:
        if not np.array_equal(payload['ids'], metadata['ids']):
            raise ValueError('Input / target alignment mismatch')
        target = torch.from_numpy(payload['target_mean'].copy()).to(device=device, dtype=torch.float64)
    return arrays, metadata, target


def subset(batch, indices):
    return {key: value[indices] for key, value in batch.items()}


def solve(batch, weights):
    from reliability_update_v2 import weighted_gaussian_update
    return weighted_gaussian_update(batch['prior_mean'], batch['prior_scale'], batch['J'],
        batch['residual'], batch['Jn'], weights.to(batch['J'].dtype), batch['valid'],
        batch['active'], batch['linearization_mean'])


def score_result(result, batch, target, families):
    with torch.no_grad():
        scores = gaussian_scores(result['mean'], result['covariance'], target)
        error = (analytic_actions(result['mean'], batch['active']) -
                 analytic_actions(target, batch['active'])).abs().mean(-1)
        active_rows = batch['active'].any(-1)
        unique = sorted(set(families))
        family_errors = {}
        for family in unique:
            mask = torch.as_tensor(families == family, device=error.device) & active_rows
            if mask.any():
                family_errors[str(family)] = float(error[mask].mean())
        selected = active_rows
        marginal = result['covariance'].diagonal(dim1=-2, dim2=-1).sqrt()
        return {'observations': len(target), 'active_observations': int(selected.sum()),
            'independent_families': len(unique), 'analytic_action_mae_m': float(np.mean(list(family_errors.values()))),
            'family_analytic_action_mae_m': family_errors,
            'active_nll_per_dimension': float(scores['nll'][selected].mean()),
            'active_crps': float(scores['crps'][selected].mean()),
            'active_joint90_coverage': float(scores['joint90'][selected].double().mean()),
            'active_marginal90_coverage': scores['marginal90'][selected].double().mean(0).tolist(),
            'active_marginal90_width': scores['width90'][selected].mean(0).tolist(),
            'all_nll_per_dimension': float(scores['nll'].mean()),
            'active_transformed_mae': (result['mean'][selected] - target[selected]).abs().mean(0).tolist(),
            'action_numerical_guard_activated_rows': int((result['mean'][:, :2].abs() > 30).any(-1).sum()),
            'all_finite': bool(torch.isfinite(result['mean']).all() and torch.isfinite(marginal).all()),
            'scope': 'development_fixed_linearization_analytic_actions_not_engine_validation'}
