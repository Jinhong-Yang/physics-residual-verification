"""Algebra-preserving implementation experiment, separate from frozen solvers.

Homogeneous groups evaluate only the observation branch selected by their mask.
Candidate Gaussian solves omit unused inverse covariance. Unit-fit covariance
is retained by default for callback compatibility; ``omit_unit_covariance=True``
is an explicit narrower callback contract (covariance key absent), suitable for
make_features and the constant-alpha M callback. Final covariance is unchanged.
No tensor is detached, no rank tolerance changes, and no prior is updated.

Asserted AST copies preserve frozen step/line-search expressions. Source files
are dependencies, never rewritten and no shared module globals are patched.
"""
import ast
import inspect
from types import SimpleNamespace
import torch
import reliability_nonlinear_v2 as reference
import reliability_nonlinear_efficient_v2 as efficient
import reliability_contact_multistart_v2 as multistart
import reliability_update_v2 as update


def observation_residual(batch, theta, nuisance):
    bouncing = batch['active'][:, 2]
    if not torch.equal(bouncing, bouncing[:1].expand_as(bouncing)):
        raise ValueError('Fast residual requires a homogeneous observation branch')
    meaningful = batch['active'].any(-1) & batch['valid'].any(-1)
    if bool(bouncing[0]):
        predicted = reference.vertical(batch['times'], nuisance[:, 0], nuisance[:, 1],
                                       nuisance[:, 2], theta[:, 2].sigmoid(), nuisance[:, 3])
    else:
        mass = theta[:, 0].clamp(-30, 30).exp()
        mu = theta[:, 1].clamp(-30, 30).exp()
        predicted = reference.slide(batch['times'], nuisance[:, 0], nuisance[:, 1],
                                    mu, 1 / mass, batch['force'], batch['force_switch'])
    axis = bouncing.long()[:, None, None].expand(-1, batch['times'].shape[1], 1)
    observed = batch['xz'].gather(2, axis).squeeze(-1)
    sigma = batch['sigma_xz'].gather(2, axis).squeeze(-1)
    if not bool((sigma > 0).all()):
        raise ValueError('Observation standard deviations must be positive')
    return torch.where(batch['valid'] & meaningful[:, None], (predicted-observed)/sigma,
                       torch.zeros_like(predicted))


def _copy_function(function, namespace, transform=None):
    tree = ast.parse(inspect.getsource(function))
    if transform is not None:
        tree = transform(tree)
    environment = dict(function.__globals__)
    environment.update(namespace)
    exec(compile(ast.fix_missing_locations(tree), __file__, 'exec'), environment)
    return environment[function.__name__]


def _without_covariance(tree):
    removed = 0
    body = tree.body[0].body
    kept = []
    for statement in body:
        if (isinstance(statement, ast.Assign) and len(statement.targets)==1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id in ['white_covariance', 'covariance']):
            removed += 1
            continue
        if isinstance(statement, ast.Return):
            result = statement.value
            assert isinstance(result, ast.Dict)
            indices = [i for i,k in enumerate(result.keys) if isinstance(k,ast.Constant) and k.value=='covariance']
            assert len(indices)==1
            index=indices[0];result.keys.pop(index);result.values.pop(index)
        kept.append(statement)
    if removed != 3:
        raise RuntimeError('Frozen Gaussian covariance expressions changed')
    tree.body[0].body = kept
    return tree


_update_no_covariance = _copy_function(update.weighted_gaussian_update, {}, _without_covariance)
_solve_no_covariance = _copy_function(reference._solve, {'weighted_gaussian_update':_update_no_covariance})
_weights_no_covariance = _copy_function(reference._weights, {'_solve':_solve_no_covariance})
_reference_branch = SimpleNamespace(**dict(vars(reference), observation_residual=observation_residual))
_relinearize_core = _copy_function(efficient.relinearize, {'reference':_reference_branch})


def relinearize(batch, theta, nuisance):
    result = _relinearize_core(batch, theta, nuisance)
    result['physical_kernel_calls'] = result['forward_calls']
    return result


def _candidate_no_covariance(tree):
    candidates = guards = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='candidate_fit' for t in node.targets):
            assert isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='_solve'
            node.value.func.id='_solve_no_covariance';candidates+=1
        if (isinstance(node,ast.Compare) and isinstance(node.left,ast.Name) and node.left.id=='iterations'
                and len(node.ops)==1 and isinstance(node.ops[0],ast.NotIn)):
            old=node.comparators[0]
            if isinstance(old,ast.List) and [n.value for n in old.elts]==[1,2,3]:
                old.elts=[ast.Constant(i) for i in range(1,9)];guards+=1
    if (candidates,guards)!=(1,1):
        raise RuntimeError('Frozen reference step/iteration guard changed')
    return tree


def _make_grouped(omit_unit_covariance):
    core = _copy_function(reference.refine, {'relinearize':relinearize,
        'observation_residual':observation_residual, '_solve_no_covariance':_solve_no_covariance,
        '_weights':_weights_no_covariance if omit_unit_covariance else reference._weights}, _candidate_no_covariance)
    return _copy_function(efficient.refine, {'_same_algebra_core':core})


_grouped_full_unit = _make_grouped(False)
_grouped_no_unit_covariance = _make_grouped(True)


def refine(batch, weights_callback=None, kind='unit', iterations=3, trust_radius=1., *, omit_unit_covariance=False):
    solver = _grouped_no_unit_covariance if omit_unit_covariance else _grouped_full_unit
    result = solver(batch, weights_callback, kind, iterations, trust_radius)
    result['physical_kernel_calls'] = result['forward_calls']
    result['efficiency_scope'] = ('structural differences and branch-specialized residuals; candidate covariance omitted; '
                                  'unit covariance omitted' if omit_unit_covariance else
                                  'structural differences and branch-specialized residuals; candidate covariance omitted; unit covariance retained')
    return result


def _refine_no_unit_covariance(*args, **kwargs):
    return refine(*args, **kwargs, omit_unit_covariance=True)


_efficient_branch = SimpleNamespace(relinearize=relinearize, refine=_refine_no_unit_covariance)
_reference_step = SimpleNamespace(**dict(vars(reference), observation_residual=observation_residual, _solve=_solve_no_covariance))
one_step = _copy_function(multistart.one_step, {'reference':_reference_step, 'efficient':_efficient_branch})
_beam_core = _copy_function(multistart.bounce_beam, {'one_step':one_step, 'efficient':_efficient_branch})


def bounce_beam(batch, trust_radius=1.):
    result = _beam_core(batch, trust_radius)
    result['physical_kernel_calls'] = result['forward_calls']
    return result


_multistart_core = _copy_function(multistart.refine, {'bounce_beam':bounce_beam, 'efficient':_efficient_branch})


def multistart_refine(batch, trust_radius=1.):
    """Same M optimizer with exactly the same implementation optimizations."""
    result = _multistart_core(batch, trust_radius)
    result['physical_kernel_calls'] = result['forward_calls']
    return result
