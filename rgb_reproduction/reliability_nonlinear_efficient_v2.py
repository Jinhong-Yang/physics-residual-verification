"""Structural-column efficiency variant; original frozen nonlinear code is intact.

Numerical algebra, finite-difference steps, optimizer objective, line search and
trust region are copied from the reference function by an asserted AST transform
that changes only its iteration-count guard. The finite-difference implementation
below skips physically structural-zero columns, preserving padded output shape.
Protocol groups run separately; closed rows return the exact prior at zero cost.
"""
import ast
import inspect

import torch

import reliability_nonlinear_v2 as reference


def relinearize(batch, theta, nuisance):
    reference._validate(batch, theta, nuisance)
    active = batch['active']
    if not torch.equal(active, active[:1].expand_as(active)):
        raise ValueError('Efficient core requires homogeneous structural protocol groups')
    columns = torch.nonzero(active[0], as_tuple=False).flatten().tolist()
    nuisance_columns = list(range(4 if active[0, 2] else 2)) if columns else []
    calls = 0
    guard = torch.zeros(len(theta), dtype=torch.int64, device=theta.device)
    def evaluate(point, q):
        nonlocal calls, guard
        calls += 1
        guard = guard + (point[:, :2].abs() > 30).any(-1).long()
        return reference.observation_residual(batch, point, q)
    residual = evaluate(theta, nuisance)
    physical = []
    for k in range(3):
        if k in columns:
            direction = torch.zeros_like(theta)
            direction[:, k] = batch['prior_scale'][:, k] * 1e-3
            physical.append((evaluate(theta+direction, nuisance)-evaluate(theta-direction, nuisance))/2e-3)
        else:
            physical.append(torch.zeros_like(residual))
    nuisance_derivatives = []
    for k in range(4):
        if k in nuisance_columns:
            direction = torch.zeros_like(nuisance); direction[:, k] = 1e-4
            nuisance_derivatives.append((evaluate(theta,nuisance+direction)-evaluate(theta,nuisance-direction))/2e-4)
        else:
            nuisance_derivatives.append(torch.zeros_like(residual))
    return {'residual':residual, 'J':torch.stack(physical,-1), 'Jn':torch.stack(nuisance_derivatives,-1),
            'linearization_mean':theta, 'nuisance':nuisance,
            'forward_calls_per_observation':calls, 'forward_calls':calls*len(theta),
            'physical_kernel_calls':2*calls*len(theta), 'numerical_log_guard_evaluations_per_observation':guard}


def _build_same_algebra_core():
    tree = ast.parse(inspect.getsource(reference.refine))
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id=='iterations':
            if len(node.ops)==1 and isinstance(node.ops[0],ast.NotIn) and len(node.comparators)==1:
                old=node.comparators[0]
                if isinstance(old,ast.List) and [item.value for item in old.elts]==[1,2,3]:
                    old.elts=[ast.Constant(value) for value in range(1,9)];count+=1
    if count!=1:
        raise RuntimeError('Reference refine no longer has the declared single iteration guard')
    namespace=dict(vars(reference));namespace['relinearize']=relinearize
    exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),namespace)
    return namespace['refine']


_same_algebra_core=_build_same_algebra_core()


def refine(batch, weights_callback=None, kind='unit', iterations=3, trust_radius=1.):
    if iterations not in range(1,9):
        raise ValueError('Efficient development variant allows 1--8 iterations only')
    count=len(batch['prior_mean']);device=batch['prior_mean'].device
    if count<1:
        raise ValueError('Empty batch')
    meaningful=batch['active'].any(-1)&batch['valid'].any(-1)
    code=(batch['active'].long()*torch.tensor([1,2,4],device=device)).sum(-1)
    # Allocate exact prior outputs first; active rows are filled by index_copy.
    mean=batch['prior_mean'];cov=torch.diag_embed(batch['prior_scale'].square())
    result={'mean':mean,'covariance':cov,'nuisance':batch['nuisance'],
            'whitened_update':torch.zeros_like(mean),'weights':torch.zeros_like(batch['times']),
            'residual':torch.zeros_like(batch['times']),
            'J':torch.zeros((*batch['times'].shape,3),dtype=mean.dtype,device=device),
            'Jn':torch.zeros((*batch['times'].shape,4),dtype=mean.dtype,device=device),
            'local_gaussian_mean':mean,
            'numerical_log_guard_evaluations_per_observation':torch.zeros(count,dtype=torch.int64,device=device),
            'final_numerical_log_guard':torch.zeros(count,dtype=torch.bool,device=device)}
    row_calls=torch.zeros(count,dtype=torch.int64,device=device)
    diagnostics=[{} for _ in range(iterations)]
    covariance_scope='conditional_final_linear_Gaussian_IRLS_approximation; excludes weight uncertainty, nuisance bounds, contact nonlinearity and trust/line-search truncation; local_gaussian_mean can differ from returned mean'
    for value in [1,2,3,4]:
        indices=torch.nonzero((code==value)&meaningful,as_tuple=False).flatten()
        if not len(indices):continue
        local={key:tensor[indices] for key,tensor in batch.items()}
        group=_same_algebra_core(local,weights_callback,kind,iterations,trust_radius)
        for key in result:
            result[key]=result[key].index_copy(0,indices,group[key])
        row_calls[indices]=group['forward_calls_per_observation']
        covariance_scope=group['covariance_scope']
        for iteration,trace in enumerate(group['diagnostics']):
            for key,item in trace.items():
                if key=='iteration':continue
                if key not in diagnostics[iteration]:
                    diagnostics[iteration][key]=torch.zeros(count,dtype=item.dtype,device=device)
                diagnostics[iteration][key]=diagnostics[iteration][key].index_copy(0,indices,item)
    if bool((meaningful&~torch.isin(code,torch.tensor([1,2,3,4],device=device))).any()):
        raise ValueError('Unsupported mixed physical protocol')
    for i,trace in enumerate(diagnostics):
        trace['iteration']=i
        if 'exact_prior_fallback' in trace:
            trace['exact_prior_fallback']=trace['exact_prior_fallback']|~meaningful
    result.update(diagnostics=diagnostics,iterations=iterations,original_prior_reused=True,
                  forward_calls_per_observation=row_calls,forward_calls=int(row_calls.sum()),
                  physical_kernel_calls=2*int(row_calls.sum()),covariance_scope=covariance_scope,
                  efficiency_scope='protocol-separated structural-zero finite differences omitted; closed rows exact prior with zero evaluations')
    return result
