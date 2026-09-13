"""Fresh RGB extraction and same initial nuisance procedure, V7 kernel only.

Isolated function copies replace the bounce kernel at every initializer layer.
The old initializer's documented operational-support seed projection remains;
original prior mean/scale are never projected or replaced. M95 subsequently
uses its existing same-prior starts. No old observation/J/nuisance arrays load.
"""
import ast
import numpy as np
import torch
import projected_observation as projected
import numerical_observation_likelihood as likelihood
import reliability_data_v2 as data
from main3d_numeric import slide
from main3d_evaluation import natural
from main3d_refinement import tracking
from reliability_nonlinear_fast_v2 import _copy_function
from reliability_canonical_v7 import vertical


def numpy_vertical(times,z0,v0,floor,e,gravity):
    t=torch.as_tensor(np.asarray(times)[None],dtype=torch.float64)
    values=[torch.tensor([v],dtype=torch.float64) for v in [z0,v0,floor,e,gravity]]
    return vertical(t,*values)[0].numpy()


def _fixed_gravity(tree,initialize=False):
    count=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name) and node.value.id in ['q','nuisance'] and isinstance(node.slice,ast.Constant) and node.slice.value==3:
            # Replace the complete subscript through a transformer below.
            count+=1
    assert count==1
    class Rewrite(ast.NodeTransformer):
        def visit_Subscript(self,node):
            if isinstance(node.value,ast.Name) and node.value.id in ['q','nuisance'] and isinstance(node.slice,ast.Constant) and node.slice.value==3:
                return ast.Constant(9.81)
            return self.generic_visit(node)
    tree=Rewrite().visit(tree)
    if initialize:
        matches=[n for n in tree.body[0].body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='bounce']
        assert len(matches)==1
        arrays=[n for n in matches[0].body if isinstance(n,ast.Assign)]
        assert len(arrays)==3
        for n in arrays:
            assert isinstance(n.value,ast.Call) and isinstance(n.value.args[0],ast.List) and len(n.value.args[0].elts)==4
            n.value.args[0].elts.pop()
    return tree


local_jacobian=_copy_function(projected.local_jacobian,{'vertical':numpy_vertical},lambda tree:_fixed_gravity(tree,True))
observation_likelihood=_copy_function(likelihood.observation_likelihood,
    {'vertical':numpy_vertical,'local_jacobian':local_jacobian},_fixed_gravity)


def _remove_local_imports(tree):
    body=tree.body[0].body;removed=[n for n in body if isinstance(n,ast.ImportFrom)]
    assert [n.module for n in removed]==['main3d_refinement','numerical_observation_likelihood','main3d_evaluation','main3d_numeric']
    tree.body[0].body=[n for n in body if not isinstance(n,ast.ImportFrom)]
    return _fixed_gravity(tree)


_extract_observation=_copy_function(data.extract_observation,dict(tracking=tracking,
    observation_likelihood=observation_likelihood,natural=natural,slide=slide,vertical=numpy_vertical),_remove_local_imports)


def extract_observation(*args,**kwargs):
    result=_extract_observation(*args,**kwargs)
    if str(result['protocol'])=='bounce':
        result['nuisance'][3]=9.81
        if bool(result['likelihood_available']):assert result['nuisance_dim']==3
        assert np.all(result['Jn'][:,3]==0)
    return result
