"""Output-schema correction only; frozen V1 optimizer runs without modification."""
import hashlib
from pathlib import Path
import reliability_setup_bounce_v7 as v1
H,G=v1.H,v1.G
V1_SHA='858e31531f4af3a282bc4711f99e28ab0a89de27280700d6d5a1c42bee5b4476'


def verify_frozen_solver():
    assert hashlib.sha256(Path(v1.__file__).read_bytes()).hexdigest()==V1_SHA


def fit_setup_bounce(*args,**kwargs):
    verify_frozen_solver()
    result=v1.fit_setup_bounce(*args,**kwargs)
    result['unit_reliability']=result['weights']
    result['weights']=result['alpha'][:,None]*result['unit_reliability']
    return result
