from pathlib import Path
from copy import deepcopy
import hashlib
import json

from . import SCIENCE

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def input_identity(c):
    return digest({k:c[k] for k in ('horizon','nodes','buoys','satellites','services','slot_s','queue_capacity','trace','physical')})

def load(path):
    path=Path(path).resolve()
    c=json.loads(path.read_text(encoding='utf-8-sig'))
    if 'physical_file' in c:
        c['physical']=json.loads((path.parent/c.pop('physical_file')).read_text(encoding='utf-8-sig'))
    if c['science'] != SCIENCE: raise ValueError('incompatible scientific identity')
    assert c['mode'] in ('certified','resource_only')
    assert c['trace'] in ('regular','stochastic_open')
    assert c['horizon']>0 and c['nodes']>0
    assert c['gamma']==c['gamma_st'] and 0<c['gamma']<=1
    assert c['lt_intervals']==[5,10,15]
    return c

def stream(seed,episode,name):
    import numpy as np
    key=hashlib.sha256(f'{seed}/{episode}/{name}'.encode()).digest()
    return np.random.default_rng(np.random.SeedSequence([int.from_bytes(key[i:i+4],'little') for i in range(0,32,4)]))
