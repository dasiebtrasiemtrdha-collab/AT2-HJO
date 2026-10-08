"""Float32 actor proposals must not create false physical capacity failures."""
from pathlib import Path

import numpy as np
import torch

from at2hjo.revision4.config import load
from at2hjo.revision4.environment import Environment
from at2hjo.revision4.learning import Learner
from at2hjo.revision4.runner import episode

ROOT=Path(__file__).resolve().parents[1]


def test_float32_actor_proposal_uses_double_precision_capacity_accounting():
    c=load(ROOT/'configs/revision4/smoke.json');e=Environment(c,2000);p=e.p
    raw=np.linspace(-.01,.01,p.action_dim,dtype=np.float32)
    idle=p.allocate(raw,np.zeros((p.m,p.n)),e.snapshot())
    oracle=p.allocate(raw.astype(np.float64),np.zeros((p.m,p.n)),e.snapshot())
    for branch in ('cpu','optical','rf'):
        assert idle[branch].dtype==np.float64
        np.testing.assert_array_equal(idle[branch],oracle[branch])
    assert np.all(idle['optical'].sum(0)<=1+1e-12)
    assert np.all(idle['rf'].sum(0)<=1+1e-12)
    assert p.evaluate(idle,e.snapshot())['safe']
    # Safety checks still reject real capacity violations.
    idle['optical'][:,0]*=1.001
    assert not p.evaluate(idle,e.snapshot())['safe']


def test_actor_controlled_certified_episode_reaches_full_horizon(tmp_path):
    torch.set_num_threads(1)
    c=load(ROOT/'configs/revision4/smoke.json')
    c.update(hidden=[8,8],attention=4,value_hidden=4,warmup=2,batch_st=2)
    learner=Learner(Environment(c,2000),'cpu',2000)
    first=episode(c,2000,0,learner,True,tmp_path)
    assert first['slots']==17 and first['operating_failures']==0
    assert learner.counters['actor_updates']>0
    second=episode(c,2000,1,learner,True,tmp_path)
    assert second['slots']==17 and second['operating_failures']==0
    assert learner.counters['environment_steps']==34
    assert first['node_temperature_violations']==second['node_temperature_violations']==0
