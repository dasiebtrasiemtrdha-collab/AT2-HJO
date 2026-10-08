from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from at2hjo.revision4.config import load
from at2hjo.revision4.environment import Environment
from at2hjo.revision4.learning import Learner
from at2hjo.revision4.runner import episode,fingerprint,generate

ROOT=Path(__file__).resolve().parents[1]

def test_exact_episode_boundary_recovery(tmp_path):
    torch.set_num_threads(1);c=load(ROOT/'configs/revision4/smoke.json');e=Environment(c,2000);a=Learner(e,'cpu',2000)
    first=tmp_path/'first';first.mkdir();episode(c,2000,0,a,True,first)
    state=a.state(True);torch.save(state,tmp_path/'resume.pt')
    restored=Learner(Environment(c,2000),'cpu',2000);restored.restore(torch.load(tmp_path/'resume.pt',weights_only=False),resume=True)
    assert fingerprint(a)==fingerprint(restored)
    for name in ['direct','continued']:(tmp_path/name).mkdir()
    x=episode(c,2000,1,a,True,tmp_path/'direct');y=episode(c,2000,1,restored,True,tmp_path/'continued')
    assert x==y and a.counters==restored.counters and fingerprint(a)==fingerprint(restored)

def test_generated_bytes_can_drive_identical_rollout(tmp_path):
    torch.set_num_threads(1);p=ROOT/'configs/revision4/stochastic_smoke.json';c=load(p)
    generate(SimpleNamespace(config=str(p),seed=4000,output=str(tmp_path/'data')))
    for name in ['generated','recorded']:(tmp_path/name).mkdir()
    a=Learner(Environment(c,4000),'cpu',2000);b=Learner(Environment(c,4000),'cpu',2000)
    x=episode(c,4000,0,a,False,tmp_path/'generated');y=episode(c,4000,0,b,False,tmp_path/'recorded',tmp_path/'data')
    assert x==y
    from at2hjo.revision4.dataset import RecordedTrace
    other=dict(c);other['algorithm']='st'
    trace=RecordedTrace(tmp_path/'data',other,4000)
    np.testing.assert_array_equal(trace.links,Environment(c,4000).trace.links)
