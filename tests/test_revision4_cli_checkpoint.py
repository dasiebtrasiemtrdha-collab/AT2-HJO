from pathlib import Path
import json
import torch
from at2hjo.revision4.config import load
from at2hjo.revision4.environment import Environment
from at2hjo.revision4.learning import Learner
from at2hjo.revision4.runner import checkpoint


def test_json_receipt_and_binary_resume_keep_separate_rng_payloads(tmp_path):
    root=Path(__file__).resolve().parents[1]
    c=load(root/'configs/revision4/smoke.json');c.update(hidden=[8,8],attention=4,value_hidden=4)
    a=Learner(Environment(c,2000),'cpu',2000)
    p=tmp_path/'resume.pt';checkpoint(p,a,c,0,True)
    receipt=json.loads(p.with_suffix('.json').read_text())
    assert receipt['role']=='resume' and 'global_rng' not in receipt
    payload=torch.load(p,weights_only=False)
    assert torch.equal(payload['global_rng']['torch_cpu'],torch.get_rng_state())
    assert set(payload['learner']['rng'])==set(a.rng)
    b=Learner(Environment(c,2000),'cpu',2000);b.restore(payload['learner'],resume=True)
    for name,value in a.models.state_dict().items():assert torch.equal(value,b.models.state_dict()[name])
