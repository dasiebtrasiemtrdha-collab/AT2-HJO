"""Small genuine revision4 gradient/target fixtures; no training campaign."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from at2hjo.revision4.config import load
from at2hjo.revision4.environment import Environment
from at2hjo.revision4.learning import Learner

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def learner():
    torch.set_num_threads(1)
    c=load(ROOT/'configs/revision4/smoke.json')
    c.update(hidden=[8,8],attention=4,value_hidden=4,batch_lt=2,batch_st=2,policy_delay=1)
    return Learner(Environment(c,2000),'cpu',2000)


def demanding_state(learner):
    """All four branches matter; the certificate rejects costly proposals."""
    e=learner.env;p=e.p
    e.dep.installed[:]=True;e.trace.present[:]=True
    e.tasks[:,0]=np.linspace(1e5,2e6,p.m)
    e.tasks[:,1]=np.linspace(1e8,2e9,p.m)
    e.tasks[:,2]=0;e.tasks[:,3]=np.linspace(.01,.04,p.m);e.wait[:]=.3
    e.theta[:]=82.;e.energy=p.reserve+p.idle*e.c['slot_s']+.1
    links=e.trace.links;links[:]=0
    for m in range(p.m):
        for b in range(p.b):links[m,p.m+b]=[1e-10,1e7,1.,1.,1.]
    for b in range(p.b):
        for s in range(p.s):
            i,j=p.m+b,p.m+p.b+s
            links[i,j]=links[j,i]=[1e-10,2e7,10.,1.,1.]
    e.dep_receipt['node_energy'][:]=.01
    state=e.observe()
    raw=torch.zeros(p.action_dim,dtype=torch.float32)
    raw[:p.sizes[0]].reshape(p.m,p.n)[:,p.b]=2.
    raw[p.sizes[0]:]=torch.linspace(-.3,.3,p.action_dim-p.sizes[0])
    return state,raw


def test_actual_certificate_forward_proposal_phi_and_all_head_gradients(learner):
    state,raw=demanding_state(learner);raw.requires_grad_()
    snap=learner.env.decode(state)
    proposal,paths=learner.p.proposal(raw.detach().numpy(),snap)
    expected_phi=learner.p.evaluate(proposal,snap,paths)['phi']
    certified,_,_=learner.p.certify(raw.detach().numpy(),snap)
    projected,phi=learner.projected(raw.unsqueeze(0),[state],gradient=True)
    expected=torch.tensor(learner.p.vector(certified),dtype=torch.float32)
    assert torch.equal(projected[0],expected)
    assert not torch.equal(projected[0],torch.tensor(learner.p.vector(proposal),dtype=torch.float32))
    assert phi.item()==pytest.approx(expected_phi,rel=1e-10,abs=1e-10)
    phi.backward()
    offset=0
    for size in learner.p.sizes:
        grad=raw.grad[offset:offset+size]
        assert torch.isfinite(grad).all() and grad.abs().sum()>0
        offset+=size


def test_proposal_continuous_phi_gradient_matches_physical_finite_difference(learner):
    state,raw=demanding_state(learner);raw=raw.double().requires_grad_()
    _,phi=learner.projected(raw.unsqueeze(0),[state],gradient=True);phi.backward()
    snap=learner.env.decode(state);p=learner.p
    indices=[p.sizes[0]+p.b,sum(p.sizes[:2]),sum(p.sizes[:3])]
    for index in indices:
        values=[]
        for delta in [-1e-5,1e-5]:
            perturbed=raw.detach().numpy().copy();perturbed[index]+=delta
            proposal,paths=p.proposal(perturbed,snap)
            values.append(p.evaluate(proposal,snap,paths)['phi'])
        numerical=(values[1]-values[0])/2e-5
        assert raw.grad[index].item()==pytest.approx(numerical,rel=5e-5,abs=1e-5)


def test_critic_uses_replayed_execution_and_terminal_target_skips_projection(learner,monkeypatch):
    state=learner.env.observe();learner.counters['environment_steps']=learner.c['warmup']
    learner.c['policy_delay']=100
    actions=[]
    for shift in [0.,.2]:
        action=np.full(learner.p.action_dim,shift,np.float32);actions.append(action)
        learner.replay_st.append((state,action,3.+shift,np.full_like(state,np.nan),True))
    observed=[]
    hook=learner.models.q1.register_forward_pre_hook(lambda module,args:observed.append(args[0].detach().clone()))
    def no_terminal_projection(*args,**kwargs):raise AssertionError('terminal bootstrap must not project or decode')
    monkeypatch.setattr(learner,'projected',no_terminal_projection)
    monkeypatch.setattr(learner.models.actor_target,'forward',no_terminal_projection)
    try:learner.td3_update()
    finally:hook.remove()
    received=observed[0][:,learner.state_dim:].numpy()
    assert all(any(np.array_equal(row,a) for a in actions) for row in received)
    assert learner.last_losses['st_target_mean']==pytest.approx(3.1)
    assert learner.last_losses['target_noise_schedule_absmax']==0.
    assert 0.<learner.last_losses['target_noise_continuous_absmax']<=learner.c['target_clip']


def test_td3_delayed_actor_has_finite_real_network_gradients(learner):
    state=learner.env.observe();action=learner.p.vector(learner.p.certify(np.zeros(learner.p.action_dim),learner.env.decode(state))[0])
    learner.counters['environment_steps']=learner.c['warmup']
    for reward in [-1.,-2.]:learner.replay_st.append((state.copy(),action.copy(),reward,state.copy(),False))
    before=[p.detach().clone() for p in learner.models.actor.parameters()]
    learner.td3_update()
    assert learner.counters['actor_updates']==1 and learner.counters['target_updates']==1
    assert learner.last_losses['actor_gradient_norm']>0.
    assert all(torch.isfinite(p.grad).all() for p in learner.models.actor.parameters())
    assert any(not torch.equal(old,new) for old,new in zip(before,learner.models.actor.parameters()))
    assert all(p.requires_grad for p in learner.models.q1.parameters())


def test_tr_literal_boundary_full_history_duration_target_and_stopped_graphs(learner):
    x,g=learner.env.boundary();bx=learner.boundary_state(x)
    assert bx.numel()==learner.state_dim-1
    altered=x.copy();altered[learner.interval_coordinate]=.9
    assert torch.equal(bx,learner.boundary_state(altered))
    observed=[]
    hook=learner.models.value.register_forward_pre_hook(lambda module,args:observed.append(args[0].detach().clone()))
    next_x=x.copy();next_x[0]+=.1;next_g=g.copy();next_g[0,0]=.2
    with torch.no_grad():
        v=learner.models.value(torch.cat([bx,learner.t(g).flatten()])).squeeze().item()
        nv=learner.models.value(torch.cat([learner.boundary_state(next_x),learner.t(next_g).flatten()])).squeeze().item()
        logp=torch.log_softmax(learner.models.tr(bx,learner.t(g)),-1)[1].item()
    interval=dict(reward=-2.,duration=3,start=7,terminal=False)
    try:learner.tr_update(x,g,1,interval,next_x,next_g)
    finally:hook.remove()
    target=-2.+learner.c['gamma']**3*nv;advantage=target-v
    assert learner.last_losses['tr']==pytest.approx(-learner.c['gamma']**7*advantage*logp,rel=1e-5)
    assert learner.last_losses['value']==pytest.approx(.5*advantage**2,rel=1e-5)
    assert torch.equal(observed[0][-g.size:],learner.t(g).flatten())
    assert learner.last_losses['tr_gradient_norm']>0.
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
               for module in (learner.models.tr.q,learner.models.tr.k,learner.models.tr.v)
               for p in module.parameters())
    # Isolate each graph independently from the actual training method.
    learner.models.zero_grad(set_to_none=True)
    v=learner.models.value(torch.cat([bx,learner.t(g).flatten()])).squeeze()
    (-v.detach()*torch.log_softmax(learner.models.tr(bx,learner.t(g)),-1)[1]).backward()
    assert all(p.grad is None for p in learner.models.value.parameters())
    learner.models.zero_grad(set_to_none=True)
    (.5*learner.models.value(torch.cat([bx,learner.t(g).flatten()])).square().mean()).backward()
    assert all(p.grad is None for p in learner.models.tr.parameters())


def test_lt_current_minibatch_adaptive_rate_and_realized_duration_target(learner):
    state=learner.env.observe();action=learner.env.dep.target.astype(np.float32)
    learner.c['gamma']=.5
    with torch.no_grad():
        for p in learner.models.lt.parameters():p.zero_()
        for p in learner.models.lt_target.parameters():p.zero_()
        learner.models.lt_target[-1].bias.fill_(8.)
    learner.replay_lt.append((state,action,0.,dict(reward=2.,duration=3,terminal=False),state,[(action,0.)]))
    learner.replay_lt.append((state,action,0.,dict(reward=-1.,duration=1,terminal=True),None,[]))
    expected_error=(abs(2.+.5**3*8.)+abs(-1.))/2
    expected_lr=learner.c['lr_min']+(learner.c['lr_max']-learner.c['lr_min'])*expected_error/(expected_error+learner.c['epsilon_eta'])
    st_rng=deepcopy(learner.rng['replay_st'].bit_generator.state)
    learner.dqn_update()
    assert learner.last_losses['lt_preupdate_mean_abs_td']==pytest.approx(expected_error)
    assert learner.optim['lt'].param_groups[0]['lr']==pytest.approx(expected_lr)
    assert learner.last_losses['lt']==pytest.approx((3.**2+(-1.)**2)/2)
    assert learner.rng['replay_st'].bit_generator.state==st_rng


def test_private_model_initialization_and_independent_replay_checkpoint_recovery(learner):
    cpu_rng=torch.get_rng_state().clone()
    clone=Learner(Environment(learner.c,2000),'cpu',2000)
    assert torch.equal(torch.get_rng_state(),cpu_rng)
    st_rng=deepcopy(learner.rng['replay_st'].bit_generator.state)
    learner.rng['replay_lt'].choice(100,10,replace=False)
    assert learner.rng['replay_st'].bit_generator.state==st_rng
    data=learner.state(True);clone.restore(data,resume=True)
    for key in learner.rng:
        assert clone.rng[key].bit_generator.state==learner.rng[key].bit_generator.state
    for key in learner.models.state_dict():
        assert torch.equal(clone.models.state_dict()[key],learner.models.state_dict()[key])
    incompatible=deepcopy(data);incompatible.pop('learner_version')
    with pytest.raises(ValueError,match='version/input signature'):clone.restore(incompatible)
    incompatible=deepcopy(data);incompatible['models']['tr.head.weight']=torch.zeros((3,learner.state_dim+learner.c['attention']))
    with pytest.raises(ValueError,match='model shape'):clone.restore(incompatible)
