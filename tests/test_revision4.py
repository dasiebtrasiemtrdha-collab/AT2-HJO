"""Meaningful physics/queue/learning invariants for the new scientific identity."""
from pathlib import Path
from copy import deepcopy
import json
import numpy as np
import pytest
import torch
from at2hjo.revision4.config import load
from at2hjo.revision4.environment import Environment,Deployment
from at2hjo.revision4.physics import thermal,torch_thermal,route
from at2hjo.revision4.learning import Learner
from at2hjo.revision4.runner import episode,fingerprint

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/revision4/smoke.json'

def env(trace='regular'):
    c=load(CONFIG);c['trace']=trace;return Environment(c,2000)

def test_four_heads_and_complete_state():
    e=env();x,g=e.boundary();e.observe()
    assert e.p.action_dim==648 and len(e.p.sizes)==4
    assert g.shape==(4,46) and len(x)>1428
    for key in ['target','installed','deployment_queue','relay_occupancy','task_queues','exogenous','controller_history','request_accumulator','pre_drop','clock','radiator','heat']:
        assert key in e.layout
    snap=e.decode(x);actual=e.snapshot()
    for key in ['theta','heat','tasks','energy','wait','links']:
        np.testing.assert_allclose(snap[key],actual[key],rtol=1e-6,atol=1e-5)

def test_fifo_wait_expiry_terminal_partition():
    e=env('stochastic_open');e.boundary();e.apply(15,e.dep.target)
    # Persist an unserved head across a slot; full FIFO descriptors survive.
    e.trace.present[:]=False;e.trace.links[:]=0;e.preview_dep,e.dep_receipt=e.dep.preview(e.trace.links,e.energy)
    old=[list(q) for q in e.queues]
    e.step(np.zeros(e.p.action_dim))
    for m,q in enumerate(old):
        if q:assert list(e.queues[m])[0]==q[0] and e.wait[m]==10
    while not e.terminal:
        if not e.remaining:e.boundary();e.apply(15,e.dep.target)
        e.step(np.zeros(e.p.action_dim))
    s=e.summary();assert s['offered']==s['completed']+s['dropped']
    assert s['expiry']+s['overflow']+s['terminal_drops']==s['dropped']

def test_thermal_power_energy_and_radiation():
    e=env();power=e.p.idle+np.arange(e.n);theta=e.theta;rad=e.radiator;heat=e.trace.heat
    a,b=thermal(theta,rad,power,heat,e.c)
    ta,tb=torch_thermal(torch.tensor(theta),torch.tensor(rad),torch.tensor(power),torch.tensor(heat),e.c)
    np.testing.assert_allclose(ta.numpy(),a,rtol=1e-12);np.testing.assert_allclose(tb.numpy(),b,rtol=1e-12)
    cold,_=thermal(theta,rad,power,np.zeros_like(heat),e.c)
    assert np.all(a[e.p.b:]>cold[e.p.b:])
    snap=e.snapshot();proposal,paths=e.p.proposal(np.zeros(e.p.action_dim),snap);ev=e.p.evaluate(proposal,snap,paths)
    np.testing.assert_allclose(ev['power']*e.c['slot_s'],e.p.idle*e.c['slot_s']+ev['compute_energy'])
    np.testing.assert_allclose(ev['use'],e.p.idle*e.c['slot_s']+snap['dep_energy']+ev['compute_energy']+ev['node_communication'])
    assert ev['node_communication'].sum()>0

def test_execution_joint_constraints_and_idle_exception():
    e=env();snap=e.snapshot();raw=np.zeros(e.p.action_dim)
    a,ev,failure=e.p.certify(raw,snap)
    assert not failure and ev['safe'];assert np.all(a['schedule'].sum(1)<=1)
    assert np.all(a['cpu'].sum(0)<=e.p.capacity*(1+1e-12))
    assert np.all(ev['latency'][ev['y']>0]+snap['wait'][ev['y']>0]<=snap['tasks'][ev['y']>0,3]+1e-7)
    assert np.all(ev['theta']<=80+1e-7)
    snap['energy']=e.p.reserve.copy();_,_,failure=e.p.certify(raw,snap);assert failure

def test_storage_without_permanent_backbone():
    e=env();zero=np.zeros_like(e.dep.target);e.dep.select(zero)
    assert not e.dep.target.any() and not e.dep.installed.any()

def deployment_fixture():
    e=env();d=Deployment(e.c,np.zeros_like(e.dep.target),e.p)
    d.sizes[:]=100.;d.buffer_limit=1000.;e.c['physical']['deployment']['gateway_rate_bps']=4.
    e.c['physical']['deployment']['write_rate_bps']=4.
    e.c['physical']['deployment']['write_energy_j_bit']=.1
    links=np.zeros_like(e.trace.links);gateway=len(links)-1;target=e.m+e.p.b
    links[gateway,target]=[1,1,2,3,1]
    matrix=d.target.copy();matrix[e.p.b,0]=True;matrix[e.p.b,1]=True;d.select(matrix)
    return e,d,links

def test_deployment_fifo_cross_boundary_and_cancel_charges():
    e,d,links=deployment_fixture();old_energy=e.energy.copy()
    d,r=d.preview(links,old_energy)
    assert not d.installed[e.p.b,0] and d.jobs[e.p.b,0]['residual']==60
    assert d.jobs[e.p.b,1]['residual']==100 # FIFO first image consumes link slot.
    energy=old_energy-r['node_energy']-e.p.idle*10
    target=d.target.copy();d.select(target)
    assert d.jobs[e.p.b,0]['residual']==60
    d,r2=d.preview(links,energy);assert d.jobs[e.p.b,0]['residual']==20
    d,r3=d.preview(links,energy-r2['node_energy']);assert d.installed[e.p.b,0]
    assert r['total_energy']>0 and sum(r['node_energy'])<r['total_energy'] # gateway tx counted globally.
    target=d.target.copy();target[e.p.b,1]=False;d.select(target)
    assert (e.p.b,1) not in d.jobs
    assert np.any(energy<old_energy) # cancellation never refunds prior debit.

def test_min_hop_lexicographic_and_pause():
    e,d,links=deployment_fixture();g=len(links)-1;a=e.m+e.p.b;b=a+1;target=a+2
    links[:]=0
    for i,j in [(g,b),(g,a),(a,target),(b,target)]:links[i,j,4]=1
    assert route(links,g,target,e.m,True)==((g,a),(a,target))
    links[:]=0;d2,r=d.preview(links,e.energy)
    assert r['total_energy']==0 and d2.jobs[e.p.b,0]['residual']==100

def test_seed_inputs_are_independent_of_learner_calls():
    a=env('stochastic_open');b=env('stochastic_open')
    for t in range(10):
        np.testing.assert_array_equal(a.trace.links,b.trace.links)
        assert a.trace.arrivals==b.trace.arrivals
        np.random.default_rng(t).normal(size=100+t)
        a.trace.advance();b.trace.advance()

def test_real_training_updates_and_read_only_evaluation(tmp_path):
    torch.set_num_threads(1);e=env();l=Learner(e,'cpu',2000);tmp_path.mkdir(exist_ok=True)
    initial=fingerprint(l)
    for i in range(2):episode(e.c,2000,i,l,True,tmp_path)
    assert fingerprint(l)!=initial
    for name in ('critic_updates','actor_updates','dqn_updates','tr_updates'):assert l.counters[name]>0
    before=fingerprint(l);episode(e.c,4000,0,l,False,tmp_path)
    assert fingerprint(l)==before

def test_reward_discount_and_regularizer_role():
    e=env();e.boundary();e.apply(5,e.dep.target)
    for i in range(5):e.step(np.zeros(e.p.action_dim))
    assert e.last_interval['reward']==pytest.approx(sum(e.c['gamma']**i*r for i,r in enumerate(e.interval_rewards)))
    assert e.last_interval['duration']==5
