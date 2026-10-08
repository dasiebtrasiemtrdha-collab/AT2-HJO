from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
from at2hjo.revision4.config import load
from at2hjo.revision4.environment import Environment
from at2hjo.revision4.reporting import audit_episode,compute_fig4a_metric,compute_fig4b_metric,compute_fig4c_metric

ROOT=Path(__file__).resolve().parents[1]


def environment(horizon=17):
    c=load(ROOT/'configs/revision4/smoke.json');c['horizon']=horizon
    return Environment(c,2000)


def rollout(e,requested):
    records=[];intervals=[]
    for duration in requested:
        e.boundary();e.apply(duration,e.dep.target)
        while e.remaining and not e.terminal:
            _,_,record=e.step(np.zeros(e.p.action_dim));records.append(record)
        intervals.append(deepcopy(e.last_interval))
    return records,intervals


def test_safe_link_encoding_and_physical_inverse():
    e=environment();x=e.observe();piece=e.layout['links'];encoded=x[piece['start']:piece['end']].reshape(piece['shape'])
    assert np.max(abs(encoded[...,0]))<2
    np.testing.assert_allclose(e.decode(x)['links'][...,0],e.trace.links[...,0],rtol=4e-6,atol=1e-30)
    assert np.max(abs(x))<100 # no accidental 10^10 inputs


@pytest.mark.parametrize('horizon,requested,actual',[(17,[10,15],[10,7]),(30,[5,10,15],[5,10,15])])
def test_interval_slots_and_raw_component_recompute(horizon,requested,actual):
    e=environment(horizon);records,intervals=rollout(e,requested)
    assert len(records)==horizon
    assert [i['duration'] for i in intervals]==actual
    assert [i['requested'] for i in intervals]==requested
    assert intervals[-1]['terminal']
    result=audit_episode(records,e.c,e.summary(),intervals)
    assert result['passed'],result
    assert compute_fig4c_metric(records,e.c)==pytest.approx(e.summary()['P0'])
    assert compute_fig4b_metric(records,e.c)==pytest.approx(e.summary()['ST_operating_cost'])
    assert compute_fig4a_metric(records,e.c)==pytest.approx(compute_fig4c_metric(records,e.c))


def test_future_changes_do_not_leak_into_current_state():
    a=environment();b=environment()
    # Ephemeris entries after current t are future materialized input.
    b.trace.ephem['eci_m'][2:]*=2
    b.trace.ephem['local_enu_m'][2:]*=2
    np.testing.assert_array_equal(a.observe(),b.observe())
    x=a.observe().copy();s=a.snapshot()
    for sign in [-1,1]:a.p.proposal(np.full(a.p.action_dim,sign),s)
    np.testing.assert_array_equal(x,a.observe())


def test_waiting_tasks_have_explicit_none_metrics_and_all_offered_denominator():
    e=environment();e.boundary();e.apply(5,np.zeros_like(e.dep.target))
    _,_,r=e.step(np.zeros(e.p.action_dim))
    assert r['completed']==0
    assert r['task_metrics']
    assert all(t['status']=='waiting_no_service' and t['processing_latency_s'] is None for t in r['task_metrics'])
    assert r['cost_task']>0 and r['holding_request_seconds']>0
    assert r['offered']==e.m
    assert e.counts['offered']==2*e.m # next-slot arrivals prepared after advancing


def test_certified_reward_not_actor_proposal_penalty():
    e=environment();e.boundary();e.apply(5,e.dep.target)
    _,_,r=e.step(np.zeros(e.p.action_dim));cost=e.c['physical']['costs'];w=e.c['weights']
    assert r['reward_st']==pytest.approx(-(w[0]*r['cost_task']/cost['xref_task']+w[1]*r['cost_temperature']/cost['xref_temperature']))
    assert r['accepted_phi']<1e-6
    assert r['proposal_phi']>0
