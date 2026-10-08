"""Manuscript-4.0 deployment adapter: physical rates and transactional slots."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from at2hjo.revision4.environment import Deployment


def fixture(sizes=(4.0,), *, buoys=1, satellites=1, slot=1.0,
            gateway_rate=10.0, write_rate=1.0, write_energy=2.0, buffer=20.0):
    n=buoys+satellites
    c={'nodes':1,'buoys':buoys,'satellites':satellites,'services':len(sizes),'slot_s':slot,
       'physical':{'services':{'sizes_bits':list(sizes)},
                   'hardware':{'bn':{'storage_bits':100.0},'leo':{'storage_bits':100.0}},
                   'channels':{'noise_psd_w_hz':1.0},
                   'deployment':{'relay_buffer_bits':buffer,'reserved_fraction':1.0,
                                 'gateway_rate_bps':gateway_rate,'write_rate_bps':write_rate,
                                 'write_energy_j_bit':write_energy}}}
    p=SimpleNamespace(m=1,b=buoys,s=satellites,n=n,idle=np.full(n,2.0),reserve=np.full(n,10.0))
    d=Deployment(c,np.zeros((n,len(sizes)),bool),p)
    links=np.zeros((n+2,n+2,5));energy=np.full(n,1e6)
    return d,links,energy


def add_link(d,links,src,dst,rate,tx=0.0,rx=0.0,available=True):
    # Construct a Shannon link whose *reserved physical* rate is exactly rate.
    b=d.p.b;bw=1.0
    if src!=len(links)-1 and ((src-d.p.m<b)!=(dst-d.p.m<b)):bw/=b
    signal=bw*(2.0**(rate/bw)-1.0)
    links[src,dst]=[signal,1.0,tx,rx,int(available)]


def select(d,n,k=0):
    target=d.target.copy();target[n,k]=True;d.select(target)


def test_write_throttling_uses_network_rx_tx_coefficients_and_write_only_charge():
    d,links,energy=fixture()
    select(d,1)
    add_link(d,links,len(links)-1,2,10.0,tx=2.0,rx=3.0)
    before=d.state_dict()
    q,r=d.preview(links,energy)
    assert d.state_dict()==before  # a discarded reservation changes no live state
    assert r['carried_bits']==[{'hop':[3,2],'bits':1.0}]
    assert r['written_bits_by_image']==[{'n':1,'k':0,'bits':1.0}]
    np.testing.assert_allclose(r['node_communication_energy'],[0.0,0.3])
    np.testing.assert_allclose(r['node_write_energy'],[0.0,2.0])
    np.testing.assert_allclose(r['node_energy'],[0.0,2.3])
    assert r['total_energy']==pytest.approx(2.5)  # includes gateway TX = 2/10 J
    assert r['pending']==1 and r['pending_residence_s']==1.0
    assert q.jobs[1,0]['written']==1.0
    assert q.descriptors()[1,0,3]==0.25
    assert not q.installed[1,0] and not d.installed[1,0]


def test_preview_completion_is_only_visible_after_commit_as_next_slot_state():
    d,links,energy=fixture(sizes=(2.0,),write_rate=2.0)
    select(d,1);add_link(d,links,3,2,10.0)
    q,r=d.preview(links,energy)
    assert not d.installed[1,0] and q.installed[1,0]
    assert r['completed_images']==[[1,0]] and r['pending']==1
    assert q.jobs=={}
    # Repeating the same uncommitted preview is identical; the clone is the
    # state the Environment commits after executing current-slot tasks.
    again,r2=d.preview(links,energy)
    assert again.state_dict()==q.state_dict()
    np.testing.assert_allclose(r2['node_energy'],r['node_energy'])
    following,r3=q.preview(links,energy)
    assert following.installed[1,0] and r3['pending']==0 and r3['total_energy']==0.0


def test_affordable_budget_pauses_without_double_counting_receiver_or_write():
    d,links,energy=fixture()
    select(d,1);add_link(d,links,3,2,10.0,tx=2.0,rx=3.0)
    energy[1]=d.p.reserve[1]+d.p.idle[1]*d.c['slot_s']+1.15
    q,r=d.preview(links,energy)
    assert r['carried_bits'][0]['bits']==pytest.approx(0.5)
    assert r['node_energy'][1]==pytest.approx(1.15)
    assert q.jobs[1,0]['written']==pytest.approx(0.5)
    assert q.jobs[1,0]['residual']==pytest.approx(3.5)
    energy[1]=d.p.reserve[1]+d.p.idle[1]*d.c['slot_s']
    paused,r2=q.preview(links,energy)
    assert r2['total_energy']==0.0
    assert paused.jobs[1,0]['written']==q.jobs[1,0]['written']


def test_store_and_forward_owns_full_relay_buffer_and_tracks_partial_installation():
    d,links,energy=fixture(sizes=(2.0,),slot=1.5,gateway_rate=2.0,write_rate=10.0,buffer=2.0)
    select(d,1)
    add_link(d,links,3,1,2.0,tx=1.0,rx=2.0)
    add_link(d,links,1,2,1.0,tx=3.0,rx=4.0)
    q,r=d.preview(links,energy)
    assert r['carried_bits']==[{'hop':[1,2],'bits':0.5},{'hop':[3,1],'bits':2.0}]
    assert r['total_energy']==pytest.approx(7.5)
    np.testing.assert_allclose(r['node_energy'],[3.5,3.0])
    assert q.jobs[1,0]['hop']==1 and q.jobs[1,0]['owner']==0
    assert q.jobs[1,0]['residual']==1.5 and q.jobs[1,0]['written']==0.5
    np.testing.assert_array_equal(q.occupancy,[2.0,0.0])
    assert q.descriptors()[1,0,3]==0.25 and q.descriptors()[1,0,4]==1
    final,r2=q.preview(links,energy)
    assert final.installed[1,0] and not q.installed[1,0]
    np.testing.assert_array_equal(final.occupancy,[0.0,0.0])
    assert r2['carried_bits']==[{'hop':[1,2],'bits':1.5}]


def test_finite_buffer_fifo_cancellation_and_retained_progress():
    d,links,energy=fixture(sizes=(4.0,2.0),gateway_rate=10.0,write_rate=10.0,buffer=3.0)
    d.select([[0,0],[1,1]])
    add_link(d,links,3,1,10.0);add_link(d,links,1,2,10.0)
    q,r=d.preview(links,energy)
    assert r['carried_bits']==[] and r['total_energy']==0.0
    assert q.jobs[1,0]['queue_position']==1 and q.jobs[1,1]['queue_position']==2
    q.select([[0,0],[0,1]])
    final,r2=q.preview(links,energy)
    assert final.installed[1,1] and not final.installed[1,0]
    assert r2['written_bits'][1]==2.0
    d,links,energy=fixture(sizes=(4.0,),gateway_rate=2.0,write_rate=10.0,buffer=4.0)
    select(d,1);add_link(d,links,3,1,2.0,rx=2.0);add_link(d,links,1,2,1.0)
    q,r=d.preview(links,energy)
    assert q.jobs[1,0]['residual']==2.0 and q.jobs[1,0]['reserved']==0
    q.select(q.target.copy())
    assert q.jobs[1,0]['residual']==2.0
    spent=q.state_dict()['engine']['cumulative_communication_energy_by_node_j']
    q.select([[0],[0]])
    assert q.jobs=={} and not q.occupancy.any()
    assert q.state_dict()['engine']['cumulative_communication_energy_by_node_j']==spent


def test_simultaneous_relay_arrivals_use_n_k_tie_instead_of_older_enqueue_ticket():
    d,links,energy=fixture(sizes=(1.0,1.0),buoys=2,satellites=2,slot=0.25,
                           gateway_rate=1.0,write_rate=10.0,write_energy=0.0,buffer=2.0)
    # Service 1 starts first on gateway -> BN1 -> LEO0 -> LEO1.
    select(d,3,1)
    for i,j in [(5,2),(2,3),(3,4)]:add_link(d,links,i,j,1.0)
    d,_=d.preview(links,energy)
    # Service 0 starts one slot later on a distinct first two directed links.
    select(d,3,0);links[5,2,4]=0
    for i,j in [(5,1),(1,3)]:add_link(d,links,i,j,1.0)
    d,_=d.preview(links,energy)
    links[5,2,4]=1
    for _ in range(7):d,_=d.preview(links,energy)
    # Both full images arrive at the shared final directed link at exactly the
    # same instant. Its FIFO gives (n,0) precedence although (n,1) was enqueued first.
    assert d.jobs[3,0]['queue_position']==1 and d.jobs[3,1]['queue_position']==2
    q,r=d.preview(links,energy)
    assert q.jobs[3,0]['written']==0.25 and q.jobs[3,1]['written']==0.0
    assert r['written_bits_by_image']==[{'n':3,'k':0,'bits':0.25}]


def test_adapter_json_state_restores_fifo_owner_and_partial_writes_exactly():
    d,links,energy=fixture(sizes=(2.0,2.0),slot=1.25,gateway_rate=2.0,write_rate=1.0,buffer=4.0)
    d.select([[0,0],[1,1]])
    add_link(d,links,3,1,2.0,tx=1.0,rx=2.0);add_link(d,links,1,2,1.0,tx=3.0,rx=4.0)
    d,_=d.preview(links,energy)
    saved=json.loads(json.dumps(d.state_dict(),allow_nan=False))
    restored=Deployment(deepcopy(d.c),np.zeros_like(d.target),deepcopy(d.p))
    restored.load_state_dict(saved)
    assert restored.state_dict()==saved
    np.testing.assert_array_equal(restored.descriptors(),d.descriptors())
    a,ra=d.preview(links,energy);b,rb=restored.preview(links,energy)
    assert a.state_dict()==b.state_dict()
    for name in ['node_energy','node_communication_energy','node_write_energy','written_bits']:
        np.testing.assert_array_equal(ra[name],rb[name])
    assert ra['events']==rb['events'] and ra['carried_bits']==rb['carried_bits']
