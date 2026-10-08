"""Scientific invariants for the active revision4 physics and input path."""
from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp
import torch

from at2hjo.geometry.orbits import orbital_ephemeris
from at2hjo.revision4.config import load
from at2hjo.revision4.inputs import Trace
from at2hjo.revision4.physics import Projector,environmental_heat,node_values,thermal,torch_thermal

ROOT=Path(__file__).resolve().parents[1]


def configuration(**overrides):
    c=load(ROOT/'configs/revision4/smoke.json');c.update(overrides);return c


def test_satellite_rk4_matches_independent_ode_and_conserves_heat():
    c=configuration(buoys=1,satellites=2);p=c['physical'];rr=p['satellite_radiation']
    theta=np.array([40.,70.,55.]);rad=np.array([20.,30.]);power=np.array([8.,25.,20.]);heat=np.array([180.,30.])
    out,radiator,balance=thermal(theta,rad,power,heat,c,return_balance=True)
    def ode(_,state):
        cpu,sink=state[:2],state[2:]
        flow=(cpu-sink)/p['thermal']['leo']['rth_k_w']
        radiation=rr['emissivity']*rr['sigma']*rr['radiating_area_m2']*((sink+273.15)**4-rr['background_k']**4)
        return np.r_[(power[1:]-flow)/p['thermal']['leo']['cth_j_k'],(flow+heat-radiation)/rr['radiator_capacity_j_k']]
    reference=solve_ivp(ode,(0,c['slot_s']),np.r_[theta[1:],rad],method='DOP853',rtol=1e-12,atol=1e-12).y[:,-1]
    np.testing.assert_allclose(np.r_[out[1:],radiator],reference,atol=1e-10,rtol=1e-11)
    np.testing.assert_allclose(balance['energy_balance_residual_j'],0,atol=2e-10)
    assert np.all(balance['radiated_energy_j']>0)
    expected=25+1.2*8+(40-25-1.2*8)*math.exp(-10/(1.2*200))
    assert out[0]==pytest.approx(expected)


def test_internal_conduction_matches_analytic_solution_without_radiative_loss():
    c=configuration(buoys=1,satellites=1);c['physical']['satellite_radiation']['emissivity']=0
    initial_cpu,initial_rad=75.,15.;cc,cr=400.,1200.;r=1.3
    out,rad,balance=thermal([40,initial_cpu],[initial_rad],[0,0],[0],c,return_balance=True)
    mean=(cc*initial_cpu+cr*initial_rad)/(cc+cr)
    delta=(initial_cpu-initial_rad)*math.exp(-c['slot_s']/r*(1/cc+1/cr))
    assert out[1]==pytest.approx(mean+cr/(cc+cr)*delta,abs=1e-10)
    assert rad[0]==pytest.approx(mean-cc/(cc+cr)*delta,abs=1e-10)
    assert balance['radiated_energy_j'][0]==0
    assert balance['stored_energy_change_j'][0]==pytest.approx(0,abs=2e-10)


def test_kelvin_radiation_equilibrium_and_node_specific_coefficients():
    c=configuration(buoys=1,satellites=2);rr=c['physical']['satellite_radiation'];sink=30.;power=15.
    c['physical']['node_parameters']={'radiator_capacity_j_k':[1000,4000],'emissivity':[.7,.9],'ambient_c':[10,5000,5000]}
    radiation=np.array([.7,.9])*rr['sigma']*rr['radiating_area_m2']*((sink+273.15)**4-rr['background_k']**4)
    cpu=sink+1.3*power
    out,rad,balance=thermal([40,cpu,cpu],[sink,sink],[8,power,power],radiation-power,c,return_balance=True)
    np.testing.assert_allclose(out[1:],[cpu,cpu],atol=1e-11)
    np.testing.assert_allclose(rad,[sink,sink],atol=1e-11)
    np.testing.assert_allclose(balance['radiated_energy_j'],radiation*10,rtol=1e-12)
    expected=10+1.2*8+(40-10-1.2*8)*math.exp(-10/(1.2*200))
    assert out[0]==pytest.approx(expected)
    changed,_=thermal([40,cpu,cpu],[sink,sink],[8,power,power],radiation-power,c,ambient_c=[35])
    assert changed[0]>out[0]
    np.testing.assert_array_equal(changed[1:],out[1:])


def test_differentiable_thermal_matches_physics_and_has_finite_heat_gradients():
    c=configuration();theta=np.r_[np.full(3,40.),np.full(6,45.)];rad=np.full(6,20.);power=np.r_[np.full(3,8.),np.full(6,15.)];heat=np.arange(6)*20.
    actual=thermal(theta,rad,power,heat,c,ambient_c=[15,25,35])
    tp=torch.tensor(power,requires_grad=True);tq=torch.tensor(heat,requires_grad=True)
    tc,tr=torch_thermal(torch.tensor(theta),torch.tensor(rad),tp,tq,c,ambient_c=[15,25,35])
    np.testing.assert_allclose(tc.detach(),actual[0],rtol=1e-12)
    np.testing.assert_allclose(tr.detach(),actual[1],rtol=1e-12)
    (tc.sum()+tr.sum()).backward()
    assert torch.isfinite(tp.grad).all() and (tp.grad>0).all()
    assert torch.isfinite(tq.grad).all() and (tq.grad>0).all()


def routed_snapshot():
    c=configuration(nodes=2,buoys=2,satellites=1,services=3);p=Projector(c);links=np.zeros((6,6,5));noise=c['physical']['channels']['noise_psd_w_hz']
    # Each optical source receives half of a 1 kHz pool. Both then share the
    # same BN relay hop at 1 kbps, so its fluid-TDMA duration doubles.
    for source in (0,1):links[source,2]=[noise*500*(2**2-1),1000,1.2,.2,1]
    relay_band=900.;links[2,3]=[noise*relay_band*(2**(1000/relay_band)-1),1000,2.2,.2,1]
    snap=dict(tasks=np.array([[100,1e9,0,10,0],[200,2e9,0,10,0]],float),wait=np.zeros(2),present=np.ones(2,bool),
              installed=np.ones((3,3),bool),links=links,theta=np.array([40,40,45.]),radiator=np.array([20.]),heat=np.array([100.]),
              energy=np.array([500000,500000,1000000.]),dep_energy=np.array([1,2,3.]),ambient=np.array([25.,25.]))
    action=dict(schedule=np.array([[0,1,0],[0,1,0]],float),cpu=np.array([[0,1e9,0],[0,1e9,0]],float),
                optical=np.array([[.5,0],[.5,0]]),rf=np.zeros((2,1)))
    return c,p,snap,action


def test_slot_average_power_all_endpoint_ledger_and_shared_relay_budget():
    c,p,snap,action=routed_snapshot();ev=p.evaluate(action,snap)
    assert ev['safe']
    np.testing.assert_allclose(ev['compute_energy'],[0,.3,0],atol=1e-12)
    np.testing.assert_allclose(ev['power'],p.idle+np.array([0,.03,0]))
    np.testing.assert_allclose(ev['transmission_latency_s'],[.3,.6],atol=1e-12)
    np.testing.assert_allclose(ev['compute_latency_s'],[1,2])
    np.testing.assert_allclose(ev['all_node_tx_energy'],[.12,.24,1.32,0,0,0],atol=1e-12)
    np.testing.assert_allclose(ev['all_node_rx_energy'],[0,0,.06,.12,0,0],atol=1e-12)
    np.testing.assert_allclose(ev['task_energy'],[.72,1.44],atol=1e-12)
    assert ev['task_hop_bits'][2,3]==300
    assert ev['all_node_communication'].sum()==pytest.approx(ev['task_energy'].sum()-ev['compute_energy'].sum())
    np.testing.assert_allclose(ev['use'],p.idle*10+snap['dep_energy']+[1.38,.42,0],atol=1e-12)
    assert ev['phi']==pytest.approx(ev['phi_deadline']+ev['phi_thermal']+ev['phi_energy'])
    # A receiving relay's budget is checked even when it computes no task.
    snap['energy'][0]=p.reserve[0]+p.idle[0]*10+snap['dep_energy'][0]+1.37
    assert not p.evaluate(action,snap)['safe']


def test_idle_slot_debits_every_node_and_zero_cpu_mask_cannot_create_work():
    _,p,snap,action=routed_snapshot();action['schedule'][:]=0;action['cpu'][:]=0
    idle=p.evaluate(action,snap)
    np.testing.assert_array_equal(idle['compute_energy'],[0,0,0])
    np.testing.assert_array_equal(idle['all_node_communication'],np.zeros(6))
    np.testing.assert_array_equal(idle['use'],p.idle*10+snap['dep_energy'])
    action['cpu'][0,1]=1e9
    invalid=p.evaluate(action,snap)
    assert not invalid['safe']
    np.testing.assert_array_equal(invalid['compute_energy'],[0,0,0])
    raw=np.zeros(p.action_dim);snap['energy']=p.reserve.copy()
    _,_,operating_failure=p.certify(raw,snap)
    assert operating_failure


def test_optional_source_budget_is_checked_against_source_tx_circuit_charge():
    _,p,snap,action=routed_snapshot();snap['source_energy']=np.array([.1,1.])
    ev=p.evaluate(action,snap)
    assert not ev['safe']
    assert ev['phi_energy']>0
    snap['source_energy'][0]=.13
    assert p.evaluate(action,snap)['safe']


def test_environment_heat_geometry_eclipse_and_absorbed_power_units():
    c=configuration(buoys=1,satellites=2);rr=c['physical']['satellite_radiation'];earth=c['physical']['orbit']['earth_radius_m'];radius=earth+1e6
    daynight=environmental_heat([[radius,0,0],[-radius,0,0]],[1,0,0],c)
    np.testing.assert_array_equal(daynight['sunlight'],[True,False])
    assert daynight['solar'][0]==pytest.approx(rr['solar_absorptivity']*rr['solar_area_m2']*rr['irradiance_w_m2'])
    assert daynight['solar'][1]==0 and daynight['albedo'][1]==0
    np.testing.assert_allclose(daynight['infrared'],rr['earth_ir_w']*(earth/radius)**2)
    farther=environmental_heat([[2*radius,0,0],[-2*radius,0,0]],[1,0,0],c)
    np.testing.assert_allclose(farther['infrared'],daynight['infrared']/4)


def test_input_clock_waypoint_current_and_arrival_timestamp_are_causal():
    c=configuration(horizon=25,nodes=3);trace=Trace(c,0);trace.t=19
    before=trace.q.copy();command=trace.cmd.copy();current=trace.current.copy();waypoint=trace.w.copy()
    r=deepcopy(trace.rng['mobility']);innovation=r.normal(0,.05,2)
    expected_current=.9*current+innovation;expected_current/=max(1,np.linalg.norm(expected_current)/.5)
    expected_waypoint=r.uniform(0,500,(3,2))
    moved=before[:,:2]+(command+current)*10;folded=np.mod(moved,1000);expected_position=np.where(folded<=500,folded,1000-folded)
    trace.advance()
    np.testing.assert_allclose(trace.q[:,:2],expected_position)
    np.testing.assert_allclose(trace.current,expected_current)
    np.testing.assert_array_equal(trace.w,expected_waypoint)
    assert not np.array_equal(trace.w,waypoint)
    assert trace.t==20 and all(task[4]==20 for node in trace.arrivals for task in node)
    assert np.linalg.norm(trace.current)<=.5+1e-12


def test_driver_schema_decodes_current_state_without_future_ephemeris_or_rng_draws():
    c=configuration(horizon=105);trace=Trace(c,0);trace.t=100;trace.popularity_shift=0
    old=trace.drivers();states={key:deepcopy(r.bit_generator.state) for key,r in trace.rng.items()}
    trace.ephem['local_enu_m'][101:]=np.nan;trace.ephem['eci_m'][101:]=np.nan;trace.ephem['ecef_m'][101:]=np.nan
    np.testing.assert_array_equal(trace.drivers(),old)
    layout=trace.driver_layout();assert max(d['end'] for d in layout.values())==len(old)
    for name,actual in [('ocean_current',trace.current),('waypoint',trace.w),('buoy_sink_temperature',trace.ambient),('satellite_velocity_enu',trace.orbital_velocity())]:
        d=layout[name];decoded=old[d['start']:d['end']].reshape(d['shape'])*d['scaling']['divisor']
        np.testing.assert_allclose(decoded,actual)
    assert all(r.bit_generator.state==states[key] for key,r in trace.rng.items())
    assert layout['satellite_position_enu']['units']=='m'
    # Independent central derivative verifies the analytic current velocity.
    reference=orbital_ephemeris(trace.orbit,[1000-.001,1000+.001])
    np.testing.assert_allclose(trace.orbital_velocity(),(reference['local_enu_m'][1]-reference['local_enu_m'][0])/.002,rtol=1e-7,atol=1e-5)


@pytest.mark.parametrize('kind',['regular','stochastic_open'])
def test_full_500_slot_inputs_follow_declared_regular_or_stochastic_laws(kind):
    c=configuration(horizon=500,trace=kind);trace=Trace(c,0);offered=0;rates={False:[],True:[]};high=0;in_area=0
    baseline=np.arange(1,21,dtype=float)**(-.8);baseline/=baseline.sum()
    for slot in range(500):
        assert trace.t==slot and np.isfinite(trace.drivers()).all() and np.isfinite(trace.links).all()
        assert np.linalg.norm(trace.current)<=.5+1e-12
        counts=np.array([len(q) for q in trace.arrivals]);offered+=int(counts.sum())
        assert np.all(counts[~trace.present]==0)
        if kind=='regular':
            assert trace.present.all() and np.all(counts==1) and not trace.traffic.any() and trace.popularity_shift==0
            np.testing.assert_allclose(trace.service_probability,baseline)
        else:
            assert trace.popularity_shift==(slot//100)%20
            expected=.7*np.roll(baseline,-trace.popularity_shift)+.3/20
            np.testing.assert_allclose(trace.service_probability,expected)
            for mode in (False,True):rates[mode].extend(counts[trace.present&(trace.traffic==mode)].tolist())
            high+=int(np.count_nonzero(trace.traffic&trace.present));in_area+=int(trace.present.sum())
        for q in trace.arrivals:
            for size,cycles,service,deadline,arrival in q:
                assert .5e6<=size<=2e6 and .5e9<=cycles<=2e9 and arrival==slot and 0<=service<20
                assert ((.5<=deadline<=2) if kind=='regular' else (20<=deadline<=40))
        if slot<499:trace.advance()
    assert offered>0
    if kind=='regular':assert offered==15000
    else:
        assert np.mean(rates[False])==pytest.approx(.5,abs=.07)
        assert np.mean(rates[True])==pytest.approx(3,abs=.2)
        assert high/in_area==pytest.approx(.2,abs=.04)
    with pytest.raises(IndexError,match='horizon'):trace.advance()


def test_source_usn_is_stationary_and_link_electrical_power_includes_circuits():
    c=configuration(nodes=2,horizon=3);c['physical']['underwater_node_types']=['USN','AUV'];trace=Trace(c,0)
    trace.q[0]=trace.bq[0]+[0,0,-50];fixed=trace.q[0].copy();trace.current[:]=[.5,0]
    links=trace._links();optical=c['physical']['channels']['optical'];circuit=c['physical']['circuit']['optical']
    assert links[0,2,2]==optical['tx_power_w']+circuit['tx_w']
    assert links[0,2,3]==circuit['rx_w'] and links[0,2,4]==1
    trace.advance();np.testing.assert_array_equal(trace.q[0],fixed);np.testing.assert_array_equal(trace.cmd[0],[0,0])


@pytest.mark.parametrize('bad',['zero_R','zero_Cr','zero_dt','absolute_zero','nan_heat'])
def test_invalid_physical_coefficients_and_units_fail_explicitly(bad):
    c=configuration(buoys=1,satellites=1);theta=[40.,45.];heat=[100.]
    if bad=='zero_R':c['physical']['thermal']['leo']['rth_k_w']=0
    if bad=='zero_Cr':c['physical']['satellite_radiation']['radiator_capacity_j_k']=0
    if bad=='zero_dt':c['slot_s']=0
    if bad=='absolute_zero':theta[1]=-273.15
    if bad=='nan_heat':heat=[np.nan]
    with pytest.raises(ValueError):thermal(theta,[20.],[8.,15.],heat,c)
