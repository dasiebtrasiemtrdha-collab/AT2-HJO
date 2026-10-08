"""One raw-component reporting contract for the manuscript 4.0 core."""
import numpy as np


def slot_components(record, config):
    p=config['physical']['costs'];dt=config['slot_s'];w=config['weights']
    task=p['beta_task_latency_per_s']*(record['actual_completed_latency_s']+record['holding_request_seconds']+4*dt*record['drops'])+p['beta_task_energy_per_j']*record['actual_task_energy_j']
    temp=float(np.maximum(np.array(record['theta_before_c'])-70,0).sum()/10)
    receipt=record['deployment']
    dep=p['beta_deployment_latency_per_s']*dt*receipt['pending']+p['beta_deployment_energy_per_j']*(0. if record['operating_failure'] else receipt['total_energy'])
    operating=w[0]*task/p['xref_task']+w[1]*temp/p['xref_temperature']
    reward=-operating
    if config['mode']=='resource_only':reward-=config['rho']*record['proposal_phi']
    return dict(task=task,temperature=temp,deployment=dep,operating=operating,reward_st=reward,p0_numerator=operating+w[2]*dep/p['xref_deployment'])


def compute_fig4a_metric(records,config):
    """Undiscounted LT task/temperature/deployment cost per actual ST slot.

    This is the explicit code reporting definition, separate from discounted
    interval returns. No undocumented historical smoothing is inferred.
    """
    return sum(slot_components(r,config)['p0_numerator'] for r in records)/max(1,len(records))


def compute_fig4b_metric(records,config):
    """Episode mean ST operating cost; residual included only in soft mode."""
    return -sum(slot_components(r,config)['reward_st'] for r in records)/max(1,len(records))


def compute_fig4c_metric(records,config):
    """P0 with the configured full-horizon 1/T denominator."""
    return sum(slot_components(r,config)['p0_numerator'] for r in records)/config['horizon']


def audit_episode(records,config,summary,intervals=()):
    components=[slot_components(r,config) for r in records]
    errors={k:abs(sum(c[k] for c in components)-summary[k]) for k in ('task','temperature','deployment')}
    errors['objective_p0']=abs(compute_fig4c_metric(records,config)-summary['P0'])
    errors['ST_operating_cost']=abs(compute_fig4b_metric(records,config)-summary['ST_operating_cost'])
    for r,c in zip(records,components):
        for raw,recomputed in [('cost_task','task'),('cost_temperature','temperature'),('cost_deployment','deployment'),('reward_st','reward_st')]:
            errors['slot_'+raw]=max(errors.get('slot_'+raw,0.),abs(r[raw]-c[recomputed]))
        if not r['operating_failure']:
            expected=np.array(r['energy_before_j'])-np.array(r['energy_use_j'])
            errors['energy_recursion']=max(errors.get('energy_recursion',0.),float(np.max(abs(expected-np.array(r['energy_after_j'])))))
    for interval in intervals:
        selected=[(r,c) for r,c in zip(records,components) if interval['start']<=r['slot']<interval['start']+interval['duration']]
        value=sum(config['gamma']**i*(c['reward_st']-config['weights'][2]*c['deployment']/config['physical']['costs']['xref_deployment']) for i,(r,c) in enumerate(selected))
        errors['LT_discounted_return']=max(errors.get('LT_discounted_return',0.),abs(value-interval['reward']))
    tolerance=1e-8
    return {'passed':all(e<=tolerance for e in errors.values()),'absolute_tolerance':tolerance,'errors':errors,'max_absolute_error':max(errors.values(),default=0.)}
