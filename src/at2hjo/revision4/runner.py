from pathlib import Path
from copy import deepcopy
import argparse
import csv
import hashlib
import json
import sys
import time
import platform
import random
import os
import numpy as np
import torch
from at2hjo.device import DeviceManager
from . import SCIENCE,VERSION
from .config import load,digest,input_identity
from .environment import Environment
from .learning import Learner

def jsonable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(i) for i in x]
    return x

def write(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(jsonable(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)

def code_hash():
    h=hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):h.update(p.name.encode());h.update(p.read_bytes())
    for name in ['device.py','geometry/orbits.py','geometry/coordinates.py','communications/channels.py','communications/configuration.py','communications/units.py']:
        p=Path(__file__).parents[1]/name;h.update(name.encode());h.update(p.read_bytes())
    return h.hexdigest()

def fingerprint(learner):
    h=hashlib.sha256()
    for k,v in learner.models.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    h.update(json.dumps(learner.counters,sort_keys=True).encode())
    return h.hexdigest()

def verify_episode_logs(output,index,c,summary,slot_offset=0,interval_offset=0):
    """Commit and check only this episode's appended bytes before completion."""
    from .reporting import audit_episode
    def rows(name,offset):
        path=output/name
        with path.open('ab') as stream:
            stream.flush();os.fsync(stream.fileno())
        with path.open('rb') as stream:
            stream.seek(offset)
            return [json.loads(line) for line in stream if line.strip()]
    records=rows('slots.jsonl',slot_offset)
    intervals=rows('intervals.jsonl',interval_offset)
    if len(records)!=summary['slots'] or [r.get('slot') for r in records]!=list(range(summary['slots'])) or any(r.get('episode')!=index for r in records):
        raise RuntimeError('Incomplete or inconsistent raw slot log; completion/checkpoint refused')
    if len(intervals)!=summary['boundaries'] or any(r.get('episode')!=index for r in intervals):
        raise RuntimeError('Incomplete or inconsistent raw interval log; completion/checkpoint refused')
    audit=audit_episode(records,c,summary,intervals)
    if not audit['passed']:
        raise RuntimeError('Raw component audit failed; completion/checkpoint refused')
    return audit

def episode(c,seed,index,learner,training,output,dataset=None):
    slot_offset=(output/'slots.jsonl').stat().st_size if (output/'slots.jsonl').exists() else 0
    interval_offset=(output/'intervals.jsonl').stat().st_size if (output/'intervals.jsonl').exists() else 0
    e=Environment(c,seed,index,dataset);learner.env=e
    x,g=e.boundary();pending=None
    while not e.terminal:
        tr_probabilities=None;attention_last_row=None
        if c['algorithm']=='joint':
            with torch.no_grad():
                tr=learner.models.tr;tg=learner.t(g)
                attention=torch.softmax(tr.q(tg)@tr.k(tg).transpose(-1,-2)/tr.d**.5,-1)
                attention_last_row=attention[-1].cpu().tolist()
                tr_probabilities=torch.softmax(tr(learner.boundary_state(x),tg),-1).cpu().tolist()
        tr_started=time.perf_counter();duration,tr_index=learner.choose_interval(x,g,training) if c['algorithm']=='joint' else (10,1);tr_ms=(time.perf_counter()-tr_started)*1000
        e.remaining=min(duration,c['horizon']-e.t);lt_state=e.observe();e.remaining=0
        lt_started=time.perf_counter();candidates=learner.candidates()
        if training and c['algorithm']!='st' and getattr(learner,'pending_lt',None) is not None:
            learner.replay_lt.append((*learner.pending_lt,lt_state.copy(),candidates));learner.pending_lt=None;learner.dqn_update()
        target,prior=learner.choose_lt(lt_state,candidates,training,index) if c['algorithm']!='st' else (e.dep.target.astype(np.float32),0.)
        with torch.no_grad():qres=float(learner.models.lt(learner.t(np.r_[lt_state,target.ravel()])).squeeze()) if c['algorithm']!='st' else None
        lt_ms=(time.perf_counter()-lt_started)*1000;state=e.apply(duration,target)
        if pending is not None:
            s,a,r=pending;learner.replay_st.append((s,a,r,state.copy(),False));pending=None
            if training and c['algorithm']!='lt':learner.td3_update()
        while e.remaining and not e.terminal:
            actor_started=time.perf_counter();raw=learner.choose_st(state,training) if c['algorithm']!='lt' else np.zeros(e.p.action_dim);actor_ms=(time.perf_counter()-actor_started)*1000
            reward,terminal,record=e.step(raw)
            record.update(actor_ms=actor_ms,tr_ms=tr_ms if record['slot']==e.interval_start else 0.,lt_ms=lt_ms if record['slot']==e.interval_start else 0.)
            if training:learner.counters['environment_steps']+=1
            with (output/'slots.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(jsonable({'episode':index,'seed':seed,**record}),allow_nan=False)+'\n')
            if training:
                if terminal:nextstate=np.zeros_like(state);learner.replay_st.append((state.copy(),np.array(record['executed_action'],np.float32),reward,nextstate,True))
                elif e.remaining:nextstate=e.observe();learner.replay_st.append((state.copy(),np.array(record['executed_action'],np.float32),reward,nextstate.copy(),False))
                else:pending=(state.copy(),np.array(record['executed_action'],np.float32),reward)
                if c['algorithm']!='lt' and (terminal or e.remaining):learner.td3_update()
            if e.remaining and not terminal:state=e.observe()
            write(output/'heartbeat.json',{'episode':index,'slot':e.t,'counters':learner.counters,'updated_at_unix':time.time()})
        interval=dict(e.last_interval)
        interval.update(tr_probabilities=tr_probabilities,attention_last_row=attention_last_row,
                        candidate_count=len(candidates),selected_q0=prior,selected_qres=qres,
                        actual_deployment=target.astype(int).tolist(),holding_time=min(duration,c['horizon']-e.interval_start))
        if e.terminal:next_x=np.zeros_like(x);next_g=np.zeros_like(g);next_state=np.zeros_like(lt_state);next_candidates=[]
        else:
            next_x,next_g=e.boundary()
            next_state=None;next_candidates=None
        if training and c['algorithm']=='joint':learner.tr_update(x,g,tr_index,interval,next_x,next_g)
        # Conditional next LT interval is sampled once and reused at the boundary.
        # Store LT replay at the following boundary, after that conditional state exists.
        if training and c['algorithm']!='st':
            learner.pending_lt=(lt_state.copy(),target.copy(),prior,interval)
            if e.terminal:
                learner.replay_lt.append((*learner.pending_lt,next_state,next_candidates));learner.pending_lt=None;learner.dqn_update()
        with (output/'intervals.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'episode':index,**interval})+'\n')
        x,g=next_x,next_g
    if pending is not None:raise RuntimeError('unflushed ST boundary transition')
    summary=e.summary()
    summary['raw_component_audit']=verify_episode_logs(output,index,c,summary,slot_offset,interval_offset)
    return summary

def checkpoint(path,learner,c,completed,resume):
    payload={'science':SCIENCE,'version':VERSION,'role':'resume' if resume else 'evaluation','config':c,'config_hash':digest(c),'code_hash':code_hash(),'completed_episodes':completed,'actual_ST_steps':learner.counters['environment_steps'],'formal_result':False,'learner':learner.state(resume)}
    if resume:
        payload['runtime_versions']={'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__}
        payload['global_rng']={'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)
    write(path.with_suffix('.json'),{k:payload[k] for k in payload if k not in ('learner','config','global_rng')})

def execute(args):
    output=Path(args.output).resolve()
    if output.exists():raise FileExistsError('Use a new output directory: '+str(output))
    device=DeviceManager(args.device);torch.set_num_threads(1)
    payload=None
    if args.checkpoint:
        # Evaluation payloads contain tensors and primitive containers only.
        # Recovery payloads also contain NumPy/Python RNG and replay state.
        if args.command=='resume' and not getattr(args,'trust_checkpoint',False):
            raise ValueError('Resume requires --trust-checkpoint for a checkpoint whose origin and checksum you have verified')
        payload=torch.load(args.checkpoint,map_location=device.device,weights_only=args.command!='resume')
        if payload.get('science')!=SCIENCE:raise ValueError('Legacy checkpoint incompatible with 4.0 queue/state semantics')
        if args.command=='resume' and payload.get('role')!='resume':raise ValueError('Evaluation checkpoint has no replay/optimizer recovery state')
        c=payload['config']
        if args.config and digest(load(args.config))!=payload['config_hash']:raise ValueError('Configuration mismatch')
        if args.command=='resume' and payload['code_hash']!=code_hash():raise ValueError('Source changed; exact recovery refused')
    else:c=load(args.config)
    if args.algorithm:
        if payload and args.algorithm!=c['algorithm']:raise ValueError('Checkpoint algorithm mismatch')
        c['algorithm']=args.algorithm
    training=args.command in ('train','resume');seed=args.seed if args.seed is not None else (payload['learner']['seed'] if payload else None)
    if seed is None:raise ValueError('--seed required')
    if training and seed not in c['training_seeds']:raise ValueError('Seed not in this profile training role')
    if not training and seed not in c['evaluation_seeds'] and seed not in c.get('validation_seeds',[]):raise ValueError('Seed not in this profile evaluation or validation role')
    seed_role='training' if training else ('validation' if seed in c.get('validation_seeds',[]) else 'evaluation')
    if args.command=='resume' and seed!=payload['learner']['seed']:raise ValueError('Resume must preserve original seed')
    start=payload['completed_episodes'] if args.command=='resume' else 0
    count=(args.episodes or c['episodes']) if training else (args.episodes or 1)
    if training and count<=start:raise ValueError('Resume target must exceed completed episodes')
    e=Environment(c,seed,0);e.observe();learner=Learner(e,device.device,seed)
    if payload:learner.restore(payload['learner'],resume=args.command=='resume')
    if args.command=='resume':
        expected={'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__}
        if payload.get('runtime_versions')!=expected:raise ValueError('Exact resume requires the saved Python/NumPy/PyTorch versions; use evaluate for read-only inference or install the matching lock file.')
        rng=payload['global_rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch_cpu'].cpu())
        if rng['torch_cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all([v.cpu() for v in rng['torch_cuda']])
    learner.models.train(training)
    output.mkdir(parents=True);write(output/'config.json',c);write(output/'network_summary.json',learner.models.summary(learner.state_dim,e.context_dim,e.p,c));write(output/'state_layout.json',e.layout)
    write(output/'run.json',{'science':SCIENCE,'version':VERSION,'command':sys.argv,'config_hash':digest(c),'code_hash':code_hash(),'seed':seed,'seed_role':seed_role,'result_scope':c['result_scope'],'mode':c['mode'],'trace':c['trace'],'formal_result':False,'runtime':{'python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'device':device.describe()},'replay_full_capacity_state_bytes':2*c['replay']*learner.state_dim*4,'legacy_compatible':False})
    before=fingerprint(learner);beg=time.perf_counter();rows=[]
    write(output/'status.json',{'status':'running','start_episode':start,'target_episode':count,'counters':learner.counters})
    try:
        for i in range(start,count):
            summary=episode(c,seed,i,learner,training,output,getattr(args,'dataset',None));rows.append({'episode':i+1,**summary,**learner.counters,'losses':learner.last_losses.copy()})
            write(output/'episode_metrics.json',rows)
            if training:checkpoint(output/'resume.pt',learner,c,i+1,True);checkpoint(output/'evaluation.pt',learner,c,i+1,False)
        after=fingerprint(learner)
        if not training and before!=after:raise RuntimeError('Evaluation mutated learner weights/counters')
        result={'status':'completed','seconds':time.perf_counter()-beg,'episodes_this_invocation':len(rows),'completed_episodes':count,'actual_ST_steps_this_invocation':sum(r['slots'] for r in rows),'counters':learner.counters,'fingerprint_before':before,'fingerprint_after':after,'evaluation_unchanged':before==after if not training else None,'formal_result':False,'rows':rows}
        write(output/'status.json',result);print(json.dumps(jsonable({k:v for k,v in result.items() if k!='rows'}),ensure_ascii=False))
    except BaseException as ex:
        write(output/'status.json',{'status':'failed','error':str(ex),'seconds':time.perf_counter()-beg,'counters':learner.counters,'formal_result':False});raise

def generate(args):
    from .inputs import Trace
    c=load(args.config);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);trace=Trace(c,args.seed,0);links=[];drivers=[];offered=[]
    for t in range(c['horizon']):
        links.append(trace.links.copy());drivers.append(trace.drivers());offered.append({'slot':t,'present':trace.present.tolist(),'arrivals':trace.arrivals,'heat_w':trace.heat.tolist(),'ambient_c':trace.ambient.tolist()})
        if t+1<c['horizon']:trace.advance()
    np.savez_compressed(out/'trace.npz',links=np.array(links),drivers=np.array(drivers))
    (out/'requests.jsonl').write_text(''.join(json.dumps(jsonable(r))+'\n' for r in offered),encoding='utf-8')
    write(out/'config.json',c)
    files={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ('trace.npz','requests.jsonl','config.json')}
    write(out/'manifest.json',{'input_schema_version':'manuscript40-input-v2','science':SCIENCE,'seed':args.seed,'horizon':c['horizon'],'trace':c['trace'],'config_hash':digest(c),'input_config_hash':input_identity(c),'files':files,'schema':{'link_axis':['received_signal_power_W','bandwidth_Hz','tx_electrical_power_W','rx_electrical_power_W','availability'],'request_fields':['input_bits','cycles','service_zero_based','deadline_s','arrival_slot_zero_based'],'drivers':trace.driver_layout(),'ambient_c':'BN heat-sink temperature, shape B, degrees Celsius'},'formal_result':False})
    print(json.dumps({'output':str(out),'files':files}))

def aggregate(args):
    rows=[]
    for name in args.runs:
        root=Path(name);run=json.loads((root/'run.json').read_text(encoding='utf-8'));c=json.loads((root/'config.json').read_text(encoding='utf-8'))
        if run['science']!=SCIENCE:raise ValueError('Legacy run excluded: '+name)
        for row in json.loads((root/'episode_metrics.json').read_text(encoding='utf-8')):
            keys=['episode','slots','P0','ST_operating_cost','penalized_cost','mean_latency_completed_s','mean_energy_completed_j','completion_fraction','drop_fraction','accepted_deadline_violations','node_temperature_violations','node_energy_violations','operating_failures']
            rows.append({'run':name,'config_hash':run['config_hash'],'science':SCIENCE,'algorithm':c['algorithm'],'mode':c['mode'],'trace':c['trace'],'seed':run['seed'],'seed_role':run['seed_role'],'scope':run['result_scope'],'formal_result':False,'performance_eligible':row['operating_failures']==0 and row['slots']==c['horizon'],**{k:row[k] for k in keys}})
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    with out.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'rows':len(rows),'output':str(out),'grouping_required':['config_hash','algorithm','mode','trace','scope','seed_role'],'operating_failures_excluded_from_performance':True}))

def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader(Path(args.table).open(encoding='utf-8')))
    groups={}
    for row in rows:
        if row['performance_eligible']!='True':continue
        key=(row['config_hash'],row['algorithm'],row['mode'],row['trace'],row['scope'],row['seed_role'])
        groups.setdefault(key,[]).append(row)
    fig,ax=plt.subplots()
    for key,values in groups.items():
        byepisode={}
        for row in values:byepisode.setdefault(int(row['episode']),[]).append(float(row[args.metric]))
        xs=sorted(byepisode);mean=np.array([np.mean(byepisode[x]) for x in xs]);std=np.array([np.std(byepisode[x],ddof=1) if len(byepisode[x])>1 else 0. for x in xs])
        ax.plot(xs,mean,marker='o',label='/'.join(key[1:]));ax.fill_between(xs,mean-std,mean+std,alpha=.15)
    ax.set(xlabel='Episode',ylabel=args.metric,title='AT2-HJO measured results');ax.legend();fig.tight_layout()
    if Path(args.output).exists():raise FileExistsError(args.output)
    fig.savefig(args.output,dpi=180);plt.close(fig)

def main(argv=None):
    parser=argparse.ArgumentParser(description='AT2-HJO public core; results labelled by profile and verification scope')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('train','resume','evaluate'):
        p=sub.add_parser(name);p.add_argument('--config');p.add_argument('--checkpoint',required=name!='train');p.add_argument('--seed',type=int);p.add_argument('--episodes',type=int);p.add_argument('--device',default='auto');p.add_argument('--output',required=True);p.add_argument('--algorithm',choices=['joint','lt','st','at2-hjo','improved-dqn','improved-td3'])
        if name=='resume':p.add_argument('--trust-checkpoint',action='store_true',help='Permit recovery-state deserialization only for a trusted, checksum-verified checkpoint')
        if name=='evaluate':p.add_argument('--dataset',help='Hash-verified materialized episode-0 input; one evaluation episode only')
    p=sub.add_parser('generate');p.add_argument('--config',required=True);p.add_argument('--seed',required=True,type=int);p.add_argument('--output',required=True)
    p=sub.add_parser('validate-data');p.add_argument('--dataset',required=True)
    p=sub.add_parser('doctor');p.add_argument('--device',default='auto')
    p=sub.add_parser('aggregate');p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('plot');p.add_argument('--table',required=True);p.add_argument('--output',required=True);p.add_argument('--metric',choices=['P0','ST_operating_cost','penalized_cost'],default='P0')
    p=sub.add_parser('audit');p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    if getattr(args,'algorithm',None):args.algorithm={'at2-hjo':'joint','improved-dqn':'lt','improved-td3':'st'}.get(args.algorithm,args.algorithm)
    if args.command=='generate':generate(args)
    elif args.command=='validate-data':
        out=Path(args.dataset);d=json.loads((out/'manifest.json').read_text());assert d['science']==SCIENCE
        for name,hash_ in d['files'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==hash_
        a=np.load(out/'trace.npz',allow_pickle=False);assert len(a['links'])==d['horizon'];assert np.isfinite(a['links']).all();print('hash/schema checks passed')
    elif args.command=='doctor':
        device=DeviceManager(args.device);description=device.describe()
        if device.device.type=='cuda':description['gpu_name']=torch.cuda.get_device_name(device.device)
        print(json.dumps({'version':VERSION,'science':SCIENCE,'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':description}))
    elif args.command=='aggregate':aggregate(args)
    elif args.command=='plot':plot(args)
    elif args.command=='audit':
        from .reporting import audit_episode
        run=Path(args.run);config=json.loads((run/'config.json').read_text());records=[json.loads(line) for line in (run/'slots.jsonl').read_text().splitlines()];intervals=[json.loads(line) for line in (run/'intervals.jsonl').read_text().splitlines()]
        audits=[]
        for summary in json.loads((run/'episode_metrics.json').read_text()):
            index=summary['episode']-1;result=audit_episode([r for r in records if r['episode']==index],config,summary,[r for r in intervals if r['episode']==index]);audits.append({'episode':index+1,**result})
        passed=all(a['passed'] for a in audits);write(args.output,{'passed':passed,'episodes':audits})
        if not passed:raise ValueError('Raw component audit failed; inspect '+args.output)
        print(json.dumps({'passed':passed,'episodes':len(audits)}))
    else:
        if args.command=='train' and not args.config:parser.error('train requires --config')
        execute(args)

if __name__=='__main__':main()
