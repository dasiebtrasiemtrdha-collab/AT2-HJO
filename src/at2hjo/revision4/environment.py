"""Persistent FIFO queues and causal LT/TR/ST state for manuscript 4.0."""
from copy import deepcopy
from collections import deque
import time
import numpy as np
from .inputs import Trace
from .physics import Projector,route,thermal,node_values

class Deployment:
    def __init__(self,c,initial,projector):
        self.c=c;self.p=projector;self.target=initial.copy();self.installed=initial.copy();self.jobs={};self.serial=0
        self.sizes=np.array(c['physical']['services']['sizes_bits'],float);self.n=projector.n;self.k=c['services']
        self.storage=node_values(c,'storage_bits','hardware','storage_bits')
        self.occupancy=np.zeros(self.n);self.buffer_limit=c['physical']['deployment']['relay_buffer_bits']
        self._engine=None;self._slot=0
        self._nodes=tuple([f'BN:{i}' for i in range(projector.b)]+
                          [f'LEO:{i}' for i in range(projector.s)])
        self._physical={node:projector.m+i for i,node in enumerate(self._nodes)}
        self._physical['GS:0']=projector.m+self.n

    def _ensure_engine(self):
        """Initialize after Environment has assigned its free installed C(0)."""
        if self._engine is not None:return
        from .deployment_engine import DeploymentEngine40
        p=self.c['physical']['deployment']
        limits=np.broadcast_to(np.asarray(self.buffer_limit,float),(self.n,))
        self._engine=DeploymentEngine40(self._nodes,self.sizes,self.storage,
            initial_installed=self.installed,gateway_ids=('GS:0',),
            relay_buffer_capacity_bits=dict(zip(self._nodes,limits)),
            write_rate_bps={node:p['write_rate_bps'] for node in self._nodes},
            write_energy_j_bit={node:p['write_energy_j_bit'] for node in self._nodes},
            node_order=self._nodes+('GS:0',))
        self._engine.boundary(self.target);self._engine.begin_slot(self._slot);self._sync()

    def _sync(self):
        e=self._engine;old=self.jobs;fresh={}
        positions={key:i+1 for queue in e.link_queues.values() for i,key in enumerate(queue)}
        for raw in e.pending_transfers:
            key=raw['destination_index'],raw['service_index']
            if key not in old:self.serial+=1
            ticket=old[key]['ticket'] if key in old else self.serial
            nodes=tuple(self._physical[node] for node in raw['route_nodes'])
            path=tuple(zip(nodes,nodes[1:]));h=raw['hop_index'];owners=set(raw['buffer_nodes'])
            source=raw['route_nodes'][h] if path and h<len(path) else None
            receiver=raw['route_nodes'][h+1] if path and h<len(path) else None
            owner=self._physical[source]-self.p.m if source in owners else -1
            reserved=self._physical[receiver]-self.p.m if receiver in owners else -1
            fresh[key]={'route':path,'hop':h,'residual':raw['residual_bits'],
                        'written':raw['installation_bits'],'owner':owner,'reserved':reserved,
                        'ticket':ticket,'queue_position':positions.get(key,0)}
        self.jobs=fresh;self.target=np.asarray(e.target,bool);self.installed=np.asarray(e.installed,bool)
        self.occupancy=np.array([e.relay_buffer_occupancy_bits.get(node,0.) for node in self._nodes])

    def select(self,matrix):
        raw=np.asarray(matrix)
        if raw.shape!=self.target.shape or not np.isin(raw,[0,1]).all():raise ValueError('storage infeasible')
        self._ensure_engine();self._engine.boundary(raw);self._sync()

    def _reserved_links(self,links):
        """Physical reserved rates supply TX/RX coefficients before write limits."""
        from .deployment_engine import DeploymentLink40
        p=self.c['physical']['deployment'];noise=self.c['physical']['channels']['noise_psd_w_hz']
        inverse={value:key for key,value in self._physical.items()};out={}
        for i in range(self.p.m,self.p.m+self.n+1):
            for j in range(self.p.m,self.p.m+self.n+1):
                if i==j or not np.any(links[i,j]):continue
                if i==self.p.m+self.n:rate=float(p['gateway_rate_bps'])
                else:
                    bw=float(links[i,j,1])*p['reserved_fraction']
                    if {inverse[i].split(':')[0],inverse[j].split(':')[0]}=={'BN','LEO'}:bw/=self.p.b
                    rate=float(bw*np.log2(1+links[i,j,0]/(noise*bw))) if bw>0 else 0.
                tx=float(links[i,j,2])/rate if rate>0 else 0.
                rx=float(links[i,j,3])/rate if rate>0 else 0.
                out[f'dep|{i}|{j}']=DeploymentLink40(inverse[i],inverse[j],rate,tx,rx,bool(links[i,j,4]))
        return out

    def preview(self,links,energy):
        """Uncommitted traffic receipt and next-slot state for the certificate.

        Only the returned clone exposes completions in t+1. The original keeps
        installed(t) for task scheduling and is unchanged if idle is infeasible.
        """
        q=deepcopy(self);q._ensure_engine();dt=float(self.c['slot_s'])
        p=self.c['physical']['deployment']
        q._engine.write_rate_bps={node:float(p['write_rate_bps']) for node in q._nodes}
        q._engine.write_energy_j_bit={node:float(p['write_energy_j_bit']) for node in q._nodes}
        affordable=np.maximum(0,np.asarray(energy)-self.p.reserve-self.p.idle*dt)
        r=q._engine.advance_slot(q._slot,dt,q._reserved_links(links),
                                energy_budget_j=dict(zip(q._nodes,affordable)))
        q._slot+=1;q._engine.begin_slot(q._slot);q._sync()
        communication=np.array([r.communication_energy_by_node_j.get(node,0.) for node in q._nodes])
        writing=np.array([r.write_energy_by_node_j.get(node,0.) for node in q._nodes])
        written=np.zeros(q.n)
        for (n,k),bits in r.written_bits_by_image.items():written[n]+=bits
        carried=[{'hop':[int(i) for i in key.split('|')[1:]],'bits':float(bits)}
                 for key,bits in r.bits_by_link.items()]
        events=[{'n':event['node_index'],'k':event['service_index'],
                 'hop':[q._physical[event['src']],q._physical[event['dst']]],
                 'bits':event['bits'],'start_s':event['start_s'],'end_s':event['end_s'],
                 'hop_index':event['hop_index'],'hop_complete':event['hop_complete'],
                 'final_hop':event['final_hop']} for event in r.events]
        return q,dict(node_energy=communication+writing,total_energy=r.total_energy_j,
                      node_communication_energy=communication,node_write_energy=writing,
                      pending=int(round(r.pending_residence_s/dt)),pending_residence_s=r.pending_residence_s,
                      written_bits=written,carried_bits=carried,events=events,
                      written_bits_by_image=[{'n':n,'k':k,'bits':bits}
                                             for (n,k),bits in r.written_bits_by_image.items()],
                      completed_images=[list(key) for key in r.completed_images])

    def descriptors(self):
        # Fixed axes; complete route, hop residual, installation and FIFO/buffer ownership.
        width=8+self.n+1;d=np.zeros((self.n,self.k,width))
        for (n,k),j in self.jobs.items():
            path=j['route'];nodes=([path[0][0]]+[h[1] for h in path]) if path else []
            d[n,k,:8]=[1,j['hop'],j['residual']/self.sizes[k],j['written']/self.sizes[k],j['owner']+1,j['reserved']+1,j.get('queue_position',0),len(nodes)]
            d[n,k,8:8+len(nodes)]=np.array(nodes)+1
        return d

    def state_dict(self):
        self._ensure_engine()
        return {'engine':self._engine.state_dict(),'slot':self._slot,'serial':self.serial,
                'jobs':[{'n':n,'k':k,'ticket':j['ticket']} for (n,k),j in sorted(self.jobs.items())]}

    def load_state_dict(self,state):
        from .deployment_engine import DeploymentEngine40
        engine=DeploymentEngine40.from_state_dict(state['engine'])
        if engine.node_ids!=self._nodes or engine.service_sizes_bits!=tuple(self.sizes) or engine.storage_capacity_bits!=tuple(self.storage):
            raise ValueError('deployment checkpoint configuration mismatch')
        self._engine=engine;self._slot=int(state['slot']);self.serial=int(state['serial'])
        if state['engine']['current_slot']!=self._slot:raise ValueError('deployment checkpoint clock mismatch')
        self.jobs={(j['n'],j['k']):{'ticket':j['ticket']} for j in state['jobs']};self._sync()

class Environment:
    def __init__(self,c,seed,episode=0,dataset=None):
        self.c=c;self.seed=seed;self.episode=episode;self.t=0;self.trace=Trace(c,seed,episode);self.p=Projector(c)
        if dataset:
            from .dataset import RecordedTrace
            if episode!=0:raise ValueError('Materialized one-episode dataset supports episode 0 only')
            self.trace=RecordedTrace(dataset,c,seed)
        self.m=self.p.m;self.n=self.p.n;self.k=c['services'];self.queues=[deque() for _ in range(self.m)]
        self.theta=node_values(c,'initial_c','thermal','initial_c')
        self.radiator=np.full(self.p.s,c['physical']['satellite_radiation']['radiator_initial_c'])
        self.energy=node_values(c,'initial_j','energy','initial_j')
        self.dep=Deployment(c,np.zeros((self.n,self.k),bool),self.p)
        # C(0) is explicit and installed. No lower bound on future selections.
        initial=np.zeros((self.n,self.k),bool)
        for k in range(self.k):
            for n in [(k+j)%self.n for j in range(self.n)]:
                if (initial[n]@self.dep.sizes)+self.dep.sizes[k]<=self.dep.storage[n]:initial[n,k]=True;break
        self.dep.target=initial.copy();self.dep.installed=initial.copy()
        pop=np.arange(1,self.k+1)**(-.8);self.pop=pop/pop.sum();self.prev_reward=0.;self.prev_links=np.zeros(4)
        self.context_dim=self.k+4+self.n+2*self.p.s+1;self.history=np.zeros((c['history'],self.context_dim))
        self.req_acc=np.zeros(self.k);self.link_acc=np.zeros(4);self.reward_acc=0.;self.interval_rewards=[];self.interval_start=0;self.remaining=0;self.boundaries=0
        self.counts=dict(offered=0,completed=0,dropped=0,expiry=0,overflow=0,terminal_drops=0,operating_failures=0,waiting_node_slots=0)
        self.sums=dict(task=0.,temperature=0.,deployment=0.,latency_completed=0.,energy_completed=0.,phi_proposal=0.,reward_st=0.,actual_phi=0.)
        self.terminal=False;self.prepare()
        self.peak_temperature=float(self.theta.max());self.deadline_violations=0;self.temperature_violations=0;self.energy_violations=0
        self._layout=None

    def prepare(self):
        self.pre_drop=np.zeros(self.m);self.arrival_counts=self.trace.new_counts.copy()
        for m,q in enumerate(self.queues):
            kept=deque()
            for task in q:
                if (self.t-task[4])*self.c['slot_s']>=task[3]:self.pre_drop[m]+=1;self.counts['expiry']+=1
                else:kept.append(task)
            self.queues[m]=kept;q=kept
            for task in self.trace.arrivals[m]:
                self.counts['offered']+=1
                if len(q)<self.c['queue_capacity']:q.append(task)
                else:self.pre_drop[m]+=1;self.counts['overflow']+=1
        self.counts['dropped']+=int(self.pre_drop.sum())
        self.tasks=np.array([q[0] if q else [0,0,0,0,0] for q in self.queues],float)
        self.wait=np.where(self.tasks[:,0]>0,(self.t-self.tasks[:,4])*self.c['slot_s'],0.)
        self.preview_dep,self.dep_receipt=self.dep.preview(self.trace.links,self.energy)

    def snapshot(self):
        return dict(tasks=self.tasks.copy(),wait=self.wait.copy(),present=self.trace.present.copy(),installed=self.dep.installed.copy(),links=self.trace.links.copy(),theta=self.theta.copy(),radiator=self.radiator.copy(),heat=self.trace.heat.copy(),ambient=np.array(getattr(self.trace,'ambient',np.full(self.p.b,25.)),copy=True),energy=self.energy.copy(),dep_energy=self.dep_receipt['node_energy'].copy())

    def context(self):
        return np.r_[self.pop,self.prev_links,self.theta/80,self.radiator/80,self.trace.heat/100,self.prev_reward/100]

    def boundary(self):
        self.history=np.roll(self.history,-1,axis=0);self.history[-1]=self.context();self.interval_start=self.t;self.boundaries+=1
        self.req_acc[:]=0;self.link_acc[:]=0;self.reward_acc=0.;self.interval_rewards=[]
        self.remaining=0
        return self.observe(),self.history.copy()

    def apply(self,duration,target):
        if self.remaining:raise RuntimeError('LT decisions only at a boundary')
        self.remaining=min(int(duration),self.c['horizon']-self.t);self.requested=int(duration)
        self.dep.select(target);self.preview_dep,self.dep_receipt=self.dep.preview(self.trace.links,self.energy)
        return self.observe()

    def observe(self):
        scale_tasks=np.array([2e6,2e9,max(1,self.k-1),self.c['slot_s']*4,self.c['horizon']])
        q=np.zeros((self.m,self.c['queue_capacity'],5))
        for m,queue in enumerate(self.queues):
            if queue:q[m,:len(queue)]=np.array(queue)/scale_tasks
        links=self.trace.links.copy()
        # A single linear RF-scale divisor made UOWC and gateway powers dominate
        # the neural state by 10^7--10^10.  Encode, rather than alter, physical W.
        links[:,:,0]=np.log1p(links[:,:,0]/1e-20)/50
        links[:,:,1]/=1e8;links[:,:,2:4]/=20
        # All simulator drivers and controller accumulators are retained, not pooled away.
        pieces={'tasks':self.tasks/scale_tasks,'wait':self.wait/(4*self.c['slot_s']),'present':self.trace.present,'target':self.dep.target,'installed':self.dep.installed,'deployment_queue':self.dep.descriptors(),'relay_occupancy':self.dep.occupancy/self.dep.buffer_limit,'storage_remaining':(self.dep.storage-self.dep.target@self.dep.sizes)/self.dep.storage,'links':links,'theta':self.theta/80,'radiator':self.radiator/80,'heat':self.trace.heat/100,'ambient':np.array(getattr(self.trace,'ambient',np.full(self.p.b,25.)))/80,'energy':self.energy/1e6,'dep_energy':self.dep_receipt['node_energy']/1e3,'task_queues':q,'exogenous':self.trace.drivers(),'popularity':self.pop,'controller_history':self.history,'request_accumulator':self.req_acc/1000,'link_accumulator':self.link_acc/100,'reward_accumulator':np.array([self.reward_acc/1e4]),'pre_drop':self.pre_drop/8,'clock':np.array([self.t/self.c['horizon'],self.interval_start/self.c['horizon'],self.remaining/15,self.boundaries/self.c['horizon'],self.prev_reward/100,*self.prev_links])}
        layout={};values=[];pos=0
        for name,arr in pieces.items():
            a=np.asarray(arr,float);layout[name]={'start':pos,'end':pos+a.size,'shape':list(a.shape)};values.append(a.ravel());pos+=a.size
        extra=np.array([sum(self.c['gamma']**i*r for i,r in enumerate(self.interval_rewards))/1e4,len(self.interval_rewards)/15,self.dep.serial/10000])
        layout['discounted_controller_accumulator']={'start':pos,'end':pos+3,'shape':[3]};values.append(extra)
        self.layout=layout
        result=np.concatenate(values).astype(np.float32)
        if not np.isfinite(result).all():raise FloatingPointError('nonfinite observation')
        return result

    def decode(self,state):
        def read(name):
            d=self.layout[name];return np.array(state[d['start']:d['end']],float).reshape(d['shape'])
        scales=np.array([2e6,2e9,max(1,self.k-1),self.c['slot_s']*4,self.c['horizon']])
        tasks=read('tasks')*scales;tasks[:,2]=np.rint(tasks[:,2]);links=read('links');links[:,:,0]=np.expm1(links[:,:,0]*50)*1e-20;links[:,:,1]*=1e8;links[:,:,2:4]*=20
        return dict(tasks=tasks,wait=read('wait')*4*self.c['slot_s'],present=read('present')>.5,installed=read('installed')>.5,links=links,theta=read('theta')*80,radiator=read('radiator')*80,heat=read('heat')*100,ambient=read('ambient')*80,energy=read('energy')*1e6,dep_energy=read('dep_energy')*1e3)

    def step(self,raw):
        if self.terminal or self.remaining<1:raise RuntimeError('deployment/interval decision required')
        snap=self.snapshot();decision_started=time.perf_counter();proposal,paths=self.p.proposal(raw,snap);proposal_eval=self.p.evaluate(proposal,snap,paths)
        action,ev,failure=self.p.certify(raw,snap) if self.c['mode']=='certified' else (proposal,proposal_eval,False)
        certificate_ms=(time.perf_counter()-decision_started)*1000
        if not np.isfinite(ev['phi']):raise FloatingPointError('invalid resource-only action; zero-resource path')
        before=[len(q) for q in self.queues]; drops=int(self.pre_drop.sum());y=ev['y']
        if failure:
            self.counts['operating_failures']+=1;y=np.zeros(self.m);ev['y']=y
        else:
            self.dep=self.preview_dep;self.theta=ev['theta'];self.radiator=ev['radiator'];self.energy-=ev['use']
            for m in np.flatnonzero(y):self.queues[m].popleft()
            self.counts['completed']+=int(y.sum())
        terminal=self.t+1>=self.c['horizon'] or failure
        self.peak_temperature=max(self.peak_temperature,float(self.theta.max()))
        if not failure:
            self.deadline_violations+=int(np.count_nonzero((ev['latency']+self.wait>self.tasks[:,3]+1e-7)&(y>0)))
            self.temperature_violations+=int(np.count_nonzero(self.theta>80+1e-7))
            self.energy_violations+=int(np.count_nonzero(self.energy<self.p.reserve-1e-7))
        terminal_drop=sum(len(q) for q in self.queues) if terminal else 0
        drops+=terminal_drop;self.counts['terminal_drops']+=terminal_drop;self.counts['dropped']+=terminal_drop
        costs=self.c['physical']['costs'];dt=self.c['slot_s'];holding=dt*(sum(before)-y.sum())
        latency=float(np.sum((ev['latency']+self.wait)*y)) if not failure else 0.
        task_energy=float(np.sum(ev['task_energy']*y)) if not failure else 0.
        task=costs['beta_task_latency_per_s']*(latency+holding+4*dt*drops)+costs['beta_task_energy_per_j']*task_energy
        temp=float(np.maximum(snap['theta']-70,0).sum()/10)
        dep=costs['beta_deployment_latency_per_s']*dt*self.dep_receipt['pending']+costs['beta_deployment_energy_per_j']*(0. if failure else self.dep_receipt['total_energy'])
        w=self.c['weights'];reward=-(w[0]*task/costs['xref_task']+w[1]*temp/costs['xref_temperature'])
        # In certified mode rho is auxiliary actor regularization, never in reward.
        if self.c['mode']=='resource_only':reward-=self.c['rho']*proposal_eval['phi']
        full_reward=reward-w[2]*dep/costs['xref_deployment'];self.interval_rewards.append(full_reward)
        self.reward_acc+=reward;self.req_acc+=self.arrival_counts
        links=self.trace.links;stats=np.array([links[:self.m,self.m:self.m+self.p.b,4].mean(),links[self.m:self.m+self.p.b,self.m:self.m+self.p.b,4].mean(),links[self.m:self.m+self.p.b,self.m+self.p.b:self.m+self.n,4].mean(),links[self.m+self.p.b:self.m+self.n,self.m+self.p.b:self.m+self.n,4].mean()])
        self.link_acc+=stats;self.counts['waiting_node_slots']+=int(np.count_nonzero(np.array(before)-y))
        for name,value in [('task',task),('temperature',temp),('deployment',dep),('latency_completed',latency),('energy_completed',task_energy),('phi_proposal',proposal_eval['phi'])]:self.sums[name]+=value
        record={'slot':self.t,'reward_st':reward,'cost_task':task,'cost_temperature':temp,'cost_deployment':dep,'holding_request_seconds':holding,'drops':drops,'offered':int(self.arrival_counts.sum()),'completed':int(y.sum()),'execution_indicator':y.tolist(),'accepted_latency_s':(ev['latency']+self.wait)[y>0].tolist(),'proposal_phi':proposal_eval['phi'],'operating_failure':failure,'certified':self.c['mode']=='certified','theta_before_c':snap['theta'].tolist(),'theta_after_c':self.theta.tolist(),'radiator_after_c':self.radiator.tolist(),'energy_use_j':(np.zeros(self.n) if failure else ev['use']).tolist(),'energy_after_j':self.energy.tolist(),'deployment':{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in self.dep_receipt.items()},'executed_action':self.p.vector(action).tolist()}
        record['proposal_and_certificate_ms']=certificate_ms
        record.update(
            raw_action=np.asarray(raw).tolist(),
            proposed_action=self.p.vector(proposal).tolist(),
            target_deployment=self.dep.target.astype(int).tolist(),
            installed_before=snap['installed'].astype(int).tolist(),
            cpu_allocation_hz=action['cpu'].tolist(),
            optical_bandwidth_share=action['optical'].tolist(),
            rf_bandwidth_share=action['rf'].tolist(),
            energy_before_j=snap['energy'].tolist(),
            actual_task_energy_j=task_energy,
            actual_completed_latency_s=latency,
            compute_energy_j=(np.zeros(self.n) if failure else ev['compute_energy']).tolist(),
            idle_energy_j=(np.zeros(self.n) if failure else self.p.idle*dt).tolist(),
            task_communication_energy_j=(np.zeros(self.n) if failure else ev['node_communication']).tolist(),
            computing_power_w=ev['power'].tolist(),
            task_queue_lengths=before,
            accepted_phi=float(ev['phi']) if not failure else None,
            residual_proposal={k:proposal_eval.get('phi_'+k) for k in ('deadline','thermal','energy')},
            task_metrics=[])
        record['actual_thermal_balance']=ev.get('thermal_balance') if not failure else None
        record['task_all_node_communication_j']=ev.get('all_node_communication',np.zeros(self.m+self.n+1)).tolist() if not failure else [0.]*(self.m+self.n+1)
        record['actual_residual']={k:ev.get('phi_'+k) if not failure else None for k in ('deadline','thermal','energy')}
        feasible_mask,all_paths=self.p.feasible(snap)
        for m,task_input in enumerate(snap['tasks']):
            if task_input[0]<=0:continue
            selected=np.flatnonzero(action['schedule'][m]); node=int(selected[0]) if len(selected) else None
            if node is not None and not failure:reason='executed'
            elif failure:reason='operating_exception'
            elif not snap['present'][m]:reason='waiting_outside_area'
            elif not snap['installed'][:,int(task_input[2])].any():reason='waiting_no_service'
            elif not feasible_mask[m].any():reason='waiting_no_feasible_route_or_energy'
            else:reason='waiting_joint_certificate_rejected'
            record['task_metrics'].append({'source':m,'service':int(task_input[2]),'input_bits':float(task_input[0]),'cycles':float(task_input[1]),'deadline_s':float(task_input[3]),'waiting_s':float(snap['wait'][m]),'status':reason,'execution_node':node,'cpu_hz':float(action['cpu'][m,node]) if node is not None and not failure else None,'path':[list(h) for h in all_paths.get((m,node),())] if node is not None else [],'processing_latency_s':float(ev['latency'][m]) if node is not None and not failure else None,'compute_latency_s':float(ev.get('compute_latency_s',np.zeros(self.m))[m]) if node is not None and not failure else None,'transmission_latency_s':float(ev.get('transmission_latency_s',np.zeros(self.m))[m]) if node is not None and not failure else None,'task_energy_j':float(ev['task_energy'][m]) if node is not None and not failure else None})
        self.sums['reward_st']+=reward
        self.sums['actual_phi']+=0. if failure else ev['phi']
        self.t+=1;self.remaining-=1;self.terminal=terminal
        if self.remaining==0 or terminal:
            d=len(self.interval_rewards);self.prev_reward=self.reward_acc/d;self.prev_links=self.link_acc/d
            self.pop=self.req_acc/self.req_acc.sum() if self.req_acc.sum() else self.pop
            self.last_interval={'duration':d,'start':self.interval_start,'requested':self.requested,'reward':float(sum(self.c['gamma']**i*r for i,r in enumerate(self.interval_rewards))),'terminal':terminal}
        if not terminal:self.trace.advance();self.prepare()
        return reward,terminal,record

    def summary(self):
        c=self.c['physical']['costs'];w=self.c['weights'];off=self.counts['offered'];done=self.counts['completed']
        assert not self.terminal or off==done+self.counts['dropped']
        p0=(w[0]*self.sums['task']/c['xref_task']+w[1]*self.sums['temperature']/c['xref_temperature']+w[2]*self.sums['deployment']/c['xref_deployment'])/self.c['horizon']
        return {**self.counts,'slots':self.t,'boundaries':self.boundaries,**self.sums,'mean_latency_completed_s':self.sums['latency_completed']/done if done else None,'mean_energy_completed_j':self.sums['energy_completed']/done if done else None,'completion_fraction':done/off if off else None,'drop_fraction':self.counts['dropped']/off if off else None,'P0':p0,'objective_p0':p0,'ST_operating_cost':-self.sums['reward_st']/max(1,self.t),'penalized_cost':p0+self.c['rho']*self.sums['actual_phi']/self.c['horizon'],'mode':self.c['mode'],'formal_result':False,'max_cpu_temperature_c':self.peak_temperature,'accepted_deadline_violations':self.deadline_violations,'node_temperature_violations':self.temperature_violations,'node_energy_violations':self.energy_violations}
