"""Executed-action TD3, residual DQN and discounted SMDP regulator."""
from collections import deque
from copy import deepcopy
import numpy as np
import torch
from .networks import Models
from .physics import torch_thermal
from .config import stream

LEARNER_VERSION='revision4-learning-v2-full-boundary-disjoint-replay'

class Learner:
    def __init__(self,env,device,seed):
        self.env=env;self.c=env.c;self.p=env.p;self.device=torch.device(device);self.seed=seed
        self.state_dim=len(env.observe());self.interval_coordinate=env.layout['clock']['start']+2
        self.boundary_state_dim=self.state_dim-1
        self.models=Models(self.state_dim,env.context_dim,self.p,self.c,seed,boundary_state_dim=self.boundary_state_dim).to(self.device)
        self.optim={name:torch.optim.Adam(getattr(self.models,name).parameters(),lr=lr) for name,lr in [('actor',self.c['actor_lr']),('q1',self.c['critic_lr']),('q2',self.c['critic_lr']),('lt',self.c['lr_min']),('tr',self.c['tr_lr']),('value',self.c['tr_lr'])]}
        self.replay_st=deque(maxlen=self.c['replay']);self.replay_lt=deque(maxlen=self.c['replay'])
        self.rng={name:stream(seed,0,name) for name in ('exploration','replay_st','replay_lt','target_noise','tr_sample','lt_exploration')}
        self.counters={'environment_steps':0,'critic_updates':0,'actor_updates':0,'target_updates':0,'dqn_updates':0,'tr_updates':0}
        self.last_losses={}

    def t(self,a):return torch.as_tensor(a,dtype=torch.float32,device=self.device)

    def update(self,name,loss):
        if not torch.isfinite(loss):raise FloatingPointError(name+' nonfinite loss')
        opt=self.optim[name];opt.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(getattr(self.models,name).parameters(),self.c['grad_clip'],error_if_nonfinite=True);opt.step()
        self.last_losses[name]=float(loss.detach())
        self.last_losses[name+'_gradient_norm']=float(norm.detach())

    def boundary_state(self,x):
        """Eq. lt_state with only the selected interval coordinate removed."""
        value=self.t(x)
        if value.shape[-1]!=self.state_dim or not torch.isfinite(value).all():
            raise ValueError('finite full boundary observation required')
        return torch.cat([value[...,:self.interval_coordinate],value[...,self.interval_coordinate+1:]],-1)

    def choose_interval(self,x,g,training):
        with torch.no_grad():probs=torch.softmax(self.models.tr(self.boundary_state(x),self.t(g)),-1).cpu().numpy()
        index=int(self.rng['tr_sample'].choice(3,p=probs)) if training else int(probs.argmax())
        return self.c['lt_intervals'][index],index

    def choose_st(self,state,training):
        if training and self.counters['environment_steps']<self.c['warmup']:
            raw=np.zeros(self.p.action_dim);mask,_=self.p.feasible(self.env.snapshot());schedule=raw[:self.p.sizes[0]].reshape(self.p.m,self.p.n);load=np.zeros(self.p.n)
            for m in range(self.p.m):
                feasible=np.flatnonzero(mask[m])
                if len(feasible):n=min(feasible,key=lambda n:(load[n],n));schedule[m,n]=1;load[n]+=1
            return raw
        with torch.no_grad():raw=self.models.actor(self.t(state)).cpu().numpy()
        if training:raw+=self.rng['exploration'].normal(0,self.c['exploration_sigma'],raw.shape)
        raw[:self.p.sizes[0]]=np.clip(raw[:self.p.sizes[0]],-5,5);raw[self.p.sizes[0]:]=np.clip(raw[self.p.sizes[0]:],-1,1)
        return raw

    def prior(self,target,snapshot,dep_descriptors):
        c=self.c;cost=c['physical']['costs'];installed=snapshot['installed'];pending=np.logical_and(target,~installed)
        remaining=np.where(dep_descriptors[...,0]>0,dep_descriptors[...,2],1.)
        estimate=cost['beta_deployment_latency_per_s']*c['slot_s']*np.sum(pending*remaining)
        temp=np.sum(np.maximum(snapshot['theta']-70,0)*(target.sum(1)>0))/10
        miss=float(np.sum(self.env.pop*np.logical_not(np.any(installed & target,axis=0))))
        return -float(np.dot(c['prior_weights'],[estimate/cost['xref_deployment'],temp/cost['xref_temperature'],miss]))

    def candidates(self):
        e=self.env;target=e.dep.target;pop=e.pop;storage=e.dep.storage;sizes=e.dep.sizes;raw=[target.copy()]
        for n in range(e.n):
            held=np.flatnonzero(target[n]);absent=np.flatnonzero(~target[n]);hot=np.argsort(-e.theta)
            if len(absent):
                k=sorted(absent,key=lambda k:(-pop[k],k))[0];a=target.copy();a[n,k]=1;raw.append(a)
                if len(held):a=target.copy();a[n,min(held,key=lambda k:(pop[k],k))]=0;a[n,k]=1;raw.append(a)
            if len(held):
                k=max(held,key=lambda k:(pop[k],-k));cool=int(np.argmin(e.theta));a=target.copy();a[n,k]=0;a[cool,k]=1;raw.append(a)
                a=target.copy();a[n,min(held,key=lambda k:(pop[k],k))]=0;raw.append(a)
        values=[];seen=set();snap=e.snapshot();desc=e.dep.descriptors()
        for a in raw:
            key=a.tobytes()
            if key not in seen and np.all(a@sizes<=storage):seen.add(key);values.append((a.astype(np.float32),self.prior(a,snap,desc)))
        return values[:64]

    def choose_lt(self,state,candidates,training,episode):
        epsilon=max(self.c['epsilon_end'],self.c['epsilon_start']-(self.c['epsilon_start']-self.c['epsilon_end'])*episode/self.c['epsilon_decay_episodes'])
        if training and self.rng['lt_exploration'].random()<epsilon:i=int(self.rng['lt_exploration'].integers(len(candidates)))
        else:
            x=self.t(np.stack([np.r_[state,a.ravel()] for a,prior in candidates]));prior=self.t([p for a,p in candidates])
            with torch.no_grad():i=int((self.models.lt(x).squeeze(-1)+prior).argmax())
        return candidates[i]

    def tr_update(self,x,g,index,interval,next_x,next_g):
        m=self.models;bx=self.boundary_state(x);v=m.value(torch.cat([bx,self.t(g).flatten()])).squeeze()
        with torch.no_grad():nv=0. if interval['terminal'] else float(m.value(torch.cat([self.boundary_state(next_x),self.t(next_g).flatten()])).squeeze())
        target=interval['reward']+self.c['gamma']**interval['duration']*nv
        advantage=self.t(target)-v
        logp=torch.log_softmax(m.tr(bx,self.t(g)),-1)[index]
        self.update('tr',-self.c['gamma']**interval['start']*advantage.detach()*logp)
        self.update('value',.5*advantage.square());self.counters['tr_updates']+=1
        self.last_losses.update(tr_target=target,tr_advantage=float(advantage.detach()),tr_initial_discount=self.c['gamma']**interval['start'],tr_duration_discount=self.c['gamma']**interval['duration'])

    def dqn_update(self):
        if len(self.replay_lt)<max(self.c['dqn_warmup'],self.c['batch_lt']):return
        ids=self.rng['replay_lt'].choice(len(self.replay_lt),self.c['batch_lt'],replace=False);batch=[self.replay_lt[i] for i in ids]
        predictions=[];targets=[]
        for state,action,prior,interval,nextstate,nextcand in batch:
            pred=self.models.lt(self.t(np.r_[state,action.ravel()])).squeeze()+prior;predictions.append(pred)
            with torch.no_grad():
                nv=0.
                if not interval['terminal']:
                    x=self.t(np.stack([np.r_[nextstate,a.ravel()] for a,p in nextcand]));ps=self.t([p for a,p in nextcand]);nv=float((self.models.lt_target(x).squeeze(-1)+ps).max())
                targets.append(interval['reward']+self.c['gamma']**interval['duration']*nv)
        errors=self.t(targets)-torch.stack(predictions);mean=float(errors.detach().abs().mean())
        # Eq.(45)-(46): current pre-update minibatch error drives this update.
        rate=self.c['lr_min']+(self.c['lr_max']-self.c['lr_min'])*mean/(mean+self.c['epsilon_eta'])
        for group in self.optim['lt'].param_groups:group['lr']=rate
        self.last_losses.update(lt_preupdate_mean_abs_td=mean,lt_learning_rate=rate)
        self.update('lt',errors.square().mean());self.counters['dqn_updates']+=1
        if self.counters['dqn_updates']%self.c['dqn_target_sync']==0:self.models.lt_target.load_state_dict(self.models.lt.state_dict())

    def projected(self,raw,states,gradient=False):
        """Certified forward; proposal ST surrogate supplies policy gradients."""
        rows=[];penalties=[]
        for r,state in zip(raw,states):
            snap=self.env.decode(state);proposal,paths=self.p.proposal(r.detach().cpu().numpy(),snap)
            if self.c['mode']=='certified':act,_,_=self.p.certify(r.detach().cpu().numpy(),snap)
            else:act=proposal
            certified=self.t(self.p.vector(act))
            if not gradient:rows.append(certified);continue
            # Predict SI quantities in float64, retaining the edge to the live
            # actor. The critic still receives its float32 executed-action
            # representation, with exactly the certificate's forward value.
            physical=r.to(torch.float64)
            def tensor(value):return torch.as_tensor(value,dtype=physical.dtype,device=r.device)
            z,f,o,rf=torch.split(physical,self.p.sizes);z=z.reshape(self.p.m,self.p.n);f=f.reshape_as(z);o=o.reshape(self.p.m,self.p.b);rf=rf.reshape(self.p.b,self.p.s)
            mask,_=self.p.feasible(snap);mt=torch.as_tensor(mask,device=r.device)
            safe_mask=mt.clone();empty=~safe_mask.any(-1);safe_mask[empty,0]=True
            prob=torch.softmax(z.masked_fill(~safe_mask,-torch.inf),-1)*tensor(~empty)[:,None]
            schedule=tensor(proposal['schedule'])+(prob-prob.detach())
            cp=torch.exp(f.clamp(-50,50))*schedule;cp=cp/(1+cp.sum(0,keepdim=True))*tensor(self.p.capacity)
            cp=tensor(proposal['cpu'])+(cp-cp.detach())
            def bandwidth(value,mask):
                mt=torch.as_tensor(mask,device=r.device)
                maximum=value.masked_fill(~mt,-torch.inf).max(0,keepdim=True).values
                maximum=torch.where(torch.isfinite(maximum),maximum,torch.zeros_like(maximum))
                exp=torch.exp((value-maximum).masked_fill(~mt,-torch.inf))
                return exp/exp.sum(0,keepdim=True).clamp_min(1e-30)
            optical=bandwidth(o,snap['links'][:self.p.m,self.p.m:self.p.m+self.p.b,4]>0)
            radio=bandwidth(rf,snap['links'][self.p.m:self.p.m+self.p.b,self.p.m+self.p.b:self.p.m+self.p.n,4]>0)
            optical=tensor(proposal['optical'])+(optical-optical.detach())
            radio=tensor(proposal['rf'])+(radio-radio.detach())
            vector=torch.cat([schedule.flatten(),(cp/tensor(self.p.capacity)).flatten(),optical.flatten(),radio.flatten()]).to(r.dtype)
            rows.append(certified+(vector-vector.detach()))
            compute=(tensor(snap['tasks'][:,1,None])*cp.square()*tensor(self.p.kappa)*schedule).sum(0)
            power=tensor(self.p.idle)+compute/self.c['slot_s']
            thermal_context={'ambient_c':tensor(snap['ambient'])} if 'ambient' in snap else {}
            theta,_=torch_thermal(tensor(snap['theta']),tensor(snap['radiator']),power,tensor(snap['heat']),self.c,**thermal_context)
            comm=physical.new_zeros(self.p.n);latency=[physical.new_zeros(()) for _ in range(self.p.m)]
            selected=list(zip(*np.nonzero(proposal['schedule'])));counts={}
            for mn in selected:
                for hop in paths[mn]:counts[hop]=counts.get(hop,0)+schedule[mn]
            for m,n in selected:
                t=physical.new_zeros(())
                for i,j in paths[m,n]:
                    link=snap['links'][i,j];bw=tensor(link[1]);rho=self.c['physical']['deployment']['reserved_fraction']
                    if i<self.p.m:bw=bw*optical[i,j-self.p.m]
                    elif (i-self.p.m<self.p.b)!=(j-self.p.m<self.p.b):
                        b=(i if i-self.p.m<self.p.b else j)-self.p.m;s=(j if i-self.p.m<self.p.b else i)-self.p.m-self.p.b;bw=bw*(1-rho)*radio[b,s]
                    else:bw=bw*(1-rho)
                    rate=bw*torch.log2(1+tensor(link[0])/(self.c['physical']['channels']['noise_psd_w_hz']*bw.clamp_min(1e-30)))
                    time=tensor(snap['tasks'][m,0])*counts[i,j]/rate.clamp_min(1e-30);t=t+time
                    index_i=i-self.p.m;index_j=j-self.p.m
                    if 0<=index_i<self.p.n:comm=comm+torch.nn.functional.one_hot(torch.tensor(index_i,device=r.device),self.p.n)*schedule[m,n]*link[2]*time
                    if 0<=index_j<self.p.n:comm=comm+torch.nn.functional.one_hot(torch.tensor(index_j,device=r.device),self.p.n)*schedule[m,n]*link[3]*time
                latency[m]=latency[m]+schedule[m,n]*(t+tensor(snap['tasks'][m,1])/cp[m,n].clamp_min(1e-30))
            # Eq. st_penalty retains y*W + sum(alpha*T) - y*Tmax.
            # Empty/inactive scheduling rows are zero and incur no deadline
            # residual. Thermal/energy terms include the reserved deployment.
            y=schedule.sum(-1);deadline=tensor(snap['tasks'][:,3]);active=tensor(snap['present'] & (snap['tasks'][:,0]>0))
            deadline_phi=(active*torch.relu(y*tensor(snap['wait'])+torch.stack(latency)-y*deadline)/deadline.clamp_min(1e-30)).sum()
            use=tensor(self.p.idle)*self.c['slot_s']+tensor(snap['dep_energy'])+compute+comm
            phi=deadline_phi+torch.relu(theta-80).sum()/10+(torch.relu(use-tensor(snap['energy'])+tensor(self.p.reserve))/(tensor(snap['energy'])+1e-9)).sum()
            # NumPy execution may round exp/resource fractions at the actor's
            # dtype. Preserve the proposal evaluator's forward residual while
            # differentiating its SI formula through the live proposal.
            phi=tensor(self.p.evaluate(proposal,snap,paths)['phi'])+(phi-phi.detach())
            penalties.append(phi)
        return torch.stack(rows),torch.stack(penalties).mean() if gradient else None

    def td3_update(self):
        if self.counters['environment_steps']<self.c['warmup'] or len(self.replay_st)<self.c['batch_st']:return
        ids=self.rng['replay_st'].choice(len(self.replay_st),self.c['batch_st'],replace=False);batch=[self.replay_st[i] for i in ids]
        states,actions,rewards,nextstates,done=zip(*batch);x=self.t(np.array(states));nx=self.t(np.array(nextstates));a=self.t(np.array(actions))
        with torch.no_grad():
            live=torch.as_tensor(np.logical_not(done),dtype=torch.bool,device=self.device)
            noise=self.rng['target_noise'].normal(0,self.c['target_sigma'],(len(batch),self.p.action_dim))
            noise[:,:self.p.sizes[0]]=0;noise=np.clip(noise,-self.c['target_clip'],self.c['target_clip'])
            target=self.t(rewards).clone()
            if live.any():
                raw=self.models.actor_target(nx[live])+self.t(noise)[live]
                raw[:,self.p.sizes[0]:].clamp_(-1,1)
                projected,_=self.projected(raw,np.array(nextstates)[np.logical_not(done)])
                target[live]+=self.c['gamma_st']*torch.minimum(self.models.q1_target(torch.cat([nx[live],projected],-1)).squeeze(-1),self.models.q2_target(torch.cat([nx[live],projected],-1)).squeeze(-1))
            self.last_losses.update(target_noise_schedule_absmax=float(np.abs(noise[:,:self.p.sizes[0]]).max()),
                                    target_noise_continuous_absmax=float(np.abs(noise[:,self.p.sizes[0]:]).max()),
                                    st_target_mean=float(target.mean()))
        for name in ('q1','q2'):self.update(name,(getattr(self.models,name)(torch.cat([x,a],-1)).squeeze(-1)-target).square().mean())
        self.counters['critic_updates']+=1
        if self.counters['critic_updates']%self.c['policy_delay']==0:
            projected,phi=self.projected(self.models.actor(x),np.array(states),gradient=True)
            for critic in (self.models.q1,self.models.q2):
                for p in critic.parameters():p.requires_grad_(False)
            loss=-self.models.q1(torch.cat([x,projected],-1)).mean()+self.c['rho']*phi
            try:self.update('actor',loss)
            finally:
                for critic in (self.models.q1,self.models.q2):
                    for p in critic.parameters():p.requires_grad_(True)
            self.last_losses.update(actor_proposal_phi=float(phi.detach()),actor_executed_Q=float(self.models.q1(torch.cat([x,projected.detach()],-1)).detach().mean()))
            with torch.no_grad():
                for name in ('actor','q1','q2'):
                    for target,p in zip(getattr(self.models,name+'_target').parameters(),getattr(self.models,name).parameters()):target.mul_(1-self.c['tau']).add_(p,alpha=self.c['tau'])
            self.counters['actor_updates']+=1;self.counters['target_updates']+=1

    def state(self,resume=True):
        data={'learner_version':LEARNER_VERSION,'input_signature':self.input_signature(),
              'models':deepcopy(self.models.state_dict()),'counters':dict(self.counters),'seed':self.seed}
        if resume:data.update(optimizers={k:v.state_dict() for k,v in self.optim.items()},replay_st=list(self.replay_st),replay_lt=list(self.replay_lt),rng={k:v.bit_generator.state for k,v in self.rng.items()})
        return deepcopy(data)

    def input_signature(self):
        return dict(state_dimension=self.state_dim,boundary_state_dimension=self.boundary_state_dim,
                    removed_interval_coordinate=self.interval_coordinate,context_dimension=self.env.context_dim,
                    history_length=self.c['history'],action_slices=list(self.p.sizes),
                    critic_action='certified normalized execution; proposal surrogate gradient',
                    replay_streams=['replay_st','replay_lt'])

    def restore(self,data,resume=False):
        if data.get('learner_version')!=LEARNER_VERSION or data.get('input_signature')!=self.input_signature():
            raise ValueError('revision4 learner version/input signature mismatch; pre-fix checkpoints are incompatible')
        expected={k:tuple(v.shape) for k,v in self.models.state_dict().items()}
        actual={k:tuple(v.shape) for k,v in data.get('models',{}).items()}
        if expected!=actual:raise ValueError('revision4 checkpoint model shape mismatch')
        if resume:
            if set(data.get('rng',{}))!=set(self.rng):raise ValueError('revision4 checkpoint requires independent LT/ST replay RNG state')
            if set(data.get('optimizers',{}))!=set(self.optim):raise ValueError('revision4 resume requires every separate optimizer state')
        self.models.load_state_dict(data['models']);self.counters.update(data['counters'])
        if resume:
            for k,opt in self.optim.items():opt.load_state_dict(data['optimizers'][k])
            self.replay_st.clear();self.replay_lt.clear()
            self.replay_st.extend(data['replay_st']);self.replay_lt.extend(data['replay_lt'])
            for k,rng in self.rng.items():rng.bit_generator.state=data['rng'][k]
