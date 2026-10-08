"""4.0 thermal, routing, proposal and joint execution certificate.

CPU power is slot-average completed-work energy / slot duration. BNs use the
effective air/water sink RC equation; satellites use the CPU/radiator ODE with
Celsius state and Kelvin radiation. Material, area, IR/albedo and integration
coefficients come from the explicit physical configuration, not paper targets.
"""
from collections import deque,Counter
import numpy as np
import torch

def node_values(c,name,default_group,default_key):
    """Per-node arrays with type-level defaults."""
    n=c['buoys']+c['satellites'];custom=c['physical'].get('node_parameters',{})
    if name in custom:
        value=np.asarray(custom[name],float)
        if value.shape!=(n,) or not np.isfinite(value).all():raise ValueError('per-node parameter axes: '+name)
    else:
        p=c['physical'][default_group]
        if name=='ambient_c':
            value=np.r_[np.full(c['buoys'],p['bn'].get(default_key,25.)),np.full(c['satellites'],p['leo'].get(default_key,20.))]
        else:value=np.r_[np.full(c['buoys'],p['bn'][default_key]),np.full(c['satellites'],p['leo'][default_key])]
    positive={'cpu_hz','storage_bits','rth_k_w','cth_j_k'}
    nonnegative={'idle_w','kappa','reserve_j','initial_j'}
    if not np.isfinite(value).all():raise ValueError('finite per-node parameters: '+name)
    if name in positive and np.any(value<=0):raise ValueError('positive per-node parameters: '+name)
    if name in nonnegative and np.any(value<0):raise ValueError('nonnegative per-node parameters: '+name)
    if name in {'ambient_c','initial_c'} and np.any(value<=-273.15):raise ValueError('temperature exceeds absolute zero: '+name)
    return value

def satellite_values(c,name):
    """Satellite-only coefficient vector; per-node overrides may be N or S."""
    b,s=c['buoys'],c['satellites'];custom=c['physical'].get('node_parameters',{})
    raw=np.asarray(custom.get(name,c['physical']['satellite_radiation'][name]),float)
    if raw.shape==(b+s,):raw=raw[b:]
    try:value=np.broadcast_to(raw,(s,)).copy()
    except ValueError as error:raise ValueError('satellite parameter axes: '+name) from error
    if not np.isfinite(value).all() or np.any(value<0):raise ValueError('finite nonnegative satellite coefficient: '+name)
    if name in {'radiator_capacity_j_k','sigma'} and np.any(value<=0):raise ValueError('positive satellite coefficient: '+name)
    if name in {'emissivity','solar_absorptivity'} and np.any(value>1):raise ValueError('satellite fraction in [0,1]: '+name)
    return value

def environmental_heat(position_eci,sun_direction,c):
    """Eq satellite_environmental_heat; freeze current geometry for one slot.

    Explicit completion: radial solar surface, parallel rays/cylindrical
    eclipse, nadir planar Earth view (Re/r)^2. Configured earth_ir_w/albedo_w
    are *absorbed powers before the view/illumination factors*, not fluxes.
    Albedo uses sub-satellite solar incidence as its illumination completion.
    """
    position=np.asarray(position_eci,float);sun=np.asarray(sun_direction,float)
    if position.shape!=(c['satellites'],3) or sun.shape!=(3,) or not np.isfinite(position).all() or not np.isfinite(sun).all():raise ValueError('finite satellite/Sun coordinate axes')
    radius=float(c['physical']['orbit']['earth_radius_m']);distance=np.linalg.norm(position,axis=1);norm=np.linalg.norm(sun)
    if not np.isfinite(radius) or radius<=0 or np.any(distance<=radius) or norm<=0:raise ValueError('Earth/Sun thermal geometry')
    sun=sun/norm;along=position@sun
    sunlight=~((along<0)&(np.linalg.norm(position-along[:,None]*sun,axis=1)<radius))
    incidence=np.clip(along/distance,0,1);view=(radius/distance)**2
    solar=satellite_values(c,'solar_absorptivity')*satellite_values(c,'solar_area_m2')*satellite_values(c,'irradiance_w_m2')*sunlight*incidence
    infrared=satellite_values(c,'earth_ir_w')*view;albedo=satellite_values(c,'albedo_w')*view*incidence
    total=solar+infrared+albedo
    if not np.isfinite(total).all():raise ValueError('environmental thermal power overflow')
    return dict(heat=total,solar=solar,infrared=infrared,albedo=albedo,sunlight=sunlight,incidence=incidence,earth_view=view)

def route(links,source,target,m,deployment=False,buoys=3):
    """Minimum-hop, lexicographic integer-node tie breaks; no future links."""
    queue=deque([(source,())]); seen={source}
    while queue:
        node,path=queue.popleft()
        if node==target:return path
        for dst in np.flatnonzero(links[node,:,4]):
            if dst in seen or dst<m:continue
            if not deployment and node==source and dst>=m+buoys:continue
            seen.add(int(dst));queue.append((int(dst),path+((node,int(dst)),)))
    return ()

def _thermal_coefficients(c):
    return dict(resistance=node_values(c,'rth_k_w','thermal','rth_k_w'),capacity=node_values(c,'cth_j_k','thermal','cth_j_k'),
        radiator_capacity=satellite_values(c,'radiator_capacity_j_k'),
        radiation_coefficient=satellite_values(c,'emissivity')*satellite_values(c,'sigma')*satellite_values(c,'radiating_area_m2'),
        background=satellite_values(c,'background_k'))

def _thermal_steps(theta,rad,power,heat,c,v):
    """Configured RK4 resolution, reduced further for custom fast thermal modes."""
    dt=float(c['slot_s']);limit=float(c['physical']['satellite_radiation']['integration_max_step_s'])
    if not np.isfinite([dt,limit]).all() or dt<=0 or limit<=0:raise ValueError('positive finite thermal slot/step duration')
    b=c['buoys']
    if c['satellites']:
        # A conservative temperature bound and conduction/radiation Jacobian
        # bound prevent a custom small R/C from destabilising explicit RK4.
        energy=(np.asarray(power)[...,b:]+np.asarray(heat))*dt
        upper=np.maximum(np.asarray(theta)[...,b:],np.asarray(rad))+273.15+energy/np.minimum(v['capacity'][b:],v['radiator_capacity'])
        rate=(1/v['resistance'][b:])*(1/v['capacity'][b:]+1/v['radiator_capacity'])+4*v['radiation_coefficient']*np.maximum(upper,0)**3/v['radiator_capacity']
        fastest=float(np.max(rate))
        if fastest>0:limit=min(limit,.5/fastest)
    steps=int(np.ceil(dt/limit))
    if steps>1000000:raise ValueError('thermal coefficients require more than one million RK4 substeps')
    return steps,dt/steps

def thermal(theta,radiator,power,heat,c,ambient_c=None,return_balance=False):
    """Exact BN RC and conservative satellite RK4 with fixed slot inputs.

    ``ambient_c`` is the current B-vector effective BN sink. The optional
    balance return reports the same RK4 quadrature used by the state update,
    making conduction cancellation and radiated joules directly auditable.
    """
    b,s=c['buoys'],c['satellites'];n=b+s;dt=c['slot_s']
    theta=np.asarray(theta,float);radiator=np.asarray(radiator,float);power=np.asarray(power,float);heat=np.asarray(heat,float)
    if theta.shape!=(n,) or power.shape!=(n,) or radiator.shape!=(s,) or heat.shape!=(s,):raise ValueError('thermal node axes')
    if not all(np.isfinite(x).all() for x in (theta,radiator,power,heat)) or np.any(power<0) or np.any(heat<0):raise ValueError('finite nonnegative thermal power inputs')
    if np.any(theta<=-273.15) or np.any(radiator<=-273.15):raise ValueError('thermal temperatures exceed absolute zero')
    ambient=node_values(c,'ambient_c','thermal','ambient_c')[:b] if ambient_c is None else np.asarray(ambient_c,float)
    if ambient.shape!=(b,) or not np.isfinite(ambient).all() or np.any(ambient<=-273.15):raise ValueError('finite B-vector ambient Celsius temperature')
    v=_thermal_coefficients(c);resistance=v['resistance'];capacitance=v['capacity'];steps,h=_thermal_steps(theta,radiator,power,heat,c,v)
    out=theta.copy();fraction=-np.expm1(-dt/(resistance[:b]*capacitance[:b]))
    out[:b]=theta[:b]+(ambient+resistance[:b]*power[:b]-theta[:b])*fraction
    cpu=theta[b:].copy();rad=radiator.copy();radiated=np.zeros(s);conducted=np.zeros(s)
    def rhs(x,y):
        conduction=(x-y)/resistance[b:];radiation=v['radiation_coefficient']*((y+273.15)**4-v['background']**4)
        return (power[b:]-conduction)/capacitance[b:],(conduction+heat-radiation)/v['radiator_capacity'],radiation,conduction
    for _ in range(steps):
        a=rhs(cpu,rad);e=rhs(cpu+h*a[0]/2,rad+h*a[1]/2);g=rhs(cpu+h*e[0]/2,rad+h*e[1]/2);k=rhs(cpu+h*g[0],rad+h*g[1])
        change=[h*(a[i]+2*e[i]+2*g[i]+k[i])/6 for i in range(4)]
        cpu+=change[0];rad+=change[1];radiated+=change[2];conducted+=change[3]
    out[b:]=cpu
    if not np.isfinite(out).all() or not np.isfinite(rad).all() or np.any(out<=-273.15) or np.any(rad<=-273.15):raise ValueError('invalid thermal transition')
    if not return_balance:return out,rad
    stored=capacitance[b:]*(cpu-theta[b:])+v['radiator_capacity']*(rad-radiator)
    balance=dict(ambient_c=ambient.copy(),radiated_energy_j=radiated,cpu_to_radiator_energy_j=conducted,
                 stored_energy_change_j=stored,energy_balance_residual_j=stored-((power[b:]+heat)*dt-radiated),integration_substeps=steps)
    return out,rad,balance

def torch_thermal(theta,rad,power,heat,c,ambient_c=None):
    """Differentiable version of the identical constant-input thermal update."""
    b,s=c['buoys'],c['satellites'];dt=c['slot_s'];v=_thermal_coefficients(c)
    if theta.shape[-1:]!=(b+s,) or power.shape[-1:]!=(b+s,) or rad.shape[-1:]!=(s,) or heat.shape[-1:]!=(s,):raise ValueError('torch thermal node axes')
    if not all(bool(torch.isfinite(x).all()) for x in (theta,rad,power,heat)) or bool((power<0).any()) or bool((heat<0).any()):raise ValueError('finite nonnegative torch thermal inputs')
    if bool((theta<=-273.15).any()) or bool((rad<=-273.15).any()):raise ValueError('thermal temperatures exceed absolute zero')
    steps,h=_thermal_steps(theta.detach().cpu().numpy(),rad.detach().cpu().numpy(),power.detach().cpu().numpy(),heat.detach().cpu().numpy(),c,v)
    tensor=lambda x:torch.as_tensor(x,device=theta.device,dtype=theta.dtype)
    resistance=tensor(v['resistance']);capacitance=tensor(v['capacity']);rcap=tensor(v['radiator_capacity']);rcoef=tensor(v['radiation_coefficient']);background=tensor(v['background'])
    ambient=tensor(node_values(c,'ambient_c','thermal','ambient_c')[:b]) if ambient_c is None else tensor(ambient_c)
    if ambient.shape[-1:]!=(b,) or not bool(torch.isfinite(ambient).all()) or bool((ambient<=-273.15).any()):raise ValueError('finite B-vector ambient Celsius temperature')
    fraction=-torch.expm1(-dt/(resistance[:b]*capacitance[:b]))
    buoy=theta[...,:b]+(ambient+resistance[:b]*power[...,:b]-theta[...,:b])*fraction
    cpu=theta[...,b:];r=rad
    def rhs(x,y):
        cond=(x-y)/resistance[b:];radiation=rcoef*((y+273.15)**4-background**4)
        return (power[...,b:]-cond)/capacitance[b:],(cond+heat-radiation)/rcap
    for _ in range(steps):
        a,d=rhs(cpu,r);e,f=rhs(cpu+h*a/2,r+h*d/2);g,j=rhs(cpu+h*e/2,r+h*f/2);k,l=rhs(cpu+h*g,r+h*j)
        cpu=cpu+h*(a+2*e+2*g+k)/6;r=r+h*(d+2*f+2*j+l)/6
    result=torch.cat([buoy,cpu],-1)
    if not bool(torch.isfinite(result).all()) or not bool(torch.isfinite(r).all()):raise ValueError('invalid torch thermal transition')
    return result,r

def softmax(x,mask,axis):
    # Physical capacity checks use double precision. Summing float32 network
    # shares as doubles can otherwise exceed one by a few ulps and reject idle.
    x=np.asarray(x,dtype=float);mask=np.asarray(mask,bool);v=np.where(mask,x,-1e30);v-=v.max(axis=axis,keepdims=True)
    exp=np.exp(v)*mask;return exp/np.maximum(exp.sum(axis=axis,keepdims=True),1e-30)

class Projector:
    def __init__(self,c):
        self.c=c;self.m=c['nodes'];self.b=c['buoys'];self.s=c['satellites'];self.n=self.b+self.s
        self.sizes=[self.m*self.n,self.m*self.n,self.m*self.b,self.b*self.s]
        self.action_dim=sum(self.sizes)
        self.capacity=node_values(c,'cpu_hz','hardware','cpu_hz')
        self.kappa=node_values(c,'kappa','thermal','kappa')
        self.idle=node_values(c,'idle_w','thermal','idle_w')
        self.reserve=node_values(c,'reserve_j','energy','reserve_j')

    def split(self,raw):
        z,f,o,r=np.split(np.asarray(raw,dtype=float),np.cumsum(self.sizes)[:-1]);return z.reshape(self.m,self.n),f.reshape(self.m,self.n),o.reshape(self.m,self.b),r.reshape(self.b,self.s)

    def feasible(self,snap):
        links=snap['links']; tasks=snap['tasks']; mask=np.zeros((self.m,self.n),bool); paths={}
        for m in range(self.m):
            if not tasks[m,0] or not snap['present'][m]:continue
            for n in range(self.n):
                path=route(links,m,self.m+n,self.m,buoys=self.b)
                if path and snap['installed'][n,int(tasks[m,2])] and snap['energy'][n]>self.reserve[n]:
                    mask[m,n]=True;paths[m,n]=path
        return mask,paths

    def allocate(self,raw,schedule,snap):
        z,f,o,r=self.split(raw);links=snap['links'];cpu=np.exp(np.clip(f,-50,50))*schedule
        cpu=cpu/(1+cpu.sum(axis=0,keepdims=True))*self.capacity
        optical=softmax(o,links[:self.m,self.m:self.m+self.b,4]>0,0)
        rf=softmax(r,links[self.m:self.m+self.b,self.m+self.b:self.m+self.n,4]>0,0)
        return dict(schedule=schedule.astype(float),cpu=cpu,optical=optical,rf=rf)

    def rates(self,action,snap):
        links=snap['links']; rates=np.zeros(links.shape[:2]);rho=self.c['physical']['deployment']['reserved_fraction'];noise=self.c['physical']['channels']['noise_psd_w_hz']
        for i,j in zip(*np.nonzero(links[:,:,4])):
            band=links[i,j,1]
            if i<self.m:band*=action['optical'][i,j-self.m]
            elif j==len(links)-1 or i==len(links)-1:continue
            elif (i-self.m<self.b)!=(j-self.m<self.b):
                b=(i if i-self.m<self.b else j)-self.m;s=(j if i-self.m<self.b else i)-self.m-self.b
                band*= (1-rho)*action['rf'][b,s]
            else:band*=1-rho
            if band>0:rates[i,j]=band*np.log2(1+links[i,j,0]/(noise*band))
        return rates

    def evaluate(self,action,snap,paths=None):
        if paths is None:_,paths=self.feasible(snap)
        schedule=np.asarray(action['schedule'],float);cpu=np.asarray(action['cpu'],float);tasks=np.asarray(snap['tasks'],float)
        if schedule.shape!=(self.m,self.n) or cpu.shape!=schedule.shape or not np.isin(schedule,[0,1]).all() or not np.isfinite(cpu).all() or np.any(cpu<0):raise ValueError('binary scheduling and finite nonnegative CPU axes')
        valid=bool(np.all(schedule.sum(1)<=1) and np.all(cpu.sum(0)<=self.capacity*(1+1e-12)) and not np.any(cpu[schedule==0]))
        for name,shape in (('optical',(self.m,self.b)),('rf',(self.b,self.s))):
            fraction=np.asarray(action[name],float)
            if fraction.shape!=shape or not np.isfinite(fraction).all() or np.any(fraction<0):raise ValueError('finite nonnegative bandwidth fraction axes: '+name)
            valid=valid and bool(np.all(fraction.sum(0)<=1+1e-12))
        rates=self.rates(action,snap)
        selected=list(zip(*np.nonzero(schedule)));counts=Counter(h for mn in selected for h in paths.get(mn,()))
        lat=np.zeros(self.m);cmp_lat=np.zeros(self.m);tx_lat=np.zeros(self.m);et=np.zeros(self.m)
        all_tx=np.zeros(len(snap['links']));all_rx=np.zeros_like(all_tx);hop_bits=np.zeros(snap['links'].shape[:2])
        # Explicit alpha masking prevents unexecuted CPU proposals from being
        # mistaken for completed work in the power/energy equations.
        compute=np.sum(self.kappa*tasks[:,1,None]*cpu**2*schedule,axis=0)
        if not np.isfinite(compute).all():raise ValueError('computing energy overflow')
        for m,n in selected:
            path=paths.get((m,n),());t=0.;energy=0.
            service=int(tasks[m,2]);eligible=bool(tasks[m,0]>0 and snap['present'][m] and 0<=service<self.c['services'] and snap['installed'][n,service])
            if not eligible or not path or cpu[m,n]<=0:valid=False;lat[m]=np.inf;cmp_lat[m]=np.inf;tx_lat[m]=np.inf;continue
            for i,j in path:
                rate=rates[i,j]/counts[i,j]
                if rate<=0:valid=False;t=np.inf;continue
                time=tasks[m,0]/rate;t+=time;tx=snap['links'][i,j,2]*time;rx=snap['links'][i,j,3]*time;energy+=tx+rx
                all_tx[i]+=tx;all_rx[j]+=rx;hop_bits[i,j]+=tasks[m,0]
            cmp_lat[m]=tasks[m,1]/cpu[m,n];tx_lat[m]=t
            lat[m]=t+cmp_lat[m];et[m]=energy+self.kappa[n]*tasks[m,1]*cpu[m,n]**2
        all_com=all_tx+all_rx;com=all_com[self.m:self.m+self.n]
        power=self.idle+compute/self.c['slot_s'];theta,rad,balance=thermal(snap['theta'],snap['radiator'],power,snap['heat'],self.c,ambient_c=snap.get('ambient'),return_balance=True)
        use=self.idle*self.c['slot_s']+snap['dep_energy']+compute+com
        y=schedule.sum(1);wait=snap['wait'];deadline=tasks[:,3]
        deadline_res=np.divide(np.maximum(0,y*wait+lat-y*deadline),np.maximum(deadline,1e-30),out=np.zeros_like(lat),where=y>0).sum()
        thermal_res=np.maximum(0,theta-80).sum()/10
        energy_res=np.sum(np.maximum(0,use-snap['energy']+self.reserve)/(snap['energy']+1e-9))
        # The manuscript specifies edge-node battery budgets. When a caller
        # also supplies underwater-source budgets, certify their actual TX/RX
        # charge without inventing a hidden source battery state.
        source_safe=True
        if 'source_energy' in snap:
            before=np.asarray(snap['source_energy'],float);reserve=np.broadcast_to(np.asarray(snap.get('source_reserve',0.),float),(self.m,))
            if before.shape!=(self.m,) or not np.isfinite(before).all() or np.any(before<0) or not np.isfinite(reserve).all() or np.any(reserve<0):raise ValueError('source energy budget axes')
            source_safe=bool(np.all(all_com[:self.m]<=before-reserve+1e-7))
            energy_res+=np.sum(np.maximum(0,all_com[:self.m]-before+reserve)/(before+1e-9))
        phi=deadline_res+thermal_res+energy_res
        safe=valid and np.all(lat[y>0]+wait[y>0]<=deadline[y>0]+1e-7) and np.all(lat<=self.c['slot_s']+1e-7) and np.all(snap['theta']<=80+1e-7) and np.all(theta<=80+1e-7) and np.all(use<=snap['energy']-self.reserve+1e-7)
        return dict(safe=bool(safe and source_safe),latency=lat,task_energy=et,node_communication=com,compute_energy=compute,power=power,theta=theta,radiator=rad,use=use,phi=float(phi),y=y,
                    phi_deadline=float(deadline_res),phi_thermal=float(thermal_res),phi_energy=float(energy_res),
                    compute_latency_s=cmp_lat,transmission_latency_s=tx_lat,all_node_communication=all_com,
                    all_node_tx_energy=all_tx,all_node_rx_energy=all_rx,task_hop_bits=hop_bits,thermal_balance=balance)

    def proposal(self,raw,snap):
        mask,paths=self.feasible(snap);z,_,_,_=self.split(raw);sched=np.zeros_like(mask,float)
        for m in range(self.m):
            if mask[m].any():sched[m,np.argmax(np.where(mask[m],z[m],-np.inf))]=1
        return self.allocate(raw,sched,snap),paths

    def certify(self,raw,snap):
        mask,paths=self.feasible(snap);z,_,_,_=self.split(raw);sched=np.zeros((self.m,self.n))
        idle=self.evaluate(self.allocate(raw,sched,snap),snap,paths)
        if not idle['safe']:return self.allocate(raw,sched,snap),idle,True
        order=sorted(np.flatnonzero(mask.any(1)),key=lambda m:(snap['tasks'][m,3]-snap['wait'][m],m))
        for m in order:
            for n in sorted(np.flatnonzero(mask[m]),key=lambda n:(-z[m,n],n)):
                trial=sched.copy();trial[m,n]=1;action=self.allocate(raw,trial,snap)
                if self.evaluate(action,snap,paths)['safe']:sched=trial;break
        action=self.allocate(raw,sched,snap);return action,self.evaluate(action,snap,paths),False

    def vector(self,action):
        return np.concatenate([action['schedule'].ravel(),(action['cpu']/self.capacity).ravel(),action['optical'].ravel(),action['rf'].ravel()])
