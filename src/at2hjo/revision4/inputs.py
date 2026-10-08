"""Appendix A generators; independent addressed environment streams.

Internal slots are zero based: slot 0 is manuscript t=1. Current movement uses
the command/current observed in the preceding slot; innovations determine only
the new state. The regular trace has fixed traffic mode and no popularity
shift. Stochastic traffic starts in its 0.2 high-state stationary distribution.
The default input-validation population consists of AUVs; explicitly supplied
USN types stay stationary. Satellite velocity is analytic at the current phase,
without reading a future slot or drawing any future innovation.
"""
from copy import deepcopy
import numpy as np
from at2hjo.geometry.orbits import orbital_ephemeris,ground_visibility,segment_min_radius
from at2hjo.geometry.coordinates import enu_rotation,enu_to_ecef,eci_to_ecef
from at2hjo.communications.channels import optical_gain,friis_gain,rician_power
from .config import stream
from .physics import environmental_heat,node_values

class Trace:
    def __init__(self,c,seed,episode=0):
        self.c=c; self.m=c['nodes']; self.b=c['buoys']; self.s=c['satellites']; self.n=self.b+self.s
        self.dt=c['slot_s']; self.t=0; self.p=c['physical']; self.rng={n:stream(seed,episode,n) for n in ('mobility','arrival','service','size','cycles','deadline','link')}
        self.outage_rng=stream(seed,episode,'rf_outage')
        self.bn_block_probability=float(self.p['channels']['bn'].get('wave_block_probability',.02))
        self.rf_outage_probability=float(self.p['channels']['rf'].get('nominal_outage_probability',0.))
        if not 0<=self.bn_block_probability<=1 or not 0<=self.rf_outage_probability<=1:raise ValueError('link outage probabilities must lie in [0,1]')
        if c['horizon']<=0 or self.dt<=0 or not np.isfinite(self.dt):raise ValueError('positive input horizon and slot duration')
        self.node_types=np.asarray(self.p.get('underwater_node_types',['AUV']*self.m))
        if self.node_types.shape!=(self.m,) or not np.isin(self.node_types,['AUV','USN']).all():raise ValueError('underwater node types require one AUV/USN label per source')
        self.mobile=self.node_types=='AUV'
        r=self.rng['mobility']; self.q=np.column_stack((r.uniform(0,500,(self.m,2)),np.full(self.m,-50.)))
        self.speed=r.uniform(1,3,self.m); self.w=np.zeros((self.m,2)); self.current=np.zeros(2)
        self.speed[~self.mobile]=0
        self.traffic=(self.rng['arrival'].random(self.m)<.2 if c['trace']=='stochastic_open' else np.zeros(self.m,bool))
        positions=np.asarray(self.p.get('buoy_positions_m',[[125,125,0],[375,125,0],[250,375,0]]),float)
        if positions.ndim!=2 or positions.shape[1]!=3 or len(positions)<self.b or not np.isfinite(positions).all():raise ValueError('explicit finite buoy positions required for this BN count')
        self.bq=positions[:self.b].copy();self.ambient=node_values(c,'ambient_c','thermal','ambient_c')[:self.b]
        self.orbit=deepcopy(self.p['orbit']); self.orbit['satellites_per_plane']=self.s
        if self.orbit['planes']!=1:raise ValueError('revision4 trace currently requires one declared Walker plane')
        self.ephem=orbital_ephemeris(self.orbit,np.arange(c['horizon']+2)*self.dt)
        self.fade=self.rng['link'].normal(size=(self.b,self.s,2)); self.bnfade=self.rng['link'].normal(size=(self.b,self.b,2)); self.block=(self.rng['link'].random((self.b,self.b))<self.bn_block_probability).astype(float)
        self._reciprocal_bn()
        self.log_turbidity=0.
        self.advance(initial=True)

    def advance(self,initial=False):
        r=self.rng['mobility']
        if not initial:
            if self.t+1>=self.c['horizon']:raise IndexError('input trace has reached its declared horizon')
            self.q[self.mobile,:2]+=(self.cmd[self.mobile]+self.current)*self.dt
            if self.c['trace']=='regular':
                folded=np.mod(self.q[:,:2],1000.); self.q[:,:2]=np.where(folded<=500,folded,1000-folded)
            self.t+=1
            cur=.9*self.current+r.normal(0,.05,2); self.current=cur/max(1,np.linalg.norm(cur)/.5)
        if self.t%20==0:
            bounds=(-150,650) if self.c['trace']=='stochastic_open' else (0,500)
            self.w=r.uniform(*bounds,(self.m,2))
        delta=self.w-self.q[:,:2]; d=np.linalg.norm(delta,axis=1)
        self.cmd=delta/np.maximum(d[:,None],1e-30)*np.minimum(self.speed,d/self.dt)[:,None]
        self.cmd[~self.mobile]=0
        self.present=np.all((self.q[:,:2]>=0)&(self.q[:,:2]<=500),axis=1)
        a=self.rng['arrival']
        if not initial and self.c['trace']=='stochastic_open':
            z=a.random(self.m); self.traffic=np.where(self.traffic,z>=.2,z<.05)
        counts=(np.ones(self.m,dtype=int) if self.c['trace']=='regular' else a.poisson(np.where(self.traffic,3.,.5))) * self.present
        k=self.c['services']; shift=self.t//100%k if self.c['trace']=='stochastic_open' else 0
        ranks=1+(np.arange(k)+shift)%k; prob=ranks**(-.8);prob/=prob.sum()
        if self.c['trace']=='stochastic_open':prob=.7*prob+.3/k
        self.popularity_shift=shift;self.service_probability=prob.copy()
        self.arrivals=[]; self.new_counts=np.zeros(k)
        for m,count in enumerate(counts):
            tasks=[]
            for _ in range(count):
                service=int(self.rng['service'].choice(k,p=prob));self.new_counts[service]+=1
                deadline=self.rng['deadline'].uniform(.5,2) if self.c['trace']=='regular' else self.rng['deadline'].uniform(2*self.dt,4*self.dt)
                tasks.append([self.rng['size'].uniform(.5e6,2e6),self.rng['cycles'].uniform(.5e9,2e9),service,deadline,self.t])
            self.arrivals.append(tasks)
        # First-order link memory, fully observed.
        ar=self.p['channels'].get('driver_ar',0.)
        if not np.isfinite(ar) or not 0<=ar<1:raise ValueError('link AR coefficient must lie in [0,1)')
        self.fade=ar*self.fade+np.sqrt(1-ar*ar)*self.rng['link'].normal(size=self.fade.shape)
        self.bnfade=ar*self.bnfade+np.sqrt(1-ar*ar)*self.rng['link'].normal(size=self.bnfade.shape)
        persistence=self.p['channels'].get('block_persistence',0.)
        p_block=self.bn_block_probability
        chance=np.where(self.block>.5,p_block+(1-p_block)*persistence,p_block*(1-persistence))
        self.block=(self.rng['link'].random((self.b,self.b))<chance).astype(float)
        self._reciprocal_bn()
        self.log_turbidity=ar*self.log_turbidity+self.rng['link'].normal(0,self.p['channels'].get('log_turbidity_noise',0.))
        self.solar_phase=self.p['satellite_radiation']['solar_phase_rad']+self.t*self.dt*self.p['satellite_radiation']['solar_phase_rate_rad_s']
        sun=np.array([np.cos(self.solar_phase),np.sin(self.solar_phase),0.])
        boundary=environmental_heat(self.ephem['eci_m'][self.t],sun,self.c)
        self.sunlight=boundary['sunlight'];self.incidence=boundary['incidence'];self.heat=boundary['heat'];self.heat_components=boundary
        self.links=self._links()

    def _reciprocal_bn(self):
        if self.p['channels'].get('reciprocal_bn_isl',False):
            lower=np.tril_indices(self.b,-1)
            self.bnfade[lower]=self.bnfade[(lower[1],lower[0])]
            self.block[lower]=self.block[(lower[1],lower[0])]

    def orbital_velocity(self):
        """Current local ENU velocity, including rotating-Earth correction."""
        phase=self.ephem['initial_anomaly_rad']+self.ephem['mean_motion_rad_s']*self.t*self.dt
        raan=self.ephem['raan_rad'];inc=np.deg2rad(self.orbit['inclination_deg']);motion=self.ephem['mean_motion_rad_s'];radius=self.ephem['orbital_radius_m']
        velocity_eci=motion*radius*np.stack((-np.cos(raan)*np.sin(phase)-np.sin(raan)*np.cos(phase)*np.cos(inc),
                                            -np.sin(raan)*np.sin(phase)+np.cos(raan)*np.cos(phase)*np.cos(inc),
                                            np.cos(phase)*np.sin(inc)),axis=-1)
        velocity_ecef=eci_to_ecef(velocity_eci,self.t*self.dt,omega=self.orbit['earth_rotation_rad_s'],angle0=self.orbit.get('earth_angle_at_epoch_rad',0))
        velocity_ecef-=np.cross([0,0,self.orbit['earth_rotation_rad_s']],self.ephem['ecef_m'][self.t])
        return velocity_ecef@enu_rotation(self.orbit['origin_lat_deg'],self.orbit['origin_lon_deg']).T

    def _links(self):
        c=self.p['channels']; d=self.p['deployment']; l=self.m+self.n+1; links=np.zeros((l,l,5))
        # [...]: received signal power, bandwidth, tx/rx electrical power, availability.
        def add(i,j,kind,dist,visible=True,gain=None):
            p=c[kind]
            if gain is None:gain=friis_gain(dist,p['carrier_hz'],10**(p['tx_gain_dbi']/10),10**(p['rx_gain_dbi']/10),10**(p['extra_loss_db']/10))
            ok=bool(visible and np.isfinite(gain) and p['tx_power_w']*gain/(c['noise_psd_w_hz']*p['bandwidth_hz'])>=c['snr_threshold_linear'])
            circuit=self.p['circuit'][kind]
            signal=gain*p['tx_power_w'] if np.isfinite(gain) else 0.
            links[i,j]=[signal,p['bandwidth_hz'],p['tx_power_w']+circuit['tx_w'],circuit['rx_w'],ok]
        for m in range(self.m):
            for b in range(self.b):
                dist=np.linalg.norm(self.q[m]-self.bq[b]); p=c['optical']
                add(m,self.m+b,'optical',dist,self.present[m] and dist<=p['max_distance_m'],optical_gain(dist,p['aperture_m2'],np.deg2rad(p['divergence_deg']),p['cw_m_inv']*np.exp(self.log_turbidity)))
        for b in range(self.b):
            for j in range(self.b):
                if b!=j:
                    dist=np.linalg.norm(self.bq[b]-self.bq[j]); p=c['bn']; gain=friis_gain(dist,p['carrier_hz'],10**(p['tx_gain_dbi']/10),10**(p['rx_gain_dbi']/10),10**(p['extra_loss_db']/10))*rician_power(self.bnfade[b,j],10**(p['rician_k_db']/10))
                    add(self.m+b,self.m+j,'bn',dist,not self.block[b,j] and dist<=p['max_distance_m'],gain)
        sq=self.ephem['local_enu_m'][self.t]; ecef=self.ephem['ecef_m'][self.t]
        observers=enu_to_ecef(self.bq,self.ephem['origin_ecef_m'],self.orbit['origin_lat_deg'],self.orbit['origin_lon_deg'])
        visible=ground_visibility(ecef,observers,self.orbit['min_elevation_deg'],self.orbit['earth_radius_m'])['visible']
        for b in range(self.b):
            for s in range(self.s):
                p=c['rf'];dist=np.linalg.norm(sq[s]-self.bq[b]);gain=friis_gain(dist,p['carrier_hz'],10**(p['tx_gain_dbi']/10),10**(p['rx_gain_dbi']/10),10**(p['extra_loss_db']/10))*rician_power(self.fade[b,s],10**(p['rician_k_db']/10))
                outage=self.rf_outage_probability>0 and self.outage_rng.random()<self.rf_outage_probability
                available=visible[b,s] and not outage
                add(self.m+b,self.m+self.b+s,'rf',dist,available,gain);add(self.m+self.b+s,self.m+b,'rf',dist,available,gain)
        for s in range(self.s):
            for j in range(self.s):
                if s!=j and min((s-j)%self.s,(j-s)%self.s)==1:
                    dist=np.linalg.norm(ecef[s]-ecef[j]);ok=(segment_min_radius(ecef[s],ecef[j])>=self.orbit['earth_radius_m']+self.orbit.get('isl_earth_clearance_m',0.) and dist<=self.orbit['isl_max_distance_m'])
                    add(self.m+self.b+s,self.m+self.b+j,'isl',dist,ok)
        gateway=self.p['orbit']['ground_station_offset_enu_m']
        gateway_ecef=enu_to_ecef(np.asarray(gateway)[None,:],self.ephem['origin_ecef_m'],self.orbit['origin_lat_deg'],self.orbit['origin_lon_deg'])
        gateway_visible=ground_visibility(ecef,gateway_ecef,self.orbit['min_elevation_deg'],self.orbit['earth_radius_m'])['visible'][0]
        for s in range(self.s):
            if gateway_visible[s]:links[-1,self.m+self.b+s]=[1.,1.,d['gateway_tx_w'],d['gateway_rx_w'],1.]
        return links

    def _driver_pieces(self):
        phase=self.ephem['initial_anomaly_rad']+self.ephem['mean_motion_rad_s']*self.t*self.dt
        # Each tuple carries the unencoded physical value, unit and divisor.
        # Existing vector order/scales are retained for environment consumers.
        return [('underwater_position',self.q,'m',1e6),('command_velocity',self.cmd,'m/s',3.),
                ('cruise_speed',self.speed,'m/s',3.),('waypoint',self.w,'m',650.),
                ('area_presence',self.present,'boolean',1.),('traffic_high_state',self.traffic,'boolean',1.),
                ('ocean_current',self.current,'m/s',.5),('slot_clock',np.array([self.t]),'zero-based ST slot',self.c['horizon']),
                ('popularity_rank_shift',np.array([self.popularity_shift]),'zero-based rank shift',1.),
                ('solar_phase',np.array([self.solar_phase]),'rad',1.),('log_turbidity',np.array([self.log_turbidity]),'log multiplier',1.),
                ('satellite_position_enu',self.ephem['local_enu_m'][self.t],'m',1e7),
                ('satellite_velocity_enu',self.orbital_velocity(),'m/s',1e4),
                ('orbital_phase_sin',np.sin(phase),'dimensionless',1.),('orbital_phase_cos',np.cos(phase),'dimensionless',1.),
                ('rf_fading_memory',self.fade,'standard-normal quadratures',1.),
                ('bn_fading_memory',self.bnfade,'standard-normal quadratures',1.),
                ('bn_blockage_memory',self.block,'boolean',1.),('buoy_sink_temperature',self.ambient,'deg C',80.),
                ('satellite_sunlit',self.sunlight,'boolean',1.),('solar_incidence',self.incidence,'dimensionless',1.)]

    def drivers(self):
        return np.concatenate([np.asarray(value,float).ravel()/scale for _,value,_,scale in self._driver_pieces()])

    def driver_layout(self):
        """JSON-ready named slices/shapes/units for decoding the driver vector."""
        layout={};start=0
        for name,value,units,scale in self._driver_pieces():
            array=np.asarray(value);end=start+array.size
            layout[name]=dict(start=start,end=end,shape=list(array.shape),units=units,
                              scaling=dict(operation='divide',divisor=float(scale)))
            start=end
        return layout
