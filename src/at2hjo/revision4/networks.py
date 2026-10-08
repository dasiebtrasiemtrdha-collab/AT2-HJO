from copy import deepcopy
import numpy as np
import torch
from torch import nn

def mlp(ins,outs,hidden=(256,256),output_scale=1e-3):
    layers=[]
    for size in hidden:layers += [nn.Linear(ins,size),nn.ReLU()];ins=size
    layers += [nn.Linear(ins,outs)]
    net=nn.Sequential(*layers)
    for layer in net:
        if isinstance(layer,nn.Linear):nn.init.kaiming_uniform_(layer.weight,nonlinearity='relu');nn.init.zeros_(layer.bias)
    nn.init.uniform_(net[-1].weight,-output_scale,output_scale)
    return net

class Actor(nn.Module):
    def __init__(self,state_dim,projector,c):
        super().__init__();self.net=mlp(state_dim,projector.action_dim,c['hidden'],c['other_init']);self.schedule=projector.sizes[0];self.temp=c['head_temperature']
        nn.init.uniform_(self.net[-1].weight[self.schedule:],-c['resource_init'],c['resource_init'])
    def forward(self,x):
        h=self.net(x);return torch.cat([5*torch.tanh(h[...,:self.schedule]),torch.tanh(h[...,self.schedule:]/self.temp)],-1)

class Regulator(nn.Module):
    def __init__(self,state_dim,context_dim,c):
        super().__init__();self.d=c['attention'];self.q=nn.Linear(context_dim,self.d,bias=False);self.k=nn.Linear(context_dim,self.d,bias=False);self.v=nn.Linear(context_dim,self.d,bias=False)
        self.head=nn.Linear(state_dim+self.d,3)
        for module in self.modules():
            if isinstance(module,nn.Linear):nn.init.xavier_uniform_(module.weight);
        nn.init.zeros_(self.head.bias)
    def forward(self,x,g):
        attention=torch.softmax(self.q(g)@self.k(g).transpose(-1,-2)/self.d**.5,-1)
        o=(attention@self.v(g))[...,-1,:]
        return self.head(torch.cat([x,o],-1))

class Models(nn.Module):
    def __init__(self,state_dim,context_dim,projector,c,seed,*,boundary_state_dim=None):
        super().__init__()
        self.boundary_state_dim=state_dim if boundary_state_dim is None else boundary_state_dim
        with torch.random.fork_rng(devices=[]):
            # Constructors are on CPU. Avoid torch.manual_seed, which also
            # changes unrelated CUDA generators outside this CPU fork.
            torch.random.default_generator.manual_seed(seed)
            self.actor=Actor(state_dim,projector,c)
            self.q1=mlp(state_dim+projector.action_dim,1,c['hidden'],c['other_init']);self.q2=mlp(state_dim+projector.action_dim,1,c['hidden'],c['other_init'])
            self.lt=mlp(state_dim+projector.n*c['services'],1,c['hidden'],c['other_init'])
            self.tr=Regulator(self.boundary_state_dim,context_dim,c)
            self.value=mlp(self.boundary_state_dim+c['history']*context_dim,1,[c['value_hidden']],c['other_init'])
        self.actor_target=deepcopy(self.actor);self.q1_target=deepcopy(self.q1);self.q2_target=deepcopy(self.q2);self.lt_target=deepcopy(self.lt)
        for net in (self.actor_target,self.q1_target,self.q2_target,self.lt_target):
            for p in net.parameters():p.requires_grad_(False)

    def summary(self,state_dim,context_dim,p,c):
        return {'state_dim_lt':state_dim,'state_dim_st':state_dim,'state_dim_tr_boundary':self.boundary_state_dim,'action_heads':['schedule','cpu','uowc','rf'],'action_slices':np.r_[0,np.cumsum(p.sizes)].tolist(),'action_dim':p.action_dim,'context_dim':context_dim,'tr_policy_input':'full boundary without interval + last attention row','tr_value_input':'full boundary without interval + flattened complete history','hidden':c['hidden'],'value_hidden':c['value_hidden'],'history':c['history'],'attention':c['attention'],'activation':f"ReLU; schedule 5*tanh; resources tanh(h/{c['head_temperature']})",'initialization':{'hidden':'Kaiming uniform','actor_resource_output':c['resource_init'],'other_output':c['other_init'],'attention':'Xavier uniform','bias':'zero'},'parameters':{name:sum(x.numel() for x in module.parameters()) for name,module in self.named_children()},'total_parameters':sum(x.numel() for x in self.parameters()),'checkpoint_compatibility':'revision4-learning-v2 only; pre-fix TR shapes and shared-replay checkpoints are incompatible'}
