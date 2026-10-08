"""Shared placement for learners, batches, evaluation and checkpoint tensors."""
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
import re
from typing import Any
import torch

class DeviceUnavailable(RuntimeError):
    pass

class DeviceManager:
    def __init__(self,spec:str='auto'):
        if spec not in {'auto','cpu','cuda'} and not re.fullmatch(r'cuda:[0-9]+',spec):
            raise ValueError('device must be auto, cpu, cuda or cuda:N')
        self.requested=spec
        if spec=='auto':spec='cuda:0' if torch.cuda.is_available() else 'cpu'
        if spec.startswith('cuda'):
            if not torch.cuda.is_available():raise DeviceUnavailable('CUDA unavailable: no usable CUDA device in the current PyTorch runtime')
            index=0 if spec=='cuda' else int(spec.split(':')[1])
            if index>=torch.cuda.device_count():raise DeviceUnavailable(f'CUDA device index {index} unavailable')
            spec=f'cuda:{index}'
        self.device=torch.device(spec)
    def tensor(self,data,*,dtype=None):return torch.as_tensor(data,dtype=dtype,device=self.device)
    def zeros(self,*shape,dtype=torch.float32):return torch.zeros(*shape,dtype=dtype,device=self.device)
    def move(self,value:Any):
        if isinstance(value,torch.Tensor):return value.to(self.device)
        if is_dataclass(value) and not isinstance(value,type):
            return replace(value,**{f.name:self.move(getattr(value,f.name)) for f in fields(value) if f.init})
        if isinstance(value,dict):return {k:self.move(v) for k,v in value.items()}
        if isinstance(value,tuple):return tuple(self.move(v) for v in value)
        if isinstance(value,list):return [self.move(v) for v in value]
        return value  # Physical NumPy arrays and scalar metadata stay on the host.
    def prepare_module(self,module:torch.nn.Module):return module.to(self.device)
    def move_optimizer(self,optimizer):
        for key,value in list(optimizer.state.items()):optimizer.state[key]=self.move(value)
    def load_checkpoint(self,path:str|Path,*,weights_only:bool=True):
        return torch.load(path,map_location=self.device,weights_only=weights_only)
    def evaluation(self,module):
        self.prepare_module(module);module.eval()
        return module
    def describe(self):return {'requested':self.requested,'selected':str(self.device),'cuda_available':torch.cuda.is_available(),'torch_cuda_version':torch.version.cuda,'cuda_device_count':torch.cuda.device_count(),'cuda_device_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
