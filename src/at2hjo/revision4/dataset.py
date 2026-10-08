"""Read-only materialized trace consumer with byte integrity checks."""
from pathlib import Path
import hashlib
import json
import numpy as np
from . import SCIENCE
from .config import digest,input_identity

class RecordedTrace:
    def __init__(self,directory,c,seed):
        self.root=Path(directory);manifest=json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('input_schema_version')!='manuscript40-input-v2':raise ValueError('Dataset uses an older driver/ambient schema; regenerate with the current core.')
        if manifest['science']!=SCIENCE or manifest.get('input_config_hash')!=input_identity(c) or manifest['seed']!=seed:raise ValueError('Dataset science/input-config/seed mismatch')
        for name,expected in manifest['files'].items():
            if hashlib.sha256((self.root/name).read_bytes()).hexdigest()!=expected:raise ValueError('Dataset hash mismatch: '+name)
        with np.load(self.root/'trace.npz',allow_pickle=False) as z:self._links=z['links'];self._drivers=z['drivers']
        self._requests=[json.loads(s) for s in (self.root/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(self._requests)!=c['horizon'] or len(self._links)!=c['horizon']:raise ValueError('Dataset horizon mismatch')
        self.c=c;self.t=0;self._load()
    def _load(self):
        record=self._requests[self.t];self.arrivals=record['arrivals'];self.present=np.array(record['present'],bool);self.heat=np.array(record['heat_w'],float);self.ambient=np.array(record['ambient_c'],float);self.links=self._links[self.t].copy()
        self.new_counts=np.zeros(self.c['services'])
        for tasks in self.arrivals:
            for task in tasks:self.new_counts[int(task[2])]+=1
    def advance(self):self.t+=1;self._load()
    def drivers(self):return self._drivers[self.t].copy()
