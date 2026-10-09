from collections import deque

import numpy as np
import torch


class PlanBuffer:
    def __init__(self, capacity, seed):
        if int(capacity) <= 0:
            raise ValueError('Plan capacity must be positive')
        self.labels = deque(maxlen=int(capacity))
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.labels)

    def add(self, label):
        if not label.valid:
            raise ValueError('Cannot train on a rejected label')
        self.labels.append(label)

    def sample(self, batch_size, device):
        if not self.labels:
            raise ValueError('Empty plan buffer')
        labels = [self.labels[int(i)] for i in self.rng.integers(0,len(self.labels),size=batch_size)]
        result = {}
        for i,mg in enumerate(labels[0].agent_ids):
            values = dict(observations=np.stack([x.observations[mg] for x in labels]),
                intents=np.stack([x.intents[:,i] for x in labels]),
                request_kw=np.stack([x.request_kw[:,i] for x in labels]),
                nominal_soc_after=np.stack([x.nominal_soc_after[:,i] for x in labels]),
                mask=np.stack([x.mask for x in labels]))
            result[mg] = {k:torch.tensor(v,device=device,dtype=torch.bool if k=='mask' else torch.float32)
                          for k,v in values.items()}
        return result
