import os
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import numpy as np

def partially_load(model, ckpt):
    pretrained_dict = torch.load(ckpt)
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def move_to_device(maybe_tensor, device):
    if torch.is_tensor(maybe_tensor):
        return maybe_tensor.to(device)
    elif isinstance(maybe_tensor, np.ndarray):
        return torch.from_numpy(maybe_tensor).to(device).contiguous()
    elif isinstance(maybe_tensor, dict):
        return {
            key: move_to_device(value, device)
            for key, value in maybe_tensor.items()
        }
    elif isinstance(maybe_tensor, list):
        return [move_to_device(x, device) for x in maybe_tensor]
    elif isinstance(maybe_tensor, tuple):
        return tuple([move_to_device(x, device) for x in maybe_tensor])
    return maybe_tensor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size

class Statistics:
    def __init__(self, key_value_dict=None, **kwargs):
        self.statistics = {'steps':0}
        if key_value_dict is not None:
            for x in key_value_dict:
                self.statistics[x] = key_value_dict[x]
        for x in kwargs:
            self.statistics[x] = kwargs[x]

    def update(self, key_or_dict, value=None):
        if value is None:
            assert isinstance(key_or_dict, dict)
            for key in key_or_dict:
                if key not in self.statistics:
                    self.statistics[key] = 0.
                self.statistics[key] += key_or_dict[key]
        else:
            assert isinstance(key_or_dict, str)
            if key_or_dict not in self.statistics:
                self.statistics[key_or_dict] = 0.
            self.statistics[key_or_dict] += value
    
    def __getitem__(self, attr):
        return self.statistics[attr]

    def step(self):
        self.statistics['steps'] += 1

def data_proc(data, queue):
    for x in data:
        queue.put(x)
    queue.put('EPOCHDONE')

def asynchronous_load(data_loader):
    queue = mp.Queue(10)
    data_generator = mp.Process(target=data_proc, args=(data_loader, queue))
    data_generator.start()
    done = False
    while not done:
        batch = queue.get()
        if isinstance(batch, str):
            done = True
        else:
            yield batch
    data_generator.join()