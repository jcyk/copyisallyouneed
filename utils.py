import os
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import numpy as np

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


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, sum=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if sum:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

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

def layer_norm(x, variance_epsilon=1e-12):
    u = x.mean(-1, keepdim=True)
    s = (x - u).pow(2).mean(-1, keepdim=True)
    x = (x - u) / torch.sqrt(s + variance_epsilon)
    return x

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