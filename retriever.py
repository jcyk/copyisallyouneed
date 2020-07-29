import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import math
import os

from transformer import Transformer, SinusoidalPositionalEmbedding, Embedding
from utils import move_to_device, label_smoothed_nll_loss, layer_norm
from generator import MonoEncoder

class MatchingModel(nn.Module):
    def __init__(self, query_encoder, response_encoder):
        super(MatchingModel, self).__init__()
        self.query_encoder = query_encoder
        self.response_encoder = response_encoder

    def forward(self, query, response):
        ''' query and response: [seq_len, batch_size]
        '''
        _, bsz = query.size()
        
        q = self.query_encoder(query)
        r = self.response_encoder(response)
 
        scores = torch.mm(q, r.t()) # bsz x bsz

        gold = torch.arange(bsz, device=scores.device)
        _, pred = torch.max(scores, -1)
        acc = torch.sum(torch.eq(gold, pred).float()) / bsz

        log_probs = F.log_softmax(scores, -1)
        loss, _ = label_smoothed_nll_loss(log_probs, gold, 0.1, sum=True)
        loss = loss / bsz

        return loss, acc, bsz

    def work(self, query, response):
        ''' query and response: [seq_len x batch_size ]
        '''
        _, bsz = query.size()
        q = self.query_encoder(query)
        r = self.response_encoder(response)

        scores = torch.sum(q * r, -1)
        return scores

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.query_encoder.state_dict(), os.path.join(output_dir, 'query_encoder'))
        torch.save(self.response_encoder.state_dict(), os.path.join(output_dir, 'response_encoder'))

    def load(self, output_dir):
        self.query_encoder.load_state_dict(os.path.join(output_dir, 'query_encoder'))
        self.response_encoder.load_state_dict(os.path.join(output_dir, 'response_encoder'))

    @classmethod
    def from_params(cls, vocabs, layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim, device):
        query_encoder = ProjEncoder(vocabs['src'], layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim, device)
        response_encoder = ProjEncoder(vocabs['tgt'], layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim, device)
        model = cls(query_encoder, response_encoder)
        return model

class ProjEncoder(nn.Module):
    def __init__(self, vocab, layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim, device):
        super(ProjEncoder, self).__init__()
        self.encoder = MonoEncoder(vocab, layers, embed_dim, ff_embed_dim, num_heads, dropout, device)
        self.proj = nn.Linear(embed_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.constant_(self.proj.bias, 0.)

    def forward(self, input_ids):
        ret, _ = self.encoder(input_ids) 
        ret = ret[0,:,:]
        ret = layer_norm(self.proj(ret))
        return ret
