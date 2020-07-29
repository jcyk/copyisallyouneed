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

    def forward(self, query, response, label_smoothing=0.):
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
        loss, _ = label_smoothed_nll_loss(log_probs, gold, label_smoothing, sum=True)
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

    def save(self, model_args, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.query_encoder.state_dict(), os.path.join(output_dir, 'query_encoder'))
        torch.save(self.response_encoder.state_dict(), os.path.join(output_dir, 'response_encoder'))
        torch.save(model_args, os.path.join(output_dir, 'args'))

    @classmethod
    def from_params(cls, vocabs, layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim):
        query_encoder = ProjEncoder(vocabs['src'], layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim)
        response_encoder = ProjEncoder(vocabs['tgt'], layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim)
        model = cls(query_encoder, response_encoder)
        return model
    
    @classmethod
    def from_pretrained(cls, vocabs, input_dir):
        model_args = torch.load(os.path.join(input_dir, 'args'))
        query_encoder = ProjEncoder.from_pretrained(vocabs['src'], model_args, os.path.join(input_dir, 'query_encoder'))
        response_encoder = ProjEncoder.from_pretrained(vocabs['src'], model_args, os.path.join(input_dir, 'response_encoder'))
        model = cls(query_encoder, response_encoder)
        return model

class ProjEncoder(nn.Module):
    def __init__(self, vocab, layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim):
        super(ProjEncoder, self).__init__()
        self.encoder = MonoEncoder(vocab, layers, embed_dim, ff_embed_dim, num_heads, dropout)
        self.proj = nn.Linear(embed_dim, output_dim)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.constant_(self.proj.bias, 0.)

    def forward(self, input_ids, batch_first=False):
        if batch_first:
            input_ids = input_ids.t()
        ret, _ = self.encoder(input_ids) 
        ret = ret[0,:,:]
        ret = F.dropout(ret, p=self.dropout, training=self.training)
        ret = layer_norm(self.proj(ret))
        return ret

    @classmethod
    def from_pretrained(cls, vocab, model_args, ckpt):
        model = cls(vocab, model_args.layers, model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout, model_args.output_dim)
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        return model
