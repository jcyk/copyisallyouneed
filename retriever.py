import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import math
import os, time

from transformer import Transformer, SinusoidalPositionalEmbedding, Embedding
from utils import move_to_device
from module import label_smoothed_nll_loss, layer_norm, MonoEncoder
from mips import MIPS, augment_query, l2_to_ip
from data import BOS, EOS, ListsToTensor, _back_to_txt_for_check

class Retriever(nn.Module):
    def __init__(self, vocabs, input_dir, nprobe, topk, gpuid, load_response_encoder=False):
        super(Retriever, self).__init__()
        model_args = torch.load(os.path.join(input_dir, 'args'))
        self.model = ProjEncoder.from_pretrained(vocabs['src'], model_args, os.path.join(input_dir, 'query_encoder'))
        if load_response_encoder:
            self.another_model = ProjEncoder.from_pretrained(vocabs['tgt'], model_args, os.path.join(input_dir, 'response_encoder'))
            
        self.mips = MIPS.from_built(os.path.join(input_dir, 'mips_index'), nprobe=nprobe)
        if gpuid >= 0:
            self.mips.to_gpu(gpuid=gpuid)
        self.mips_max_norm = torch.load(os.path.join(input_dir, 'max_norm.pt'))
        self.mem_pool = [line.strip().split() for line in open(os.path.join(input_dir, 'candidates.txt')).readlines()]
        self.mem_feat = torch.load(os.path.join(input_dir, 'feat.pt'))
        self.topk = topk
        self.vocabs = vocabs

    def work(self, inp, allow_hit):
        src_tokens = inp['src_tokens'] 
        src_feat, src, src_mask = self.model(src_tokens, return_src=True)
        vecsq = src_feat.detach().cpu().numpy()
        retrieval_start = time.time()
        vecsq = augment_query(vecsq)
        D, I = self.mips.search(vecsq, self.topk + 1)
        D = l2_to_ip(D, vecsq, self.mips_max_norm) / (self.mips_max_norm * self.mips_max_norm)
        mem_sents = []
        for i, (Ii, Di) in enumerate(zip(I, D)):
            tmplist = []
            for pred, s in zip(Ii, Di):
                mem = self.mem_pool[pred]
                if allow_hit or mem!=inp['tgt_raw_sents'][i]:
                    mem_feat = self.mem_feat[pred]
                    score = float(s)
                    #ip = torch.sum(src_feat[i] * mem_feat.to(src_feat.device))
                    #norm_times_norm = torch.norm(src_feat[i]) * torch.norm(mem_feat)
                    #print ('single score', ip/norm_times_norm, score)
                    tmplist.append([mem, mem_feat, score])
            tmplist = tmplist[:self.topk]
            assert len(tmplist) == self.topk
            mem_sents.append(tmplist)
        retrieval_cost = time.time() - retrieval_start
        #print ('retrieval_cost', retrieval_cost)

        # put all memory tokens (across the same batch) in a single list:
        # No.1 memory for sentence 1, ..., No.1 memory for sentence N, No.2 memory for sentence 1, ...
        # then convert to tensors:
        # all_mem_tokens -> seq_len x ( num_mem_sents_per_instance * bsz )
        # all_mem_scores -> num_mem_sents_per_instance * bsz
        # all_mem_feats -> num_mem_sents_per_instance * bsz x dim
        all_mem_tokens = []
        all_mem_scores = []
        all_mem_feats = []
        for t in zip(*mem_sents):
            all_mem_tokens.extend([tokens+[EOS] for tokens, _, _ in t])
            all_mem_scores.extend([score for _, _, score in t])
            all_mem_feats.extend([feat for _, feat, _ in t])
        bsz, dim = src_feat.size()
        src_feat = src_feat.view(1, bsz, dim)
        all_mem_feats = torch.stack(all_mem_feats, dim=0).to(src_feat.device).view(-1, bsz, dim)

        #all_mem_scores_ = torch.tensor(all_mem_scores, dtype=torch.float, device=src_feat.device)
        all_mem_scores = torch.sum(src_feat * all_mem_feats, dim=-1).view(-1) / (self.mips_max_norm ** 2)

        all_mem_tokens = ListsToTensor(all_mem_tokens, self.vocabs['tgt'])
        # to avoid GPU OOM issue, truncate the mem to the max. length of 1.5 x src_tokens
        max_mem_len = int(1.5 * src_tokens.shape[0])
        all_mem_tokens = move_to_device(all_mem_tokens[:max_mem_len,:], inp['src_tokens'].device)

        mem_ret = {}
        mem_ret['top1_retrieval_raw_sents'] = [x[0][0] for x in mem_sents]
        mem_ret['all_mem_tokens'] = all_mem_tokens
        mem_ret['all_mem_scores'] = all_mem_scores
        mem_ret['num_mem_sents_per_instance'] = self.topk
        return src, src_mask, mem_ret

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
        response_encoder = ProjEncoder.from_pretrained(vocabs['tgt'], model_args, os.path.join(input_dir, 'response_encoder'))
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

    def forward(self, input_ids, batch_first=False, return_src=False):
        if batch_first:
            input_ids = input_ids.t()
        src, src_mask = self.encoder(input_ids) 
        ret = src[0,:,:]
        ret = F.dropout(ret, p=self.dropout, training=self.training)
        ret = layer_norm(self.proj(ret))
        if return_src:
            return ret, src, src_mask
        return ret

    @classmethod
    def from_pretrained(cls, vocab, model_args, ckpt):
        model = cls(vocab, model_args.layers, model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout, model_args.output_dim)
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        return model
