import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import math
import os, time, random, logging

from transformer import Transformer, SinusoidalPositionalEmbedding, Embedding
from utils import move_to_device, asynchronous_load
from module import label_smoothed_nll_loss, layer_norm, MonoEncoder
from mips import MIPS, augment_query, augment_data, l2_to_ip
from data import BOS, EOS, ListsToTensor, _back_to_txt_for_check

logger = logging.getLogger(__name__)
class Retriever(nn.Module):
    def __init__(self, vocabs, model, mips, mips_max_norm, mem_pool, mem_feat_or_feat_maker, num_heads, topk, gpuid):
        super(Retriever, self).__init__()
        self.model = model
        self.mem_pool = mem_pool
        self.mem_feat_or_feat_maker = mem_feat_or_feat_maker
        self.num_heads = num_heads
        self.topk = topk
        self.vocabs = vocabs
        self.gpuid = gpuid
        self.mips = mips
        if self.gpuid >= 0:
            self.mips.to_gpu(gpuid=self.gpuid)
        self.mips_max_norm = mips_max_norm

    @classmethod
    def from_pretrained(cls, num_heads, vocabs, input_dir, nprobe, topk, gpuid, use_response_encoder=False):
        model_args = torch.load(os.path.join(input_dir, 'args'))
        model = MultiProjEncoder.from_pretrained_projencoder(num_heads, vocabs['src'], model_args, os.path.join(input_dir, 'query_encoder'))
        mem_pool = [line.strip().split() for line in open(os.path.join(input_dir, 'candidates.txt')).readlines()]
       
        if use_response_encoder:
            mem_feat_or_feat_maker = ProjEncoder.from_pretrained(vocabs['tgt'], model_args, os.path.join(input_dir, 'response_encoder'))
        else:
            mem_feat_or_feat_maker = torch.load(os.path.join(input_dir, 'feat.pt'))
        
        mips = MIPS.from_built(os.path.join(input_dir, 'mips_index'), nprobe=nprobe)
        mips_max_norm = torch.load(os.path.join(input_dir, 'max_norm.pt'))
        retriever = cls(vocabs, model, mips, mips_max_norm, mem_pool, mem_feat_or_feat_maker, num_heads, topk, gpuid)
        return retriever

    def drop_index(self):
        self.mips.reset()
        self.mips = None
        self.mips_max_norm = None

    def update_index(self, index_dir, nprobe):
        self.mips = MIPS.from_built(os.path.join(index_dir, 'mips_index'), nprobe=nprobe)
        if self.gpuid >= 0:
            self.mips.to_gpu(gpuid=self.gpuid)
        self.mips_max_norm = torch.load(os.path.join(index_dir, 'max_norm.pt'))

    def rebuild_index(self, index_dir, batch_size=2048, add_every=1000000, index_type='IVF1024_HNSW32,SQ8', norm_th=999, max_training_instances=1000000, max_norm_cf=1.0, nprobe=64, efSearch=128):
        if not os.path.exists(index_dir):
            os.mkdir(index_dir)
        max_norm = None
        data = [ [' '.join(x), i] for i, x in enumerate(self.mem_pool) ]
        random.shuffle(data)
        used_data = [x[0] for x in data[:max_training_instances]]
        used_ids = np.array([x[1] for x in data[:max_training_instances]])
        logger.info('Computing feature for training')
        used_data, used_ids, max_norm = get_features(batch_size, norm_th, self.vocabs['tgt'], self.mem_feat_or_feat_maker, used_data, used_ids, max_norm_cf=max_norm_cf)
        torch.cuda.empty_cache()
        logger.info('Using %d instances for training', used_data.shape[0])
        mips = MIPS(self.model.output_dim+1, index_type, efSearch=efSearch, nprobe=nprobe) 
        mips.to_gpu()
        mips.train(used_data)
        mips.to_cpu()
        mips.add_with_ids(used_data, used_ids)
        data = data[max_training_instances:]
        torch.save(max_norm, os.path.join(index_dir, 'max_norm.pt'))
        
        cur = 0
        while cur < len(data):
            used_data = [x[0] for x in data[cur:cur+add_every]]
            used_ids = np.array([x[1] for x in data[cur:cur+add_every]])
            cur += add_every
            logger.info('Computing feature for indexing')
            used_data, used_ids, _ = get_features(batch_size, norm_th, vocab, self.mem_feat_or_feat_maker, used_data, used_ids, max_norm)
            logger.info('Adding %d instances to index', used_data.shape[0])
            mips.add_with_ids(used_data, used_ids)
        mips.save(os.path.join(index_dir, 'mips_index'))

    def work(self, inp, allow_hit):
        src_tokens = inp['src_tokens']
        src_feat, src, src_mask = self.model(src_tokens, return_src=True)
        num_heads, bsz, dim = src_feat.size()
        assert num_heads == self.num_heads
        topk = self.topk
        vecsq = src_feat.reshape(num_heads * bsz, -1).detach().cpu().numpy() 
        #retrieval_start = time.time()
        vecsq = augment_query(vecsq)
        D, I = self.mips.search(vecsq, topk + 1)
        D = l2_to_ip(D, vecsq, self.mips_max_norm) / (self.mips_max_norm * self.mips_max_norm)
        # I, D: (bsz * num_heads x (topk + 1) )
        indices = torch.zeros(topk, num_heads, bsz, dtype=torch.long)
        for i, (Ii, Di) in enumerate(zip(I, D)):
            bid, hid = i % bsz, i // bsz
            tmp_list = []
            for pred, _ in zip(Ii, Di):
                if allow_hit or self.mem_pool[pred]!=inp['tgt_raw_sents'][bid]:
                    tmp_list.append(pred)
            tmp_list = tmp_list[:topk]
            assert len(tmp_list) == topk
            indices[:, hid, bid] = torch.tensor(tmp_list)
        #retrieval_cost = time.time() - retrieval_start
        #print ('retrieval_cost', retrieval_cost)
        # convert to tensors:
        # all_mem_tokens -> seq_len x ( topk * num_heads * bsz )
        # all_mem_feats -> topk * num_heads * bsz x dim
        all_mem_tokens = []
        for idx in indices.view(-1).tolist():
            #TODO self.mem_pool[idx] +[EOS]
            all_mem_tokens.append([BOS] + self.mem_pool[idx])
        all_mem_tokens = ListsToTensor(all_mem_tokens, self.vocabs['tgt'])
        
        # to avoid GPU OOM issue, truncate the mem to the max. length of 1.5 x src_tokens
        max_mem_len = int(1.5 * src_tokens.shape[0])
        all_mem_tokens = move_to_device(all_mem_tokens[:max_mem_len,:], inp['src_tokens'].device)
       
        if torch.is_tensor(self.mem_feat_or_feat_maker):
            all_mem_feats = self.mem_feat_or_feat_maker[indices].to(src_feat.device)
        else:
            all_mem_feats = self.mem_feat_or_feat_maker(all_mem_tokens).view(topk, num_heads, bsz, dim)

        # all_mem_scores -> topk x num_heads x bsz
        all_mem_scores = torch.sum(src_feat.unsqueeze(0) * all_mem_feats, dim=-1) / (self.mips_max_norm ** 2)

        mem_ret = {}
        indices = indices.view(-1, bsz).transpose(0, 1).tolist()
        mem_ret['retrieval_raw_sents'] = [ [self.mem_pool[idx] for idx in ind] for ind in indices]
        mem_ret['all_mem_tokens'] = all_mem_tokens
        mem_ret['all_mem_scores'] = all_mem_scores
        return src, src_mask, mem_ret

class BOWModel(nn.Module):
    def __init__(self, tgt_embed):
        ## bag of words autoencoder
        super(BOWModel, self).__init__()
        vocab_size, embed_dim = tgt_embed.weight.shape
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.output_projection = nn.Linear(
                embed_dim,
                vocab_size,
                bias=False,
        )
        self.output_projection.weight = tgt_embed.weight
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.constant_(self.proj.bias, 0.)

    def forward(self, outs, label):
        # bow loss
        bsz, seq_len = label.shape
        label_mask = torch.le(label, 3) # except for PAD UNK BOS EOS
        logits = self.output_projection(self.proj(outs))
        lprobs = F.log_softmax(logits, dim=-1)
        #bsz x vocab
        loss = torch.gather(-lprobs, -1, label).masked_fill(label_mask, 0.)
        loss = loss.sum(dim=-1).mean()

        return loss

class MatchingModel(nn.Module):
    def __init__(self, query_encoder, response_encoder, bow=False):
        super(MatchingModel, self).__init__()
        self.query_encoder = query_encoder
        self.response_encoder = response_encoder
        self.bow = bow
        if self.bow:
            self.query_bow = BOWModel(query_encoder.encoder.src_embed)
            self.response_bow = BOWModel(response_encoder.encoder.src_embed)

    def forward(self, query, response, label_smoothing=0.):
        ''' query and response: [seq_len, batch_size]
        '''
        _, bsz = query.size()
        
        q, q_src, _ = self.query_encoder(query, return_src=True)
        r, r_src, _ = self.response_encoder(response, return_src=True)
        q_src = q_src[0,:,:]
        r_src = r_src[0,:,:]
 
        scores = torch.mm(q, r.t()) # bsz x (bsz + adt)

        gold = torch.arange(bsz, device=scores.device)
        _, pred = torch.max(scores, -1)
        acc = torch.sum(torch.eq(gold, pred).float()) / bsz

        log_probs = F.log_softmax(scores, -1)
        loss, _ = label_smoothed_nll_loss(log_probs, gold, label_smoothing, sum=True)
        loss = loss / bsz

        if self.bow:
            loss_bow_q = self.query_bow(r_src, query.transpose(0, 1))
            loss_bow_r = self.response_bow(q_src, response.transpose(0, 1))
            loss = loss + loss_bow_q + loss_bow_r
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
    def from_params(cls, vocabs, layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim, bow):
        query_encoder = ProjEncoder(vocabs['src'], layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim)
        response_encoder = ProjEncoder(vocabs['tgt'], layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim)
        model = cls(query_encoder, response_encoder, bow)
        return model
    
    @classmethod
    def from_pretrained(cls, vocabs, input_dir):
        model_args = torch.load(os.path.join(input_dir, 'args'))
        query_encoder = ProjEncoder.from_pretrained(vocabs['src'], model_args, os.path.join(input_dir, 'query_encoder'))
        response_encoder = ProjEncoder.from_pretrained(vocabs['tgt'], model_args, os.path.join(input_dir, 'response_encoder'))
        model = cls(query_encoder, response_encoder)
        return model

class MultiProjEncoder(nn.Module):
    def __init__(self, num_proj_heads, vocab, layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim):
        super(MultiProjEncoder, self).__init__()
        self.encoder = MonoEncoder(vocab, layers, embed_dim, ff_embed_dim, num_heads, dropout)
        self.proj = nn.Linear(embed_dim, num_proj_heads*output_dim)
        self.num_proj_heads = num_proj_heads
        self.output_dim = output_dim
        self.dropout = dropout

    def forward(self, input_ids, batch_first=False, return_src=False):
        if batch_first:
            input_ids = input_ids.t()
        src, src_mask = self.encoder(input_ids) 
        ret = src[0,:,:]
        ret = F.dropout(ret, p=self.dropout, training=self.training)
        ret = self.proj(ret).view(-1, self.num_proj_heads, self.output_dim).transpose(0, 1)
        ret = layer_norm(F.dropout(ret, p=self.dropout, training=self.training))
        if return_src:
            return ret, src, src_mask
        return ret

    @classmethod
    def from_pretrained_projencoder(cls, num_proj_heads, vocab, model_args, ckpt):
        model = cls(num_proj_heads, vocab, model_args.layers, model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout, model_args.output_dim)
        state_dict = torch.load(ckpt, map_location='cpu')
        model.encoder.load_state_dict({k[len('encoder.'):]:v for k,v in state_dict.items() if k.startswith('encoder.')})
        weight = state_dict['proj.weight'].repeat(num_proj_heads, 1)
        bias = state_dict['proj.bias'].repeat(num_proj_heads)
        model.proj.weight = nn.Parameter(weight)
        model.proj.bias = nn.Parameter(bias)
        return model

class ProjEncoder(nn.Module):
    def __init__(self, vocab, layers, embed_dim, ff_embed_dim, num_heads, dropout, output_dim):
        super(ProjEncoder, self).__init__()
        self.encoder = MonoEncoder(vocab, layers, embed_dim, ff_embed_dim, num_heads, dropout)
        self.proj = nn.Linear(embed_dim, output_dim)
        self.dropout = dropout
        self.output_dim = output_dim
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
        ret = self.proj(ret)
        ret = layer_norm(ret)
        if return_src:
            return ret, src, src_mask
        return ret

    @classmethod
    def from_pretrained(cls, vocab, model_args, ckpt):
        model = cls(vocab, model_args.layers, model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout, model_args.output_dim)
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        return model


def batchify(data, vocab):

    tokens = [[BOS] + x for x in data]

    token = ListsToTensor(tokens, vocab)

    return token

class DataLoader(object):
    def __init__(self, used_data, vocab, batch_size, max_seq_len=256):
        self.vocab = vocab
        self.batch_size = batch_size

        data = []
        for x in used_data:
            x = x.split()[:max_seq_len]
            data.append(x)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        indices = np.arange(len(self))

        cur = 0
        while cur < len(indices):
            data = [self.data[i] for i in indices[cur:cur+self.batch_size]]
            cur += self.batch_size
            yield batchify(data, self.vocab)

@torch.no_grad()
def get_features(batch_size, norm_th, vocab, model, used_data, used_ids, max_norm=None, max_norm_cf=1.0):
    vecs, ids = [], []
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.eval()
    data_loader = DataLoader(used_data, vocab, batch_size)
    cur, tot = 0, len(used_data)
    for batch in asynchronous_load(data_loader):
        batch = move_to_device(batch, torch.device('cuda', 0)).t()
        bsz = batch.size(0)
        cur_vecs = model(batch, batch_first=True).detach().cpu().numpy()
        valid = np.linalg.norm(cur_vecs, axis=1) <= norm_th
        vecs.append(cur_vecs[valid])
        ids.append(used_ids[cur:cur+batch_size][valid])
        cur += bsz
        logger.info("%d / %d", cur, tot)
    vecs = np.concatenate(vecs, 0)
    ids = np.concatenate(ids, 0)
    out, max_norm = augment_data(vecs, max_norm, max_norm_cf)
    return out, ids, max_norm
