import torch
from torch import nn
import torch.nn.functional as F

from transformer import MultiheadAttention, FeedForwardLayer
from utils import label_smoothed_nll_loss

class TokenDecoder(nn.Module):
    def __init__(self, vocabs, tgt_embed, label_smoothing):
        super(TokenDecoder, self).__init__()
        self.output_projection = nn.Linear(
                tgt_embed.weight.shape[1],
                tgt_embed.weight.shape[0],
                bias=False,
        )
        self.output_projection.weight = tgt_embed.weight
        self.vocabs = vocabs
        self.label_smoothing = label_smoothing

    def forward(self, outs, data, work=False):
        lprobs = F.log_softmax(self.output_projection(outs), -1)
        
        if work:
            return lprobs
        loss, nll_loss = label_smoothed_nll_loss(lprobs, data['tgt_tokens_out'], self.label_smoothing, ignore_index=self.vocabs['tgt'].padding_idx, sum=True)
        top1 = torch.argmax(lprobs, -1)
        acc = torch.eq(top1, data['tgt_tokens_out']).float().sum().item()
        loss = loss / data['tgt_num_tokens']
        return loss, acc

class CopyTokenDecoder(nn.Module):
    def __init__(self, vocabs, tgt_embed, label_smoothing, embed_dim, ff_embed_dim, dropout):
        super(CopyTokenDecoder, self).__init__()
        self.output_projection = nn.Linear(
                tgt_embed.weight.shape[1],
                tgt_embed.weight.shape[0],
                bias=False,
        )
        self.alignment_layer = MultiheadAttention(embed_dim, 1, dropout, weights_dropout=False)
        self.alignment_layer_norm = nn.LayerNorm(embed_dim)
        self.ff_layer = FeedForwardLayer(embed_dim, ff_embed_dim, dropout)
        self.ff_layer_norm = nn.LayerNorm(embed_dim)
        self.diverter = nn.Linear(2*embed_dim, 2)
        self.output_projection.weight = tgt_embed.weight
        self.vocabs = vocabs
        self.dropout = dropout
        self.label_smoothing = label_smoothing
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.diverter.weight, std=0.02)
        nn.init.constant_(self.diverter.bias, 0.)   

    def forward(self, outs, mem, mem_mask, copy_seq, data, work=False):
        attn, alignment_weight = self.alignment_layer(outs, mem, mem,
                                                    key_padding_mask=mem_mask,
                                                    need_weights='one')
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        attn_normalized = self.alignment_layer_norm(attn)

        gates = F.softmax(self.diverter(torch.cat([outs, attn_normalized], -1)), -1)
        gen_gate, copy_gate = gates.chunk(2, dim=-1)

        outs = self.alignment_layer_norm(outs + attn)
        outs = self.ff_layer(outs)
        outs = F.dropout(outs, p=self.dropout, training=self.training)
        outs = self.ff_layer_norm(outs)

        seq_len, bsz, _ = outs.size()
        probs = gen_gate * F.softmax(self.output_projection(outs), -1)

        #copy_seq: src_len x bsz
        #copy_gate: tgt_len x bsz 
        #alignment_weight: tgt_len x bsz x src_len
        #index: tgt_len x bsz
        index = copy_seq.transpose(0, 1).contiguous().view(1, bsz, -1).expand(seq_len, -1, -1)
        # -> tgt_len x bsz x src_len
        copy_probs = (copy_gate * alignment_weight).view(seq_len, bsz, -1)
        # -> tgt_len x bsz x src_len
        probs = probs.scatter_add_(-1, index, copy_probs)
        lprobs = torch.log(probs + 1e-12)

        if work:
            return lprobs
        loss, nll_loss = label_smoothed_nll_loss(lprobs, data['tgt_tokens_out'], self.label_smoothing, ignore_index=self.vocabs['tgt'].padding_idx, sum=True)
        top1 = torch.argmax(lprobs, -1)
        acc = torch.eq(top1, data['tgt_tokens_out']).float().sum().item()
        loss = loss / data['tgt_num_tokens']
        return loss, acc
