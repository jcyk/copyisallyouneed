import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from transformer import MultiheadAttention, Transformer

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

    def forward(self, outs, data, label_smoothing=0., work=False):
        lprobs = F.log_softmax(self.output_projection(outs), -1)
        
        if work:
            return lprobs
        loss, nll_loss = label_smoothed_nll_loss(lprobs, data['tgt_tokens_out'], self.label_smoothing, ignore_index=self.vocabs['tgt'].padding_idx, sum=True)
        top1 = torch.argmax(lprobs, -1)
        acc = torch.eq(top1, data['tgt_tokens_out']).float().sum().item()
        loss = loss / data['tgt_num_tokens']
        return loss, acc
