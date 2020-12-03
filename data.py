import random, logging
import torch
from torch import nn
import numpy as np
from utils import move_to_device

PAD, UNK, BOS, EOS = '<PAD>', '<UNK>', '<BOS>', '<EOS>'

logger = logging.getLogger(__name__)

class Vocab(object):
    def __init__(self, filename, min_occur_cnt, specials = None):
        idx2token = [PAD, UNK] + (specials if specials is not None else [])
        num_tot_tokens = 0
        num_invocab_tokens = 0
        for line in open(filename).readlines():
            try:
                token, cnt = line.rstrip('\n').split('\t')
                cnt = int(cnt)
                num_tot_tokens += cnt
            except:
                logger.info("(Vocab)Illegal line:", line)
            if cnt >= min_occur_cnt:
                idx2token.append(token)
                num_invocab_tokens += cnt
        self.coverage = num_invocab_tokens/num_tot_tokens
        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        self._idx2token = idx2token
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]

    @property
    def size(self):
        return len(self._idx2token)

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def padding_idx(self):
        return self._padding_idx

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self._token2idx.get(x, self.unk_idx)

def _back_to_txt_for_check(tensor, vocab, local_idx2token=None):
    for bid, xs in enumerate(tensor.t().tolist()):
        txt = []
        for x in xs:
            if x == vocab.padding_idx:
                break
            if x >= vocab.size:
                assert local_idx2token is not None
                assert local_idx2token[bid] is not None
                tok = local_idx2token[bid][x]
            else:
                tok = vocab.idx2token(x)
            txt.append(tok)
        txt = ' '.join(txt)
        print (txt)
        print ('-'*55)
    print ('='*55)

def ListsToTensor(xs, vocab=None, worddrop=0., local_vocabs=None):
    pad = vocab.padding_idx if vocab else 0

    def toIdx(w, i):
        if vocab is None:
            return w
        if isinstance(w, list):
            return [toIdx(_, i) for _ in w]
        if random.random() < worddrop:
            return vocab.unk_idx
        if local_vocabs is not None:
            local_vocab = local_vocabs[i]
            if (local_vocab is not None) and (w in local_vocab):
                return local_vocab[w]
        return vocab.token2idx(w)

    max_len = max(len(x) for x in xs)
    ys = []
    for i, x in enumerate(xs):
        y = toIdx(x, i) + [pad]*(max_len-len(x))
        ys.append(y)
    data = np.transpose(np.array(ys))
    return data

def ArraysToTensor(xs):
    "list of numpy array, each has the same demonsionality"
    x = np.array([ list(x.shape) for x in xs])
    shape = [len(xs)] + list(x.max(axis = 0))
    data = np.zeros(shape, dtype=np.int)
    for i, x in enumerate(xs):
        slicing_shape = list(x.shape)
        slices = tuple([slice(i, i+1)]+[slice(0, x) for x in slicing_shape])
        data[slices] = x
        #tensor = torch.from_numpy(data).long()
    return data

def batchify(data, vocabs, max_seq_len):

    src_tokens = [ [BOS]+x['src_tokens'][:max_seq_len] for x in data]
    tgt_tokens_in = [[BOS]+x['tgt_tokens'][:max_seq_len] for x in data]
    tgt_tokens_out = [x['tgt_tokens'][:max_seq_len]+[EOS] for x in data]

    src_token = ListsToTensor(src_tokens, vocabs['src'])

    tgt_token_in = ListsToTensor(tgt_tokens_in, vocabs['tgt'])
    tgt_token_out = ListsToTensor(tgt_tokens_out, vocabs['tgt'])

    not_padding = (tgt_token_out != vocabs['tgt'].padding_idx).astype(np.int64)
    tgt_lengths = np.sum(not_padding, axis=0)
    tgt_num_tokens = int(np.sum(tgt_lengths))


    #not_padding = (src_token != vocabs['src'].padding_idx).astype(np.int64)
    #src_lengths = np.sum(not_padding, axis=0)
    ret = {
        'src_tokens': src_token,
        #'src_lengths': src_lengths,
        'tgt_tokens_in': tgt_token_in,
        'tgt_tokens_out': tgt_token_out,
        'tgt_num_tokens': tgt_num_tokens,
        #'tgt_lengths': tgt_lengths,
        'tgt_raw_sents': [x['tgt_tokens'] for x in data],
        'indices': [x['index'] for x in data]
    }

    # only if there is some memory input
    if data[0]['mem_sents']:
        num_mem_sents = len(data[0]['mem_sents'])
        for x in data:
            assert len(x['mem_sents']) == len(x['mem_scores']) == num_mem_sents
        # put all memory tokens (across the same batch) in a single list:
        # No.1 memory for sentence 1, ..., No.1 memory for sentence N,
        # No.2 memory for sentence 1, ...
        all_mem_tokens = []
        all_mem_scores = []
        for i in range(num_mem_sents):
            all_mem_tokens.extend([ [BOS]+x['mem_sents'][i][:max_seq_len] for x in data])
            all_mem_scores.extend([x['mem_scores'][i] for x in data])
        # then convert to tensors:
        # all_mem_tokens -> seq_len x (num_mem_sents * bsz)
        # all_mem_scores -> num_mem_sents * bsz
        ret['all_mem_tokens'] = ListsToTensor(all_mem_tokens, vocabs['tgt'])
        ret['all_mem_scores'] = np.array(all_mem_scores, dtype=np.float32)
        # to avoid GPU OOM issue, truncate the mem to the max. length of 1.5 x src_tokens
        max_mem_len = int(1.5 * src_token.shape[0])
        ret['all_mem_tokens'] = ret['all_mem_tokens'][:max_mem_len,:]
        #ret['retrieval_raw_sents'] = [x['mem_sents'] for x in data]

    return ret

class DataLoader(object):
    def __init__(self, vocabs, filename, batch_size, for_train, max_seq_len=256, rank=0, num_replica=1):
        self.vocabs = vocabs
        self.batch_size = batch_size
        self.train = for_train

        src_tokens, tgt_tokens = [], []
        src_sizes, tgt_sizes = [], []
        mem_sents, mem_scores = [], []
        for line in open(filename).readlines()[rank::num_replica]:
            src, tgt, *mem = line.strip().split('\t')
            src, tgt = src.split(), tgt.split()
            src_sizes.append(len(src))
            tgt_sizes.append(len(tgt))
            src_tokens.append(src)
            tgt_tokens.append(tgt)

            mem_sents.append([ref.split() for ref in mem[:-1:2]])
            mem_scores.append([float(score) for score in mem[1::2]])
        self.src = src_tokens
        self.tgt = tgt_tokens
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes)
        self.max_seq_len = max_seq_len
        self.mem_sents = mem_sents
        self.mem_scores = mem_scores
        logger.info("(DataLoader rank %d) read %s file with %d paris. max src len: %d, max tgt len: %d", rank, filename, len(self.src), self.src_sizes.max(), self.tgt_sizes.max())

    def __len__(self):
        return len(self.src)

    def __iter__(self):
        if self.train:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        
        indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        indices = indices[np.argsort(self.src_sizes[indices], kind='mergesort')]
        
        batches = []
        num_tokens, batch = 0, []
        for i in indices:
            num_tokens += 1 + max(self.src_sizes[i], self.tgt_sizes[i])
            if num_tokens > self.batch_size:
                batches.append(batch)
                num_tokens, batch = 1 + max(self.src_sizes[i], self.tgt_sizes[i]), [i]
            else:
                batch.append(i)

        if not self.train or num_tokens > self.batch_size/2:
            batches.append(batch)

        if self.train:
            random.shuffle(batches)

        for batch in batches:
            data = []
            for i in batch:
                src_tokens = self.src[i]
                tgt_tokens = self.tgt[i]
                mem_sents = self.mem_sents[i]
                mem_scores = self.mem_scores[i]
                item = {'src_tokens':src_tokens, 'tgt_tokens':tgt_tokens, 'mem_sents':mem_sents, 'mem_scores':mem_scores, 'index':i}
                data.append(item)
            yield batchify(data, self.vocabs, self.max_seq_len)

def parse_config():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_vocab', type=str, default='es.vocab')
    parser.add_argument('--tgt_vocab', type=str, default='en.vocab')

    parser.add_argument('--train_data', type=str, default='dev.mem.txt')
    parser.add_argument('--train_batch_size', type=int, default=4096)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_config()
    vocabs = dict()
    vocabs['src'] = Vocab(args.src_vocab, 0, [EOS])
    vocabs['tgt'] = Vocab(args.tgt_vocab, 0, [BOS, EOS])

    train_data = DataLoader(vocabs, args.train_data, args.train_batch_size, for_train=True)
    for d in train_data:
        d = move_to_device(d, torch.device('cpu'))
        for k, v in d.items():
            if 'raw' in k:
                continue
            try:
                print (k, v.shape)
            except:
                print (k, v)
        _back_to_txt_for_check(d['src_tokens'][:,5:6], vocabs['src'])
        _back_to_txt_for_check(d['tgt_tokens_in'][:,5:6], vocabs['tgt'])
        _back_to_txt_for_check(d['tgt_tokens_out'][:,5:6], vocabs['tgt'])
        bsz = d['tgt_tokens_out'].size(1)
        _back_to_txt_for_check(d['all_mem_tokens'][:,5::bsz], vocabs['tgt'])
        break


