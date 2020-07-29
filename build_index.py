import argparse
import random, os
import logging
import torch
import numpy as np

from mips import MIPS, augment_data
from utils import move_to_device, asynchronous_load
from retriever import ProjEncoder
from data import Vocab, BOS, EOS, ListsToTensor

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str)
    parser.add_argument('--args_path', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--vocab_path', type=str)

    parser.add_argument('--index_path', type=str,
        help='can be saving path if train_index == True else loading path')
    parser.add_argument('--train_index', type=bool, default=True, 
        help='whether to train an index from scratch')
    parser.add_argument('--add_to_index', type=bool, default=True,
        help='whether to add instances to the to-be-trained/exsiting index')

    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--index_type', type=str, default='IVF1024_HNSW32,SQ8')
    parser.add_argument('--efSearch', type=int, default=128)
    parser.add_argument('--nprobe', type=int, default=64)
    parser.add_argument('--max_training_instances', type=int, default=1000000)
    
    parser.add_argument('--max_norm', type=float, default=None,
        help='if given, use it as max_norm in ip_to_l2 tranformation')
    parser.add_argument('--max_norm_cf', type=float, default=1.0,
        help='if max_norm is not given, max_norm = max_norm_in_training * max_norm_cf')
    parser.add_argument('--norm_th', type=float, default=999,
        help='will discard a vector if its norm is bigger than this value')
    parser.add_argument('--dump_every', type=int, default=100000)
    return parser.parse_args()

def batchify(data, vocab):

    tokens = [[EOS] + x for x in data]

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
def get_features(args, vocab, model, used_data, used_ids, max_norm=None, max_norm_cf=1.0):
    vecs, ids = [], []
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.eval()
    data_loader = DataLoader(used_data, vocab, args.batch_size)
    cur, tot = 0, len(used_data)
    for batch in asynchronous_load(data_loader):
        batch = move_to_device(batch, torch.device('cuda', 0)).t()
        bsz = batch.size(0)
        cur_vecs = model(batch, batch_first=True).detach().cpu().numpy()
        valid = np.linalg.norm(cur_vecs, axis=1) <= args.norm_th
        vecs.append(cur_vecs[valid])
        ids.append(used_ids[cur:cur+args.batch_size][valid])
        cur += bsz
        logger.info("%d / %d", cur, tot)
    vecs = np.concatenate(vecs, 0)
    ids = np.concatenate(ids, 0)
    out, max_norm = augment_data(vecs, max_norm, max_norm_cf)
    return out, ids, max_norm

def main(args):
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
    logger.info('Loading model...')
    logger.info("using %d gpus", torch.cuda.device_count())
    device = torch.device('cuda', 0)

    vocab = Vocab(args.vocab_path, 0, [BOS, EOS])
    model_args = torch.load(args.args_path)
    model = ProjEncoder.from_pretrained(vocab, model_args, args.ckpt_path)
    model.to(device)

    logger.info('Collecting data...')
    data = []
    line_id = -1
    with open(args.input_file) as f:
        for line in f.readlines():
            r = line.strip()
            line_id += 1
            data.append([r, line_id])

    logger.info('Collected %d instances', len(data))
    max_norm = args.max_norm
    if args.train_index:
        random.shuffle(data)
        used_data = [x[0] for x in data[:args.max_training_instances]]
        used_ids = np.array([x[1] for x in data[:args.max_training_instances]])
        logger.info('Computing feature for training')
        used_data, _, max_norm = get_features(args, vocab, model, used_data, used_ids, max_norm_cf=args.max_norm_cf)
        logger.info('Using %d instances for training', used_data.shape[0])
        mips = MIPS(model_args.output_dim+1, args.index_type, efSearch=args.efSearch, nprobe=args.nprobe) 
        mips.to_gpu()
        mips.train(used_data)
        mips.to_cpu()
        mips.save(args.index_path)
        torch.save(max_norm, os.path.join(os.path.dirname(args.index_path), 'max_norm.pt'))
    else:
        mips = MIPS.from_built(args.index_path, nprobe=args.nprobe)
        max_norm = torch.load(os.path.join(os.path.dirname(args.index_path), 'max_norm.pt'))

    if args.add_to_index:
        cur = 0
        while cur < len(data):
            used_data = [x[0] for x in data[cur:cur+args.dump_every]]
            used_ids = np.array([x[1] for x in data[cur:cur+args.dump_every]])
            cur += args.dump_every
            logger.info('Computing feature for indexing')
            used_data, used_ids, _ = get_features(args, vocab, model, used_data, used_ids, max_norm)
            logger.info('Adding %d instances to index', used_data.shape[0])
            mips.add_with_ids(used_data, used_ids)
        mips.save(args.index_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
