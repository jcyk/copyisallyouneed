import argparse
import random, os
import logging
import torch
import numpy as np

from mips import MIPS
from retriever import ProjEncoder, get_features
from data import Vocab, BOS, EOS, ListsToTensor

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--only_dump_feat', action='store_true')
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
    parser.add_argument('--add_every', type=int, default=1000000)
    return parser.parse_args() 

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

    if args.only_dump_feat:
        max_norm = torch.load(os.path.join(os.path.dirname(args.index_path), 'max_norm.pt'))
        used_data = [x[0] for x in data]
        used_ids = np.array([x[1] for x in data])
        used_data, used_ids, _ = get_features(args.batch_size, args.norm_th, vocab, model, used_data, used_ids, max_norm)
        used_data = used_data[:,1:]
        assert (used_ids == np.sort(used_ids)).all()
        logger.info('Dumping %d instances', used_data.shape[0])
        torch.save(torch.from_numpy(used_data), os.path.join(os.path.dirname(args.index_path), 'feat.pt')) 
        exit(0)


    logger.info('Collected %d instances', len(data))
    max_norm = args.max_norm
    if args.train_index:
        random.shuffle(data)
        used_data = [x[0] for x in data[:args.max_training_instances]]
        used_ids = np.array([x[1] for x in data[:args.max_training_instances]])
        logger.info('Computing feature for training')
        used_data, used_ids, max_norm = get_features(args.batch_size, args.norm_th, vocab, model, used_data, used_ids, max_norm_cf=args.max_norm_cf)
        logger.info('Using %d instances for training', used_data.shape[0])
        mips = MIPS(model_args.output_dim+1, args.index_type, efSearch=args.efSearch, nprobe=args.nprobe) 
        mips.to_gpu()
        mips.train(used_data)
        mips.to_cpu()
        if args.add_to_index:
            mips.add_with_ids(used_data, used_ids)
            data = data[args.max_training_instances:]
        mips.save(args.index_path)
        torch.save(max_norm, os.path.join(os.path.dirname(args.index_path), 'max_norm.pt'))
    else:
        mips = MIPS.from_built(args.index_path, nprobe=args.nprobe)
        max_norm = torch.load(os.path.join(os.path.dirname(args.index_path), 'max_norm.pt'))

    if args.add_to_index:
        cur = 0
        while cur < len(data):
            used_data = [x[0] for x in data[cur:cur+args.add_every]]
            used_ids = np.array([x[1] for x in data[cur:cur+args.add_every]])
            cur += args.add_every
            logger.info('Computing feature for indexing')
            used_data, used_ids, _ = get_features(args.batch_size, args.norm_th, vocab, model, used_data, used_ids, max_norm)
            logger.info('Adding %d instances to index', used_data.shape[0])
            mips.add_with_ids(used_data, used_ids)
        mips.save(args.index_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
