import argparse
import random, os
import logging
import numpy as np
import torch
import json

from mips import MIPS, augment_query
from model import BERTEncoder
from data import ArraysToTensor, MAX_LEN, MonoDataLoader, asynchronous_load 
from utils import move_to_device

logger = logging.getLogger(__name__)
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--topk', type=int, default=100)

    parser.add_argument('--vocab_path', type=str, default='../douban/best/best_ckpt/query_encoder')
    parser.add_argument('--ckpt_path', type=str, default='../douban/best/best_ckpt/query_encoder')
    parser.add_argument('--batch_size', type=int, default=1024)

    parser.add_argument('--nprobe', type=int, default=64)
    parser.add_argument('--index_file', type=str, default='../douban/candidate.txt')
    parser.add_argument('--index_path', type=str, default='../douban/best/mips_index')
    
    
    return parser.parse_args()

def main(args):
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
    logger.info('Loading model...')
    device = torch.device('cuda', 0)
    
    vocab = Vocab(args.vocab_path, 0, [EOS])
    model_args = torch.load(args.args_path)
    model = ProjEncoder.from_pretrained(vocab, model_args, ckpt)
    model.to(device)
    
    logger.info('Collecting data...')
    
    data_r = []
    with open(args.index_file) as f:
        for line in f.readlines():
            r = line.strip()
            data_r.append(r)

    data_q = []
    SEP = model.tokenizer.sep_token
    with open(args.input_file, 'r') as f:
        #last_q = None
        for line in f.readlines():
            q = line.strip()
            data_q.append(q)
            #xs = line.strip().split('\t')
            #label = int(xs[0])
            #q = SEP.join(xs[1:-1])
            #if q != last_q:
            #    data_q.append(q)
            #last_q = q

    logger.info('Collected %d instances', len(data_q))
    textq, textr = data_q, data_r
    data_loader = MonoDataLoader(textq, model.tokenizer, args.batch_size, args.trim_left, eval=True)

    mips = MIPS.from_built(args.index_path, nprobe=args.nprobe)
    mips.to_gpu() 
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        
    cur = 0 

    model.eval()
    with open(args.output_file, 'w') as fo:
        for batch in asynchronous_load(data_loader):
            with torch.no_grad():
                q = move_to_device(batch, torch.device('cuda'))
                bsz = q.size(0)
                vecsq = model(q).detach().cpu().numpy()
            vecsq = augment_query(vecsq)
            D, I = mips.search(vecsq, args.topk)
            for i, Ii in enumerate(I):
                item = {'query':textq[cur+i]}
                item['retrieval'] = [{'response':textr[pred]} for pred in Ii]
                fo.write(json.dumps(item)+'\n')
            cur += bsz

if __name__ == "__main__":
    args = parse_args()
    main(args)
