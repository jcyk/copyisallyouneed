import argparse
import random, os
import logging
import numpy as np
import torch
import json

from mips import MIPS, augment_query, l2_to_ip
from retriever import ProjEncoder
from build_index import DataLoader 
from utils import move_to_device, asynchronous_load
from data import Vocab, EOS

logger = logging.getLogger(__name__)
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--topk', type=int, default=10)

    parser.add_argument('--vocab_path', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--args_path', type=str)
    parser.add_argument('--batch_size', type=int, default=1024)

    parser.add_argument('--nprobe', type=int, default=64)
    parser.add_argument('--index_file', type=str)
    parser.add_argument('--index_path', type=str)
    
    
    return parser.parse_args()

def main(args):
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
    logger.info('Loading model...')
    device = torch.device('cuda', 0)
    
    vocab = Vocab(args.vocab_path, 0, [EOS])
    model_args = torch.load(args.args_path)
    model = ProjEncoder.from_pretrained(vocab, model_args, args.ckpt_path)
    model.to(device)
    
    logger.info('Collecting data...')
    
    data_r = []
    with open(args.index_file) as f:
        for line in f.readlines():
            r = line.strip()
            data_r.append(r)

    data_q = []
    with open(args.input_file, 'r') as f:
        for line in f.readlines():
            q = line.strip()
            data_q.append(q)

    logger.info('Collected %d instances', len(data_q))
    textq, textr = data_q, data_r
    data_loader = DataLoader(data_q, vocab, args.batch_size) 

    mips = MIPS.from_built(args.index_path, nprobe=args.nprobe)
    max_norm = torch.load(os.path.dirname(args.index_path)+'/max_norm.pt')
    mips.to_gpu() 
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.eval()

    cur = 0
    with open(args.output_file, 'w') as fo:
        for batch in asynchronous_load(data_loader):
            with torch.no_grad():
                q = move_to_device(batch, torch.device('cuda')).t()
                bsz = q.size(0)
                vecsq = model(q, batch_first=True).detach().cpu().numpy()
            vecsq = augment_query(vecsq)
            D, I = mips.search(vecsq, args.topk)
            D = l2_to_ip(D, vecsq, max_norm) / (max_norm * max_norm)
            for i, (Ii, Di) in enumerate(zip(I, D)):
                item = {'query':textq[cur+i]}
                item['retrieval'] = [{'response':textr[pred], 'score':float(s)} for pred, s in zip(Ii, Di)]
                fo.write(json.dumps(item)+'\n')
            cur += bsz

if __name__ == "__main__":
    args = parse_args()
    main(args)
