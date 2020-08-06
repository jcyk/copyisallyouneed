import argparse
import random, os
import logging
import torch
import json

from retriever import MatchingModel
from data import Vocab, BOS, EOS

logger = logging.getLogger(__name__)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_vocab', type=str, default='../esen/src.vocab')
    parser.add_argument('--tgt_vocab', type=str, default='../esen/tgt.vocab')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
    args = parse_args()
    logger.info('Loading model...')
    vocabs = dict()
    vocabs['src'] = Vocab(args.src_vocab, 0, [BOS, EOS])
    vocabs['tgt'] = Vocab(args.tgt_vocab, 0, [BOS, EOS])
    
    model = MatchingModel.from_pretrained(vocabs, args.input_path)
    torch.save(model.query_encoder.encoder.state_dict(), args.output_path)
    

