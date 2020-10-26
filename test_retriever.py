import torch
import argparse, os, logging, time
import random
import torch.multiprocessing as mp
import torch.distributed as dist

from data import Vocab, DataLoader, BOS, EOS
from optim import Adam, get_inverse_sqrt_schedule_with_warmup
from utils import move_to_device, set_seed, average_gradients, Statistics
from generator import Generator, MemGenerator, RetrieverGenerator
from work import validate
from retriever import Retriever, MatchingModel
from pretrain import DataLoader as RetrieverDataLoader
logger = logging.getLogger(__name__)

def parse_config():
    parser = argparse.ArgumentParser()
    # vocabs
    parser.add_argument('--src_vocab', type=str, default='es.vocab')
    parser.add_argument('--tgt_vocab', type=str, default='en.vocab')

    # architecture
    parser.add_argument('--arch', type=str, choices=['vanilla', 'mem', 'rg'], default='vanilla')
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--ff_embed_dim', type=int, default=2048)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--enc_layers', type=int, default=6)
    parser.add_argument('--dec_layers', type=int, default=6)
    parser.add_argument('--mem_enc_layers', type=int, default=4)

    # retriever
    parser.add_argument('--add_retrieval_loss', action='store_true')
    parser.add_argument('--share_encoder', action='store_true')
    parser.add_argument('--retriever', type=str, default=None)
    parser.add_argument('--nprobe', type=int, default=64)
    parser.add_argument('--num_retriever_heads', type=int, default=1)
    parser.add_argument('--topk', type=int, default=5)
 
    # dropout / label_smoothing
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--mem_dropout', type=float, default=0.1)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    
    # IO
    parser.add_argument('--dev_data', type=str, default='dev.txt')
    parser.add_argument('--dev_batch_size', type=int, default=2048)

    return parser.parse_args()

def main(args, local_rank=0):

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

    vocabs = dict()
    vocabs['src'] = Vocab(args.src_vocab, 0, [BOS, EOS])
    vocabs['tgt'] = Vocab(args.tgt_vocab, 0, [BOS, EOS])

    logger.info(args)
    for name in vocabs:
        logger.info("vocab %s, size %d, coverage %.3f", name, vocabs[name].size, vocabs[name].coverage)

    set_seed(19940117)

    #device = torch.device('cpu')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    

    logger.info("start building model")
    logger.info("building retriever")
    if args.add_retrieval_loss:
        retriever, another_model = Retriever.from_pretrained(args.num_retriever_heads, vocabs, args.retriever, args.nprobe, args.topk, local_rank, load_response_encoder=True)
        matchingmodel = MatchingModel(retriever.model, another_model)
        matchingmodel = matchingmodel.to(device)
    else:
        retriever = Retriever.from_pretrained(args.num_retriever_heads, vocabs, args.retriever, args.nprobe, args.topk, local_rank)

    logger.info("building retriever + generator")
    model = RetrieverGenerator(vocabs, retriever, args.share_encoder,
            args.embed_dim, args.ff_embed_dim, args.num_heads, args.dropout, args.mem_dropout,
            args.enc_layers, args.dec_layers, args.mem_enc_layers, args.label_smoothing)


    model = model.to(device)

    model.eval()
    dev_data = DataLoader(vocabs, args.dev_data, args.dev_batch_size, for_train=False)
    bleu = validate(device, model, dev_data, beam_size=5, alpha=0.6, max_time_step=10)


if __name__ == "__main__":
    args = parse_config()
    main(args)
