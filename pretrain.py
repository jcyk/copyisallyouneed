import torch
import argparse, os, logging
import random
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np

from data import Vocab, BOS, EOS, ListsToTensor, _back_to_txt_for_check
from optim import Adam, get_linear_schedule_with_warmup
from utils import move_to_device, set_seed, average_gradients, Statistics
from retriever import MatchingModel


logger = logging.getLogger(__name__)

def parse_config():
    parser = argparse.ArgumentParser()
    # vocabs
    parser.add_argument('--src_vocab', type=str, default='es.vocab')
    parser.add_argument('--tgt_vocab', type=str, default='en.vocab')

    # architecture
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--ff_embed_dim', type=int, default=2048)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--output_dim', type=int, default=256)

    # dropout / label_smoothing
    parser.add_argument('--worddrop', type=float, default=0.33)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    # training
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--additional_negs', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--total_train_steps', type=int, default=100000)
    parser.add_argument('--warmup_steps', type=int, default=4000)
    parser.add_argument('--per_gpu_train_batch_size', type=int, default=4096)
    parser.add_argument('--dev_batch_size', type=int, default=4096)

    # IO
    parser.add_argument('--train_data', type=str, default='dev.txt')
    parser.add_argument('--dev_data', type=str, default='dev.txt')
    parser.add_argument('--ckpt', type=str, default='ckpt')
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=5000)

    # distributed training
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--MASTER_ADDR', type=str, default='localhost')
    parser.add_argument('--MASTER_PORT', type=str, default='55555')
    parser.add_argument('--start_rank', type=int, default=0)

    return parser.parse_args()

def batchify(data, vocabs, worddrop):

    src_tokens = [[BOS] + x['src_tokens'] for x in data]
    tgt_tokens = [[BOS] + x['tgt_tokens'] for x in data]

    if 'adt_tokens' in data[0]:
        adt_tokens = [[BOS] + x['adt_tokens'] for x in data]
        tgt_tokens = tgt_tokens + adt_tokens

    src_token = ListsToTensor(src_tokens, vocabs['src'], worddrop)
    tgt_token = ListsToTensor(tgt_tokens, vocabs['tgt'], worddrop)

    ret = {
        'src_tokens': src_token,
        'tgt_tokens': tgt_token,
    }
    return ret

class DataLoader(object):
    def __init__(self, vocabs, filename, batch_size, worddrop=0., max_seq_len=256, addition=True):
        self.vocabs = vocabs
        self.batch_size = batch_size
        self.worddrop = worddrop
        self.addition = addition

        src_tokens, tgt_tokens = [], []
        adt_sents = []
        for line in open(filename).readlines():
            if self.addition:
                src, tgt, *adt = line.strip().split('\t')
                adt = [ x.split()[:max_seq_len] for x in adt[::2] ]
                adt_sents.append(adt)
            else:
                src, tgt = line.strip().split('\t')
            src, tgt = src.split()[:max_seq_len], tgt.split()[:max_seq_len]
            src_tokens.append(src)
            tgt_tokens.append(tgt)

        self.src = src_tokens
        self.tgt = tgt_tokens
        self.adt = adt_sents

    def __len__(self):
        return len(self.src)

    def __iter__(self):
        if self.train:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        
        cur = 0
        while cur < len(indices):
            if self.addition:
                data = [{'src_tokens':self.src[i], 'tgt_tokens':self.tgt[i], 'adt_tokens':random.choice(self.adt[i])} for i in indices[cur:cur+self.batch_size]]
            else:
                data = [{'src_tokens':self.src[i], 'tgt_tokens':self.tgt[i]} for i in indices[cur:cur+self.batch_size]]
            cur += self.batch_size
            yield batchify(data, self.vocabs, self.worddrop)

@torch.no_grad()
def validate(model, dev_data, device):
    model.eval()
    q_list = []
    r_list = []
    for batch in dev_data:
        batch = move_to_device(batch, device)
        q = model.query_encoder(batch['src_tokens'])
        r = model.response_encoder(batch['tgt_tokens'])
        q_list.append(q)
        r_list.append(r)
    q = torch.cat(q_list, dim=0)
    r = torch.cat(r_list, dim=0)

    bsz = q.size(0)
    scores = torch.mm(q, r.t()) # bsz x bsz
    gold = torch.arange(bsz, device=scores.device)
    _, pred = torch.max(scores, -1)
    acc = torch.sum(torch.eq(gold, pred).float()) / bsz
    return acc

def main(args, local_rank):

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

    vocabs = dict()
    vocabs['src'] = Vocab(args.src_vocab, 0, [BOS, EOS])
    vocabs['tgt'] = Vocab(args.tgt_vocab, 0, [BOS, EOS])

    if args.world_size == 1 or (dist.get_rank() == 0):
        logger.info(args)
        for name in vocabs:
            logger.info("vocab %s, size %d, coverage %.3f", name, vocabs[name].size, vocabs[name].coverage)

    set_seed(19940117)

    #device = torch.device('cpu')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    
    if args.resume_ckpt:
        model = MatchingModel.from_pretrained(vocabs, args.resume_ckpt)
    else:
        model = MatchingModel.from_params(vocabs, args.layers, args.embed_dim, args.ff_embed_dim, args.num_heads, args.dropout, args.output_dim)

    if args.world_size > 1:
        set_seed(19940117 + dist.get_rank())

    model = model.to(device)

    if args.resume_ckpt:
        dev_data = DataLoader(vocabs, args.dev_data, args.dev_batch_size, addition=True)
        acc = validate(model, dev_data, device)
        logger.info("initialize from %s, initial acc %.2f", args.resume_ckpt, acc)

    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    lr_schedule = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, args.total_train_steps)
    train_data = DataLoader(vocabs, args.train_data, args.per_gpu_train_batch_size, worddrop=args.worddrop, addition=True)
    global_step, step, epoch = 0, 0, 0
    tr_stat = Statistics()
    logger.info("start training")
    model.train()
    while global_step <= args.total_train_steps:
        for batch in train_data:
            batch = move_to_device(batch, device)
            loss, acc, bsz = model(batch['src_tokens'], batch['tgt_tokens'], args.label_smoothing)
            tr_stat.update({'loss':loss.item() * bsz,
                            'nsamples': bsz,
                            'acc':acc * bsz})
            tr_stat.step()
            loss.backward()

            step += 1
            if not (step % args.gradient_accumulation_steps == -1 % args.gradient_accumulation_steps):
                continue

            if args.world_size > 1:
                average_gradients(model)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_schedule.step()
            optimizer.zero_grad()
            global_step += 1

            if args.world_size == 1 or (dist.get_rank() == 0):
                if global_step % args.print_every == -1 % args.print_every:
                    logger.info("epoch %d, step %d, loss %.3f, acc %.3f", epoch, global_step, tr_stat['loss']/tr_stat['nsamples'], tr_stat['acc']/tr_stat['nsamples'])
                    tr_stat = Statistics()
                if global_step > args.warmup_steps and global_step % args.eval_every == -1 % args.eval_every:
                    dev_data = DataLoader(vocabs, args.dev_data, args.dev_batch_size, addition=True)
                    acc = validate(model, dev_data, device)
                    logger.info("epoch %d, step %d, dev, dev acc %.2f", epoch, global_step, acc)
                    save_path = '%s/epoch%d_batch%d_acc%.2f'%(args.ckpt, epoch, global_step, acc)
                    model.save(args, save_path)
                    model.train()
            if global_step > args.total_train_steps:
                break
        epoch += 1
    logger.info('rank %d, finish training after %d steps', local_rank, global_step)

def init_processes(local_rank, args, backend='nccl'):
    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT
    dist.init_process_group(backend, rank=args.start_rank+local_rank, world_size=args.world_size)
    main(args, local_rank)

if __name__ == "__main__":
    args = parse_config()
    if not os.path.exists(args.ckpt):
        os.mkdir(args.ckpt)

    if args.world_size == 1:
        main(args, 0)
        exit(0)

    mp.spawn(init_processes, args=(args,), nprocs=args.gpus)
