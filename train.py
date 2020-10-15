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

    # training
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--total_train_steps', type=int, default=100000)
    parser.add_argument('--warmup_steps', type=int, default=4000)
    parser.add_argument('--per_gpu_train_batch_size', type=int, default=4096)
    parser.add_argument('--dev_batch_size', type=int, default=4096)
    parser.add_argument('--in_batch_negatives', type=int, default=128)
    parser.add_argument('--worddrop', type=float, default=0.33)
    parser.add_argument('--update_retriever_after', default=5000)
    
    # IO
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--train_data', type=str, default='dev.txt')
    parser.add_argument('--dev_data', type=str, default='dev.txt')
    parser.add_argument('--ckpt', type=str, default='ckpt')
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=1000)

    # distributed training
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--MASTER_ADDR', type=str, default='localhost')
    parser.add_argument('--MASTER_PORT', type=str, default='55555')
    parser.add_argument('--start_rank', type=int, default=0)

    return parser.parse_args()

def partially_load(model, ckpt):
    pretrained_dict = torch.load(ckpt)
    model_dict = model.state_dict()
    #model_dict_keys = set(model_dict.keys())
    #pretrained_dict_keys = set(pretrained_dict.keys())
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

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
    
    if args.arch == 'vanilla':
        model = Generator(vocabs,
                args.embed_dim, args.ff_embed_dim, args.num_heads, args.dropout,
                args.enc_layers, args.dec_layers, args.label_smoothing)
    elif args.arch == 'mem':
        model = MemGenerator(vocabs,
                args.embed_dim, args.ff_embed_dim, args.num_heads, args.dropout, args.mem_dropout,
                args.enc_layers, args.dec_layers, args.mem_enc_layers, args.label_smoothing)
    elif args.arch == 'rg':
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

    if args.resume_ckpt:
        model.load_state_dict(torch.load(args.resume_ckpt)['model'])
        global_step = 5000
    else:
        global_step = 0

    if args.world_size > 1:
        set_seed(19940117 + dist.get_rank())

    model = model.to(device)

    retriever_params = [ v for k, v in model.named_parameters() if k.startswith('retriever.')]
    other_params = [ v for k, v in model.named_parameters() if not k.startswith('retriever.')]

    optimizer = Adam([ {'params':retriever_params, 'lr':args.embed_dim**-0.5*0.1},
                       {'params':other_params, 'lr': args.embed_dim**-0.5}], betas=(0.9, 0.98), eps=1e-9)
    lr_schedule = get_inverse_sqrt_schedule_with_warmup(optimizer, args.warmup_steps, args.total_train_steps)
    train_data = DataLoader(vocabs, args.train_data, args.per_gpu_train_batch_size,
                            for_train=True, rank=local_rank, num_replica=args.world_size)
    if args.add_retrieval_loss:

        retr_train_data = RetrieverDataLoader(vocabs, args.train_data, args.in_batch_negatives, worddrop=args.worddrop, for_train=True)
        it_retr_train_data = iter(retr_train_data)
    model.eval()
    dev_data = DataLoader(vocabs, args.dev_data, args.dev_batch_size, for_train=False)
    bleu = validate(device, model, dev_data, beam_size=5, alpha=0.6, max_time_step=10)

    step, epoch = 0, 0
    tr_stat = Statistics()
    logger.info("start training")
    model.train()

    if args.add_retrieval_loss:
        matchingmodel.train()

        for x in matchingmodel.response_encoder.parameters():
            x.requires_grad = False

    
    while global_step <= args.total_train_steps:
        for batch in train_data:
            step_start = time.time()
            batch = move_to_device(batch, device)
            if args.arch == 'rg':
                loss, acc = model(batch, update_mem_bias=(global_step > args.update_retriever_after))
            else:
                loss, acc = model(batch)
            #tr_stat.update({'loss':0., 'tokens':1, 'acc':0.})
            tr_stat.update({'loss':loss.item() * batch['tgt_num_tokens'],
                            'tokens':batch['tgt_num_tokens'],
                            'acc':acc})
            tr_stat.step()
            loss.backward()
            if args.add_retrieval_loss:
                try:
                    batch = next(it_retr_train_data)
                except StopIteration:
                    it_retr_train_data = iter(retr_train_data)
                    batch = next(it_retr_train_data)
                batch = move_to_device(batch, device)
                loss_retrieval, acc_retrieval, bsz_retrieval = matchingmodel(batch['src_tokens'], batch['tgt_tokens'], label_smoothing=args.label_smoothing)
                tr_stat.update({'loss_retrieval':loss_retrieval.item() * bsz_retrieval,
                                'acc_retrieval':acc_retrieval * bsz_retrieval,
                                'bsz_retrieval':bsz_retrieval})
                loss_retrieval.backward()
            step_cost = time.time() - step_start
            #print ('step_cost', step_cost)
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
                    logger.info("epoch %d, step %d, loss %.3f, acc %.3f", epoch, global_step, tr_stat['loss']/tr_stat['tokens'], tr_stat['acc']/tr_stat['tokens'])
                    if args.add_retrieval_loss:
                        logger.info("retrieval loss %.3f, acc %.3f", tr_stat['loss_retrieval']/tr_stat['bsz_retrieval'], tr_stat['acc_retrieval']/tr_stat['bsz_retrieval'])
                    tr_stat = Statistics()
                if global_step % args.eval_every == -1 % args.eval_every:
                    model.eval()
                    if args.add_retrieval_loss:
                        matchingmodel.eval()
                    dev_data = DataLoader(vocabs, args.dev_data, args.dev_batch_size, for_train=False) 
                    max_time_step = 100 if global_step > args.warmup_steps else 5
                    bleu = validate(device, model, dev_data, beam_size=5, alpha=0.6, max_time_step=max_time_step)
                    logger.info("epoch %d, step %d, dev bleu %.2f", epoch, global_step, bleu)
                    torch.save({'args':args, 'model':model.state_dict()}, '%s/epoch%d_batch%d_devbleu%.2f'%(args.ckpt, epoch, global_step, bleu))
                    model.train()
                    if args.add_retrieval_loss:
                        matchingmodel.train()
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
