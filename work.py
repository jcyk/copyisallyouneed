import torch
import sacrebleu
import json, re, logging
import numpy as np

from data import Vocab, DataLoader, BOS, EOS
from generator import Generator, MemGenerator, RetrieverGenerator
from utils import move_to_device
from retriever import Retriever
import argparse, os, time

logger = logging.getLogger(__name__)

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str)
    parser.add_argument('--index_path', type=str, default=None)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--test_batch_size', type=int, default=4096)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--max_time_step', type=int, default=256)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--bt', action='store_true')
    parser.add_argument('--retain_bpe', action='store_true')
    parser.add_argument('--comp_bleu', action='store_true')
    parser.add_argument('--src_vocab_path', type=str, default=None)
    parser.add_argument('--tgt_vocab_path', type=str, default=None)
    # Only for debug and analyze
    parser.add_argument('--dump_path', default=None)
    parser.add_argument('--hot_index', default=None)

    return parser.parse_args()

def generate_batch(model, batch, beam_size, alpha, max_time_step):
    token_batch = []
    beams = model.work(batch, beam_size, max_time_step)
    for beam in beams:
        best_hyp = beam.get_k_best(1, alpha)[0]
        predicted_token = [token for token in best_hyp.seq[1:-1]]
        token_batch.append(predicted_token)
    return token_batch, batch['indices']

def validate(device, model, test_data, beam_size=5, alpha=0.6, max_time_step=100, dump_path=None):
    """For Development Only"""

    ref_stream = []
    sys_stream = []
    topk_sys_retr_stream = []
    for batch in test_data:
        batch = move_to_device(batch, device)
        res, _ = generate_batch(model, batch, beam_size, alpha, max_time_step)
        sys_stream.extend(res)
        ref_stream.extend(batch['tgt_raw_sents'])
        sys_retr = batch.get('retrieval_raw_sents', None)
        if sys_retr:
            topk_sys_retr_stream.extend(sys_retr)

    assert len(sys_stream) == len(ref_stream)

    sys_stream = [ re.sub(r'(@@ )|(@@ ?$)', '', ' '.join(o)) for o in sys_stream]
    ref_stream = [ re.sub(r'(@@ )|(@@ ?$)', '', ' '.join(o)) for o in ref_stream]
    ref_streams = [ref_stream]

    bleu = sacrebleu.corpus_bleu(sys_stream, ref_streams, 
                          force=True, lowercase=False,
                          tokenize='none').score
    sys_retr_streams = []
    if topk_sys_retr_stream:
        assert len(topk_sys_retr_stream) == len(ref_stream)
        topk = len(topk_sys_retr_stream[0])
        for i in range(topk):
            sys_retr_stream = [ re.sub(r'(@@ )|(@@ ?$)', '', ' '.join(o[i])) for o in topk_sys_retr_stream]
            lratio = []
            for aa, bb in zip(sys_retr_stream, ref_stream):
                laa = len(aa.split())
                lbb = len(bb.split())
                lratio.append(max(laa/lbb, lbb/laa))
            bleu_retr = sacrebleu.corpus_bleu(sys_retr_stream, ref_streams, 
                              force=True, lowercase=False,
                              tokenize='none').score
            sys_retr_streams.append(sys_retr_stream)
            logger.info("Retrieval top%d bleu %.2f length ratio %.2f", i+1, bleu_retr, sum(lratio)/len(lratio))
        # logger.info("show some examples >>>")
        # for sample_id in [5, 6, 11, 22, 33, 44, 55, 66, 555, 666]:
        #     retrieval = [ "%d: %s"%(i, sys_retr_streams[i][sample_id]) for i in range(topk)]
        #     logger.info("%d: %s###\n generation: %s###\nretrieval:\n %s", sample_id, ref_stream[sample_id], sys_stream[sample_id], '\n'.join(retrieval))
        # logger.info("<<< show some examples")
    if dump_path is not None:
        results = {'sys_stream': sys_stream,
        'ref_stream' : ref_stream,
        'sys_retr_streams': sys_retr_streams}
        json.dump(results, open(dump_path, 'w'))
    return bleu

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

    args = parse_config()
    if args.bt:
        args.retain_bpe = True
        args.comp_bleu = False

    test_models = []
    if os.path.isdir(args.load_path):
        for file in os.listdir(args.load_path):
            fname = os.path.join(args.load_path, file)
            if os.path.isfile(fname):
                test_models.append(fname)
        model_args = torch.load(fname)['args']  
    else:
        test_models.append(args.load_path)
        model_args = torch.load(args.load_path)['args']
    vocabs = dict()
    vocabs['src'] = Vocab(args.src_vocab_path if args.src_vocab_path else model_args.src_vocab, 0, [BOS, EOS])
    vocabs['tgt'] = Vocab(args.tgt_vocab_path if args.tgt_vocab_path else model_args.tgt_vocab, 0, [BOS, EOS])

    if args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)

    if model_args.arch == 'vanilla':
        model = Generator(vocabs,
            model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout,
            model_args.enc_layers, model_args.dec_layers, model_args.label_smoothing)
    elif model_args.arch == 'mem':
        model = MemGenerator(vocabs,
            model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout, model_args.mem_dropout,
            model_args.enc_layers, model_args.dec_layers, model_args.mem_enc_layers, model_args.label_smoothing, model_args.use_mem_score)
    elif model_args.arch == 'rg':
        retriever = Retriever.from_pretrained(model_args.num_retriever_heads, vocabs, args.index_path if args.index_path else model_args.retriever, model_args.nprobe, model_args.topk, args.device, use_response_encoder=(model_args.rebuild_every > 0))
        model = RetrieverGenerator(vocabs, retriever, model_args.share_encoder,
                model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout, model_args.mem_dropout,
                model_args.enc_layers, model_args.dec_layers, model_args.mem_enc_layers, model_args.label_smoothing)

        if args.hot_index is not None:
            model.retriever.drop_index()
            torch.cuda.empty_cache()
            model.retriever.update_index(args.hot_index, model_args.nprobe)

    test_data = DataLoader(vocabs, args.test_data, args.test_batch_size, for_train=False)

    for test_model in test_models:
        model.load_state_dict(torch.load(test_model)['model'])
        model = model.to(device)
        model.eval()
        if args.comp_bleu:
            bleu = validate(device, model, test_data, beam_size=args.beam_size, alpha=args.alpha, max_time_step=args.max_time_step, dump_path=args.dump_path)
            logger.info("%s %s %.2f", test_model, args.test_data, bleu)
        

        if args.output_path is not None:
            start_time = time.time()
            TOT = len(test_data)
            DONE = 0
            logger.info("%d/%d", DONE, TOT)
            outs, indices = [], []
            for batch in test_data:
                batch = move_to_device(batch, device)
                res, ind = generate_batch(model, batch, args.beam_size, args.alpha, args.max_time_step)
                for out_tokens, index in zip(res, ind):
                    if args.retain_bpe:
                        out_line = ' '.join(out_tokens)
                    else:
                        out_line = re.sub(r'(@@ )|(@@ ?$)', '', ' '.join(out_tokens))
                    DONE += 1
                    if DONE % 10000 == -1 % 10000:
                        logger.info("%d/%d", DONE, TOT)
                    outs.append(out_line)
                    indices.append(index)
            end_time = time.time()
            logger.info("Time elapsed: %f", end_time - start_time)
            order = np.argsort(np.array(indices))
            with open(args.output_path, 'w') as fo:
                for i in order:
                    out_line = outs[i]
                    fo.write(out_line+'\n')
