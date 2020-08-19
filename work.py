import torch
import sacrebleu
import json, re, logging
import numpy as np

from data import Vocab, DataLoader, BOS, EOS
from generator import Generator, MemGenerator
from utils import move_to_device
import argparse, os

logger = logging.getLogger(__name__)

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--test_batch_size', type=int, default=4096)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--max_time_step', type=int, default=256)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--device', type=int, default=0)

    return parser.parse_args()

def generate_batch(device, model, batch, beam_size, alpha, max_time_step):
    batch = move_to_device(batch, device)
    token_batch = []
    beams = model.work(batch, beam_size, max_time_step)
    for beam in beams:
        best_hyp = beam.get_k_best(1, alpha)[0]
        predicted_token = [token for token in best_hyp.seq[1:-1]]
        token_batch.append(predicted_token)
    return token_batch, batch['indices']

def validate(device, model, test_data, beam_size=5, alpha=0.6, max_time_step=100):
    """For Development Only"""

    ref_stream = []
    sys_stream = []
    sys_retr_stream = []
    for batch in test_data:
        res, _ = generate_batch(device, model, batch, beam_size, alpha, max_time_step)
        sys_stream.extend(res)
        ref_stream.extend(batch['tgt_raw_sents'])
        sys_retr = batch.get('top1_retrieval_raw_sents', None)
        if sys_retr:
            sys_retr_stream.extend(sys_retr)
    assert len(sys_stream) == len(ref_stream)

    sys_stream = [ re.sub(r'(@@ )|(@@ ?$)', '', ' '.join(o)) for o in sys_stream]
    ref_stream = [ re.sub(r'(@@ )|(@@ ?$)', '', ' '.join(o)) for o in ref_stream]
    ref_streams = [ref_stream]

    bleu = sacrebleu.corpus_bleu(sys_stream, ref_streams, 
                          force=True, lowercase=False,
                          tokenize='none').score
    if sys_retr_stream:
        sys_retr_stream = [ re.sub(r'(@@ )|(@@ ?$)', '', ' '.join(o)) for o in sys_retr_stream]
        assert len(sys_retr_stream) == len(ref_stream)
        bleu_retr = sacrebleu.corpus_bleu(sys_retr_stream, ref_streams, 
                          force=True, lowercase=False,
                          tokenize='none').score
        logger.info("Retrieval top1 bleu %.2f", bleu_retr)
    return bleu

if __name__ == "__main__":

    args = parse_config()

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
    vocabs['src'] = Vocab(model_args.src_vocab, 0, [BOS, EOS])
    vocabs['tgt'] = Vocab(model_args.tgt_vocab, 0, [BOS, EOS])

    if args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)

    if model_args.arch == 'mem':
        model = MemGenerator(vocabs,
            model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout, model_args.mem_dropout,
            model_args.enc_layers, model_args.dec_layers, model_args.mem_enc_layers, model_args.label_smoothing)
    else:
        model = Generator(vocabs,
            model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout,
            model_args.enc_layers, model_args.dec_layers, model_args.label_smoothing)

    test_data = DataLoader(vocabs, args.test_data, args.test_batch_size, for_train=False)

    for test_model in test_models:
        print (test_model)
        model.load_state_dict(torch.load(test_model)['model'])
        model = model.to(device)
        model.eval()
        bleu = validate(device, model, test_data, beam_size=args.beam_size, alpha=args.alpha, max_time_step=args.max_time_step)
        print (bleu)
        
        outs, indices = [], []
        for batch in test_data:
            res, ind = generate_batch(device, model, batch, args.beam_size, args.alpha, args.max_time_step)
            for out_tokens, index in zip(res, ind):
                out_line = re.sub(r'(@@ )|(@@ ?$)', '', ' '.join(out_tokens))
                outs.append(out_line)
                indices.append(index)

        order = np.argsort(np.array(indices))
        with open(args.output_path, 'w') as fo:
            for i in order:
                out_line = outs[i]
                fo.write(out_line+'\n')
