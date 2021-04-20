set -e

ckpt=${MTPATH}/mt.ckpts/esen/ckpt.exp.dynamic/epoch29_batch88999_devbleu67.66_testbleu67.16
dataset=${MTPATH}/esen
python3 work.py --load_path ${ckpt} \
       --test_data ${dataset}/test.txt \
       --src_vocab_path ${dataset}/src.vocab \
       --tgt_vocab_path ${dataset}/tgt.vocab \
       --output_path ${dataset}/test.out.fixed.txt \
       --comp_bleu

ckpt=${MTPATH}/mt.ckpts/enes/ckpt.exp.dynamic/epoch27_batch81999_devbleu63.73_testbleu63.22
dataset=${MTPATH}/enes
python3 work.py --load_path ${ckpt} \
       --test_data ${dataset}/test.txt \
       --src_vocab_path ${dataset}/src.vocab \
       --tgt_vocab_path ${dataset}/tgt.vocab \
       --output_path ${dataset}/test.out.fixed.txt  \
       --comp_bleu

ckpt=${MTPATH}/mt.ckpts/ende/ckpt.exp.dynamic/epoch32_batch98999_devbleu58.12_testbleu57.92
dataset=${MTPATH}/ende
python3 work.py --load_path ${ckpt} \
       --test_data ${dataset}/test.txt \
       --src_vocab_path ${dataset}/src.vocab \
       --tgt_vocab_path ${dataset}/tgt.vocab \
       --output_path ${dataset}/test.out.fixed.txt  \
       --comp_bleu

ckpt=${MTPATH}/mt.ckpts/deen/ckpt.exp.dynamic/epoch32_batch98999_devbleu64.39_testbleu64.01
dataset=${MTPATH}/deen
python3 work.py --load_path ${ckpt} \
       --test_data ${dataset}/test.txt \
       --src_vocab_path ${dataset}/src.vocab \
       --tgt_vocab_path ${dataset}/tgt.vocab \
       --output_path ${dataset}/test.out.fixed.txt  \
       --comp_bleu

#--index_path ${MTPATH}/mt.ckpts/multi_domain/ckpt.exp.pretrain/epoch40_batch99999_acc0.80_full
