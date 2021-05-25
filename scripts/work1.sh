set -e

ckpt=${MTPATH}/mt.ckpts/esen/ckpt.exp.dynamic.qr/epoch29_batch89999_devbleu67.73_testbleu67.42
dataset=${MTPATH}/esen
index_path=${MTPATH}/mt.ckpts/esen/ckpt.exp.dynamic.qr/batch86999
python3 work.py --load_path ${ckpt} \
       --hot_index ${index_path} \
       --test_data ${dataset}/test.txt \
       --src_vocab_path ${dataset}/src.vocab \
       --tgt_vocab_path ${dataset}/tgt.vocab \
       --output_path ${dataset}/test.out.dynamic.txt \
       --comp_bleu

ckpt=${MTPATH}/mt.ckpts/enes/ckpt.exp.dynamic.qr/epoch28_batch86999_devbleu64.18_testbleu63.86
dataset=${MTPATH}/enes
index_path=${MTPATH}/mt.ckpts/enes/ckpt.exp.dynamic.qr/batch83999
python3 work.py --load_path ${ckpt} \
       --hot_index ${index_path} \
       --test_data ${dataset}/test.txt \
       --src_vocab_path ${dataset}/src.vocab \
       --tgt_vocab_path ${dataset}/tgt.vocab \
       --output_path ${dataset}/test.out.dynamic.txt \
       --comp_bleu

ckpt=${MTPATH}/mt.ckpts/deen/ckpt.exp.dynamic.qr/epoch24_batch74999_devbleu64.48_testbleu64.62
dataset=${MTPATH}/deen
index_path=${MTPATH}/mt.ckpts/deen/ckpt.exp.dynamic.qr/batch71999
python3 work.py --load_path ${ckpt} \
       --hot_index ${index_path} \
       --test_data ${dataset}/test.txt \
       --src_vocab_path ${dataset}/src.vocab \
       --tgt_vocab_path ${dataset}/tgt.vocab \
       --output_path ${dataset}/test.out.dynamic.txt \
       --comp_bleu

ckpt=${MTPATH}/mt.ckpts/ende/ckpt.exp.dynamic.qr/epoch24_batch75999_devbleu58.77_testbleu58.42
dataset=${MTPATH}/ende
index_path=${MTPATH}/mt.ckpts/ende/ckpt.exp.dynamic.qr/batch74999
python3 work.py --load_path ${ckpt} \
       --hot_index ${index_path} \
       --test_data ${dataset}/test.txt \
       --src_vocab_path ${dataset}/src.vocab \
       --tgt_vocab_path ${dataset}/tgt.vocab \
       --output_path ${dataset}/test.out.dynamic.txt \
       --comp_bleu
