set -e

ckpt=${MTPATH}/mt.ckpts/esen/ckpt.vanilla/epoch30_batch92999_devbleu64.25_testbleu64.07
dataset=${MTPATH}/esen
python3 work.py --load_path ${ckpt} \
       --test_data ${dataset}/test.txt \
       --dump_path ${dataset}/vanilla.test.dump.json \
       --comp_bleu

ckpt=${MTPATH}/mt.ckpts/enes/ckpt.vanilla/epoch31_batch95999_devbleu62.27_testbleu61.54
dataset=${MTPATH}/enes
python3 work.py --load_path ${ckpt} \
       --test_data ${dataset}/test.txt \
       --dump_path ${dataset}/vanilla.test.dump.json \
       --comp_bleu

ckpt=${MTPATH}/mt.ckpts/ende/ckpt.vanilla/epoch32_batch97999_devbleu55.01_testbleu54.90
dataset=${MTPATH}/ende
python3 work.py --load_path ${ckpt} \
       --test_data ${dataset}/test.txt \
       --dump_path ${dataset}/vanilla.test.dump.json \
       --comp_bleu

ckpt=${MTPATH}/mt.ckpts/deen/ckpt.vanilla/epoch32_batch97999_devbleu59.82_testbleu60.76
dataset=${MTPATH}/deen
python3 work.py --load_path ${ckpt} \
       --test_data ${dataset}/test.txt \
       --dump_path ${dataset}/vanilla.ctest.dump.json \
       --comp_bleu

#--index_path ${MTPATH}/mt.ckpts/multi_domain/ckpt.exp.pretrain/epoch40_batch99999_acc0.80_full
