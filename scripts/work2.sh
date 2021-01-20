set -e

ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/esen/ckpt.vanilla/epoch30_batch92999_devbleu64.25_testbleu64.07
dataset=/apdcephfs/private_jcykcai/esen
python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
       --test_data ${dataset}/test.txt \
       --dump_path ${dataset}/vanilla.test.dump.json \
       --comp_bleu

ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/enes/ckpt.vanilla/epoch31_batch95999_devbleu62.27_testbleu61.54
dataset=/apdcephfs/private_jcykcai/enes
python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
       --test_data ${dataset}/test.txt \
       --dump_path ${dataset}/vanilla.test.dump.json \
       --comp_bleu

ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/ende/ckpt.vanilla/epoch32_batch97999_devbleu55.01_testbleu54.90
dataset=/apdcephfs/private_jcykcai/ende
python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
       --test_data ${dataset}/test.txt \
       --dump_path ${dataset}/vanilla.test.dump.json \
       --comp_bleu

ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/deen/ckpt.vanilla/epoch32_batch97999_devbleu59.82_testbleu60.76
dataset=/apdcephfs/private_jcykcai/deen
python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
       --test_data ${dataset}/test.txt \
       --dump_path ${dataset}/vanilla.ctest.dump.json \
       --comp_bleu

#--index_path /apdcephfs/share_916081/jcykcai/mt.ckpts/multi_domain/ckpt.exp.pretrain/epoch40_batch99999_acc0.80_full
