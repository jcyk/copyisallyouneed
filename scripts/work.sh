set -e

ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/esen/ckpt.exp.dynamic/epoch29_batch88999_devbleu67.66_testbleu67.16
dataset=/apdcephfs/private_jcykcai/esen
python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
       --test_data ${dataset}/test.txt \
       --dump_path ${dataset}/test.dump.json \
       --comp_bleu

ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/enes/ckpt.exp.dynamic/epoch27_batch81999_devbleu63.73_testbleu63.22
dataset=/apdcephfs/private_jcykcai/enes
python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
       --test_data ${dataset}/test.txt \
       --dump_path ${dataset}/test.dump.json \
       --comp_bleu

ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/ende/ckpt.exp.dynamic/epoch32_batch98999_devbleu58.12_testbleu57.92
dataset=/apdcephfs/private_jcykcai/ende
python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
       --test_data ${dataset}/test.txt \
       --dump_path ${dataset}/test.dump.json \
       --comp_bleu

ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/deen/ckpt.exp.dynamic/epoch32_batch98999_devbleu64.39_testbleu64.01
dataset=/apdcephfs/private_jcykcai/deen
python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
       --test_data ${dataset}/test.txt \
       --dump_path ${dataset}/test.dump.json \
       --comp_bleu

#--index_path /apdcephfs/share_916081/jcykcai/mt.ckpts/multi_domain/ckpt.exp.pretrain/epoch40_batch99999_acc0.80_full
