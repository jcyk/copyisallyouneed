set -e

ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/esen/ckpt.exp.dynamic.qr/epoch29_batch89999_devbleu67.73_testbleu67.42
dataset=/apdcephfs/private_jcykcai/esen
index_path=/apdcephfs/share_916081/jcykcai/mt.ckpts/esen/ckpt.exp.dynamic.qr/batch86999
python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
       --hot_index ${index_path} \
       --test_data ${dataset}/test.txt \
       --dump_path ${dataset}/test.dump.full.json \
       --comp_bleu

ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/enes/ckpt.exp.dynamic.qr/epoch28_batch86999_devbleu64.18_testbleu63.86
dataset=/apdcephfs/private_jcykcai/enes
index_path=/apdcephfs/share_916081/jcykcai/mt.ckpts/enes/ckpt.exp.dynamic.qr/batch83999
python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
       --hot_index ${index_path} \
       --test_data ${dataset}/test.txt \
       --dump_path ${dataset}/test.dump.full.json \
       --comp_bleu

ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/deen/ckpt.exp.dynamic.qr/epoch24_batch74999_devbleu64.48_testbleu64.62
dataset=/apdcephfs/private_jcykcai/deen
index_path=/apdcephfs/share_916081/jcykcai/mt.ckpts/deen/ckpt.exp.dynamic.qr/batch71999
python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
       --hot_index ${index_path} \
       --test_data ${dataset}/test.txt \
       --dump_path ${dataset}/test.dump.full.json \
       --comp_bleu

ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/ende/ckpt.exp.dynamic.qr/epoch24_batch75999_devbleu58.77_testbleu58.42
dataset=/apdcephfs/private_jcykcai/ende
index_path=/apdcephfs/share_916081/jcykcai/mt.ckpts/ende/ckpt.exp.dynamic.qr/batch74999
python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
       --hot_index ${index_path} \
       --test_data ${dataset}/test.txt \
       --dump_path ${dataset}/test.dump.full.json \
       --comp_bleu
