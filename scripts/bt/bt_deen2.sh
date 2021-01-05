set -e



dataset=/apdcephfs/private_jcykcai/deen
ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/deen/ckpt.vanilla.2.4/epoch64_batch97999_devbleu57.94_testbleu58.47




python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
        --test_data ${dataset}/train.txt \
        --output_path ${dataset}/bt.greedy.2.4.train.txt \
        --bt \
        --beam_size 1

python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
        --test_data ${dataset}/train.txt \
        --output_path ${dataset}/bt.beam.2.4.train.txt \
        --bt \
        --beam_size 5
