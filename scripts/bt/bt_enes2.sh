set -e

dataset=/apdcephfs/private_jcykcai/enes
ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/enes/ckpt.vanilla.2.4/epoch65_batch97999_devbleu60.40_testbleu59.78

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