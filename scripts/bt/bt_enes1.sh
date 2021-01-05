set -e


dataset=/apdcephfs/private_jcykcai/enes
ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/enes/ckpt.vanilla.1.4/epoch103_batch77999_devbleu57.21_testbleu56.19

python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
        --test_data ${dataset}/train.txt \
        --output_path ${dataset}/bt.greedy.1.4.train.txt \
        --bt \
        --beam_size 1

python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
        --test_data ${dataset}/train.txt \
        --output_path ${dataset}/bt.beam.1.4.train.txt \
        --bt \
        --beam_size 5
