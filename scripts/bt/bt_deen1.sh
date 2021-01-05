set -e



dataset=/apdcephfs/private_jcykcai/deen
ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/deen/ckpt.vanilla.1.4/epoch125_batch95999_devbleu54.41_testbleu54.81


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
