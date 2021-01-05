set -e


dataset=/apdcephfs/private_jcykcai/esen
ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/esen/ckpt.vanilla.2.4/epoch60_batch91999_devbleu62.76_testbleu61.79


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
