set -e



dataset=/apdcephfs/private_jcykcai/esen
ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/esen/ckpt.vanilla.1.4/epoch112_batch84999_devbleu59.60_testbleu58.69


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