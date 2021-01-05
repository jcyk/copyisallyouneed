set -e



dataset=/apdcephfs/private_jcykcai/ende
ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/ende/ckpt.vanilla.1.4/epoch116_batch88999_devbleu49.56_testbleu50.14


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
