dataset=/apdcephfs/private_jcykcai/fren
ckpt=${dataset}/ckpt.2gpus/epoch19_batch59999
python3 /apdcephfs/private_jcykcai/copy/work.py --load_path ${ckpt} \
        --test_data ${dataset}/dev.txt \
        --output_path ${dataset}/dev.out.txt


python3 /apdcephfs/private_jcykcai/copy/work.py --load_path ${ckpt} \
        --test_data ${dataset}/test.txt \
        --output_path ${dataset}/test.out.txt

dataset=/apdcephfs/private_jcykcai/deen
ckpt=${dataset}/ckpt.2gpus/epoch19_batch60999
python3 /apdcephfs/private_jcykcai/copy/work.py --load_path ${ckpt} \
        --test_data ${dataset}/dev.txt \
        --output_path ${dataset}/dev.out.txt

python3 /apdcephfs/private_jcykcai/copy/work.py --load_path ${ckpt} \
        --test_data ${dataset}/test.txt \
        --output_path ${dataset}/test.out.txt
