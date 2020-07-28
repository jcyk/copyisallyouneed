dataset=/apdcephfs/private_jcykcai/esen
ckpt=${dataset}/ckpt/epoch27_batch1999
python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
        --test_data ${dataset}/dev.mem.txt \
        --output_path ${dataset}/dev.mem.out.txt \
        --test_batch_size 32

