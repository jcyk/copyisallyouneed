dataset=/apdcephfs/private_jcykcai/esen
ckpt=${dataset}/ckpt.rg/epoch1_batch5999_devbleu53.41
python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
        --test_data ${dataset}/small.dev.txt \
        --output_path ${dataset}/small.dev.final.out.txt \
        --test_batch_size 32

