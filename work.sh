dataset=/apdcephfs/private_jcykcai/esen
ckpt=${dataset}/ckpt.mem.4gpus.memdrop0.1.new/epoch25_batch78999_devbleu65.06
python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
        --test_data ${dataset}/dev.mem.txt \
        --output_path ${dataset}/dev.mem.out.txt \
        --test_batch_size 32

