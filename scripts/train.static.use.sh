dataset=/apdcephfs/private_jcykcai/esen/full
python3 /apdcephfs/private_jcykcai/copyisallyouneed/train.py --train_data ${dataset}/train.mem.txt \
        --dev_data ${dataset}/dev.mem.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --ckpt /apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.exp.static.use \
        --world_size 2 \
        --gpus 2 \
        --arch mem \
        --use_mem_score \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 4096
