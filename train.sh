dataset=/apdcephfs/private_jcykcai/esen
python3 /apdcephfs/private_jcykcai/copyisallyouneed/train.py --train_data ${dataset}/train.mem.txt \
        --dev_data ${dataset}/dev.mem.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --ckpt ${dataset}/ckpt.mem.4gpus.memdrop0.1.new \
        --world_size 4 \
        --gpus 4 \
        --arch mem \
        --mem_dropout 0.1 \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 2048
