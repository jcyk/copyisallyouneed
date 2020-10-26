dataset=/apdcephfs/private_jcykcai/esen/full
python3 /apdcephfs/private_jcykcai/copyisallyouneed/pretrain.py --train_data ${dataset}/train.txt \
        --dev_data ${dataset}/dev.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --ckpt /apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.pretrain.rnd.0 \
        --world_size 1 \
        --gpus 1 \
        --dev_batch_size 128 \
        --layers 6 \
        --per_gpu_train_batch_size 128 \
        --worddrop 0.
