dataset=/apdcephfs/private_jcykcai/esen
python3 /apdcephfs/private_jcykcai/copyisallyouneed/train.py --train_data ${dataset}/train.txt \
        --dev_data ${dataset}/dev.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --ckpt ${dataset}/ckpt.share6 \
        --world_size 2 \
        --gpus 2 \
        --arch vanilla \
        --dev_batch_size 2048 \
        --shared_encoder ${dataset}/ckpt.pretrain.6layers/epoch19_batch99999_acc0.98/shared_encoder \
        --per_gpu_train_batch_size 4096
