dataset=${MTPATH}/ende/1.4
python3 pretrain.py --train_data ${dataset}/train.txt \
        --dev_data ${dataset}/dev.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --ckpt ${MTPATH}/mt.ckpts/ende/ckpt.exp.pretrain1.4 \
        --world_size 1 \
        --gpus 1 \
        --dev_batch_size 128 \
        --layers 3 \
        --per_gpu_train_batch_size 128 \
        --bow
