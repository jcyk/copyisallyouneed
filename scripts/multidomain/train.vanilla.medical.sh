dataset=${MTPATH}/multi_domain
python3 train.py --train_data ${dataset}/train/medical.train.txt \
        --dev_data ${dataset}/dev/medical.dev.txt \
        --test_data ${dataset}/test/medical.test.txt \
        --src_vocab ${dataset}/train/src.vocab \
        --tgt_vocab ${dataset}/train/tgt.vocab \
        --ckpt ${MTPATH}/mt.ckpts/multi_domain/ckpt.vanilla.medical \
        --world_size 2 \
        --gpus 2 \
        --arch vanilla \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 4096
