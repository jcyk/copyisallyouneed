dataset=pdcephfs/share_916081/jcykcai/multi_domain
/train.py --train_data ${dataset}/train/it.train.txt \
        --dev_data ${dataset}/dev/it.dev.txt \
        --test_data ${dataset}/test/it.test.txt \
        --src_vocab ${dataset}/train/src.vocab \
        --tgt_vocab ${dataset}/train/tgt.vocab \
        --ckpt ${MTPATH}/mt.ckpts/multi_domain/ckpt.vanilla.it \
        --world_size 2 \
        --gpus 2 \
        --arch vanilla \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 4096
