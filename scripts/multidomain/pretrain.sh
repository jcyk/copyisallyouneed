dataset=pdcephfs/share_916081/jcykcai/multi_domain
/pretrain.py --train_data ${dataset}/train/train.txt \
        --dev_data ${dataset}/dev.txt \
        --src_vocab ${dataset}/train/src.vocab \
        --tgt_vocab ${dataset}/train/tgt.vocab \
        --ckpt ${MTPATH}/mt.ckpts/multi_domain/ckpt.exp.pretrain \
        --world_size 1 \
        --gpus 1 \
        --dev_batch_size 128 \
        --layers 3 \
        --per_gpu_train_batch_size 128 \
        --bow
