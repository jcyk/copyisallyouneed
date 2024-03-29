dataset=${MTPATH}/multi_domain
python3 train.py --train_data ${dataset}/train.txt \
        --dev_data ${dataset}/dev \
        --test_data ${dataset}/test \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --retriever ${MTPATH}/mt.ckpts/multi_domain/ckpt.exp.pretrain/epoch40_batch99999_acc0.80 \
        --ckpt ${MTPATH}/mt.ckpts/multi_domain/ckpt.exp.dynamic \
        --world_size 2 \
        --gpus 2 \
        --arch rg \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 4096 \
        --num_retriever_heads 1 \
        --topk 5
