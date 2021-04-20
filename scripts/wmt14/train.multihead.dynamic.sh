dataset=${MTPATH}/wmt14_gl
python3 train.py --train_data ${dataset}/train.txt \
        --dev_data ${dataset}/dev.txt \
        --test_data ${dataset}/test.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --retriever ${MTPATH}/mt.ckpts/wmt14/ckpt.exp.pretrain/epoch3_batch99999_acc0.97_ext1 \
        --ckpt ${MTPATH}/mt.ckpts/wmt14/ckpt.exp.dynamic.ext1 \
        --world_size 4 \
        --gpus 4 \
        --arch rg \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 2048 \
        --num_retriever_heads 1 \
        --topk 5
