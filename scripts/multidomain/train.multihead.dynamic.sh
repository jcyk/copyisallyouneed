dataset=pdcephfs/share_916081/jcykcai/multi_domain/train
dev_test_path=pdcephfs/share_916081/jcykcai/multi_domain
/train.py --train_data ${dataset}/train.txt \
        --dev_data ${dev_test_path}/dev \
        --test_data ${dev_test_path}/test \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --retriever ${MTPATH}/mt.ckpts/multi_domain/ckpt.exp.pretrain/epoch40_batch99999_acc0.80_train \
        --ckpt ${MTPATH}/mt.ckpts/multi_domain/ckpt.exp.dynamic \
        --world_size 2 \
        --gpus 2 \
        --arch rg \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 4096 \
        --num_retriever_heads 1 \
        --topk 5
