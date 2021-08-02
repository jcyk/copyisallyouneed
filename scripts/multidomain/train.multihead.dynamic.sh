dataset=${MTPATH}/multi_domain/train
<<<<<<< HEAD
dev_test_path=pdcephfs/share_916081/jcykcai/multi_domain
=======
dev_test_path=${MTPATH}/multi_domain
>>>>>>> 7e5fbc0a6b1e6326dce4b3e540e255ebc4af9485
python3 train.py --train_data ${dataset}/train.txt \
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
