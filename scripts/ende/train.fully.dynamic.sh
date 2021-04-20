dataset=pdcephfs/share_916081/jcykcai/ende
/train.py --train_data ${dataset}/train.txt \
        --dev_data ${dataset}/dev.txt \
        --test_data ${dataset}/test.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --retriever ${MTPATH}/mt.ckpts/ende/ckpt.exp.pretrain/epoch19_batch99999_acc0.97 \
        --ckpt ${MTPATH}/mt.ckpts/ende/ckpt.exp.dynamic.qr \
        --world_size 2 \
        --gpus 2 \
        --arch rg \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 4096 \
        --num_retriever_heads 1 \
        --topk 5 \
        --rebuild_every 3000
