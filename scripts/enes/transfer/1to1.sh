dataset=/apdcephfs/private_jcykcai/enes/1.4
python3 /apdcephfs/private_jcykcai/copyisallyouneed/train.py --train_data ${dataset}/train.txt \
        --dev_data ${dataset}/dev.txt \
        --test_data ${dataset}/test.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --retriever /apdcephfs/share_916081/jcykcai/mt.ckpts/enes/ckpt.exp.pretrain1.4/epoch78_batch99999_acc0.99_1.4 \
        --ckpt /apdcephfs/share_916081/jcykcai/mt.ckpts/enes/transfer/1to1 \
        --world_size 2 \
        --gpus 2 \
        --arch rg \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 4096 \
        --num_retriever_heads 1 \
        --topk 5
