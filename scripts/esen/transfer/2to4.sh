dataset=/apdcephfs/private_jcykcai/esen/2.4
python3 /apdcephfs/private_jcykcai/copyisallyouneed/train.py --train_data ${dataset}/train.txt \
        --dev_data ${dataset}/dev.txt \
        --test_data ${dataset}/test.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --retriever /apdcephfs/share_916081/jcykcai/mt.ckpts/esen/ckpt.exp.pretrain2.4/epoch39_batch99999_acc0.99_full \
        --ckpt /apdcephfs/share_916081/jcykcai/mt.ckpts/esen/transfer/2to4 \
        --world_size 2 \
        --gpus 2 \
        --arch rg \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 4096 \
        --num_retriever_heads 1 \
        --topk 5
