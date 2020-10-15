dataset=/apdcephfs/private_jcykcai/esen/2.5
python3 /apdcephfs/private_jcykcai/copyisallyouneed/train.py --train_data ${dataset}/train.txt \
        --dev_data ${dataset}/dev.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --retriever /apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.pretrain.2.5.old/epoch48_batch99999_acc0.98 \
        --ckpt /apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.singlehead.dynamic.2.5.old \
        --world_size 2 \
        --gpus 2 \
        --arch rg \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 4096 \
        --num_retriever_heads 1 \
        --topk 5
