dataset=/apdcephfs/private_jcykcai/esen/full
python3 /apdcephfs/private_jcykcai/copyisallyouneed/train.py --train_data ${dataset}/train.txt \
        --dev_data ${dataset}/dev.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --retriever /apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.exp.pretrain/exp7/epoch19_batch99999_acc0.99 \
        --ckpt /apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.exp7.translation \
        --world_size 2 \
        --gpus 2 \
        --arch rg \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 4096 \
        --num_retriever_heads 1 \
        --topk 5
#${dataset}/ckpt.pretrain.6layers/epoch19_batch99999_acc0.98 
