dataset=/apdcephfs/private_jcykcai/esen
python3 /apdcephfs/private_jcykcai/copyisallyouneed/train.py --train_data ${dataset}/train.txt \
        --dev_data ${dataset}/dev.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --retriever ${dataset}/ckpt.pretrain.6layers/epoch19_batch99999_acc0.98 \
        --ckpt /apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.dynamic \
        --world_size 2 \
        --gpus 2 \
        --arch rg \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 4096 \
        --resume_ckpt ${dataset}/epoch1_batch4999_devbleu52.94
