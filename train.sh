dataset=/apdcephfs/private_jcykcai/esen
python3 /apdcephfs/private_jcykcai/copyisallyouneed/train.py --train_data ${dataset}/train.txt \
        --dev_data ${dataset}/dev.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --retriever ${dataset}/ckpt.pretrain.6layers/epoch19_batch99999_acc0.98 \
        --ckpt ${dataset}/ckpt.rg \
        --world_size 1 \
        --gpus 1 \
        --arch rg \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 4096 \
        --add_retrieval_loss \
        --resume_ckpt ${dataset}/epoch1_batch4999_devbleu52.94


