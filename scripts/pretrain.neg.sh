dataset=/apdcephfs/private_jcykcai/esen/full
python3 /apdcephfs/private_jcykcai/copyisallyouneed/pretrain.py --train_data ${dataset}/train.ir.txt \
        --dev_data ${dataset}/dev.ir.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --ckpt /apdcephfs/share_916081/jcykcai/mt.ckpts/ckpt.pretrain.neg \
        --resume_ckpt ${dataset}/ckpt.pretrain.6layers/epoch19_batch99999_acc0.98 \
        --additional_negs \
        --world_size 1 \
        --gpus 1 \
        --dev_batch_size 128 \
        --layers 6 \
        --per_gpu_train_batch_size 128 \
        --worddrop 0.33
