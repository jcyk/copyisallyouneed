dataset=/apdcephfs/private_jcykcai/fren
python3 /apdcephfs/private_jcykcai/copy/train.py --train_data ${dataset}/train.txt \
        --dev_data ${dataset}/dev.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --ckpt ${dataset}/ckpt.2gpus \
        --world_size 2 \
        --gpus 2
