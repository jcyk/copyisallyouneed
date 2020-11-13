dataset=/apdcephfs/private_jcykcai/deen
python3 /apdcephfs/private_jcykcai/copyisallyouneed/train.py --train_data ${dataset}/train.bm25.txt \
        --dev_data ${dataset}/dev.bm25.txt \
        --test_data ${dataset}/test.bm25.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --ckpt /apdcephfs/share_916081/jcykcai/mt.ckpts/deen/ckpt.bm25 \
        --world_size 2 \
        --gpus 2 \
        --arch mem \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 4096
