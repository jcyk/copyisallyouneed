dataset=/apdcephfs/private_jcykcai/multi_domain/full
dev_test_path=/apdcephfs/private_jcykcai/multi_domain
python3 /apdcephfs/private_jcykcai/copyisallyouneed/train.py --train_data ${dataset}/train.all.txt \
        --dev_data ${dev_test_path}/dev \
        --test_data ${dev_test_path}/test \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --ckpt /apdcephfs/share_916081/jcykcai/mt.ckpts/multi_domain/full/ckpt.vanilla.2 \
        --world_size 2 \
        --gpus 2 \
        --arch vanilla \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 4096
