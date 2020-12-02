dataset=/apdcephfs/private_jcykcai/multi_domain/2.4
dev_test_path=/apdcephfs/private_jcykcai/multi_domain
python3 /apdcephfs/private_jcykcai/copyisallyouneed/train.py --train_data ${dataset}/train.txt \
        --dev_data ${dev_test_path}/dev \
        --test_data ${dev_test_path}/test \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --retriever /apdcephfs/share_916081/jcykcai/mt.ckpts/multi_domain/2.4/ckpt.exp.pretrain/epoch20_batch99999_acc0.81 \
        --ckpt /apdcephfs/share_916081/jcykcai/mt.ckpts/multi_domain/2.4/ckpt.exp.dynamic \
        --world_size 2 \
        --gpus 2 \
        --arch rg \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 4096 \
        --num_retriever_heads 1 \
        --topk 5
