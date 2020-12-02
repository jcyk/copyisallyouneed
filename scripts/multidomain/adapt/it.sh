dataset=/apdcephfs/private_jcykcai/multi_domain/1.4
dev_test_path=/apdcephfs/private_jcykcai/multi_domain
domain=it
python3 /apdcephfs/private_jcykcai/copyisallyouneed/train.py --train_data ${dataset}/train.txt \
        --dev_data ${dev_test_path}/dev/${domain}.dev.txt \
        --test_data ${dev_test_path}/test/${domain}.test.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --retriever /apdcephfs/share_916081/jcykcai/mt.ckpts/multi_domain/ckpt.exp.pretrain/epoch40_batch99999_acc0.80_${domain} \
        --ckpt /apdcephfs/share_916081/jcykcai/mt.ckpts/multi_domain/ckpt.exp.${domain}.dynamic \
        --world_size 2 \
        --gpus 2 \
        --arch rg \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 4096 \
        --num_retriever_heads 1 \
        --topk 5
