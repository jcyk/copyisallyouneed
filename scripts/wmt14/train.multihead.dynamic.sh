dataset=/apdcephfs/private_jcykcai/wmt14_gl
python3 /apdcephfs/private_jcykcai/copyisallyouneed/train.py --train_data ${dataset}/train.txt \
        --dev_data ${dataset}/dev.txt \
        --test_data ${dataset}/test.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --retriever /apdcephfs/share_916081/jcykcai/mt.ckpts/wmt14/ckpt.exp.pretrain.gl/epoch3_batch99999_acc0.97 \
        --ckpt /apdcephfs/share_916081/jcykcai/mt.ckpts/wmt14/ckpt.exp.dynamic.gl \
        --world_size 2 \
        --gpus 2 \
        --arch rg \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 4096 \
        --num_retriever_heads 1 \
        --topk 5
