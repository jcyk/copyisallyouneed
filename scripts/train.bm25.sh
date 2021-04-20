for direction in enes esen deen ende; do
dataset=pdcephfs/share_916081/jcykcai/${direction}
/train.py --train_data ${dataset}/train.bm25.txt \
        --dev_data ${dataset}/dev.bm25.txt \
        --test_data ${dataset}/test.bm25.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --ckpt ${MTPATH}/mt.ckpts/${direction}/ckpt.bm25 \
        --world_size 2 \
        --gpus 2 \
        --arch mem \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 4096
done