
set -e

#directions=(esen deen enes ende)
#ckpts=(epoch78_batch99999_acc0.99 epoch77_batch99999_acc0.97 epoch78_batch99999_acc0.99 epoch77_batch99999_acc0.97)
train=1.4
mem=full
name=btmm1
dir=enes
pt=epoch78_batch99999_acc0.99
dataset=${MTPATH}/${dir}/${train}
retriever=${MTPATH}/mt.ckpts/${dir}/ckpt.exp.pretrain${train}/${pt}_${mem}
ckpt=${MTPATH}/mt.ckpts/${dir}/transfer/${name}
python3 train.py --train_data ${dataset}/bt.beam.train.txt \
        --dev_data ${dataset}/dev.txt \
        --test_data ${dataset}/test.txt \
        --src_vocab ${dataset}/src.vocab \
        --tgt_vocab ${dataset}/tgt.vocab \
        --retriever ${retriever} \
        --ckpt ${ckpt} \
        --world_size 2 \
        --gpus 2 \
        --arch rg \
        --dev_batch_size 2048 \
        --per_gpu_train_batch_size 4096 \
        --num_retriever_heads 1 \
        --topk 5
