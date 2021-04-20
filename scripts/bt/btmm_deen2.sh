
set -e

#directions=(esen deen enes ende)
#ckpts=(epoch39_batch99999_acc0.99 epoch38_batch99999_acc0.98 epoch39_batch99999_acc0.99 eepoch38_batch99999_acc0.97)
train=2.4
mem=full
name=btmm2
dir=deen
pt=epoch38_batch99999_acc0.98
dataset=pdcephfs/share_916081/jcykcai/${dir}/${train}
retriever=${MTPATH}/mt.ckpts/${dir}/ckpt.exp.pretrain${train}/${pt}_${mem}
ckpt=${MTPATH}/mt.ckpts/${dir}/transfer/${name}
/train.py --train_data ${dataset}/bt.beam.train.txt \
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

