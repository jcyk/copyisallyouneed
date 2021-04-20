
set -e

directions=(ende)
ckpts=(epoch77_batch99999_acc0.97)
train=1.4
mem=1.4
name=1to1
total=${#directions[@]}

for (( i=0; i<${total}; i++ )); do
    dir=${directions[$i]}
    pt=${ckpts[$i]}
    dataset=pdcephfs/share_916081/jcykcai/${dir}/${train}
    retriever=${MTPATH}/mt.ckpts/${dir}/ckpt.exp.pretrain${train}/${pt}_${mem}
    ckpt=${MTPATH}/mt.ckpts/${dir}/transfer/${name}
    /train.py --train_data ${dataset}/train.txt \
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
done
