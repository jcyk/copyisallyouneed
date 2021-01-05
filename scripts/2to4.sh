
set -e

directions=(esen deen enes ende)
ckpts=(epoch39_batch99999_acc0.99 epoch38_batch99999_acc0.98 epoch39_batch99999_acc0.99 eepoch38_batch99999_acc0.97)
train=2.4
mem=full
name=2to4
total=${#directions[@]}

for (( i=0; i<${total}; i++ )); do
    dir=${directions[$i]}
    pt=${ckpts[$i]}
    dataset=/apdcephfs/private_jcykcai/${dir}/${train}
    retriever=/apdcephfs/share_916081/jcykcai/mt.ckpts/${dir}/ckpt.exp.pretrain${train}/${pt}_${mem}
    ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/${dir}/transfer/${name}
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/train.py --train_data ${dataset}/train.txt \
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
