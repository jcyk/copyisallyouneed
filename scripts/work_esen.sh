set -e

dataset=${MTPATH}/esen
ckpt_prefix=${MTPATH}/mt.ckpts/esen

ckpt=${ckpt_prefix}/transfer/1to1/epoch123_batch92999_devbleu58.99_testbleu58.30
index=${ckpt_prefix}/ckpt.exp.pretrain1.4/epoch78_batch99999_acc0.99
for set in 1.4 2.4 3.4 full; do
    for split in dev test; do
        python3 work.py --load_path ${ckpt} \
        --test_data ${dataset}/${split}.txt \
        --index_path ${index}_${set}
    done
done


ckpt=${ckpt_prefix}/transfer/2to2/epoch54_batch82999_devbleu63.34_testbleu62.75
index=${ckpt_prefix}/ckpt.exp.pretrain2.4/epoch39_batch99999_acc0.99
for set in 2.4 3.4 full; do
    for split in dev test; do
        python3 work.py --load_path ${ckpt} \
        --test_data ${dataset}/${split}.txt \
        --index_path ${index}_${set}
    done
done
