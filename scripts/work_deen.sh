set -e

dataset=pdcephfs/share_916081/jcykcai/deen
ckpt_prefix=${MTPATH}/mt.ckpts/deen

ckpt=${ckpt_prefix}/transfer/1to1/epoch70_batch53999_devbleu54.09_testbleu54.64
index=${ckpt_prefix}/ckpt.exp.pretrain1.4/epoch77_batch99999_acc0.97
for set in 1.4 2.4 3.4 full; do
    for split in dev test; do
        python3 work.py --load_path ${ckpt} \
        --test_data ${dataset}/${split}.txt \
        --index_path ${index}_${set}
    done
done


ckpt=${ckpt_prefix}/transfer/2to2/epoch43_batch65999_devbleu59.39_testbleu60.15
index=${ckpt_prefix}/ckpt.exp.pretrain2.4/epoch38_batch99999_acc0.98
for set in 2.4 3.4 full; do
    for split in dev test; do
        python3 work.py --load_path ${ckpt} \
        --test_data ${dataset}/${split}.txt \
        --index_path ${index}_${set}
    done
done
