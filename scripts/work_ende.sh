set -e

dataset=/apdcephfs/private_jcykcai/ende
ckpt_prefix=/apdcephfs/share_916081/jcykcai/mt.ckpts/ende

ckpt=${ckpt_prefix}/transfer/1to1/epoch124_batch94999_devbleu49.03_testbleu49.81
index=${ckpt_prefix}/ckpt.exp.pretrain1.4/epoch77_batch99999_acc0.97
for set in 1.4 2.4 3.4 full; do
    for split in dev test; do
        python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
        --test_data ${dataset}/${split}.txt \
        --index_path ${index}_${set}
    done
done


ckpt=${ckpt_prefix}/transfer/2to2/epoch40_batch61999_devbleu53.71_testbleu53.90
index=${ckpt_prefix}/ckpt.exp.pretrain2.4/epoch38_batch99999_acc0.97
for set in 2.4 3.4 full; do
    for split in dev test; do
        python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
        --test_data ${dataset}/${split}.txt \
        --index_path ${index}_${set}
    done
done
