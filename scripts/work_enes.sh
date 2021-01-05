set -e

dataset=/apdcephfs/private_jcykcai/enes
ckpt_prefix=/apdcephfs/share_916081/jcykcai/mt.ckpts/enes

#ckpt=${ckpt_prefix}/transfer/1to1/epoch101_batch75999_devbleu56.54_testbleu55.96
#index=${ckpt_prefix}/ckpt.exp.pretrain1.4/epoch78_batch99999_acc0.99
#for set in 1.4 2.4 3.4 full; do
#    for split in dev test; do
#        python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
#        --test_data ${dataset}/${split}.txt \
#        --index_path ${index}_${set}
#    done
#done


ckpt=${ckpt_prefix}/transfer/2to2/epoch28_batch42999_devbleu60.34_testbleu59.67
index=${ckpt_prefix}/ckpt.exp.pretrain2.4/epoch39_batch99999_acc0.99
for set in 2.4 3.4 full; do
    for split in dev test; do
        python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
        --test_data ${dataset}/${split}.txt \
        --index_path ${index}_${set}
    done
done
