set -e

### Multidomain testing###


dataset=/apdcephfs/private_jcykcai/multi_domain
ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/multi_domain/2.4/ckpt.vanilla/epoch46_batch85999_devbleu36.95_testbleu37.37

#for domain in medical law it koran subtitles; do
#    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
#        --test_data ${dataset}/${domain}/test.txt \
#        --output_path ${dataset}/vanilla.${domain}.test.out.txt
#done


ckpt=/apdcephfs/share_916081/jcykcai/mt.ckpts/multi_domain/2.4/ckpt.exp.dynamic/epoch26_batch48999_devbleu37.55_testbleu38.19

for domain in medical law it koran subtitles; do
    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
        --test_data ${dataset}/${domain}/test.txt \
        --output_path ${dataset}/dynamic.${domain}.test.out.txt
done


#for domain in medical law it koran subtitles; do
#    python3 /apdcephfs/private_jcykcai/copyisallyouneed/work.py --load_path ${ckpt} \
#        --index_path /apdcephfs/share_916081/jcykcai/mt.ckpts/multi_domain/ckpt.exp.pretrain/epoch40_batch99999_acc0.80_full \
#        --test_data ${dataset}/${domain}/test.txt \
#        --output_path ${dataset}/dynamic.full.${domain}.test.out.txt
#done
